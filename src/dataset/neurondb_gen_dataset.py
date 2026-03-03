#!/usr/bin/env python3

"""
Simple dataset ingestor: creates schemas, downloads text datasets from stable URLs,
and imports them line-by-line into a single TEXT column using COPY.

Config via environment variables (defaults shown):
  PGHOST=localhost PGPORT=5432 PGUSER=$USER PGPASSWORD= PGDATABASE=postgres

Datasets:
  - SQUAD_V1: train-v1.1.json
  - AG_NEWS:  train.csv
  - WIKITEXT2: train.txt
  - MOBY_DICK: pg2701.txt

Note: This script avoids binary formats to keep the import generic and robust.
"""

import os
import sys
from datetime import datetime
import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Optional

import requests
import psycopg2
from psycopg2 import sql


def log(ok: bool, msg: str) -> None:
    tick = "✔" if ok else "✘"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tick} {msg}", flush=True)


# ---------- DB helpers ----------
def get_conn(dbname: Optional[str] = None) -> psycopg2.extensions.connection:
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    user = os.getenv("PGUSER") or os.getenv("USER") or os.getenv("LOGNAME")
    password = os.getenv("PGPASSWORD")
    database = dbname or os.getenv("PGDATABASE", "postgres")
    conn = psycopg2.connect(dbname=database, user=user, password=password, host=host, port=port)
    try:
        conn.set_client_encoding('UTF8')
    except Exception:
        pass
    return conn


def admin_drop_database(dbname: str) -> None:
    """Drop a database using admin connection (PGDB_ADMIN or 'postgres')."""
    admin_db = os.getenv("PGDB_ADMIN", "postgres")
    conn = None
    try:
        conn = get_conn(admin_db)
        try:
            conn.set_session(autocommit=True)
        except Exception:
            conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
            if cur.fetchone():
                # Terminate existing connections to allow DROP
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid();
                    """,
                    (dbname,)
                )
                cur.execute(psycopg2.sql.SQL("DROP DATABASE {};").format(psycopg2.sql.Identifier(dbname)))
                log(True, f"Dropped database '{dbname}'")
            else:
                log(True, f"Database '{dbname}' does not exist (skip drop)")
    except Exception as e:
        log(False, f"Failed to drop database '{dbname}': {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def admin_create_database(dbname: str, owner: Optional[str] = None) -> None:
    """Create a database using admin connection if it doesn't already exist."""
    admin_db = os.getenv("PGDB_ADMIN", "postgres")
    conn = None
    try:
        conn = get_conn(admin_db)
        # Ensure CREATE DATABASE runs outside a transaction
        try:
            conn.set_session(autocommit=True)
        except Exception:
            conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
            if cur.fetchone():
                log(True, f"Database '{dbname}' already exists (skip create)")
                return
            if owner:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} WITH OWNER {};")
                    .format(sql.Identifier(dbname), sql.Identifier(owner))
                )
            else:
                cur.execute(
                    sql.SQL("CREATE DATABASE {};").format(sql.Identifier(dbname))
                )
            log(True, f"Created database '{dbname}'")
    except Exception as e:
        log(False, f"Failed to create database '{dbname}': {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def ensure_schema_and_table(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.data (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT
                );
                """
            ).format(sql.Identifier(schema))
        )
    conn.commit()
    log(True, f"Schema/table ready: {schema}")


def ensure_ms_marco(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};" ).format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.data (
                    docid TEXT PRIMARY KEY,
                    content TEXT
                );
                """
            ).format(sql.Identifier(schema))
        )
    conn.commit()
    log(True, f"Schema/table ready: {schema}.data (MS MARCO)")


def ensure_wiki_embeddings(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};" ).format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.items (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    text TEXT,
                    embedding REAL[]
                );
                """
            ).format(sql.Identifier(schema))
        )
    conn.commit()
    log(True, f"Schema/table ready: {schema}.items (wiki embeddings)")


def ensure_hotpotqa(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};" ).format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.qa (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    question TEXT,
                    context TEXT,
                    answer TEXT
                );
                """
            ).format(sql.Identifier(schema))
        )
    conn.commit()
    log(True, f"Schema/table ready: {schema}.qa (HotpotQA)")


def ensure_vectors(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};" ).format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.vectors (
                    id BIGINT PRIMARY KEY,
                    embedding REAL[]
                );
                """
            ).format(sql.Identifier(schema))
        )
    conn.commit()
    log(True, f"Schema/table ready: {schema}.vectors (vector embeddings)")


def ensure_datasets_catalog(conn) -> None:
    """Create a public.datasets catalog to track dataset metadata."""
    ddl = """
    CREATE TABLE IF NOT EXISTS public.datasets (
        name TEXT PRIMARY KEY,
        url TEXT,
        purpose TEXT,
        schema_name TEXT,
        rows BIGINT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    log(True, "Catalog table ensured: public.datasets")


# ---------- Download ----------
def download_file(url: str, dest: Path, retries: int = 2) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "pge-datasets/1.0"}
    # skip if already downloaded and non-empty
    try:
        if dest.exists() and dest.stat().st_size > 0:
            log(True, f"Already downloaded, skipping: {dest}")
            return True
    except Exception:
        pass
    attempt = 0
    while attempt <= retries:
        try:
            log(False, f"Downloading {url} -> {dest} (attempt {attempt+1})")
            with requests.get(url, stream=True, timeout=(15, 180), allow_redirects=True, headers=headers) as r:
                if r.status_code != 200:
                    log(False, f"HTTP {r.status_code} for {url}")
                    attempt += 1
                    continue
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            log(True, f"Saved: {dest}")
            return True
        except Exception as e:
            log(False, f"Download error for {url}: {e}")
            attempt += 1
    return False


# ---------- Import ----------
def import_file(conn, schema: str, path: Path) -> bool:
    try:
        with conn.cursor() as cur, open(path, "r", encoding="utf-8", errors="replace") as f:
            cur.copy_expert(
                sql.SQL("COPY {}.data(content) FROM STDIN").format(sql.Identifier(schema)),
                f,
            )
        conn.commit()
        log(True, f"Imported into {schema} from {path}")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"Import error for {schema}: {e}")
        return False


# ---------- Advanced dataset loaders ----------
def load_msmarco_collection(conn, schema: str, archive_path: Path, limit: int = 50000) -> bool:
    """Load MS MARCO collection.tar.gz (expects a collection.tsv inside: docid\ttext)."""
    try:
        ensure_ms_marco(conn, schema)
        inserted = 0
        with tarfile.open(archive_path, mode="r:gz") as tar:
            # Heuristic: find first .tsv entry named collection.tsv
            member = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith("collection.tsv")), None)
            if not member:
                log(False, f"collection.tsv not found in {archive_path}")
                return False
            f = tar.extractfile(member)
            assert f is not None
            batch = []
            with conn.cursor() as cur:
                for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                    parts = raw.rstrip("\n").split("\t", 1)
                    if len(parts) != 2:
                        continue
                    docid, text = parts
                    batch.append((docid, text))
                    if len(batch) >= 1000:
                        cur.executemany(
                            sql.SQL("INSERT INTO {}.data(docid, content) VALUES (%s, %s) ON CONFLICT (docid) DO NOTHING;")
                            .format(sql.Identifier(schema)),
                            batch,
                        )
                        inserted += len(batch)
                        batch.clear()
                    if inserted >= limit:
                        break
                if batch:
                    cur.executemany(
                        sql.SQL("INSERT INTO {}.data(docid, content) VALUES (%s, %s) ON CONFLICT (docid) DO NOTHING;")
                        .format(sql.Identifier(schema)),
                        batch,
                    )
                    inserted += len(batch)
            conn.commit()
        log(True, f"MS MARCO loaded: {inserted} docs into {schema}.data")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"MS MARCO load failed: {e}")
        return False


def load_msmarco_hf(conn, schema: str, limit: int = 50000) -> bool:
    """Fallback: stream MS MARCO corpus from Hugging Face and insert into schema.data."""
    try:
        ensure_ms_marco(conn, schema)
        try:
            # Preferred: sentence-transformers/msmarco, 'corpus' config
            from datasets import load_dataset  # lazy import
            ds = load_dataset("sentence-transformers/msmarco", "corpus", split="train", streaming=True)
        except Exception:
            try:
                # Alternative split name sometimes used
                from datasets import load_dataset  # lazy import
                ds = load_dataset("sentence-transformers/msmarco", "corpus", split="corpus", streaming=True)
            except Exception as e2:
                log(False, f"HF load_dataset failed for MS MARCO: {e2}")
                return False
        inserted = 0
        batch = []
        with conn.cursor() as cur:
            for ex in ds:
                docid = str(ex.get("corpus_id") or ex.get("id") or inserted)
                text = ex.get("text") or ex.get("document") or ex.get("passage_text")
                if not text:
                    continue
                batch.append((docid, text))
                if len(batch) >= 1000:
                    cur.executemany(
                        sql.SQL("INSERT INTO {}.data(docid, content) VALUES (%s, %s) ON CONFLICT (docid) DO NOTHING;")
                        .format(sql.Identifier(schema)),
                        batch,
                    )
                    inserted += len(batch)
                    batch.clear()
                if inserted >= limit:
                    break
            if batch:
                cur.executemany(
                    sql.SQL("INSERT INTO {}.data(docid, content) VALUES (%s, %s) ON CONFLICT (docid) DO NOTHING;")
                    .format(sql.Identifier(schema)),
                    batch,
                )
                inserted += len(batch)
        conn.commit()
        log(True, f"MS MARCO (HF) loaded: {inserted} docs into {schema}.data")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"MS MARCO (HF) load failed: {e}")
        return False


def load_wikipedia_ndjson_gz(conn, schema: str, gz_path: Path, limit: int = 50000) -> bool:
    """Load NDJSON.GZ with fields id, title, text, embedding (array of floats)."""
    try:
        ensure_wiki_embeddings(conn, schema)
        inserted = 0
        with gzip.open(gz_path, mode="rt", encoding="utf-8", errors="replace") as f:
            batch = []
            with conn.cursor() as cur:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    row_id = str(obj.get("id") or obj.get("_id") or obj.get("uuid") or inserted)
                    title = obj.get("title")
                    text = obj.get("text") or obj.get("content")
                    emb = obj.get("embedding") or obj.get("vector")
                    if emb is not None and not isinstance(emb, list):
                        # best-effort normalization
                        try:
                            emb = list(emb)
                        except Exception:
                            emb = None
                    batch.append((row_id, title, text, emb))
                    if len(batch) >= 1000:
                        cur.executemany(
                            sql.SQL("""
                                INSERT INTO {}.items(id, title, text, embedding)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (id) DO NOTHING;
                            """)
                            .format(sql.Identifier(schema)),
                            batch,
                        )
                        inserted += len(batch)
                        batch.clear()
                    if inserted >= limit:
                        break
                if batch:
                    cur.executemany(
                        sql.SQL("""
                            INSERT INTO {}.items(id, title, text, embedding)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING;
                        """)
                        .format(sql.Identifier(schema)),
                        batch,
                    )
                    inserted += len(batch)
            conn.commit()
        log(True, f"Wikipedia embeddings loaded: {inserted} rows into {schema}.items")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"Wikipedia embeddings load failed: {e}")
        return False


def load_hotpotqa_json(conn, schema: str, json_path: Path, limit: int = 50000) -> bool:
    try:
        ensure_hotpotqa(conn, schema)
        max_rows = int(os.getenv("MAX_ROWS", str(limit)))
        inserted = 0
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if not isinstance(data, list):
            log(False, f"Unexpected HotpotQA format (list expected)")
            return False
        batch = []
        with conn.cursor() as cur:
            for i, obj in enumerate(data):
                if inserted >= max_rows:
                    break
                qid = str(obj.get("_id") or obj.get("id") or i)
                title = None
                if isinstance(obj.get("supporting_facts"), list) and obj["supporting_facts"]:
                    # try to derive a title from supporting facts
                    sf = obj["supporting_facts"][0]
                    if isinstance(sf, list) and len(sf) > 0:
                        title = str(sf[0])
                question = obj.get("question")
                answer = obj.get("answer")
                # Concatenate context paragraphs if present
                context_text = None
                if isinstance(obj.get("context"), list):
                    parts = []
                    for c in obj["context"]:
                        if isinstance(c, list) and len(c) >= 2:
                            parts.append(f"{c[0]}: {' '.join(map(str, c[1]))}")
                    if parts:
                        context_text = "\n".join(parts)
                batch.append((qid, title, question, context_text, answer))
                if len(batch) >= 1000:
                    cur.executemany(
                        sql.SQL("""
                            INSERT INTO {}.qa(id, title, question, context, answer)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING;
                        """)
                        .format(sql.Identifier(schema)),
                        batch,
                    )
                    inserted += len(batch)
                    batch.clear()
            if batch:
                cur.executemany(
                    sql.SQL("""
                        INSERT INTO {}.qa(id, title, question, context, answer)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING;
                    """)
                    .format(sql.Identifier(schema)),
                    batch,
                )
                inserted += len(batch)
        conn.commit()
        log(True, f"HotpotQA loaded: {inserted} rows into {schema}.qa")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"HotpotQA load failed: {e}")
        return False


def load_sift_hdf5(conn, schema: str, h5_path: Path, limit: int = 100000) -> bool:
    try:
        import numpy as np  # noqa: F401
        import h5py  # type: ignore
    except Exception as e:
        log(False, f"Missing numpy/h5py for HDF5 loaders: {e}")
        return False
    try:
        ensure_vectors(conn, schema)
        max_rows = int(os.getenv("MAX_ROWS", str(limit)))
        inserted = 0
        with h5py.File(h5_path, "r") as h5:
            ds = h5.get("train") or h5.get("features")
            if ds is None:
                log(False, "No 'train' or 'features' dataset in HDF5")
                return False
            dim = ds.shape[1]
            with conn.cursor() as cur:
                batch = []
                for idx in range(min(ds.shape[0], max_rows)):
                    vec = ds[idx].astype(float).tolist()
                    batch.append((idx, vec))
                    if len(batch) >= 1000:
                        cur.executemany(
                            sql.SQL("INSERT INTO {}.vectors(id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;")
                            .format(sql.Identifier(schema)),
                            batch,
                        )
                        inserted += len(batch)
                        batch.clear()
                if batch:
                    cur.executemany(
                        sql.SQL("INSERT INTO {}.vectors(id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;")
                        .format(sql.Identifier(schema)),
                        batch,
                    )
                    inserted += len(batch)
        conn.commit()
        log(True, f"SIFT HDF5 loaded: {inserted} vectors into {schema}.vectors")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"SIFT HDF5 load failed: {e}")
        return False


def load_deep1b_hdf5(conn, schema: str, h5_path: Path, limit: int = 100000) -> bool:
    try:
        import numpy as np  # noqa: F401
        import h5py  # type: ignore
    except Exception as e:
        log(False, f"Missing numpy/h5py for HDF5 loaders: {e}")
        return False
    try:
        ensure_vectors(conn, schema)
        max_rows = int(os.getenv("MAX_ROWS", str(limit)))
        inserted = 0
        with h5py.File(h5_path, "r") as h5:
            ds = h5.get("train") or h5.get("features")
            if ds is None:
                log(False, "No 'train' or 'features' dataset in HDF5")
                return False
            with conn.cursor() as cur:
                batch = []
                for idx in range(min(ds.shape[0], max_rows)):
                    vec = ds[idx].astype(float).tolist()
                    batch.append((idx, vec))
                    if len(batch) >= 1000:
                        cur.executemany(
                            sql.SQL("INSERT INTO {}.vectors(id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;")
                            .format(sql.Identifier(schema)),
                            batch,
                        )
                        inserted += len(batch)
                        batch.clear()
                if batch:
                    cur.executemany(
                        sql.SQL("INSERT INTO {}.vectors(id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;")
                        .format(sql.Identifier(schema)),
                        batch,
                    )
                    inserted += len(batch)
        conn.commit()
        log(True, f"DEEP1B HDF5 loaded: {inserted} vectors into {schema}.vectors")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"DEEP1B HDF5 load failed: {e}")
        return False


# ---------- Indexes and utility ----------
def create_fts_index(conn, schema: str) -> bool:
    try:
        idx_name = f"idx_{schema}_data_fts"
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.data USING GIN (to_tsvector('english', content));")
                .format(sql.Identifier(idx_name), sql.Identifier(schema))
            )
        conn.commit()
        log(True, f"FTS index ensured: {schema}.{idx_name}")
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        log(False, f"FTS index creation failed for {schema}: {e}")
        return False


def count_rows(conn, schema: str) -> int:
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}.data").format(sql.Identifier(schema)))
            (cnt,) = cur.fetchone()
            return int(cnt)
    except Exception:
        return -1


def count_rows_any(conn, schema: str) -> int:
    """Count rows from a known table in schema: prefers items, data, qa, vectors (in that order)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = %s AND tablename IN ('items','data','qa','vectors')
                    ORDER BY CASE tablename
                        WHEN 'items' THEN 0
                        WHEN 'data' THEN 1
                        WHEN 'qa' THEN 2
                        WHEN 'vectors' THEN 3
                        ELSE 9 END
                    LIMIT 1
                """),
                (schema,)
            )
            row = cur.fetchone()
            if not row:
                return -1
            table = row[0]
            cur.execute(
                sql.SQL("SELECT count(*) FROM {}.{}")
                .format(sql.Identifier(schema), sql.Identifier(table))
            )
            (cnt,) = cur.fetchone()
            return int(cnt)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return -1


def main() -> int:
    # 1) Normalize: drop both variants if they exist, then create target DB 'nurondb_dataset'
    admin_drop_database("nuerondb_dataset")
    admin_drop_database("nurondb_dataset")

    owner = os.getenv("PGUSER") or os.getenv("USER") or os.getenv("LOGNAME")
    admin_create_database("nurondb_dataset", owner=owner)

    datasets = [
        # Text datasets (line-based into schema.data)
        {"name": "SQUAD_V1", "schema": "squad_v1", "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json", "filename": "train-v1.1.json", "loader": "text", "purpose": "Question answering (JSON)"},
        {"name": "TINY_SHAKESPEARE", "schema": "tiny_shakespeare", "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "filename": "input.txt", "loader": "text", "purpose": "Language modeling"},
        {"name": "WIKITEXT2", "schema": "wikitext2", "url": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt", "filename": "train.txt", "loader": "text", "purpose": "Language modeling"},
        {"name": "MOBY_DICK", "schema": "moby_dick", "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt", "filename": "pg2701.txt", "loader": "text", "purpose": "Classic literature text"},
        # Other datasets at same level
        {"name": "MS_MARCO", "schema": "ms_marco", "url": "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz", "filename": "collection.tar.gz", "loader": "msmarco", "purpose": "Passage ranking corpus (TSV in tar.gz)"},
        {"name": "Wikipedia_Embeddings", "schema": "wikipedia_embeddings", "url": "https://huggingface.co/datasets/Supabase/wikipedia-en-embeddings/resolve/main/wiki_minilm.ndjson.gz", "filename": "wiki_minilm.ndjson.gz", "loader": "wiki_embeddings", "purpose": "Wikipedia text + MiniLM embeddings (NDJSON.GZ)"},
        {"name": "HotpotQA", "schema": "hotpotqa", "url": "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.json", "filename": "hotpot_train_v1.json", "loader": "hotpotqa", "purpose": "Multi-hop QA training set (JSON)"},
        {"name": "SIFT1M", "schema": "sift1m", "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5", "filename": "sift-128-euclidean.hdf5", "loader": "hdf5_sift", "purpose": "SIFT1M 128-d vectors (HDF5)"},
        {"name": "DEEP1B", "schema": "deep1b", "url": "https://ann-benchmarks.com/deep-image-96-angular.hdf5", "filename": "deep-image-96-angular.hdf5", "loader": "hdf5_deep", "purpose": "Deep1B 96-d vectors subset (HDF5)"},
    ]

    data_root = Path(os.getenv("DATA_ROOT", "datasets")).resolve()

    # Connect once to the target DB we just created
    try:
        conn = get_conn("nurondb_dataset")
    except Exception as e:
        log(False, f"DB connection failed: {e}")
        return 1

    # Prepare catalog and schemas/tables for text datasets; others will be ensured by loaders
    for ds in datasets:
        if ds["loader"] == "text":
            ensure_schema_and_table(conn, ds["schema"])
    ensure_datasets_catalog(conn)
    # Ensure empty schemas for non-text datasets; specific tables created by loaders
    with conn.cursor() as cur:
        for ds in datasets:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};" ).format(sql.Identifier(ds["schema"])))
    conn.commit()

    # Download and ingest all datasets according to their loader
    for ds in datasets:
        schema = ds["schema"]
        url = ds["url"]
        dest = data_root / ds["name"].replace(" ", "_") / ds["filename"]
        downloaded = download_file(url, dest)
        ok = False
        if ds["loader"] == "text":
            if downloaded and import_file(conn, schema, dest):
                # FTS index best-effort
                create_fts_index(conn, schema)
                ok = True
        elif ds["loader"] == "msmarco":
            if downloaded:
                ok = load_msmarco_collection(conn, schema, dest)
            if not ok:
                ok = load_msmarco_hf(conn, schema)
        elif ds["loader"] == "wiki_embeddings":
            if downloaded:
                ok = load_wikipedia_ndjson_gz(conn, schema, dest)
        elif ds["loader"] == "hotpotqa":
            if downloaded:
                ok = load_hotpotqa_json(conn, schema, dest)
        elif ds["loader"] == "hdf5_sift":
            if os.getenv("SKIP_HDF5", "false").lower() in {"1","true","yes","on"}:
                log(True, f"Skipping HDF5 load for {ds['name']} due to SKIP_HDF5")
                ok = True
            elif downloaded:
                ok = load_sift_hdf5(conn, schema, dest)
        elif ds["loader"] == "hdf5_deep":
            if os.getenv("SKIP_HDF5", "false").lower() in {"1","true","yes","on"}:
                log(True, f"Skipping HDF5 load for {ds['name']} due to SKIP_HDF5")
                ok = True
            elif downloaded:
                ok = load_deep1b_hdf5(conn, schema, dest)
        else:
            log(False, f"Unknown loader for {ds['name']}, skipped")
        # Catalog update
        if ok:
            cnt = count_rows_any(conn, schema)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO public.datasets(name, url, purpose, schema_name, rows, updated_at)
                            VALUES (%s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (name) DO UPDATE SET
                              url = EXCLUDED.url,
                              purpose = EXCLUDED.purpose,
                              schema_name = EXCLUDED.schema_name,
                              rows = EXCLUDED.rows,
                              updated_at = NOW()
                            """
                        ),
                        (ds["name"], url, ds.get("purpose"), schema, cnt)
                    )
                    conn.commit()
                    log(True, f"Catalog updated: {ds['name']} rows={cnt}")
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                log(False, f"Failed to update catalog for {ds['name']}: {e}")

    # Show dataset tables at the end
    try:
        with conn.cursor() as cur:
            # List all dataset schemas
            all_schemas = [d["schema"] for d in datasets]
            if all_schemas:
                placeholders = ", ".join(["%s"] * len(all_schemas))
                cur.execute(
                    f"""
                    SELECT schemaname, tablename
                    FROM pg_tables
                    WHERE schemaname IN ({placeholders})
                    ORDER BY schemaname, tablename
                    """,
                    tuple(all_schemas)
                )
            else:
                cur.execute("SELECT schemaname, tablename FROM pg_tables WHERE false")
            rows = cur.fetchall()
            for s, t in rows:
                log(True, f"table: {s}.{t}")
    except Exception as e:
        log(False, f"Failed listing tables: {e}")

    conn.close()
    log(True, "Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
 

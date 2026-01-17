#!/usr/bin/env python3

"""
Enhanced NeurondB Dataset Generator
====================================
Comprehensive dataset loader for NeurondB regression testing

Supports:
  - MS MARCO passages (document retrieval)
  - Wikipedia embeddings (general knowledge)
  - HotpotQA (question answering)
  - SIFT/Deep1B vectors (high-dimensional benchmarks)
  - Synthetic test data generation

Usage:
  python3 neurondb_dataset.py --recreate-db
  python3 neurondb_dataset.py --load-msmarco --limit 10000
  python3 neurondb_dataset.py --load-all
  python3 neurondb_dataset.py --show-stats
"""

import argparse
import os
import sys
from pathlib import Path

# Version
__version__ = "2.1.0"

# Import all functions from the original neurondb_gen_dataset.py
# We'll import it as a module
sys.path.insert(0, str(Path(__file__).parent))
try:
    import neurondb_gen_dataset as gd
except ImportError:
    print("Error: Could not import neurondb_gen_dataset.py", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Load datasets for NeurondB regression testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recreate database
  %(prog)s --recreate-db

  # Load specific datasets
  %(prog)s --load-msmarco --limit 10000
  %(prog)s --load-wikipedia --limit 5000
  
  # Load all datasets
  %(prog)s --load-all

  # Create synthetic test data
  %(prog)s --create-synthetic

  # Show statistics
  %(prog)s --show-stats
        """
    )

    # Database operations
    parser.add_argument('--recreate-db', action='store_true',
                        help='Drop and recreate the test database')
    parser.add_argument('--dbname', default='neurondb_test',
                        help='Database name (default: neurondb_test)')

    # Dataset loading
    parser.add_argument('--load-all', action='store_true',
                        help='Load all available datasets')
    parser.add_argument('--load-msmarco', action='store_true',
                        help='Load MS MARCO passages dataset')
    parser.add_argument('--load-wikipedia', action='store_true',
                        help='Load Wikipedia embeddings dataset')
    parser.add_argument('--load-hotpotqa', action='store_true',
                        help='Load HotpotQA dataset')
    parser.add_argument('--load-sift', action='store_true',
                        help='Load SIFT1M vectors dataset')
    parser.add_argument('--load-deep1b', action='store_true',
                        help='Load Deep1B vectors dataset')

    # Options
    parser.add_argument('--limit', type=int, default=50000,
                        help='Limit number of rows to load (default: 50000)')
    parser.add_argument('--data-dir', default='datasets',
                        help='Data directory for downloads (default: datasets)')
    parser.add_argument('--skip-hdf5', action='store_true',
                        help='Skip HDF5 datasets (SIFT, Deep1B)')

    # Synthetic data
    parser.add_argument('--create-synthetic', action='store_true',
                        help='Create synthetic test datasets')

    # Utilities
    parser.add_argument('--create-fts-indexes', action='store_true',
                        help='Create full-text search indexes')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show dataset statistics')
    
    # Version
    parser.add_argument('--version', action='version', version=f'neurondb_dataset.py version {__version__}')

    args = parser.parse_args()

    # Set environment variables
    os.environ['PGDATABASE'] = args.dbname
    if args.skip_hdf5:
        os.environ['SKIP_HDF5'] = '1'
    os.environ['DATA_ROOT'] = args.data_dir

    # Execute operations
    if args.recreate_db:
        gd.log(True, f"Recreating database: {args.dbname}")
        gd.admin_drop_database(args.dbname)
        gd.admin_create_database(args.dbname)
        gd.log(True, f"Database {args.dbname} recreated")
        return 0

    # Get connection
    try:
        conn = gd.get_conn(args.dbname)
    except Exception as e:
        gd.log(False, f"Failed to connect to {args.dbname}: {e}")
        return 1

    # Create schema
    schema = 'neurondb_datasets'
    try:
        with conn.cursor() as cur:
            cur.execute(gd.sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(gd.sql.Identifier(schema)))
        conn.commit()
        gd.log(True, f"Schema {schema} ready")
    except Exception as e:
        gd.log(False, f"Failed to create schema: {e}")
        return 1

    # Ensure catalog
    gd.ensure_datasets_catalog(conn)

    data_root = Path(args.data_dir).resolve()

    # Load datasets based on arguments
    if args.load_all or args.load_msmarco:
        gd.log(True, "Loading MS MARCO dataset...")
        gd.ensure_ms_marco(conn, schema)
        # Try HuggingFace first (easier)
        ok = gd.load_msmarco_hf(conn, schema, limit=args.limit)
        if not ok:
            # Fallback to download
            dest = data_root / "msmarco" / "collection.tar.gz"
            if gd.download_file("https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz", dest):
                ok = gd.load_msmarco_collection(conn, schema, dest, limit=args.limit)
        if ok:
            gd.update_catalog(conn, "MS_MARCO", schema, "MS MARCO passages")

    if args.load_all or args.load_wikipedia:
        gd.log(True, "Loading Wikipedia embeddings...")
        gd.ensure_wiki_embeddings(conn, schema)
        dest = data_root / "wikipedia" / "wiki_minilm.ndjson.gz"
        if gd.download_file("https://huggingface.co/datasets/Supabase/wikipedia-en-embeddings/resolve/main/wiki_minilm.ndjson.gz", dest):
            ok = gd.load_wikipedia_ndjson_gz(conn, schema, dest, limit=args.limit)
            if ok:
                gd.update_catalog(conn, "Wikipedia", schema, "Wikipedia with embeddings")

    if args.load_all or args.load_hotpotqa:
        gd.log(True, "Loading HotpotQA...")
        gd.ensure_hotpotqa(conn, schema)
        dest = data_root / "hotpotqa" / "hotpot_train_v1.json"
        if gd.download_file("https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.json", dest):
            ok = gd.load_hotpotqa_json(conn, schema, dest, limit=args.limit)
            if ok:
                gd.update_catalog(conn, "HotpotQA", schema, "Multi-hop QA dataset")

    if (args.load_all or args.load_sift) and not args.skip_hdf5:
        gd.log(True, "Loading SIFT1M vectors...")
        gd.ensure_vectors(conn, f"{schema}_sift")
        dest = data_root / "sift" / "sift-128-euclidean.hdf5"
        if gd.download_file("https://ann-benchmarks.com/sift-128-euclidean.hdf5", dest):
            ok = gd.load_sift_hdf5(conn, f"{schema}_sift", dest, limit=args.limit)
            if ok:
                gd.update_catalog(conn, "SIFT1M", f"{schema}_sift", "SIFT 128-d vectors")

    if (args.load_all or args.load_deep1b) and not args.skip_hdf5:
        gd.log(True, "Loading Deep1B vectors...")
        gd.ensure_vectors(conn, f"{schema}_deep1b")
        dest = data_root / "deep1b" / "deep-image-96-angular.hdf5"
        if gd.download_file("https://ann-benchmarks.com/deep-image-96-angular.hdf5", dest):
            ok = gd.load_deep1b_hdf5(conn, f"{schema}_deep1b", dest, limit=args.limit)
            if ok:
                gd.update_catalog(conn, "Deep1B", f"{schema}_deep1b", "Deep1B 96-d vectors")

    if args.create_synthetic:
        gd.log(True, "Creating synthetic test datasets...")
        create_synthetic_datasets(conn, schema)

    if args.create_fts_indexes:
        gd.log(True, "Creating full-text search indexes...")
        try:
            gd.create_fts_index(conn, schema)
            gd.log(True, "FTS indexes created")
        except Exception as e:
            gd.log(False, f"FTS index creation failed: {e}")

    if args.show_stats:
        show_statistics(conn)

    conn.close()
    gd.log(True, "Done")
    return 0


def create_synthetic_datasets(conn, schema):
    """Create synthetic test datasets for various ML algorithms"""
    try:
        with conn.cursor() as cur:
            # Synthetic clustering data (3 well-separated clusters)
            cur.execute(gd.sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.synthetic_clusters (
                    id SERIAL PRIMARY KEY,
                    vec REAL[],
                    cluster_id INT
                );
            """).format(gd.sql.Identifier(schema)))

            # Insert 300 points in 3 clusters
            for cluster in range(3):
                for i in range(100):
                    import random
                    base = [cluster * 10.0] * 10
                    noise = [random.gauss(0, 0.5) for _ in range(10)]
                    vec = [b + n for b, n in zip(base, noise)]
                    cur.execute(
                        gd.sql.SQL("INSERT INTO {}.synthetic_clusters (vec, cluster_id) VALUES (%s, %s)").format(
                            gd.sql.Identifier(schema)
                        ),
                        (vec, cluster)
                    )

            # Synthetic outlier data
            cur.execute(gd.sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.synthetic_outliers (
                    id SERIAL PRIMARY KEY,
                    vec REAL[],
                    is_outlier BOOLEAN
                );
            """).format(gd.sql.Identifier(schema)))

            # Insert normal points + outliers
            import random
            for i in range(95):
                vec = [random.gauss(0, 1) for _ in range(5)]
                cur.execute(
                    gd.sql.SQL("INSERT INTO {}.synthetic_outliers (vec, is_outlier) VALUES (%s, false)").format(
                        gd.sql.Identifier(schema)
                    ),
                    (vec,)
                )
            # Add 5 outliers
            for i in range(5):
                vec = [random.gauss(10, 1) for _ in range(5)]
                cur.execute(
                    gd.sql.SQL("INSERT INTO {}.synthetic_outliers (vec, is_outlier) VALUES (%s, true)").format(
                        gd.sql.Identifier(schema)
                    ),
                    (vec,)
                )

        conn.commit()
        gd.log(True, "Synthetic datasets created")
    except Exception as e:
        conn.rollback()
        gd.log(False, f"Failed to create synthetic datasets: {e}")


def update_catalog(conn, name, schema, purpose):
    """Update dataset catalog"""
    try:
        cnt = gd.count_rows_any(conn, schema)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.datasets(name, schema_name, purpose, rows, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (name) DO UPDATE SET
                  schema_name = EXCLUDED.schema_name,
                  purpose = EXCLUDED.purpose,
                  rows = EXCLUDED.rows,
                  updated_at = NOW()
                """,
                (name, schema, purpose, cnt)
            )
        conn.commit()
        gd.log(True, f"Catalog: {name} ({cnt} rows)")
    except Exception as e:
        conn.rollback()
        gd.log(False, f"Catalog update failed for {name}: {e}")


def show_statistics(conn):
    """Show statistics for all loaded datasets"""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name, schema_name, rows, updated_at FROM public.datasets ORDER BY name")
            rows = cur.fetchall()
            print("\n" + "=" * 70)
            print("Dataset Statistics")
            print("=" * 70)
            for name, schema, row_count, updated in rows:
                print(f"{name:30s} | {schema:25s} | {row_count:10d} rows | {updated}")
            print("=" * 70)

            # Schema table counts
            cur.execute("""
                SELECT schemaname, tablename, 
                       (SELECT COUNT(*) FROM pg_tables pt WHERE pt.schemaname = t.schemaname AND pt.tablename = t.tablename) 
                FROM pg_tables t 
                WHERE schemaname LIKE 'neurondb%'
                ORDER BY schemaname, tablename
            """)
            tables = cur.fetchall()
            if tables:
                print("\nTables:")
                for schema, table, _ in tables:
                    print(f"  {schema}.{table}")
    except Exception as e:
        gd.log(False, f"Failed to show statistics: {e}")


# Add update_catalog to gd module
gd.update_catalog = update_catalog


if __name__ == "__main__":
    sys.exit(main())


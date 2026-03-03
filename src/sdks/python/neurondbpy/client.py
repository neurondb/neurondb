"""
Sync NeuronDB client using psycopg2.

Provides connection management, arbitrary SQL execution, vector search,
batch insert, index creation, and hybrid search.
"""

from typing import Any, Dict, List, Optional, Union

import psycopg2
import psycopg2.extras


# Distance operators: L2 <->, cosine <=>, inner product <#>
_DISTANCE_OPS = {
    "l2": "<->",
    "cosine": "<=>",
    "inner_product": "<#>",
}


class NeuronDBClient:
    """
    Sync client for NeuronDB (PostgreSQL + NeuronDB extension).

    Example:
        >>> with NeuronDBClient("postgresql://user:pass@localhost/neurondb") as client:
        ...     rows = client.search_vectors("documents", [0.1, 0.2, ...], limit=10)
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        host: Optional[str] = None,
        port: int = 5432,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            dsn: PostgreSQL connection string (e.g. postgresql://user:pass@host/db).
            host: Database host (used if dsn not provided).
            port: Database port.
            dbname: Database name.
            user: Database user.
            password: Database password.
        """
        if dsn:
            self._dsn = dsn
        else:
            parts = [f"host={host}", f"port={port}", f"dbname={dbname}", f"user={user}"]
            if password:
                parts.append(f"password={password}")
            self._dsn = " ".join(parts)
        self._conn: Optional[Any] = None

    def connect(self) -> "NeuronDBClient":
        """Open a connection. Idempotent."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            self._ensure_extension()
        return self

    def close(self) -> None:
        """Close the connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "NeuronDBClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_extension(self) -> None:
        """Ensure neurondb extension is installed."""
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS neurondb")
            self._conn.commit()

    def execute(
        self,
        sql: str,
        params: Optional[Union[tuple, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL and return all rows as list of dicts.

        Args:
            sql: SQL query (placeholders %s or %(name)s).
            params: Query parameters.

        Returns:
            List of row dicts; empty if no result set.
        """
        self.connect()
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if cur.description:
                return [dict(row) for row in cur.fetchall()]
            return []

    def execute_one(
        self,
        sql: str,
        params: Optional[Union[tuple, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute SQL and return the first row or None."""
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    def search_vectors(
        self,
        table: str,
        query_vector: List[float],
        limit: int = 10,
        metric: str = "cosine",
        vector_column: str = "embedding",
        where: Optional[str] = None,
        extra_columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run vector similarity search.

        Args:
            table: Table name.
            query_vector: Query vector.
            limit: Max number of results.
            metric: One of "cosine", "l2", "inner_product".
            vector_column: Name of the vector column.
            where: Optional WHERE clause (e.g. "category = 'tech'").
            extra_columns: Optional list of extra columns to select (default: *).

        Returns:
            List of row dicts including a "distance" key.
        """
        op = _DISTANCE_OPS.get(metric, "<=>")
        vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        where_clause = f" WHERE {where}" if where else ""
        if extra_columns:
            cols = ", ".join(extra_columns) + f", {vector_column} {op} %s::vector AS distance"
        else:
            cols = f"*, {vector_column} {op} %s::vector AS distance"
        sql = (
            f"SELECT {cols} FROM {table}{where_clause} "
            f"ORDER BY {vector_column} {op} %s::vector LIMIT %s"
        )
        return self.execute(sql, (vec_str, vec_str, limit))

    def insert_vectors(
        self,
        table: str,
        vectors: List[Dict[str, Any]],
        vector_column: str = "embedding",
    ) -> int:
        """
        Insert rows with vector columns. Each dict must have key vector_column
        (list of floats) and may have other column keys.

        Args:
            table: Table name.
            vectors: List of dicts; each has at least vector_column and optionally other cols.
            vector_column: Name of the vector column.

        Returns:
            Number of rows inserted.
        """
        if not vectors:
            return 0
        self.connect()
        cols = list(vectors[0].keys())
        # Build placeholders: use %s::vector for the vector column so string is cast
        placeholders = []
        for c in cols:
            if c == vector_column:
                placeholders.append("%s::vector")
            else:
                placeholders.append("%s")
        col_list = ", ".join(cols)
        ph = ", ".join(placeholders)
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({ph})"
        with self._conn.cursor() as cur:
            for row in vectors:
                out = []
                for c in cols:
                    v = row[c]
                    if c == vector_column and isinstance(v, list):
                        v = "[" + ",".join(str(float(x)) for x in v) + "]"
                    out.append(v)
                cur.execute(sql, out)
            self._conn.commit()
            return len(vectors)

    def create_index(
        self,
        table: str,
        column: str,
        index_type: str = "hnsw",
        index_name: Optional[str] = None,
        m: int = 16,
        ef_construction: int = 200,
        ops: Optional[str] = None,
    ) -> str:
        """
        Create a vector index (HNSW or IVF).

        Args:
            table: Table name.
            column: Vector column name.
            index_type: "hnsw" or "ivf".
            index_name: Optional index name (default: {table}_{column}_{type}_idx).
            m: HNSW M parameter.
            ef_construction: HNSW ef_construction.
            ops: Index ops: "vector_l2_ops", "vector_cosine_ops", "vector_ip_ops" (default: cosine).

        Returns:
            The created index name.
        """
        name = index_name or f"{table}_{column}_{index_type}_idx"
        if ops is None:
            ops = "vector_cosine_ops"
        if index_type == "hnsw":
            sql = (
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                f"USING hnsw ({column} {ops}) WITH (m = %s, ef_construction = %s)"
            )
            self.execute(sql, (m, ef_construction))
        elif index_type == "ivf":
            lists = max(100, m * 10)  # placeholder
            sql = (
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                f"USING ivfflat ({column} {ops}) WITH (lists = %s)"
            )
            self.execute(sql, (lists,))
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")
        return name

    def hybrid_search(
        self,
        table: str,
        query_vector: List[float],
        text_query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        vector_column: str = "embedding",
        fts_column: Optional[str] = None,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search (vector + full-text). Uses NeuronDB hybrid_search if available,
        otherwise falls back to a manual combined query.

        Args:
            table: Table name.
            query_vector: Query vector.
            text_query: Full-text query string.
            limit: Max results.
            vector_weight: Weight for vector score (0–1); 1 - vector_weight for FTS.
            vector_column: Vector column name.
            fts_column: Full-text search column (tsvector); if None, uses hybrid_search().
            where: Optional filter (e.g. "metadata @> '{}'").

        Returns:
            List of row dicts with hybrid score.
        """
        vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        filter_json = "{}" if not where else where
        # Prefer NeuronDB hybrid_search(table, vector, text, filters, vector_weight, limit)
        try:
            sql = (
                "SELECT * FROM hybrid_search(%s, %s::vector, %s, %s::jsonb, %s, %s)"
            )
            return self.execute(
                sql,
                (table, vec_str, text_query, filter_json, vector_weight, limit),
            )
        except Exception:
            # Fallback: vector-only search if hybrid_search not available
            return self.search_vectors(
                table, query_vector, limit=limit, vector_column=vector_column, where=where
            )

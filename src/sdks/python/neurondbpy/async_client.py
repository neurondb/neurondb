"""
Async NeuronDB client using asyncpg.

Same API surface as NeuronDBClient but all methods are async.
"""

from typing import Any, Dict, List, Optional, Union

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

from neurondbpy.client import _DISTANCE_OPS


class AsyncNeuronDBClient:
    """
    Async client for NeuronDB (PostgreSQL + NeuronDB extension).

    Requires the [async] extra: pip install neurondbpy[async]

    Example:
        >>> async with AsyncNeuronDBClient("postgresql://...") as client:
        ...     rows = await client.search_vectors("documents", [0.1, 0.2], limit=10)
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
        min_size: int = 1,
        max_size: int = 10,
    ):
        if asyncpg is None:
            raise ImportError("asyncpg is required for AsyncNeuronDBClient. Install with: pip install neurondbpy[async]")
        if dsn:
            self._dsn = dsn
        else:
            self._dsn = (
                f"postgresql://{user}:{password or ''}@{host}:{port}/{dbname}"
            )
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> "AsyncNeuronDBClient":
        """Create connection pool. Idempotent."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
            )
            await self._ensure_extension()
        return self

    async def close(self) -> None:
        """Close the pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> "AsyncNeuronDBClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _ensure_extension(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS neurondb")

    async def execute(
        self,
        sql: str,
        params: Optional[Union[tuple, List[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute SQL and return all rows as list of dicts."""
        await self.connect()
        params = params or ()
        async with self._pool.acquire() as conn:
            if params:
                rows = await conn.fetch(sql, *params)
            else:
                rows = await conn.fetch(sql)
            if rows:
                return [dict(r) for r in rows]
            return []

    async def execute_one(
        self,
        sql: str,
        params: Optional[Union[tuple, List[Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute SQL and return the first row or None."""
        rows = await self.execute(sql, params)
        return rows[0] if rows else None

    async def search_vectors(
        self,
        table: str,
        query_vector: List[float],
        limit: int = 10,
        metric: str = "cosine",
        vector_column: str = "embedding",
        where: Optional[str] = None,
        extra_columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run vector similarity search."""
        op = _DISTANCE_OPS.get(metric, "<=>")
        vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        where_clause = f" WHERE {where}" if where else ""
        if extra_columns:
            cols = ", ".join(extra_columns) + f", {vector_column} {op} $1::vector AS distance"
        else:
            cols = f"*, {vector_column} {op} $1::vector AS distance"
        sql = (
            f"SELECT {cols} FROM {table}{where_clause} "
            f"ORDER BY {vector_column} {op} $1::vector LIMIT $2"
        )
        return await self.execute(sql, (vec_str, limit))

    async def insert_vectors(
        self,
        table: str,
        vectors: List[Dict[str, Any]],
        vector_column: str = "embedding",
    ) -> int:
        """Insert rows with vector columns."""
        if not vectors:
            return 0
        await self.connect()
        cols = list(vectors[0].keys())
        async with self._pool.acquire() as conn:
            for row in vectors:
                out = []
                for c in cols:
                    v = row[c]
                    if c == vector_column and isinstance(v, list):
                        v = "[" + ",".join(str(float(x)) for x in v) + "]"
                    out.append(v)
                placeholders = []
                for i, c in enumerate(cols):
                    if c == vector_column:
                        placeholders.append(f"${i+1}::vector")
                    else:
                        placeholders.append(f"${i+1}")
                ph = ", ".join(placeholders)
                col_list = ", ".join(cols)
                sql = f"INSERT INTO {table} ({col_list}) VALUES ({ph})"
                await conn.execute(sql, *out)
        return len(vectors)

    async def create_index(
        self,
        table: str,
        column: str,
        index_type: str = "hnsw",
        index_name: Optional[str] = None,
        m: int = 16,
        ef_construction: int = 200,
        ops: Optional[str] = None,
    ) -> str:
        """Create a vector index (HNSW or IVF)."""
        name = index_name or f"{table}_{column}_{index_type}_idx"
        if ops is None:
            ops = "vector_cosine_ops"
        if index_type == "hnsw":
            sql = (
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                f"USING hnsw ({column} {ops}) WITH (m = $1, ef_construction = $2)"
            )
            await self.execute(sql, (m, ef_construction))
        elif index_type == "ivf":
            lists = max(100, m * 10)
            sql = (
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                f"USING ivfflat ({column} {ops}) WITH (lists = $1)"
            )
            await self.execute(sql, (lists,))
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")
        return name

    async def hybrid_search(
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
        """Hybrid search (vector + full-text)."""
        vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        filter_json = "{}" if not where else where
        try:
            sql = "SELECT * FROM hybrid_search($1, $2::vector, $3, $4::jsonb, $5, $6)"
            return await self.execute(
                sql,
                (table, vec_str, text_query, filter_json, vector_weight, limit),
            )
        except Exception:
            return await self.search_vectors(
                table, query_vector, limit=limit, vector_column=vector_column, where=where
            )

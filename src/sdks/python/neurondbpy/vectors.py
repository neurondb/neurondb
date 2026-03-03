"""
Vector search, insert, and index helpers.

Convenience functions that delegate to NeuronDBClient.
"""

from typing import Any, Dict, List, Optional, Union

from neurondbpy.client import NeuronDBClient


def search_vectors(
    client: NeuronDBClient,
    table: str,
    query_vector: List[float],
    limit: int = 10,
    metric: str = "cosine",
    vector_column: str = "embedding",
    where: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run vector similarity search.

    Args:
        client: NeuronDBClient instance.
        table: Table name.
        query_vector: Query vector.
        limit: Max results.
        metric: "cosine", "l2", or "inner_product".
        vector_column: Vector column name.
        where: Optional WHERE clause.

    Returns:
        List of row dicts with "distance" key.
    """
    return client.search_vectors(
        table=table,
        query_vector=query_vector,
        limit=limit,
        metric=metric,
        vector_column=vector_column,
        where=where,
    )


def insert_vectors(
    client: NeuronDBClient,
    table: str,
    vectors: List[Dict[str, Any]],
    vector_column: str = "embedding",
) -> int:
    """
    Insert rows with vector columns.

    Args:
        client: NeuronDBClient instance.
        table: Table name.
        vectors: List of dicts; each has vector_column (list of floats) and optional other cols.
        vector_column: Name of the vector column.

    Returns:
        Number of rows inserted.
    """
    return client.insert_vectors(
        table=table,
        vectors=vectors,
        vector_column=vector_column,
    )


def create_index(
    client: NeuronDBClient,
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
        client: NeuronDBClient instance.
        table: Table name.
        column: Vector column name.
        index_type: "hnsw" or "ivf".
        index_name: Optional index name.
        m: HNSW M parameter.
        ef_construction: HNSW ef_construction.
        ops: "vector_l2_ops", "vector_cosine_ops", or "vector_ip_ops".

    Returns:
        Created index name.
    """
    return client.create_index(
        table=table,
        column=column,
        index_type=index_type,
        index_name=index_name,
        m=m,
        ef_construction=ef_construction,
        ops=ops,
    )


def hybrid_search(
    client: NeuronDBClient,
    table: str,
    query_vector: List[float],
    text_query: str,
    limit: int = 10,
    vector_weight: float = 0.7,
    vector_column: str = "embedding",
    where: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search (vector + full-text).

    Args:
        client: NeuronDBClient instance.
        table: Table name.
        query_vector: Query vector.
        text_query: Full-text query.
        limit: Max results.
        vector_weight: Weight for vector score (0–1).
        vector_column: Vector column name.
        where: Optional filter.

    Returns:
        List of row dicts with hybrid score.
    """
    return client.hybrid_search(
        table=table,
        query_vector=query_vector,
        text_query=text_query,
        limit=limit,
        vector_weight=vector_weight,
        vector_column=vector_column,
        where=where,
    )

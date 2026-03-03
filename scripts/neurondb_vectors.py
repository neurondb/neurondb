"""
NeuronDB VectorStore

Vector search and similarity operations.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
from .client import Client


class VectorStore:
	"""
	High-level interface for vector search operations.
	
	Example:
		>>> store = VectorStore(client, table="documents", column="embedding")
		>>> results = store.search(query_vector, k=10)
	"""
	
	def __init__(
		self,
		client: Client,
		table: str,
		column: str = "embedding",
		index_name: Optional[str] = None,
	):
		"""
		Initialize VectorStore.
		
		Args:
			client: NeuronDB client instance
			table: Table name containing vectors
			column: Vector column name
			index_name: Optional index name for faster search
		"""
		self.client = client
		self.table = table
		self.column = column
		self.index_name = index_name
	
	def search(
		self,
		query_vector: Union[List[float], np.ndarray],
		k: int = 10,
		distance_metric: str = "cosine",
		where: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		"""
		Search for similar vectors.
		
		Args:
			query_vector: Query vector (list or numpy array)
			k: Number of results to return
			distance_metric: Distance metric ("cosine", "l2", "inner_product")
			where: Optional WHERE clause for filtering
		
		Returns:
			List of results with similarity scores
		"""
		# Convert to list if numpy array
		if isinstance(query_vector, np.ndarray):
			query_vector = query_vector.tolist()
		
		# Build vector literal
		vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
		
		# Build query
		where_clause = f"WHERE {where}" if where else ""
		
		if distance_metric == "cosine":
			operator = "<->"
		elif distance_metric == "l2":
			operator = "<->"
		elif distance_metric == "inner_product":
			operator = "<#>"
		else:
			operator = "<->"
		
		query = f"""
			SELECT *, {self.column} {operator} %s::vector AS distance
			FROM {self.table}
			{where_clause}
			ORDER BY distance
			LIMIT %s
		"""
		
		vector_pg = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
		results = self.client.execute(query, (vector_pg, k))
		return results
	
	def create_index(
		self,
		index_type: str = "hnsw",
		m: int = 16,
		ef_construction: int = 200,
	) -> str:
		"""
		Create vector index for faster search.
		
		Args:
			index_type: Index type ("hnsw" or "ivf")
			m: HNSW parameter M
			ef_construction: HNSW parameter ef_construction
		
		Returns:
			Index name
		"""
		if not self.index_name:
			self.index_name = f"{self.table}_{self.column}_idx"
		
		if index_type == "hnsw":
			query = f"""
				SELECT hnsw_create_index(
					%s,  -- table_name
					%s,  -- column_name
					%s,  -- index_name
					%s,  -- m
					%s   -- ef_construction
				) as index_name
			"""
			result = self.client.execute_one(
				query,
				(self.table, self.column, self.index_name, m, ef_construction)
			)
		else:
			raise ValueError(f"Unsupported index type: {index_type}")
		
		return self.index_name if result else None
	
	def insert(self, vector: Union[List[float], np.ndarray], **kwargs) -> int:
		"""
		Insert a vector into the table.
		
		Args:
			vector: Vector to insert
			**kwargs: Additional column values
		
		Returns:
			Inserted row ID
		"""
		# Convert to list if numpy array
		if isinstance(vector, np.ndarray):
			vector = vector.tolist()
		
		vector_str = "[" + ",".join(str(float(x)) for x in vector) + "]"
		
		# Build columns and values
		columns = [self.column] + list(kwargs.keys())
		values = [vector_str] + [f"'{v}'" if isinstance(v, str) else str(v) for v in kwargs.values()]
		
		columns_str = ", ".join(columns)
		values_str = ", ".join(values)
		
		query = f"""
			INSERT INTO {self.table} ({columns_str})
			VALUES ({values_str})
			RETURNING id
		"""
		
		result = self.client.execute_one(query)
		return result["id"] if result else None
	
	def batch_insert(
		self,
		vectors: List[Union[List[float], np.ndarray]],
		**kwargs_list: List[Any],
	) -> List[int]:
		"""
		Insert multiple vectors.
		
		Args:
			vectors: List of vectors to insert
			**kwargs_list: Additional column values (lists matching vectors length)
		
		Returns:
			List of inserted row IDs
		"""
		ids = []
		for i, vector in enumerate(vectors):
			kwargs = {k: v[i] for k, v in kwargs_list.items()}
			row_id = self.insert(vector, **kwargs)
			ids.append(row_id)
		return ids
	
	def hybrid_search(
		self,
		query_vector: Union[List[float], np.ndarray],
		query_text: str,
		vector_weight: float = 0.7,
		k: int = 10,
		where: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		"""
		Hybrid search combining vector similarity and full-text search.
		
		Args:
			query_vector: Query vector
			query_text: Query text for full-text search
			vector_weight: Weight for vector component (0-1)
			k: Number of results
			where: Optional WHERE clause
		
		Returns:
			List of results with combined scores
		"""
		# Convert to list if numpy array
		if isinstance(query_vector, np.ndarray):
			query_vector = query_vector.tolist()
		
		vector_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
		
		query = f"""
			SELECT * FROM hybrid_search(
				%s,  -- table_name
				%s::vector,  -- query_vector
				%s,  -- query_text
				%s::jsonb,  -- filters
				%s,  -- vector_weight
				%s   -- k
			)
		"""
		
		filters = where if where else "{}"
		results = self.client.execute(
			query,
			(self.table, vector_str, query_text, filters, vector_weight, k)
		)
		return results


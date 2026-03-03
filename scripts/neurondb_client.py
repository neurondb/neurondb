"""
NeuronDB Client

Connection management and basic operations.
"""

import asyncio
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.pool import ThreadedConnectionPool
import psycopg2.extras


class Client:
	"""
	Main client for interacting with NeuronDB.
	
	Example:
		>>> client = Client("postgresql://user:pass@localhost/dbname")
		>>> model = client.train("random_forest", table="data", target="label")
	"""
	
	def __init__(
		self,
		connection_string: Optional[str] = None,
		host: Optional[str] = None,
		port: int = 5432,
		database: Optional[str] = None,
		user: Optional[str] = None,
		password: Optional[str] = None,
		min_connections: int = 1,
		max_connections: int = 10,
	):
		"""
		Initialize NeuronDB client.
		
		Args:
			connection_string: PostgreSQL connection string (DSN format)
			host: Database host (if not using connection_string)
			port: Database port
			database: Database name
			user: Database user
			password: Database password
			min_connections: Minimum connection pool size
			max_connections: Maximum connection pool size
		"""
		if connection_string:
			self.connection_string = connection_string
		else:
			self.connection_string = f"host={host} port={port} dbname={database} user={user} password={password}"
		
		self.pool = ThreadedConnectionPool(
			min_connections,
			max_connections,
			self.connection_string
		)
		self._ensure_extension()
	
	def _get_connection(self):
		"""Get connection from pool."""
		return self.pool.getconn()
	
	def _return_connection(self, conn):
		"""Return connection to pool."""
		self.pool.putconn(conn)
	
	def _ensure_extension(self):
		"""Ensure neurondb extension is installed."""
		conn = self._get_connection()
		try:
			with conn.cursor() as cur:
				cur.execute("CREATE EXTENSION IF NOT EXISTS neurondb")
				conn.commit()
		finally:
			self._return_connection(conn)
	
	def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
		"""
		Execute SQL query and return results.
		
		Args:
			query: SQL query string
			params: Query parameters
		
		Returns:
			List of dictionaries representing rows
		"""
		conn = self._get_connection()
		try:
			with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
				cur.execute(query, params)
				if cur.description:
					return [dict(row) for row in cur.fetchall()]
				return []
		finally:
			self._return_connection(conn)
	
	def execute_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
		"""
		Execute SQL query and return first result.
		
		Args:
			query: SQL query string
			params: Query parameters
		
		Returns:
			Dictionary representing first row, or None
		"""
		results = self.execute(query, params)
		return results[0] if results else None
	
	def train(
		self,
		algorithm: str,
		table: str,
		target: str,
		features: Optional[List[str]] = None,
		project: str = "default",
		model_name: Optional[str] = None,
		params: Optional[Dict[str, Any]] = None,
	) -> "Model":
		"""
		Train a machine learning model.
		
		Args:
			algorithm: Algorithm name (e.g., "random_forest", "linear_regression")
			table: Training table name
			target: Target column name
			features: List of feature column names (None = all columns except target)
			project: Project name
			model_name: Model name (auto-generated if None)
			params: Hyperparameters as dictionary
		
		Returns:
			Model object
		"""
		from neurondb.models import Model
		
		if features is None:
			# Get all columns except target
			query = """
				SELECT column_name 
				FROM information_schema.columns 
				WHERE table_name = %s AND column_name != %s
			"""
			rows = self.execute(query, (table, target))
			features = [row["column_name"] for row in rows]
		
		if model_name is None:
			model_name = f"{algorithm}_{table}_{target}"
		
		if params is None:
			params = {}
		
		# Build SQL function call
		features_array = "ARRAY[" + ",".join(f"'{f}'" for f in features) + "]::text[]"
		params_json = str(params).replace("'", '"') if params else "{}"
		
		query = f"""
			SELECT neurondb.train(
				%s,  -- project
				%s,  -- algorithm
				%s,  -- table_name
				%s,  -- target_column
				{features_array},  -- feature_columns
				%s::jsonb  -- hyperparameters
			) as model_id
		"""
		
		result = self.execute_one(query, (project, algorithm, table, target, params_json))
		model_id = result["model_id"]
		
		return Model(self, model_id, algorithm, model_name, project)
	
	def predict(self, model_id: int, data: Any) -> Any:
		"""
		Make predictions using a trained model.
		
		Args:
			model_id: Model ID
			data: Input data (table name, array, or feature dict)
		
		Returns:
			Predictions
		"""
		# This will be implemented in Model class
		from neurondb.models import Model
		model = Model(self, model_id)
		return model.predict(data)
	
	def close(self):
		"""Close connection pool."""
		if self.pool:
			self.pool.closeall()
	
	def __enter__(self):
		"""Context manager entry."""
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Context manager exit."""
		self.close()
	
	async def aexecute(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
		"""
		Execute SQL query asynchronously.
		
		Args:
			query: SQL query string
			params: Query parameters
		
		Returns:
			List of dictionaries representing rows
		"""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self.execute, query, params)


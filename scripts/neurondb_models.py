"""
NeuronDB Model

Model lifecycle management (train, predict, deploy).
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .client import Client


@dataclass
class ModelMetrics:
	"""Model performance metrics."""
	accuracy: Optional[float] = None
	precision: Optional[float] = None
	recall: Optional[float] = None
	f1_score: Optional[float] = None
	mse: Optional[float] = None
	mae: Optional[float] = None
	r2_score: Optional[float] = None
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
		"""Create ModelMetrics from dictionary."""
		return cls(
			accuracy=data.get("accuracy"),
			precision=data.get("precision"),
			recall=data.get("recall"),
			f1_score=data.get("f1_score"),
			mse=data.get("mse"),
			mae=data.get("mae"),
			r2_score=data.get("r2_score"),
		)


class Model:
	"""
	Represents a trained ML model.
	
	Example:
		>>> model = client.train("random_forest", table="data", target="label")
		>>> predictions = model.predict("test_data")
		>>> model.deploy()
	"""
	
	def __init__(
		self,
		client: Client,
		model_id: int,
		algorithm: Optional[str] = None,
		model_name: Optional[str] = None,
		project: Optional[str] = None,
	):
		"""
		Initialize Model object.
		
		Args:
			client: NeuronDB client instance
			model_id: Model ID from database
			algorithm: Algorithm name (optional, loaded from DB if None)
			model_name: Model name (optional, loaded from DB if None)
			project: Project name (optional, loaded from DB if None)
		"""
		self.client = client
		self.model_id = model_id
		
		# Load model info from database
		if not algorithm or not model_name or not project:
			info = self._load_info()
			self.algorithm = algorithm or info.get("algorithm")
			self.model_name = model_name or info.get("model_name")
			self.project = project or info.get("project_name")
			self.version = info.get("version", 1)
			self.status = info.get("status")
			self.metrics = ModelMetrics.from_dict(info.get("metrics", {}))
		else:
			self.algorithm = algorithm
			self.model_name = model_name
			self.project = project
			self.version = 1
			self.status = "trained"
			self.metrics = ModelMetrics()
	
	def _load_info(self) -> Dict[str, Any]:
		"""Load model information from database."""
		query = """
			SELECT 
				m.model_id,
				m.algorithm,
				m.model_name,
				p.project_name,
				m.version,
				m.status,
				m.metrics
			FROM neurondb.ml_models m
			JOIN neurondb.ml_projects p ON m.project_id = p.project_id
			WHERE m.model_id = %s
		"""
		result = self.client.execute_one(query, (self.model_id,))
		if not result:
			raise ValueError(f"Model {self.model_id} not found")
		return result
	
	def predict(self, data: Any) -> List[Any]:
		"""
		Make predictions using this model.
		
		Args:
			data: Input data - can be:
				- Table name (string): Predict on all rows
				- Feature array (list): Single prediction
				- Feature dict: Single prediction
		
		Returns:
			List of predictions
		"""
		if isinstance(data, str):
			# Table name - predict on all rows
			query = f"""
				SELECT neurondb.predict(%s, %s) as prediction
			"""
			# This is a simplified version - actual implementation would
			# need to handle table-based prediction differently
			results = self.client.execute(query, (self.model_id, data))
			return [r["prediction"] for r in results]
		elif isinstance(data, dict):
			# Feature dictionary - single prediction
			features = list(data.values())
			return self.predict(features)
		elif isinstance(data, list):
			# Feature array - single prediction
			features_array = "ARRAY[" + ",".join(str(f) for f in data) + "]"
			query = f"""
				SELECT neurondb.predict(%s, {features_array}) as prediction
			"""
			result = self.client.execute_one(query, (self.model_id,))
			return result["prediction"] if result else None
		else:
			raise ValueError(f"Unsupported data type: {type(data)}")
	
	def deploy(self, version: Optional[int] = None) -> "Model":
		"""
		Deploy this model (or specific version) to production.
		
		Args:
			version: Version to deploy (None = latest)
		
		Returns:
			Deployed Model instance
		"""
		if version:
			query = "SELECT neurondb.deploy_model(%s, %s) as deployed_model_id"
			result = self.client.execute_one(query, (self.model_id, version))
		else:
			query = "SELECT neurondb.deploy_model(%s) as deployed_model_id"
			result = self.client.execute_one(query, (self.model_id,))
		
		if result:
			deployed_id = result["deployed_model_id"]
			return Model(self.client, deployed_id)
		return self
	
	def rollback(self) -> "Model":
		"""
		Rollback to previous version of this model.
		
		Returns:
			Previous Model instance
		"""
		query = "SELECT neurondb.rollback_model(%s) as previous_model_id"
		result = self.client.execute_one(query, (self.model_id,))
		
		if result:
			previous_id = result["previous_model_id"]
			return Model(self.client, previous_id)
		raise ValueError("No previous version available")
	
	def compare(self, other_model_id: int, test_table: str) -> Dict[str, Any]:
		"""
		Compare this model with another model on test data.
		
		Args:
			other_model_id: Other model ID to compare with
			test_table: Test table name
		
		Returns:
			Comparison metrics
		"""
		query = """
			SELECT neurondb.compare_models(%s, %s, %s) as comparison
		"""
		result = self.client.execute_one(query, (self.model_id, other_model_id, test_table))
		return result["comparison"] if result else {}
	
	def get_metrics(self) -> ModelMetrics:
		"""Get model performance metrics."""
		return self.metrics
	
	def update(self):
		"""Refresh model information from database."""
		info = self._load_info()
		self.version = info.get("version", self.version)
		self.status = info.get("status", self.status)
		self.metrics = ModelMetrics.from_dict(info.get("metrics", {}))
	
	def __repr__(self) -> str:
		return f"Model(id={self.model_id}, algorithm={self.algorithm}, name={self.model_name})"


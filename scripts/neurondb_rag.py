"""
NeuronDB RAG

Retrieval Augmented Generation pipeline operations.
"""

from typing import Optional, List, Dict, Any
from .client import Client
from .vectors import VectorStore


class RAG:
	"""
	RAG (Retrieval Augmented Generation) pipeline.
	
	Example:
		>>> rag = RAG(client, table="documents", embedding_model="all-MiniLM-L6-v2")
		>>> results = rag.retrieve("What is machine learning?", k=5)
		>>> answer = rag.generate("What is machine learning?", context=results)
	"""
	
	def __init__(
		self,
		client: Client,
		table: str,
		content_column: str = "content",
		embedding_column: str = "embedding",
		embedding_model: str = "all-MiniLM-L6-v2",
		pipeline_name: Optional[str] = None,
	):
		"""
		Initialize RAG pipeline.
		
		Args:
			client: NeuronDB client instance
			table: Table name containing documents
			content_column: Column name containing document content
			embedding_column: Column name containing embeddings
			embedding_model: Embedding model name
			pipeline_name: Optional pipeline name for configuration
		"""
		self.client = client
		self.table = table
		self.content_column = content_column
		self.embedding_column = embedding_column
		self.embedding_model = embedding_model
		self.pipeline_name = pipeline_name or f"rag_{table}"
		self.vector_store = VectorStore(client, table, embedding_column)
		
		self._ensure_pipeline_config()
	
	def _ensure_pipeline_config(self):
		"""Ensure RAG pipeline configuration exists."""
		query = """
			INSERT INTO neurondb.rag_pipelines (
				pipeline_name, embedding_model, configuration
			) VALUES (%s, %s, %s::jsonb)
			ON CONFLICT (pipeline_name) DO NOTHING
		"""
		config = {
			"table": self.table,
			"content_column": self.content_column,
			"embedding_column": self.embedding_column,
		}
		self.client.execute(query, (self.pipeline_name, self.embedding_model, str(config)))
	
	def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
		"""
		Generate embeddings for texts.
		
		Args:
			texts: List of text strings
		
		Returns:
			List of embedding vectors
		"""
		embeddings = []
		for text in texts:
			query = f"""
				SELECT embed_text(%s, %s)::vector as embedding
			"""
			result = self.client.execute_one(query, (text, self.embedding_model))
			if result:
				# Parse vector string to list
				vec_str = result["embedding"]
				# Remove brackets and split
				vec_str = vec_str.strip("[]")
				embedding = [float(x) for x in vec_str.split(",")]
				embeddings.append(embedding)
		return embeddings
	
	def retrieve(
		self,
		query: str,
		k: int = 5,
		rerank: bool = False,
		rerank_model: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		"""
		Retrieve relevant documents for query.
		
		Args:
			query: Query text
			k: Number of documents to retrieve
			rerank: Whether to rerank results
			rerank_model: Reranking model name (optional)
		
		Returns:
			List of retrieved documents with scores
		"""
		# Generate query embedding
		query_embeddings = self.generate_embeddings([query])
		query_vector = query_embeddings[0]
		
		# Vector search
		results = self.vector_store.search(query_vector, k=k * 2 if rerank else k)
		
		# Rerank if requested
		if rerank and results:
			texts = [r.get(self.content_column, "") for r in results]
			if rerank_model:
				query_sql = """
					SELECT idx, score FROM rerank_cross_encoder(
						%s,  -- query
						%s,  -- texts array
						%s,  -- model
						%s   -- k
					)
				"""
				texts_array = "ARRAY[" + ",".join(f"'{t}'" for t in texts) + "]::text[]"
				rerank_results = self.client.execute(
					query_sql.format(texts_array),
					(query, rerank_model, k)
				)
				# Reorder results based on reranking
				reranked = []
				for rr in rerank_results:
					idx = rr["idx"]
					if idx < len(results):
						result = results[idx]
						result["rerank_score"] = rr["score"]
						reranked.append(result)
				results = reranked[:k]
		
		return results
	
	def generate(
		self,
		query: str,
		context: Optional[List[Dict[str, Any]]] = None,
		model: str = "gpt-3.5-turbo",
		max_tokens: int = 500,
	) -> str:
		"""
		Generate answer using LLM with retrieved context.
		
		Args:
			query: Query text
			context: Retrieved documents (if None, will retrieve automatically)
			model: LLM model name
			max_tokens: Maximum tokens in response
		
		Returns:
			Generated answer text
		"""
		# Retrieve context if not provided
		if context is None:
			context = self.retrieve(query, k=5)
		
		# Build context text
		context_text = "\n\n".join([
			f"Document {i+1}:\n{r.get(self.content_column, '')}"
			for i, r in enumerate(context)
		])
		
		# Build prompt
		prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
		
		# Generate using LLM (simplified - would need actual LLM integration)
		query_sql = """
			SELECT llm_complete(%s, %s, %s) as completion
		"""
		result = self.client.execute_one(
			query_sql,
			(model, prompt, max_tokens)
		)
		
		return result["completion"] if result else ""
	
	def chat(
		self,
		messages: List[Dict[str, str]],
		k: int = 5,
	) -> str:
		"""
		Chat interface with RAG context retrieval.
		
		Args:
			messages: List of message dicts with "role" and "content"
			k: Number of documents to retrieve
		
		Returns:
			Response text
		"""
		# Get last user message
		last_user_msg = None
		for msg in reversed(messages):
			if msg.get("role") == "user":
				last_user_msg = msg.get("content")
				break
		
		if not last_user_msg:
			return ""
		
		# Retrieve context
		context = self.retrieve(last_user_msg, k=k, rerank=True)
		
		# Generate response
		response = self.generate(last_user_msg, context=context)
		return response


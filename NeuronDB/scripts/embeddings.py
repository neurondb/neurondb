#!/usr/bin/env python3
"""
embeddings.py
    Embedding generation and vector index management

Copyright (c) 2024-2025, neurondb, Inc.
"""

import psycopg2
from typing import Tuple, Optional


class EmbeddingGenerator:
	"""Generate embeddings for document chunks"""
	
	def __init__(self, conn, table_prefix: str = 'docs', model: str = 'all-MiniLM-L6-v2'):
		self.conn = conn
		self.table_prefix = table_prefix
		self.model = model
	
	def get_pending_count(self) -> int:
		"""Get count of chunks without embeddings"""
		cur = self.conn.cursor()
		try:
			cur.execute(f"""
				SELECT COUNT(*) 
				FROM {self.table_prefix}_chunks 
				WHERE embedding IS NULL
			""")
			return cur.fetchone()[0]
		finally:
			cur.close()
	
	def generate_embeddings(self, batch_size: int = 100, 
		verbose: bool = False) -> Tuple[int, int]:
		"""
		Generate embeddings for all chunks without embeddings
		
		Args:
			batch_size: Number of chunks to process per batch
			verbose: Print progress messages
		
		Returns:
			(processed_count, error_count)
		"""
		cur = self.conn.cursor()
		
		total_chunks = self.get_pending_count()
		if total_chunks == 0:
			if verbose:
				print("All chunks already have embeddings")
			return 0, 0
		
		if verbose:
			print(f"Generating embeddings for {total_chunks} chunks...")
		
		processed = 0
		errors = 0
		
		# Process in batches
		while True:
			cur.execute(f"""
				SELECT chunk_id, chunk_text 
				FROM {self.table_prefix}_chunks 
				WHERE embedding IS NULL
				ORDER BY chunk_id
				LIMIT %s
			""", (batch_size,))
			
			chunks = cur.fetchall()
			if not chunks:
				break
			
			for chunk_id, chunk_text in chunks:
				try:
					cur.execute(f"""
						UPDATE {self.table_prefix}_chunks
						SET embedding = embed_text(%s, %s)
						WHERE chunk_id = %s
					""", (chunk_text, self.model, chunk_id))
					processed += 1
				except Exception as e:
					errors += 1
					if verbose:
						print(f"Error generating embedding for chunk {chunk_id}: {e}")
			
			self.conn.commit()
			
			if verbose and processed % (batch_size * 10) == 0:
				print(f"Processed {processed}/{total_chunks} chunks...")
		
		cur.close()
		
		if verbose:
			print(f"Embedding generation complete: {processed} processed, {errors} errors")
		
		return processed, errors
	
	def get_embedding_statistics(self) -> Optional[Tuple]:
		"""Get embedding statistics"""
		cur = self.conn.cursor()
		try:
			cur.execute(f"""
				SELECT 
					COUNT(*) AS total_chunks,
					COUNT(*) FILTER (WHERE embedding IS NOT NULL) 
						AS chunks_with_embeddings,
					COUNT(*) FILTER (WHERE embedding IS NULL) 
						AS chunks_without_embeddings
				FROM {self.table_prefix}_chunks
			""")
			return cur.fetchone()
		finally:
			cur.close()


class VectorIndexManager:
	"""Manage vector indexes for similarity search"""
	
	def __init__(self, conn, table_prefix: str = 'docs'):
		self.conn = conn
		self.table_prefix = table_prefix
	
	def create_index(self, m: int = 16, ef_construction: int = 200) -> bool:
		"""
		Create HNSW vector index
		
		Args:
			m: Number of bi-directional links for each element
			ef_construction: Size of the dynamic candidate list
		
		Returns:
			True if successful, False otherwise
		"""
		cur = self.conn.cursor()
		
		try:
			# Drop existing index
			cur.execute(f"""
				DROP INDEX IF EXISTS idx_{self.table_prefix}_chunks_embedding
			""")
			
			# Create HNSW index
			cur.execute(f"""
				CREATE INDEX idx_{self.table_prefix}_chunks_embedding 
					ON {self.table_prefix}_chunks 
					USING hnsw (embedding vector_l2_ops)
					WITH (m = {m}, ef_construction = {ef_construction})
			""")
			
			# Analyze table for query planner
			cur.execute(f"ANALYZE {self.table_prefix}_chunks")
			
			self.conn.commit()
			cur.close()
			return True
		except Exception as e:
			self.conn.rollback()
			cur.close()
			raise e
	
	def index_exists(self) -> bool:
		"""Check if vector index exists"""
		cur = self.conn.cursor()
		try:
			cur.execute(f"""
				SELECT EXISTS (
					SELECT 1 FROM pg_indexes 
					WHERE indexname = 'idx_{self.table_prefix}_chunks_embedding'
				)
			""")
			return cur.fetchone()[0]
		finally:
			cur.close()


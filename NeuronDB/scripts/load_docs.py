#!/usr/bin/env python3
"""
load_docs.py
    Main entry point for loading documentation files and creating embeddings
    using NeurondB

Usage:
    python3 load_docs.py [OPTIONS]

Options:
    -d, --directory DIR     Directory containing documentation files
    -D, --database DB       Database name (default: postgres)
    -U, --user USER         Database user (default: postgres)
    -H, --host HOST         Database host (default: localhost)
    -p, --port PORT         Database port (default: 5432)
    -m, --model MODEL       Embedding model (default: all-MiniLM-L6-v2)
    -c, --chunk-size SIZE   Chunk size in characters (default: 1000)
    -o, --overlap SIZE      Chunk overlap in characters (default: 200)
    -t, --table-prefix PFX   Table name prefix (default: docs)
    --skip-embeddings        Skip embedding generation
    --skip-index            Skip index creation
    --batch-size SIZE       Batch size for embeddings (default: 100)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Example:
    python3 load_docs.py -d /path/to/docs -D mydb -U myuser

Copyright (c) 2024-2025, neurondb, Inc.
"""

import sys
import os
import argparse
import psycopg2
from pathlib import Path

# Import modular components
from doc_processor import FileProcessor
from chunking import SQLChunker
from db_operations import DatabaseSchema, DocumentLoader, StatisticsManager
from embeddings import EmbeddingGenerator, VectorIndexManager


class DocumentationLoader:
	"""Main class for loading documentation"""
	
	def __init__(self, db_host: str, db_port: int, db_name: str, 
		db_user: str, table_prefix: str = 'docs'):
		"""Initialize with database connection parameters"""
		self.db_host = db_host
		self.db_port = db_port
		self.db_name = db_name
		self.db_user = db_user
		self.table_prefix = table_prefix
		self.conn = None
	
	def connect(self):
		"""Establish database connection"""
		self.conn = psycopg2.connect(
			host=self.db_host,
			port=self.db_port,
			database=self.db_name,
			user=self.db_user,
			password=os.environ.get('PGPASSWORD', '')
		)
	
	def disconnect(self):
		"""Close database connection"""
		if self.conn:
			self.conn.close()
	
	def load_files(self, doc_dir: Path, verbose: bool = False) -> tuple:
		"""
		Load files from directory into database
		
		Returns:
			(loaded_count, skipped_count, error_count)
		"""
		processor = FileProcessor()
		loader = DocumentLoader(self.conn, self.table_prefix)
		
		files = processor.find_documentation_files(doc_dir)
		total = len(files)
		
		if total == 0:
			print("No documentation files found")
			return 0, 0, 0
		
		print(f"Processing {total} files...")
		
		loaded = 0
		skipped = 0
		errors = 0
		
		for i, filepath in enumerate(files, 1):
			if verbose and i % 100 == 0:
				print(f"Progress: {i}/{total} files processed...")
			
			result = processor.process_file(filepath)
			if result is None:
				skipped += 1
				continue
			
			title, content, metadata = result
			
			try:
				loader.insert_document(
					filepath=str(filepath),
					filename=filepath.name,
					title=title,
					content=content,
					file_size=filepath.stat().st_size,
					file_type=metadata['file_type'],
					metadata=metadata
				)
				loaded += 1
			except Exception as e:
				errors += 1
				if verbose:
					print(f"Error loading {filepath}: {e}")
		
		return loaded, skipped, errors
	
	def chunk_documents(self, chunk_size: int, chunk_overlap: int, 
		verbose: bool = False) -> tuple:
		"""
		Chunk all documents in database
		
		Returns:
			Statistics tuple (total_chunks, avg_tokens, min_tokens, max_tokens)
		"""
		chunker = SQLChunker(self.table_prefix)
		cur = self.conn.cursor()
		
		# Create chunking function
		if verbose:
			print("Creating chunking function...")
		cur.execute(chunker.create_chunking_function())
		
		# Chunk documents
		if verbose:
			print("Chunking documents...")
		cur.execute(chunker.chunk_documents_sql(chunk_size, chunk_overlap))
		
		# Get statistics
		cur.execute(chunker.get_chunk_statistics_sql())
		stats = cur.fetchone()
		
		self.conn.commit()
		cur.close()
		
		return stats
	
	def generate_embeddings(self, model: str, batch_size: int = 100,
		verbose: bool = False) -> tuple:
		"""
		Generate embeddings for all chunks
		
		Returns:
			(processed_count, error_count)
		"""
		generator = EmbeddingGenerator(self.conn, self.table_prefix, model)
		return generator.generate_embeddings(batch_size, verbose)
	
	def create_index(self, verbose: bool = False) -> bool:
		"""Create vector index for similarity search"""
		index_manager = VectorIndexManager(self.conn, self.table_prefix)
		
		if verbose:
			print("Creating vector index...")
		
		try:
			index_manager.create_index()
			return True
		except Exception as e:
			if verbose:
				print(f"Warning: Index creation failed: {e}")
			return False
	
	def get_statistics(self) -> tuple:
		"""Get final statistics"""
		stats_manager = StatisticsManager(self.conn, self.table_prefix)
		return stats_manager.get_statistics()


def main():
	parser = argparse.ArgumentParser(
		description='Load documentation files and create embeddings using NeurondB',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Basic usage
  python3 load_docs.py -d /path/to/docs

  # Custom database and model
  python3 load_docs.py -d /path/to/docs -D mydb -U myuser -m all-mpnet-base-v2

  # Skip embeddings (just load and chunk)
  python3 load_docs.py -d /path/to/docs --skip-embeddings
		"""
	)
	
	parser.add_argument('-d', '--directory', required=True,
		help='Directory containing documentation files')
	parser.add_argument('-D', '--database', default='postgres',
		help='Database name (default: postgres)')
	parser.add_argument('-U', '--user', default='postgres',
		help='Database user (default: postgres)')
	parser.add_argument('-H', '--host', default='localhost',
		help='Database host (default: localhost)')
	parser.add_argument('-p', '--port', type=int, default=5432,
		help='Database port (default: 5432)')
	parser.add_argument('-m', '--model', default='all-MiniLM-L6-v2',
		help='Embedding model (default: all-MiniLM-L6-v2)')
	parser.add_argument('-c', '--chunk-size', type=int, default=1000,
		help='Chunk size in characters (default: 1000)')
	parser.add_argument('-o', '--overlap', type=int, default=200,
		help='Chunk overlap in characters (default: 200)')
	parser.add_argument('-t', '--table-prefix', default='docs',
		help='Table name prefix (default: docs)')
	parser.add_argument('--skip-embeddings', action='store_true',
		help='Skip embedding generation')
	parser.add_argument('--skip-index', action='store_true',
		help='Skip index creation')
	parser.add_argument('--batch-size', type=int, default=100,
		help='Batch size for embeddings (default: 100)')
	parser.add_argument('-v', '--verbose', action='store_true',
		help='Verbose output')
	
	args = parser.parse_args()
	
	# Validate directory
	doc_dir = Path(args.directory)
	if not doc_dir.exists() or not doc_dir.is_dir():
		print(f"Error: Directory does not exist: {doc_dir}")
		sys.exit(1)
	
	# Initialize loader
	loader = DocumentationLoader(
		db_host=args.host,
		db_port=args.port,
		db_name=args.database,
		db_user=args.user,
		table_prefix=args.table_prefix
	)
	
	try:
		# Connect to database
		loader.connect()
		
		# Create schema
		print("Creating schema...")
		schema = DatabaseSchema(args.table_prefix)
		schema.create_schema(loader.conn)
		
		# Load files
		print("Loading files...")
		loaded, skipped, errors = loader.load_files(doc_dir, args.verbose)
		print(f"Loaded: {loaded}, Skipped: {skipped}, Errors: {errors}")
		
		# Chunk documents
		print("Chunking documents...")
		chunk_stats = loader.chunk_documents(
			args.chunk_size, 
			args.overlap, 
			args.verbose
		)
		print(f"Created {chunk_stats[0]} chunks "
			f"(avg tokens: {chunk_stats[1]})")
		
		# Generate embeddings
		if not args.skip_embeddings:
			print("Generating embeddings...")
			processed, embed_errors = loader.generate_embeddings(
				args.model,
				args.batch_size,
				args.verbose
			)
			print(f"Embeddings: {processed} processed, {embed_errors} errors")
		else:
			print("Skipping embedding generation")
		
		# Create index
		if not args.skip_index and not args.skip_embeddings:
			if loader.create_index(args.verbose):
				print("Index created successfully")
		else:
			print("Skipping index creation")
		
		# Final statistics
		print("\n=== Final Statistics ===")
		stats = loader.get_statistics()
		if stats:
			print(f"Total documents: {stats[0]}")
			print(f"Total chunks: {stats[1]}")
			print(f"Chunks with embeddings: {stats[2]}")
			print(f"Average document length: {int(stats[3])} characters")
			print(f"Total content size: {stats[4]} characters")
		
		print("\nDocumentation loading complete!")
		print(f"\nQuery your documentation using:")
		print(f"  SELECT * FROM {args.table_prefix}_documents LIMIT 10;")
		print(f"  SELECT * FROM {args.table_prefix}_chunks LIMIT 10;")
		print(f"\nFor semantic search:")
		print(f"  WITH q AS (SELECT embed_text('your query', '{args.model}') AS emb)")
		print(f"  SELECT c.chunk_text, c.embedding <-> q.emb AS distance")
		print(f"  FROM {args.table_prefix}_chunks c, q")
		print(f"  ORDER BY distance LIMIT 10;")
		
	except Exception as e:
		print(f"Error: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		sys.exit(1)
	finally:
		loader.disconnect()


if __name__ == '__main__':
	main()

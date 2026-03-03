"""
NeuronDB Semantic Search Example
Complete working example for semantic search over document collections

Requirements:
    pip install psycopg2-binary sentence-transformers numpy
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuration (defaults match Docker Compose setup)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


class DocumentIngester:
    """Ingests documents and generates embeddings"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.conn = psycopg2.connect(**db_config)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._setup_schema()
    
    def _setup_schema(self):
        """Create necessary tables"""
        with self.conn.cursor() as cur:
            # Create documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    chunk_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create HNSW index for fast similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            self.conn.commit()
            print("✓ Schema created")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = ' '.join(words[i:i + CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def ingest_document(self, filepath: Path):
        """Ingest a single document"""
        print(f"Processing: {filepath.name}")
        
        # Read document
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk document
        chunks = self.chunk_text(content)
        print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings and store
        with self.conn.cursor() as cur:
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.model.encode(chunk)
                embedding_list = embedding.tolist()
                
                # Store in database
                cur.execute("""
                    INSERT INTO documents (filename, content, embedding, chunk_index)
                    VALUES (%s, %s, %s, %s)
                """, (filepath.name, chunk, embedding_list, idx))
        
        self.conn.commit()
        print(f"✓ Ingested {filepath.name}")
    
    def ingest_directory(self, dirpath: Path):
        """Ingest all text/markdown files in directory"""
        files = list(dirpath.glob('**/*.txt')) + list(dirpath.glob('**/*.md'))
        
        print(f"\nFound {len(files)} documents to ingest\n")
        
        for filepath in files:
            try:
                self.ingest_document(filepath)
            except Exception as e:
                print(f"✗ Error processing {filepath.name}: {e}")
        
        print(f"\n✓ Ingestion complete!")
    
    def close(self):
        self.conn.close()


class SemanticSearcher:
    """Performs semantic search over documents"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.conn = psycopg2.connect(**db_config)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.model.encode(query)
        query_embedding_list = query_embedding.tolist()
        
        # Search database
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    filename,
                    content,
                    chunk_index,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding_list, query_embedding_list, limit))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'filename': row[1],
                    'content': row[2],
                    'chunk_index': row[3],
                    'similarity': float(row[4])
                })
            
            return results
    
    def close(self):
        self.conn.close()


def create_sample_documents():
    """Create sample documents for testing"""
    sample_dir = Path('sample_docs')
    sample_dir.mkdir(exist_ok=True)
    
    documents = {
        'machine_learning.md': """
# Machine Learning

Machine learning is a subset of artificial intelligence that focuses on 
developing algorithms and statistical models that enable computers to improve 
their performance on a specific task through experience.

## Types of Machine Learning

1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through interaction with an environment

## Applications

Machine learning is used in various applications including:
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles
""",
        'databases.md': """
# Database Systems

A database is an organized collection of structured information or data, 
typically stored electronically in a computer system. Database management 
systems (DBMS) are software that interact with users, applications, and 
the database itself to capture and analyze data.

## Types of Databases

1. Relational Databases: Use tables with rows and columns
2. NoSQL Databases: Non-relational, flexible schema
3. Vector Databases: Optimized for similarity search
4. Time-series Databases: Optimized for time-stamped data

## PostgreSQL

PostgreSQL is a powerful, open-source object-relational database system with 
over 30 years of active development. It has a strong reputation for reliability, 
feature robustness, and performance.
""",
        'vector_search.md': """
# Vector Search

Vector search, also known as similarity search, is a method of finding similar 
items in a dataset by representing items as vectors in high-dimensional space 
and using distance metrics to measure similarity.

## How It Works

1. Convert items (text, images, etc.) to vectors using embeddings
2. Store vectors in a specialized index structure
3. Query by converting the query to a vector
4. Find nearest neighbors using distance metrics

## Distance Metrics

- Cosine Similarity: Measures angle between vectors
- Euclidean Distance: Straight-line distance
- Dot Product: Inner product of vectors

## Use Cases

Vector search enables:
- Semantic search over documents
- Image similarity search
- Recommendation systems
- Anomaly detection
"""
    }
    
    for filename, content in documents.items():
        filepath = sample_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✓ Created {len(documents)} sample documents in {sample_dir}/")
    return sample_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuronDB Semantic Search Example')
    parser.add_argument('command', choices=['ingest', 'search', 'demo'],
                       help='Command to run')
    parser.add_argument('--input-dir', type=str, default='sample_docs',
                       help='Input directory for documents')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--limit', type=int, default=5,
                       help='Number of results to return')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        print("=" * 70)
        print("  NeuronDB Semantic Search Demo")
        print("=" * 70)
        print()
        
        # Create sample documents
        sample_dir = create_sample_documents()
        print()
        
        # Ingest documents
        print("=" * 70)
        print("  Step 1: Ingesting Documents")
        print("=" * 70)
        ingester = DocumentIngester(DB_CONFIG)
        ingester.ingest_directory(sample_dir)
        ingester.close()
        print()
        
        # Perform searches
        print("=" * 70)
        print("  Step 2: Semantic Search")
        print("=" * 70)
        
        searcher = SemanticSearcher(DB_CONFIG)
        
        queries = [
            "What is machine learning?",
            "Tell me about database systems",
            "How does similarity search work?"
        ]
        
        for query in queries:
            print(f"\nQuery: \"{query}\"")
            print("-" * 70)
            
            results = searcher.search(query, limit=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['filename']} (chunk {result['chunk_index']})")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   {result['content'][:200]}...")
        
        searcher.close()
        print()
        print("=" * 70)
        print("  Demo Complete!")
        print("=" * 70)
    
    elif args.command == 'ingest':
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)
        
        ingester = DocumentIngester(DB_CONFIG)
        ingester.ingest_directory(input_dir)
        ingester.close()
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query is required for search command")
            sys.exit(1)
        
        searcher = SemanticSearcher(DB_CONFIG)
        results = searcher.search(args.query, limit=args.limit)
        
        print(f"\nSearch Results for: \"{args.query}\"")
        print("=" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']} (chunk {result['chunk_index']})")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   {result['content'][:300]}...")
        
        searcher.close()


if __name__ == '__main__':
    main()




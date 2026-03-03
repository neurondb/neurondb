#!/usr/bin/env python3
"""
NeuronDB RAGAS (Retrieval Augmented Generation Assessment) Integration

This benchmark tests RAG answer quality using the RAGAS framework.
RAGAS evaluates:
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevance
- Answer Correctness
- Answer Semantic Similarity
"""

import sys
import os
import argparse
import json
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
        answer_semantic_similarity,
    )
    from datasets import Dataset
except ImportError:
    print("Error: ragas not installed. Install with: pip install ragas")
    sys.exit(1)


class NeuronDBRAGSystem:
    """
    RAG system using NeuronDB for retrieval and optional LLM for generation.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 5432,
        database: str = "neurondb",
        user: str = "pge",
        password: Optional[str] = None,
        table_name: str = "rag_documents",
        index_type: str = "hnsw",
        top_k: int = 5,
    ):
        """
        Initialize NeuronDB RAG system.
        
        Args:
            model_name: Embedding model name
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            table_name: Table name for documents
            index_type: Index type
            top_k: Number of documents to retrieve
        """
        self.model_name = model_name
        self.table_name = table_name
        self.index_type = index_type
        self.top_k = top_k
        
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        self.conn.autocommit = True
        
        # Verify NeuronDB extension
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'neurondb';")
            if not cur.fetchone():
                raise RuntimeError("NeuronDB extension not installed")
            
            # Enable fail_open for benchmarks to allow fallback embeddings if API fails
            try:
                cur.execute("SET neurondb.llm_fail_open = true;")
            except Exception:
                pass  # Ignore if setting doesn't exist
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of dicts with 'id' and 'text' keys
        """
        if not documents:
            return
        
        # Get embedding dimension from first document
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT neurondb_embed(%s, %s)",
                (documents[0]['text'][:100], self.model_name)  # Sample for dimension
            )
            vec_str = cur.fetchone()[0]
            dim = len(self._parse_vector(vec_str))
        
        with self.conn.cursor() as cur:
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    doc_id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding vector({dim})
                );
            """)
            
            # Insert documents
            for doc in documents:
                # Generate embedding
                cur.execute(
                    "SELECT neurondb_embed(%s, %s)",
                    (doc['text'], self.model_name)
                )
                vec_str = cur.fetchone()[0]
                
                # Insert document
                cur.execute(f"""
                    INSERT INTO {self.table_name} (doc_id, text, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding;
                """, (doc['id'], doc['text'], vec_str))
            
            # Create index
            self._create_index(cur, dim)
    
    def _create_index(self, cursor, dim: int):
        """Create vector index."""
        index_name = f"{self.table_name}_embedding_idx"
        
        cursor.execute(f"""
            DROP INDEX IF EXISTS {index_name};
        """)
        
        if self.index_type == "hnsw":
            cursor.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200);
            """)
        elif self.index_type == "ivfflat":
            cursor.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
        
        Returns:
            List of retrieved document texts
        """
        with self.conn.cursor() as cur:
            # Generate query embedding
            cur.execute(
                "SELECT neurondb_embed(%s, %s)",
                (query, self.model_name)
            )
            vec_str = cur.fetchone()[0]
            
            # Search
            cur.execute(f"""
                SELECT text
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (vec_str, self.top_k))
            
            results = [row[0] for row in cur.fetchall()]
            return results
    
    def _parse_vector(self, vec_str: str) -> List[float]:
        """Parse vector string to list of floats."""
        if vec_str.startswith('[') and vec_str.endswith(']'):
            vec_str = vec_str[1:-1]
        return [float(x.strip()) for x in vec_str.split(',')]


def run_ragas_benchmark(
    dataset_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "./results",
    host: str = "localhost",
    port: int = 5432,
    database: str = "neurondb",
    user: str = "pge",
    password: Optional[str] = None,
    index_type: str = "hnsw",
    top_k: int = 5,
    **kwargs
) -> Dict:
    """
    Run RAGAS benchmark using NeuronDB RAG system.
    
    Args:
        dataset_path: Path to dataset JSON file with questions, answers, contexts
        model_name: Embedding model name
        output_dir: Directory for results
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        index_type: Index type
        top_k: Number of documents to retrieve
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Expected format: list of dicts with keys:
    # - question: str
    # - answer: str (ground truth or generated)
    # - contexts: List[str] (retrieved contexts)
    # - ground_truth: str (optional, for answer correctness)
    
    # Initialize RAG system
    print(f"Initializing NeuronDB RAG system with model: {model_name}")
    rag_system = NeuronDBRAGSystem(
        model_name=model_name,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        index_type=index_type,
        top_k=top_k,
    )
    
    # If dataset has documents, add them
    if 'documents' in data:
        print(f"Adding {len(data['documents'])} documents to RAG system...")
        rag_system.add_documents(data['documents'])
    
    # Prepare evaluation dataset
    eval_data = []
    for item in data['examples']:
        question = item['question']
        
        # Retrieve contexts if not provided
        if 'contexts' not in item or not item['contexts']:
            contexts = rag_system.retrieve(question)
        else:
            contexts = item['contexts']
        
        eval_item = {
            'question': question,
            'answer': item.get('answer', ''),
            'contexts': contexts,
        }
        
        if 'ground_truth' in item:
            eval_item['ground_truth'] = item['ground_truth']
        
        eval_data.append(eval_item)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(eval_data)
    
    # Define metrics
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
    
    # Add answer correctness if ground truth available
    if 'ground_truth' in eval_data[0]:
        metrics.append(answer_correctness)
        metrics.append(answer_semantic_similarity)
    
    # Run evaluation
    print(f"Running RAGAS evaluation on {len(eval_data)} examples...")
    start_time = time.time()
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        **kwargs
    )
    
    eval_time = time.time() - start_time
    
    # Save results
    summary = {
        "model": model_name,
        "index_type": index_type,
        "top_k": top_k,
        "num_examples": len(eval_data),
        "eval_time": eval_time,
        "metrics": result,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    summary_file = os.path.join(output_dir, f"ragas_{summary['timestamp']}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nEvaluation completed in {eval_time:.2f} seconds")
    print(f"Results saved to: {summary_file}")
    
    # Print metrics
    print("\n" + "="*60)
    print("RAGAS Benchmark Results")
    print("="*60)
    for metric_name, metric_value in result.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")
    print("="*60)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS benchmark with NeuronDB")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Database host")
    parser.add_argument("--port", type=int, default=5432,
                       help="Database port")
    parser.add_argument("--database", type=str, default="neurondb",
                       help="Database name")
    parser.add_argument("--user", type=str, default="pge",
                       help="Database user")
    parser.add_argument("--password", type=str, default=None,
                       help="Database password")
    parser.add_argument("--index", type=str, default="hnsw",
                       choices=["hnsw", "ivfflat"],
                       help="Index type")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_ragas_benchmark(
        dataset_path=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        index_type=args.index,
        top_k=args.top_k,
    )


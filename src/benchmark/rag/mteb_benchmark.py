#!/usr/bin/env python3
"""
NeuronDB MTEB (Massive Text Embedding Benchmark) Integration

This benchmark tests embedding quality using the MTEB framework.
MTEB evaluates embeddings across multiple tasks including:
- Classification
- Clustering
- Pair Classification
- Reranking
- Retrieval
- STS (Semantic Text Similarity)
- Summarization
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
    from mteb import MTEB
    import mteb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install mteb sentence-transformers")
    sys.exit(1)


class NeuronDBEmbeddingModel:
    """
    Wrapper for NeuronDB embedding functions to work with MTEB.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 5432,
        database: str = "neurondb",
        user: str = "pge",
        password: Optional[str] = None,
    ):
        """
        Initialize NeuronDB embedding model wrapper.
        
        Args:
            model_name: Name of the embedding model to use
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.model_name = model_name
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
    
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences into embeddings using NeuronDB.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress (not used)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            numpy array of embeddings (n_sentences, embedding_dim)
        """
        embeddings = []
        
        # Process in batches - create new cursor for each batch
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Use batch embedding if available, otherwise individual calls
            try:
                cur = self.conn.cursor()
                try:
                    # Try batch embedding function
                    placeholders = ','.join(['%s'] * len(batch))
                    cur.execute(f"""
                        SELECT neurondb_embed_batch(ARRAY[{placeholders}], %s)
                    """, batch + [self.model_name])
                    
                    result = cur.fetchone()[0]
                    if result:
                        # Parse array of vectors
                        batch_embeddings = self._parse_vector_array(result)
                        embeddings.extend(batch_embeddings)
                    else:
                        # Fallback to individual embeddings
                        for sentence in batch:
                            cur.execute(
                                "SELECT neurondb_embed(%s, %s)",
                                (sentence, self.model_name)
                            )
                            vec_str = cur.fetchone()[0]
                            embeddings.append(self._parse_vector(vec_str))
                finally:
                    cur.close()
            except Exception:
                # Fallback to individual embeddings
                cur = self.conn.cursor()
                try:
                    for sentence in batch:
                        cur.execute(
                            "SELECT neurondb_embed(%s, %s)",
                            (sentence, self.model_name)
                        )
                        vec_str = cur.fetchone()[0]
                        embeddings.append(self._parse_vector(vec_str))
                finally:
                    cur.close()
        
        return np.array(embeddings)
    
    def _parse_vector(self, vec_str: str) -> List[float]:
        """Parse vector string format [1.0, 2.0, 3.0] to list of floats."""
        if vec_str.startswith('[') and vec_str.endswith(']'):
            vec_str = vec_str[1:-1]
        return [float(x.strip()) for x in vec_str.split(',')]
    
    def _parse_vector_array(self, vec_array_str: str) -> List[List[float]]:
        """Parse array of vectors."""
        # This is a simplified parser - adjust based on actual format
        vectors = []
        # Remove outer braces
        if vec_array_str.startswith('{') and vec_array_str.endswith('}'):
            vec_array_str = vec_array_str[1:-1]
        
        # Split by vector boundaries (rough approximation)
        # This may need adjustment based on actual PostgreSQL array format
        parts = vec_array_str.split('],[')
        for part in parts:
            part = part.strip('[]')
            vectors.append(self._parse_vector(part))
        
        return vectors


def run_mteb_benchmark(
    model_name: str = "all-MiniLM-L6-v2",
    tasks: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    output_dir: str = "./results",
    host: str = "localhost",
    port: int = 5432,
    database: str = "neurondb",
    user: str = "pge",
    password: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Run MTEB benchmark using NeuronDB embeddings.
    
    Args:
        model_name: Embedding model name
        tasks: Specific tasks to run (None = all)
        task_types: Task types to run (None = all)
        output_dir: Directory for results
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing NeuronDB embedding model: {model_name}")
    model = NeuronDBEmbeddingModel(
        model_name=model_name,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
    
    # Wrap for MTEB - Use SentenceTransformer instance and replace encode method
    # This ensures MTEB recognizes it as an Encoder, not a SearchProtocol
    # SentenceTransformer takes model name as positional argument
    embedding_model = SentenceTransformer(model_name)
    # Replace the encode method with our NeuronDB implementation
    embedding_model.encode = model.encode
    
    # Get tasks for MTEB
    # For text-only embedding models, we need to filter to text-to-text tasks only
    # Exclude: retrieval (needs SearchInterface), image tasks (i2i, i2t, etc.), multimodal
    if tasks:
        # Get specific tasks
        task_list = mteb.get_tasks(tasks)
    elif task_types:
        # Get tasks by type
        all_tasks = mteb.get_tasks(task_types=task_types)
    else:
        # Default: Get text-based tasks only (Classification, Clustering, STS, Summarization)
        # Exclude retrieval and image/multimodal tasks
        all_tasks = mteb.get_tasks(
            task_types=['Classification', 'Clustering', 'STS', 'Summarization', 'PairClassification'],
            modalities=['text']  # Only text modality
        )
        task_list = all_tasks
    
    # Filter out tasks that require SearchInterface or are not text-to-text
    filtered_tasks = []
    for task in task_list:
        # Skip retrieval tasks
        if task.metadata.type in ['Retrieval', 'Any2AnyRetrieval', 'Any2AnyMultilingualRetrieval', 'Reranking']:
            continue
        # Skip image/multimodal tasks (check categories)
        if task.metadata.category and any(cat in task.metadata.category for cat in ['i2i', 'i2t', 't2i', 'it2i', 'it2t', 'i2it', 't2it', 'it2it']):
            continue
        # Only keep text-to-text or text-to-class tasks
        if task.metadata.category and all(cat in ['t2t', 't2c'] for cat in task.metadata.category):
            filtered_tasks.append(task)
        elif not task.metadata.category:  # Some tasks don't have category, include them
            filtered_tasks.append(task)
    
    task_list = filtered_tasks
    
    # Initialize MTEB with tasks
    evaluation = MTEB(tasks=task_list)
    
    # Run evaluation
    print(f"Running MTEB benchmark with {len(evaluation.tasks)} tasks...")
    start_time = time.time()
    
    results = evaluation.run(
        embedding_model,
        output_folder=output_dir,
        verbosity=2,
        **kwargs
    )
    
    total_time = time.time() - start_time
    
    # Save summary
    summary = {
        "model": model_name,
        "total_time": total_time,
        "num_tasks": len(evaluation.tasks),
        "results": results,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    summary_file = os.path.join(output_dir, f"mteb_summary_{summary['timestamp']}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nBenchmark completed in {total_time:.2f} seconds")
    print(f"Results saved to: {summary_file}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTEB benchmark with NeuronDB")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                       help="Specific tasks to run")
    parser.add_argument("--task-types", type=str, nargs="+", default=None,
                       choices=["Classification", "Clustering", "PairClassification",
                               "Reranking", "Retrieval", "STS", "Summarization"],
                       help="Task types to run")
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
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_mteb_benchmark(
        model_name=args.model,
        tasks=args.tasks,
        task_types=args.task_types,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("MTEB Benchmark Summary")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"Tasks: {results['num_tasks']}")
    print("="*60)


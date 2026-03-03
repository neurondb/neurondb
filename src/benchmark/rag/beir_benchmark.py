#!/usr/bin/env python3
"""
NeuronDB BEIR (Benchmarking IR) Integration

This benchmark tests retrieval quality using the BEIR framework.
BEIR evaluates retrieval systems across diverse domains and tasks.
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
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
except ImportError:
    print("Error: beir not installed. Install with: pip install beir")
    sys.exit(1)


class NeuronDBRetriever(DRES):
    """
    BEIR-compatible retriever using NeuronDB for dense retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 5432,
        database: str = "neurondb",
        user: str = "pge",
        password: Optional[str] = None,
        table_name: str = "beir_documents",
        index_type: str = "hnsw",
        index_params: Optional[dict] = None,
    ):
        """
        Initialize NeuronDB retriever.
        
        Args:
            model_name: Embedding model name
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            table_name: Table name for storing documents
            index_type: Index type ("hnsw", "ivfflat")
            index_params: Index parameters
        """
        self.model_name = model_name
        self.table_name = table_name
        self.index_type = index_type
        self.index_params = index_params or {}
        
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
        
        self._ensure_table()
    
    def _ensure_table(self):
        """Create table for storing documents and embeddings."""
        with self.conn.cursor() as cur:
            # Table will be created when we know the dimension
            pass
    
    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode queries into embeddings."""
        return self._encode_texts(queries, batch_size)
    
    def encode_corpus(
        self,
        corpus: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode corpus documents into embeddings."""
        return self._encode_texts(corpus, batch_size)
    
    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode texts using NeuronDB."""
        embeddings = []
        
        with self.conn.cursor() as cur:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                for text in batch:
                    cur.execute(
                        "SELECT neurondb_embed(%s, %s)",
                        (text, self.model_name)
                    )
                    vec_str = cur.fetchone()[0]
                    embeddings.append(self._parse_vector(vec_str))
        
        return np.array(embeddings)
    
    def _parse_vector(self, vec_str: str) -> List[float]:
        """Parse vector string to list of floats."""
        if vec_str.startswith('[') and vec_str.endswith(']'):
            vec_str = vec_str[1:-1]
        return [float(x.strip()) for x in vec_str.split(',')]
    
    def index(
        self,
        corpus: Dict[str, Dict[str, str]],
        corpus_embeddings: np.ndarray,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Index corpus documents in NeuronDB.
        
        Args:
            corpus: Dictionary mapping doc_id to document dict with 'text' key
            corpus_embeddings: Pre-computed embeddings for corpus
        """
        doc_ids = list(corpus.keys())
        n_docs = len(doc_ids)
        
        if n_docs == 0:
            return
        
        # Get embedding dimension
        dim = corpus_embeddings.shape[1]
        
        with self.conn.cursor() as cur:
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    doc_id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding vector({dim})
                );
            """)
            
            # Clear existing data
            cur.execute(f"TRUNCATE TABLE {self.table_name};")
            
            # Insert documents and embeddings in batches
            batch_size = 1000
            for i in range(0, n_docs, batch_size):
                batch_ids = doc_ids[i:i+batch_size]
                batch_embeddings = corpus_embeddings[i:i+batch_size]
                
                values = []
                for doc_id, embedding in zip(batch_ids, batch_embeddings):
                    text = corpus[doc_id].get('text', '')
                    vec_str = '[' + ','.join(str(float(x)) for x in embedding) + ']'
                    values.append((doc_id, text, vec_str))
                
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (doc_id, text, embedding)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding;
                    """,
                    values,
                )
            
            # Create index
            self._create_index(cur, dim)
    
    def _create_index(self, cursor, dim: int):
        """Create vector index."""
        index_name = f"{self.table_name}_embedding_idx"
        
        cursor.execute(f"""
            DROP INDEX IF EXISTS {index_name};
        """)
        
        if self.index_type == "hnsw":
            m = self.index_params.get('m', 16)
            ef_construction = self.index_params.get('ef_construction', 200)
            
            cursor.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction});
            """)
        elif self.index_type == "ivfflat":
            lists = self.index_params.get('lists', 100)
            
            cursor.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Search for documents using query embeddings.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            top_k: Number of results per query
        
        Returns:
            Dictionary mapping query_idx to dict of doc_id to score
        """
        results = {}
        
        with self.conn.cursor() as cur:
            for query_idx, query_emb in enumerate(query_embeddings):
                vec_str = '[' + ','.join(str(float(x)) for x in query_emb) + ']'
                
                cur.execute(f"""
                    SELECT doc_id, 1 - (embedding <=> %s::vector) AS score
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (vec_str, vec_str, top_k))
                
                query_results = {}
                for row in cur.fetchall():
                    doc_id, score = row
                    query_results[doc_id] = float(score)
                
                results[str(query_idx)] = query_results
        
        return results


def run_beir_benchmark(
    dataset: str,
    model_name: str = "all-MiniLM-L6-v2",
    data_path: str = "./beir_data",
    output_dir: str = "./results",
    host: str = "localhost",
    port: int = 5432,
    database: str = "neurondb",
    user: str = "pge",
    password: Optional[str] = None,
    index_type: str = "hnsw",
    top_k: int = 100,
    **kwargs
) -> Dict:
    """
    Run BEIR benchmark using NeuronDB.
    
    Args:
        dataset: BEIR dataset name (e.g., "msmarco", "nq", "scifact")
        model_name: Embedding model name
        data_path: Path to download/store BEIR datasets
        output_dir: Directory for results
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        index_type: Index type
        top_k: Number of top results to retrieve
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    
    # Download dataset if needed
    print(f"Loading BEIR dataset: {dataset}")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = os.path.join(data_path, dataset)
    
    if not os.path.exists(data_path):
        print(f"Downloading dataset to {data_path}...")
        util.download_and_unzip(url, data_path)
    
    # Load dataset
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")
    
    # Initialize retriever
    print(f"Initializing NeuronDB retriever with model: {model_name}")
    retriever = NeuronDBRetriever(
        model_name=model_name,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        index_type=index_type,
    )
    
    # Encode corpus
    print("Encoding corpus...")
    start_time = time.time()
    corpus_embeddings = retriever.encode_corpus(
        [corpus[doc_id]['text'] for doc_id in corpus.keys()],
        show_progress_bar=True,
    )
    encode_time = time.time() - start_time
    print(f"Corpus encoded in {encode_time:.2f} seconds")
    
    # Index corpus
    print("Indexing corpus...")
    start_time = time.time()
    retriever.index(corpus, corpus_embeddings)
    index_time = time.time() - start_time
    print(f"Corpus indexed in {index_time:.2f} seconds")
    
    # Encode queries
    print("Encoding queries...")
    start_time = time.time()
    query_embeddings = retriever.encode_queries(
        [queries[qid] for qid in queries.keys()],
        show_progress_bar=True,
    )
    query_encode_time = time.time() - start_time
    print(f"Queries encoded in {query_encode_time:.2f} seconds")
    
    # Search
    print(f"Searching (top_k={top_k})...")
    start_time = time.time()
    results = retriever.search(query_embeddings, top_k=top_k)
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.2f} seconds")
    
    # Evaluate
    print("Evaluating results...")
    evaluator = EvaluateRetrieval(k_values=[1, 3, 5, 10, 100])
    metrics = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])
    
    # Save results
    summary = {
        "dataset": dataset,
        "model": model_name,
        "index_type": index_type,
        "num_documents": len(corpus),
        "num_queries": len(queries),
        "top_k": top_k,
        "timing": {
            "corpus_encode_time": encode_time,
            "index_time": index_time,
            "query_encode_time": query_encode_time,
            "search_time": search_time,
            "total_time": encode_time + index_time + query_encode_time + search_time,
        },
        "metrics": metrics,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    summary_file = os.path.join(output_dir, f"beir_{dataset}_{summary['timestamp']}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nBenchmark completed")
    print(f"Results saved to: {summary_file}")
    
    # Print metrics
    print("\n" + "="*60)
    print("BEIR Benchmark Results")
    print("="*60)
    for metric_name, metric_values in metrics.items():
        print(f"{metric_name}:")
        for k, v in metric_values.items():
            print(f"  k={k}: {v:.4f}")
    print("="*60)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEIR benchmark with NeuronDB")
    parser.add_argument("--dataset", type=str, required=True,
                       help="BEIR dataset name (e.g., msmarco, nq, scifact)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--data-path", type=str, default="./beir_data",
                       help="Path for BEIR datasets")
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
    parser.add_argument("--top-k", type=int, default=100,
                       help="Number of top results to retrieve")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_beir_benchmark(
        dataset=args.dataset,
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        index_type=args.index,
        top_k=args.top_k,
    )


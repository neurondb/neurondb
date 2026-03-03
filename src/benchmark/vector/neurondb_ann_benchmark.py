#!/usr/bin/env python3
"""
NeuronDB ANN-Benchmarks Compatibility Wrapper

This module implements the ANN-Benchmarks interface for NeuronDB,
allowing comparison against other vector search libraries using
standard datasets and metrics (recall vs QPS).
"""

import sys
import os
import time
import numpy as np
from typing import Optional, Tuple, List
import psycopg2
import psycopg2.extras

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

# Add ann-benchmarks to path if available
ann_benchmarks_paths = [
    '/tmp/ann-benchmarks',
    os.path.join(os.path.dirname(__file__), '../../ann-benchmarks'),
    os.path.join(os.path.expanduser('~'), 'ann-benchmarks'),
]
for path in ann_benchmarks_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

try:
    from neurondb.client import Client
except ImportError:
    # Fallback if not installed as package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "neurondb.client",
        os.path.join(os.path.dirname(__file__), '../../python/neurondb/client.py')
    )
    client_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client_module)
    Client = client_module.Client


class NeuronDBANN:
    """
    ANN-Benchmarks compatible wrapper for NeuronDB.
    
    This class implements the interface expected by ANN-Benchmarks
    for evaluating approximate nearest neighbor search algorithms.
    """
    
    def __init__(
        self,
        metric: str = "euclidean",
        connection_string: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "neurondb",
        user: str = "pge",
        password: Optional[str] = None,
        table_name: str = "benchmark_vectors",
        index_type: str = "hnsw",
        index_params: Optional[dict] = None,
    ):
        """
        Initialize NeuronDB ANN wrapper.
        
        Args:
            metric: Distance metric ("euclidean", "angular", "inner_product")
            connection_string: PostgreSQL connection string (optional)
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            table_name: Table name for storing vectors
            index_type: Index type ("hnsw", "ivfflat", "none")
            index_params: Index-specific parameters
        """
        self.metric = metric
        self.table_name = table_name
        self.index_type = index_type
        self.index_params = index_params or {}
        
        # Map ANN-Benchmarks metrics to NeuronDB operators
        self.metric_map = {
            "euclidean": ("<->", "vector_l2_distance"),
            "angular": ("<=>", "vector_cosine_distance"),
            "inner_product": ("<#>", "vector_inner_product"),
        }
        
        if metric not in self.metric_map:
            raise ValueError(f"Unsupported metric: {metric}. Supported: {list(self.metric_map.keys())}")
        
        self.operator, self.distance_func = self.metric_map[metric]
        
        # Initialize database connection
        if connection_string:
            self.client = Client(connection_string=connection_string)
        else:
            self.client = Client(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )
        
        self.dimension = None
        self._ensure_table()
    
    def _get_connection(self):
        """Get direct database connection for operations."""
        if hasattr(self.client, 'pool'):
            return self.client.pool.getconn()
        else:
            return psycopg2.connect(self.client.connection_string)
    
    def _ensure_table(self):
        """Create table if it doesn't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (self.table_name,))
                exists = cur.fetchone()[0]
                
                if not exists:
                    # Table will be created when we know the dimension
                    pass
        finally:
            if hasattr(self.client, 'pool'):
                self.client.pool.putconn(conn)
            else:
                conn.close()
    
    def fit(self, X: np.ndarray):
        """
        Build index on training data.
        
        Args:
            X: Training vectors (n_samples, n_features)
        """
        n_samples, n_dim = X.shape
        self.dimension = n_dim
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create table if needed
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        vector vector({n_dim})
                    );
                """)
                
                # Clear existing data
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
                
                # Insert vectors in batches
                batch_size = 1000
                for i in range(0, n_samples, batch_size):
                    batch = X[i:i+batch_size]
                    vectors = []
                    for vec in batch:
                        vec_str = '[' + ','.join(str(float(x)) for x in vec) + ']'
                        vectors.append(vec_str)
                    
                    # Bulk insert
                    values = ','.join(f"('{v}')" for v in vectors)
                    cur.execute(f"""
                        INSERT INTO {self.table_name} (vector)
                        VALUES {values};
                    """)
                
                conn.commit()
                
                # Create index if specified
                if self.index_type != "none":
                    self._create_index(cur)
                    conn.commit()
        finally:
            if hasattr(self.client, 'pool'):
                self.client.pool.putconn(conn)
            else:
                conn.close()
    
    def _create_index(self, cursor):
        """Create index on vector column."""
        index_name = f"{self.table_name}_vector_idx"
        
        # Drop existing index if any
        cursor.execute(f"""
            DROP INDEX IF EXISTS {index_name};
        """)
        
        if self.index_type == "hnsw":
            # HNSW index parameters
            m = self.index_params.get('m', 16)
            ef_construction = self.index_params.get('ef_construction', 200)
            
            # Determine operator class based on metric
            if self.metric == "euclidean":
                opclass = "vector_l2_ops"
            elif self.metric == "angular":
                opclass = "vector_cosine_ops"
            else:
                opclass = "vector_l2_ops"  # default
            
            cursor.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING hnsw (vector {opclass})
                WITH (m = {m}, ef_construction = {ef_construction});
            """)
        elif self.index_type == "ivfflat":
            # IVFFlat index parameters
            lists = self.index_params.get('lists', 100)
            
            # Determine operator class based on metric
            if self.metric == "euclidean":
                opclass = "vector_l2_ops"
            elif self.metric == "angular":
                opclass = "vector_cosine_ops"
            else:
                opclass = "vector_l2_ops"  # default
            
            cursor.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING ivfflat (vector {opclass})
                WITH (lists = {lists});
            """)
    
    def query(self, v: np.ndarray, k: int) -> List[int]:
        """
        Query for k nearest neighbors.
        
        Args:
            v: Query vector (n_features,)
            k: Number of neighbors to return
        
        Returns:
            List of neighbor indices
        """
        vec_str = '[' + ','.join(str(float(x)) for x in v) + ']'
        
        query = f"""
            SELECT id, vector {self.operator} %s::vector AS distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT %s;
        """
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (vec_str, k))
                results = cur.fetchall()
                return [row[0] - 1 for row in results]  # Convert to 0-indexed
        finally:
            if hasattr(self.client, 'pool'):
                self.client.pool.putconn(conn)
            else:
                conn.close()
    
    def batch_query(self, X: np.ndarray, k: int, num_threads: int = 1) -> List[List[int]]:
        """
        Batch query for k nearest neighbors.
        
        Args:
            X: Query vectors (n_queries, n_features)
            k: Number of neighbors per query
            num_threads: Number of threads (for compatibility, not used)
        
        Returns:
            List of lists of neighbor indices
        """
        results = []
        for query_vec in X:
            neighbors = self.query(query_vec, k)
            results.append(neighbors)
        return results
    
    def get_additional(self) -> dict:
        """Get additional information about the index."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Get table size
                cur.execute(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}'));
                """)
                size = cur.fetchone()[0]
                
                # Get index info if exists
                index_name = f"{self.table_name}_vector_idx"
                cur.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = %s AND indexname = %s;
                """, (self.table_name, index_name))
                index_info = cur.fetchone()
                
                return {
                    "table_size": size,
                    "index_type": self.index_type,
                    "index_params": self.index_params,
                    "has_index": index_info is not None,
                }
        finally:
            if hasattr(self.client, 'pool'):
                self.client.pool.putconn(conn)
            else:
                conn.close()
    
    def __str__(self):
        return f"NeuronDB({self.index_type}, {self.metric})"


def run_benchmark(
    dataset_name: str,
    metric: str = "euclidean",
    index_type: str = "hnsw",
    index_params: Optional[dict] = None,
    k: int = 10,
    connection_string: Optional[str] = None,
    **db_kwargs
):
    """
    Run a benchmark using ANN-Benchmarks dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "sift-128-euclidean")
        metric: Distance metric
        index_type: Index type
        index_params: Index parameters
        k: Number of neighbors
        connection_string: Database connection string
        **db_kwargs: Additional database connection parameters
    
    Returns:
        Dictionary with benchmark results
    """
    try:
        import ann_benchmarks.datasets as datasets
    except ImportError:
        # Try alternative import paths
        try:
            sys.path.insert(0, '/tmp/ann-benchmarks')
            import ann_benchmarks.datasets as datasets
        except ImportError:
            raise ImportError(
                "ann_benchmarks not found. Please clone: "
                "git clone https://github.com/erikbern/ann-benchmarks.git /tmp/ann-benchmarks"
            )
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    hdf5_file, dimension = datasets.get_dataset(dataset_name)
    
    # Extract train and test data
    # For initial testing, use a subset to avoid server crashes
    max_train_size = 100000  # Limit to 100k vectors for testing
    X_train_full = hdf5_file['train'][:]
    if len(X_train_full) > max_train_size:
        print(f"Using subset: {max_train_size} vectors (full dataset has {len(X_train_full)})")
        X_train = X_train_full[:max_train_size]
    else:
        X_train = X_train_full
    
    if 'test' in hdf5_file:
        X_test = hdf5_file['test'][:]
        # Limit test set too
        if len(X_test) > 1000:
            X_test = X_test[:1000]
    else:
        # Some datasets only have train, use a subset for testing
        X_test = X_train[:min(1000, len(X_train)//10)]
        X_train = X_train[len(X_test):]
    
    # Get distance metric from dataset attributes
    distance = hdf5_file.attrs.get('distance', 'euclidean')
    hdf5_file.close()
    
    if distance != metric:
        print(f"Warning: Dataset metric ({distance}) != requested metric ({metric})")
    
    # Initialize NeuronDB
    print(f"Initializing NeuronDB with {index_type} index...")
    neurondb = NeuronDBANN(
        metric=metric,
        index_type=index_type,
        index_params=index_params,
        connection_string=connection_string,
        **db_kwargs
    )
    
    # Build index
    print("Building index...")
    start_time = time.time()
    neurondb.fit(X_train)
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds")
    
    # Run queries
    print(f"Running queries (k={k})...")
    query_times = []
    all_results = []
    
    for i, query_vec in enumerate(X_test):
        start = time.time()
        neighbors = neurondb.query(query_vec, k)
        query_time = time.time() - start
        query_times.append(query_time)
        all_results.append(neighbors)
        
        if (i + 1) % 100 == 0:
            avg_time = np.mean(query_times[-100:])
            qps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"  Processed {i+1}/{len(X_test)} queries, avg QPS: {qps:.2f}")
    
    avg_query_time = np.mean(query_times)
    qps = 1.0 / avg_query_time if avg_query_time > 0 else 0
    
    # Get ground truth from FAISS
    print("Computing ground truth with FAISS...")
    try:
        import faiss
        import faiss.contrib
        
        # Build FAISS index for exact search
        if metric == "euclidean":
            faiss_index = faiss.IndexFlatL2(X_train.shape[1])
        elif metric == "angular":
            faiss_index = faiss.IndexFlatIP(X_train.shape[1])
            # Normalize for cosine similarity
            faiss.normalize_L2(X_train)
            X_test_norm = X_test.copy()
            faiss.normalize_L2(X_test_norm)
        else:
            faiss_index = faiss.IndexFlatIP(X_train.shape[1])
        
        faiss_index.add(X_train.astype('float32'))
        
        # Get ground truth
        _, ground_truth = faiss_index.search(
            X_test_norm.astype('float32') if metric == "angular" else X_test.astype('float32'),
            k
        )
        
        # Calculate recall
        recalls = []
        for i, (result, truth) in enumerate(zip(all_results, ground_truth)):
            intersection = len(set(result) & set(truth))
            recall = intersection / len(truth) if len(truth) > 0 else 0.0
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        print(f"Average recall: {avg_recall:.4f}")
        
    except ImportError:
        print("Warning: FAISS not available, skipping recall calculation")
        avg_recall = None
        ground_truth = None
    
    # Get additional info
    additional = neurondb.get_additional()
    
    results = {
        "dataset": dataset_name,
        "metric": metric,
        "index_type": index_type,
        "index_params": index_params,
        "k": k,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dimension": X_train.shape[1],
        "build_time": build_time,
        "avg_query_time": avg_query_time,
        "qps": qps,
        "avg_recall": avg_recall,
        "additional": additional,
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NeuronDB ANN-Benchmarks")
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean",
                       help="Dataset name")
    parser.add_argument("--metric", type=str, default="euclidean",
                       choices=["euclidean", "angular", "inner_product"],
                       help="Distance metric")
    parser.add_argument("--index", type=str, default="hnsw",
                       choices=["hnsw", "ivfflat", "none"],
                       help="Index type")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of neighbors")
    parser.add_argument("--m", type=int, default=16,
                       help="HNSW m parameter")
    parser.add_argument("--ef-construction", type=int, default=200,
                       help="HNSW ef_construction parameter")
    parser.add_argument("--lists", type=int, default=100,
                       help="IVFFlat lists parameter")
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
    
    # Build index parameters
    index_params = {}
    if args.index == "hnsw":
        index_params = {"m": args.m, "ef_construction": args.ef_construction}
    elif args.index == "ivfflat":
        index_params = {"lists": args.lists}
    
    # Run benchmark
    results = run_benchmark(
        dataset_name=args.dataset,
        metric=args.metric,
        index_type=args.index,
        index_params=index_params,
        k=args.k,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )
    
    # Print results
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    for key, value in results.items():
        if key != "additional":
            print(f"{key}: {value}")
    print("="*60)


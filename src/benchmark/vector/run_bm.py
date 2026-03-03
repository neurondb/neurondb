#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  NeuronDB Vector Search Benchmark Suite                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  A comprehensive benchmark framework for evaluating NeuronDB's vector       ║
║  search capabilities against industry-standard datasets and metrics.        ║
║                                                                              ║
║  Features:                                                                   ║
║    • Automatic dataset downloading and database loading                     ║
║    • Support for multiple index types (HNSW, IVFFlat, Sequential)          ║
║    • Comprehensive performance metrics (QPS, recall, latency)               ║
║    • Real-time progress tracking                                            ║
║    • Professional CLI with extensive help documentation                     ║
║                                                                              ║
║  Supported Datasets:                                                         ║
║    • SIFT-128 (Euclidean)          • GIST-960 (Euclidean)                  ║
║    • GloVe-100 (Cosine/Angular)    • Deep1B (Euclidean)                    ║
║                                                                              ║
║  Author: NeuronDB Team                                                       ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback

# Check if running with --help or --version (don't require dependencies)
HELP_ARGS = {'--help', '-h', '--version', '--list-datasets', '--list-configs'}
if not any(arg in HELP_ARGS for arg in sys.argv[1:]):
    try:
        import numpy as np
        import psycopg2
        from tqdm import tqdm
        import h5py
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}", file=sys.stderr)
        print("\nPlease install dependencies:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
else:
    # Dummy imports for help display
    np = None
    psycopg2 = None
    tqdm = None
    h5py = None

# Version information
__version__ = "2.1.0"
__author__ = "NeuronDB Team"

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ANSI color codes for beautiful terminal output
class Colors:
    """Terminal color codes for enhanced user experience."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Dataset registry with download URLs and specifications
DATASET_REGISTRY = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "dimension": 128,
        "metric": "euclidean",
        "description": "SIFT features (128-dim vectors with Euclidean distance)"
    },
    "gist-960-euclidean": {
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "dimension": 960,
        "metric": "euclidean",
        "description": "GIST features (960-dim vectors with Euclidean distance)"
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "dimension": 100,
        "metric": "angular",
        "description": "GloVe word embeddings (100-dim vectors with angular distance)"
    },
}

# Index configuration presets
INDEX_CONFIGS = {
    "hnsw": {
        "type": "hnsw",
        "params": {"m": 16, "ef_construction": 200},
        "search_params": {"ef_search": 100},
        "description": "Hierarchical Navigable Small World graph (high recall, fast)"
    },
    "hnsw_fast": {
        "type": "hnsw",
        "params": {"m": 8, "ef_construction": 100},
        "search_params": {"ef_search": 50},
        "description": "HNSW optimized for speed"
    },
    "hnsw_balanced": {
        "type": "hnsw",
        "params": {"m": 16, "ef_construction": 200},
        "search_params": {"ef_search": 100},
        "description": "HNSW balanced for recall and speed"
    },
    "ivfflat": {
        "type": "ivfflat",
        "params": {"lists": 100},
        "search_params": {"probes": 10},
        "description": "Inverted File with Flat compression (good balance)"
    },
    "none": {
        "type": "none",
        "params": {},
        "search_params": {},
        "description": "Sequential scan (baseline, no index)"
    },
}

# Database defaults
DEFAULT_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "neurondb",
    "user": "pge",
    "password": None,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print a beautiful welcome banner."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║             NeuronDB Vector Search Benchmark Suite v{:<25}║".format(__version__))
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

def print_status(message: str, status: str = "info"):
    """Print formatted status message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "success":
        icon = f"{Colors.OKGREEN}✓{Colors.ENDC}"
    elif status == "error":
        icon = f"{Colors.FAIL}✗{Colors.ENDC}"
    elif status == "warning":
        icon = f"{Colors.WARNING}⚠{Colors.ENDC}"
    else:
        icon = f"{Colors.OKBLUE}ℹ{Colors.ENDC}"
    
    print(f"[{timestamp}] {icon} {message}")

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'─' * 80}{Colors.ENDC}\n")

def print_progress(current: int, total: int, prefix: str = "", suffix: str = ""):
    """Print progress bar (fallback if tqdm is not used)."""
    percent = 100 * (current / float(total))
    filled = int(50 * current // total)
    bar = '█' * filled + '░' * (50 - filled)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if current == total:
        print()

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetManager:
    """Manages dataset downloading, loading, and caching."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get local path for dataset file."""
        return self.data_dir / f"{dataset_name}.hdf5"
    
    def is_dataset_cached(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded."""
        dataset_path = self.get_dataset_path(dataset_name)
        return dataset_path.exists() and dataset_path.stat().st_size > 0
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> Path:
        """
        Download dataset if not already cached.
        
        Args:
            dataset_name: Name of the dataset to download
            force: Force re-download even if cached
            
        Returns:
            Path to the downloaded dataset file
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {', '.join(DATASET_REGISTRY.keys())}")
        
        dataset_path = self.get_dataset_path(dataset_name)
        dataset_info = DATASET_REGISTRY[dataset_name]
        
        if not force and self.is_dataset_cached(dataset_name):
            print_status(f"Dataset '{dataset_name}' already cached", "success")
            return dataset_path
        
        print_section(f"Downloading Dataset: {dataset_name}")
        print(f"  Description: {dataset_info['description']}")
        print(f"  URL: {dataset_info['url']}")
        print(f"  Destination: {dataset_path}\n")
        
        try:
            # Download with progress bar
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f'\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', 
                          end='', flush=True)
            
            urllib.request.urlretrieve(
                dataset_info['url'],
                dataset_path,
                reporthook=download_progress
            )
            print()  # New line after progress
            
            # Verify download
            if not dataset_path.exists() or dataset_path.stat().st_size == 0:
                raise RuntimeError("Download failed: file is empty or missing")
            
            print_status(f"Dataset downloaded successfully ({dataset_path.stat().st_size / (1024*1024):.1f} MB)", 
                        "success")
            return dataset_path
            
        except Exception as e:
            print_status(f"Download failed: {e}", "error")
            if dataset_path.exists():
                dataset_path.unlink()  # Remove partial download
            raise
    
    def load_dataset(self, dataset_name: str) -> Dict:
        """
        Load dataset from HDF5 file.
        
        Returns:
            Dictionary with train, test, neighbors, and distances data
        """
        dataset_path = self.get_dataset_path(dataset_name)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run with --prepare flag first.")
        
        print_status(f"Loading dataset from {dataset_path}...", "info")
        
        try:
            with h5py.File(dataset_path, 'r') as hf:
                data = {
                    'train': np.array(hf['train']),
                    'test': np.array(hf['test']),
                    'neighbors': np.array(hf['neighbors']),
                    'distances': np.array(hf['distances']) if 'distances' in hf else None,
                }
            
            print_status(f"Loaded {len(data['train'])} training vectors, "
                        f"{len(data['test'])} test queries", "success")
            return data
            
        except Exception as e:
            print_status(f"Failed to load dataset: {e}", "error")
            raise

class DatabaseLoader:
    """Handles loading vector data into PostgreSQL/NeuronDB."""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**{k: v for k, v in self.db_config.items() if v is not None})
            self.conn.autocommit = True
            
            # Verify NeuronDB extension
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'neurondb';")
                if not cur.fetchone():
                    print_status("NeuronDB extension not found. Attempting to create...", "warning")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS neurondb;")
            
            print_status("Database connection established", "success")
            
        except Exception as e:
            print_status(f"Database connection failed: {e}", "error")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_table(self, table_name: str, dimension: int):
        """Create table for vector storage."""
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding vector({dimension})
                );
            """)
        print_status(f"Created table '{table_name}' with dimension {dimension}", "success")
    
    def load_vectors(self, table_name: str, vectors: np.ndarray, batch_size: int = 1000):
        """
        Load vectors into database in batches.
        
        Args:
            table_name: Target table name
            vectors: Numpy array of vectors
            batch_size: Number of vectors per batch
        """
        print_section(f"Loading {len(vectors)} vectors into database")
        
        num_batches = (len(vectors) + batch_size - 1) // batch_size
        
        with tqdm(total=len(vectors), desc="Loading vectors", unit="vec") as pbar:
            with self.conn.cursor() as cur:
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    
                    # Convert vectors to PostgreSQL array format
                    values = []
                    for vec in batch:
                        vec_str = '[' + ','.join(map(str, vec)) + ']'
                        values.append(f"('{vec_str}')")
                    
                    # Bulk insert
                    cur.execute(f"""
                        INSERT INTO {table_name} (embedding)
                        VALUES {','.join(values)}
                    """)
                    
                    pbar.update(len(batch))
        
        # Get final count
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cur.fetchone()[0]
        
        print_status(f"Loaded {count} vectors successfully", "success")
    
    def create_index(self, table_name: str, index_config: Dict):
        """Create vector index on table."""
        index_type = index_config.get("type", "hnsw")
        params = index_config.get("params", {})
        index_name = f"{table_name}_{index_type}_idx"
        
        print_section(f"Building {index_type.upper()} index")
        print(f"  Parameters: {params}")
        
        with self.conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {index_name};")
            
            start_time = time.time()
            
            if index_type == "hnsw":
                m = params.get('m', 16)
                ef_construction = params.get('ef_construction', 200)
                cur.execute(f"""
                    CREATE INDEX {index_name}
                    ON {table_name}
                    USING hnsw (embedding vector_l2_ops)
                    WITH (m = {m}, ef_construction = {ef_construction});
                """)
            elif index_type == "ivfflat":
                lists = params.get('lists', 100)
                cur.execute(f"""
                    CREATE INDEX {index_name}
                    ON {table_name}
                    USING ivfflat (embedding vector_l2_ops)
                    WITH (lists = {lists});
                """)
            elif index_type == "none":
                print_status("Skipping index creation (using sequential scan)", "warning")
                return 0
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            build_time = time.time() - start_time
        
        print_status(f"Index built in {build_time:.2f} seconds", "success")
        return build_time

# ═══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK EXECUTION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class VectorBenchmark:
    """Executes vector search benchmarks and collects metrics."""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(**{k: v for k, v in self.db_config.items() if v is not None})
        self.conn.autocommit = True
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def run_benchmark(self, table_name: str, queries: np.ndarray, ground_truth: np.ndarray,
                     k: int = 10, index_config: Dict = None, max_queries: Optional[int] = None) -> Dict:
        """
        Run benchmark queries and calculate metrics.
        
        Args:
            table_name: Table containing vectors
            queries: Query vectors
            ground_truth: Ground truth nearest neighbor IDs
            k: Number of nearest neighbors to retrieve
            index_config: Index configuration for search parameters
            
        Returns:
            Dictionary with benchmark results
        """
        print_section(f"Running Benchmark (k={k})")
        
        # Set search parameters if provided
        search_params = index_config.get("search_params", {}) if index_config else {}
        if search_params:
            print(f"  Search parameters: {search_params}")
            with self.conn.cursor() as cur:
                for param, value in search_params.items():
                    if param == "ef_search":
                        cur.execute(f"SET hnsw.ef_search = {value};")
                    elif param == "probes":
                        cur.execute(f"SET ivfflat.probes = {value};")
        
        # Limit queries if specified
        if max_queries and max_queries < len(queries):
            queries = queries[:max_queries]
            ground_truth = ground_truth[:max_queries]
            print(f"  Limiting to {max_queries} queries for quick test")
        
        latencies = []
        recalls = []
        
        with tqdm(total=len(queries), desc="Running queries", unit="query") as pbar:
            with self.conn.cursor() as cur:
                for i, query in enumerate(queries):
                    query_str = '[' + ','.join(map(str, query)) + ']'
                    
                    # Execute query and measure time
                    start_time = time.time()
                    cur.execute(f"""
                        SELECT id
                        FROM {table_name}
                        ORDER BY embedding <-> '{query_str}'::vector
                        LIMIT {k};
                    """)
                    results = [row[0] - 1 for row in cur.fetchall()]  # Adjust for 0-based indexing
                    latency = time.time() - start_time
                    
                    latencies.append(latency * 1000)  # Convert to milliseconds
                    
                    # Calculate recall
                    gt = set(ground_truth[i][:k])
                    found = set(results)
                    recall = len(gt & found) / k
                    recalls.append(recall)
                    
                    pbar.update(1)
        
        # Calculate metrics
        metrics = {
            "k": k,
            "num_queries": len(queries),
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "qps": 1000.0 / np.mean(latencies),  # Queries per second
            "recall": np.mean(recalls),
            "min_recall": np.min(recalls),
            "max_recall": np.max(recalls),
        }
        
        # Print results
        print("\n  Results:")
        print(f"    Recall@{k}:        {metrics['recall']:.4f}")
        print(f"    Avg Latency:      {metrics['avg_latency_ms']:.2f} ms")
        print(f"    P95 Latency:      {metrics['p95_latency_ms']:.2f} ms")
        print(f"    QPS:              {metrics['qps']:.2f}")
        
        return metrics

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT MANAGEMENT MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class ResultManager:
    """Manages benchmark results and report generation."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict, dataset_name: str, config_name: str) -> Path:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{dataset_name}_{config_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print_status(f"Results saved to {filepath}", "success")
        return filepath
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable text report."""
        report = []
        report.append("=" * 80)
        report.append("NEURONDB VECTOR BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"\nTimestamp: {results.get('timestamp', 'N/A')}")
        report.append(f"Dataset: {results.get('dataset', 'N/A')}")
        report.append(f"Index Config: {results.get('config_name', 'N/A')}")
        report.append(f"\nDataset Statistics:")
        report.append(f"  Training vectors: {results.get('num_train_vectors', 0):,}")
        report.append(f"  Test queries: {results.get('num_test_queries', 0):,}")
        report.append(f"  Vector dimension: {results.get('dimension', 0)}")
        
        if 'timings' in results:
            timings = results['timings']
            report.append(f"\nTimings:")
            report.append(f"  Data loading: {timings.get('load_time', 0):.2f}s")
            report.append(f"  Index building: {timings.get('index_build_time', 0):.2f}s")
            report.append(f"  Benchmark execution: {timings.get('benchmark_time', 0):.2f}s")
            report.append(f"  Total time: {timings.get('total_time', 0):.2f}s")
        
        if 'metrics' in results:
            report.append(f"\nPerformance Metrics:")
            for k, metrics in sorted(results['metrics'].items()):
                report.append(f"\n  k={k}:")
                report.append(f"    Recall:         {metrics.get('recall', 0):.4f}")
                report.append(f"    Avg Latency:    {metrics.get('avg_latency_ms', 0):.2f} ms")
                report.append(f"    P50 Latency:    {metrics.get('p50_latency_ms', 0):.2f} ms")
                report.append(f"    P95 Latency:    {metrics.get('p95_latency_ms', 0):.2f} ms")
                report.append(f"    P99 Latency:    {metrics.get('p99_latency_ms', 0):.2f} ms")
                report.append(f"    QPS:            {metrics.get('qps', 0):.2f}")
        
        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkOrchestrator:
    """Main orchestrator for the benchmark suite."""
    
    def __init__(self, args):
        self.args = args
        self.dataset_manager = DatasetManager(args.data_dir)
        self.result_manager = ResultManager(args.output_dir)
    
    def prepare_data(self):
        """Prepare (download) datasets."""
        print_banner()
        print_section("Data Preparation")
        
        datasets = self.args.datasets.split(',')
        
        for dataset_name in datasets:
            try:
                self.dataset_manager.download_dataset(dataset_name.strip(), force=self.args.force_download)
            except Exception as e:
                print_status(f"Failed to download {dataset_name}: {e}", "error")
                if not self.args.continue_on_error:
                    return False
        
        print_status("Data preparation completed", "success")
        return True
    
    def load_to_database(self):
        """Load datasets into database."""
        print_banner()
        print_section("Database Loading")
        
        datasets = self.args.datasets.split(',')
        db_config = {
            "host": self.args.db_host,
            "port": self.args.db_port,
            "database": self.args.db_name,
            "user": self.args.db_user,
            "password": self.args.db_password,
        }
        
        db_loader = DatabaseLoader(db_config)
        
        try:
            db_loader.connect()
            
            for dataset_name in datasets:
                dataset_name = dataset_name.strip()
                
                try:
                    # Load dataset from file
                    data = self.dataset_manager.load_dataset(dataset_name)
                    
                    # Create table and load vectors
                    table_name = f"vectors_{dataset_name.replace('-', '_')}"
                    dimension = data['train'].shape[1]
                    
                    db_loader.create_table(table_name, dimension)
                    db_loader.load_vectors(table_name, data['train'], batch_size=self.args.batch_size)
                    
                    # Create index if specified
                    if self.args.index_config:
                        index_config = INDEX_CONFIGS.get(self.args.index_config)
                        if index_config:
                            db_loader.create_index(table_name, index_config)
                    
                except Exception as e:
                    print_status(f"Failed to load {dataset_name}: {e}", "error")
                    if not self.args.continue_on_error:
                        return False
            
            print_status("Database loading completed", "success")
            return True
            
        finally:
            db_loader.disconnect()
    
    def run_benchmarks(self):
        """Execute benchmarks."""
        print_banner()
        print_section("Benchmark Execution")
        
        datasets = self.args.datasets.split(',')
        configs = self.args.configs.split(',')
        k_values = [int(k.strip()) for k in self.args.k_values.split(',')]
        
        db_config = {
            "host": self.args.db_host,
            "port": self.args.db_port,
            "database": self.args.db_name,
            "user": self.args.db_user,
            "password": self.args.db_password,
        }
        
        benchmark = VectorBenchmark(db_config)
        
        try:
            benchmark.connect()
            
            for dataset_name in datasets:
                dataset_name = dataset_name.strip()
                
                # Load dataset
                data = self.dataset_manager.load_dataset(dataset_name)
                table_name = f"vectors_{dataset_name.replace('-', '_')}"
                
                for config_name in configs:
                    config_name = config_name.strip()
                    index_config = INDEX_CONFIGS.get(config_name, {})
                    
                    print_section(f"Benchmarking: {dataset_name} with {config_name}")
                    
                    # Collect results
                    results = {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dataset_name,
                        "config_name": config_name,
                        "index_config": index_config,
                        "num_train_vectors": len(data['train']),
                        "num_test_queries": len(data['test']),
                        "dimension": data['train'].shape[1],
                        "metrics": {},
                    }
                    
                    # Run for each k value
                    for k in k_values:
                        metrics = benchmark.run_benchmark(
                            table_name,
                            data['test'],
                            data['neighbors'],
                            k=k,
                            index_config=index_config,
                            max_queries=self.args.max_queries
                        )
                        results['metrics'][k] = metrics
                    
                    # Save results
                    self.result_manager.save_results(results, dataset_name, config_name)
                    
                    # Print report
                    report = self.result_manager.generate_report(results)
                    print("\n" + report)
            
            print_status("Benchmark execution completed", "success")
            return True
            
        except Exception as e:
            print_status(f"Benchmark failed: {e}", "error")
            traceback.print_exc()
            return False
            
        finally:
            benchmark.disconnect()

# ═══════════════════════════════════════════════════════════════════════════════
#  COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser with professional help."""
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data (download datasets)
  %(prog)s --prepare --datasets sift-128-euclidean,glove-100-angular
  
  # Load data into database
  %(prog)s --load --datasets sift-128-euclidean --index-config hnsw
  
  # Run benchmarks
  %(prog)s --run --datasets sift-128-euclidean --configs hnsw,ivfflat --k-values 10,50,100
  
  # Full pipeline (prepare + load + run)
  %(prog)s --prepare --load --run --datasets sift-128-euclidean
  
  # Quick test with defaults
  %(prog)s --prepare --load --run

For more information, visit: https://github.com/neurondb/neurondb
        """
    )
    
    # Execution modes
    mode_group = parser.add_argument_group('Execution Modes')
    mode_group.add_argument('--prepare', action='store_true',
                           help='Download and prepare datasets (required before --load)')
    mode_group.add_argument('--load', action='store_true',
                           help='Load datasets into database (required before --run)')
    mode_group.add_argument('--run', action='store_true',
                           help='Execute benchmarks on loaded data')
    
    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--datasets', type=str, default='sift-128-euclidean',
                              help='Comma-separated list of datasets (default: sift-128-euclidean)\n'
                                   'Available: ' + ', '.join(DATASET_REGISTRY.keys()))
    dataset_group.add_argument('--data-dir', type=str, default='./data',
                              help='Directory for storing downloaded datasets (default: ./data)')
    dataset_group.add_argument('--force-download', action='store_true',
                              help='Force re-download of datasets even if cached')
    
    # Database configuration
    db_group = parser.add_argument_group('Database Configuration')
    db_group.add_argument('--db-host', type=str, default=DEFAULT_DB_CONFIG['host'],
                         help=f'Database host (default: {DEFAULT_DB_CONFIG["host"]})')
    db_group.add_argument('--db-port', type=int, default=DEFAULT_DB_CONFIG['port'],
                         help=f'Database port (default: {DEFAULT_DB_CONFIG["port"]})')
    db_group.add_argument('--db-name', type=str, default=DEFAULT_DB_CONFIG['database'],
                         help=f'Database name (default: {DEFAULT_DB_CONFIG["database"]})')
    db_group.add_argument('--db-user', type=str, default=DEFAULT_DB_CONFIG['user'],
                         help=f'Database user (default: {DEFAULT_DB_CONFIG["user"]})')
    db_group.add_argument('--db-password', type=str, default=None,
                         help='Database password (optional)')
    
    # Index configuration
    index_group = parser.add_argument_group('Index Configuration')
    index_group.add_argument('--index-config', type=str, default='hnsw',
                            help='Index configuration to use during --load (default: hnsw)\n'
                                 'Available: ' + ', '.join(INDEX_CONFIGS.keys()))
    index_group.add_argument('--configs', type=str, default='hnsw',
                            help='Comma-separated list of index configs to benchmark during --run '
                                 '(default: hnsw)')
    
    # Benchmark parameters
    bench_group = parser.add_argument_group('Benchmark Parameters')
    bench_group.add_argument('--k-values', type=str, default='10',
                            help='Comma-separated list of k values for k-NN search '
                                 '(default: 10)')
    bench_group.add_argument('--batch-size', type=int, default=1000,
                            help='Batch size for loading vectors (default: 1000)')
    bench_group.add_argument('--max-queries', type=int, default=None,
                            help='Maximum number of queries to run (for quick testing, default: all)')
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output-dir', type=str, default='./results',
                             help='Directory for saving results (default: ./results)')
    output_group.add_argument('--continue-on-error', action='store_true',
                             help='Continue execution even if individual steps fail')
    
    # Information
    info_group = parser.add_argument_group('Information')
    info_group.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    info_group.add_argument('--list-datasets', action='store_true',
                           help='List all available datasets and exit')
    info_group.add_argument('--list-configs', action='store_true',
                           help='List all available index configurations and exit')
    
    return parser

def list_datasets():
    """Print list of available datasets."""
    print(f"\n{Colors.BOLD}Available Datasets:{Colors.ENDC}\n")
    for name, info in DATASET_REGISTRY.items():
        print(f"  {Colors.OKBLUE}{name}{Colors.ENDC}")
        print(f"    Description: {info['description']}")
        print(f"    Dimension: {info['dimension']}")
        print(f"    Metric: {info['metric']}")
        print()

def list_configs():
    """Print list of available index configurations."""
    print(f"\n{Colors.BOLD}Available Index Configurations:{Colors.ENDC}\n")
    for name, config in INDEX_CONFIGS.items():
        print(f"  {Colors.OKBLUE}{name}{Colors.ENDC}")
        print(f"    Type: {config['type']}")
        print(f"    Description: {config['description']}")
        print(f"    Parameters: {config['params']}")
        print()

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle information requests
    if args.list_datasets:
        list_datasets()
        return 0
    
    if args.list_configs:
        list_configs()
        return 0
    
    # Validate execution mode
    if not (args.prepare or args.load or args.run):
        parser.error("At least one execution mode (--prepare, --load, --run) must be specified")
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(args)
    
    # Execute requested operations
    try:
        if args.prepare:
            if not orchestrator.prepare_data():
                return 1
        
        if args.load:
            if not orchestrator.load_to_database():
                return 1
        
        if args.run:
            if not orchestrator.run_benchmarks():
                return 1
        
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ All operations completed successfully!{Colors.ENDC}\n")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal error: {e}{Colors.ENDC}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

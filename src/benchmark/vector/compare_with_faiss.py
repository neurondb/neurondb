#!/usr/bin/env python3
"""
Compare NeuronDB against FAISS for ANN-Benchmarks compatibility.

This script:
1. Loads standard ANN-Benchmarks datasets
2. Runs queries on both NeuronDB and FAISS
3. Computes recall using FAISS Flat (exact search) as ground truth
4. Measures QPS (queries per second) for both
5. Generates comparison plots (recall vs QPS)
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
sys.path.insert(0, os.path.dirname(__file__))

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

from neurondb_ann_benchmark import NeuronDBANN, run_benchmark

try:
    import faiss
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: {e}")
    print("Some features may not be available. Install with: pip install faiss-cpu matplotlib seaborn")


def compute_ground_truth(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute ground truth using FAISS Flat (exact search).
    
    Args:
        X_train: Training vectors
        X_test: Test query vectors
        k: Number of neighbors
        metric: Distance metric
    
    Returns:
        Ground truth indices (n_queries, k)
    """
    import faiss
    
    if metric == "euclidean":
        index = faiss.IndexFlatL2(X_train.shape[1])
    elif metric == "angular":
        index = faiss.IndexFlatIP(X_train.shape[1])
        # Normalize for cosine similarity
        X_train_norm = X_train.copy().astype('float32')
        X_test_norm = X_test.copy().astype('float32')
        faiss.normalize_L2(X_train_norm)
        faiss.normalize_L2(X_test_norm)
        X_train = X_train_norm
        X_test = X_test_norm
    else:
        index = faiss.IndexFlatIP(X_train.shape[1])
    
    index.add(X_train.astype('float32'))
    
    # Search
    distances, indices = index.search(X_test.astype('float32'), k)
    
    return indices


def compute_recall(results: List[List[int]], ground_truth: np.ndarray) -> float:
    """
    Compute average recall.
    
    Args:
        results: List of result indices for each query
        ground_truth: Ground truth indices (n_queries, k)
    
    Returns:
        Average recall
    """
    recalls = []
    for result, truth in zip(results, ground_truth):
        intersection = len(set(result) & set(truth))
        recall = intersection / len(truth) if len(truth) > 0 else 0.0
        recalls.append(recall)
    
    return np.mean(recalls)


def benchmark_neurondb(
    X_train: np.ndarray,
    X_test: np.ndarray,
    metric: str,
    index_type: str,
    index_params: Optional[dict],
    k: int,
    connection_string: Optional[str] = None,
    **db_kwargs
) -> Tuple[List[List[int]], float, Dict]:
    """
    Benchmark NeuronDB.
    
    Returns:
        (results, qps, metadata)
    """
    neurondb = NeuronDBANN(
        metric=metric,
        index_type=index_type,
        index_params=index_params,
        connection_string=connection_string,
        **db_kwargs
    )
    
    # Build index
    print("Building NeuronDB index...")
    build_start = time.time()
    neurondb.fit(X_train)
    build_time = time.time() - build_start
    
    # Run queries
    print("Running NeuronDB queries...")
    query_times = []
    results = []
    
    for i, query_vec in enumerate(X_test):
        start = time.time()
        neighbors = neurondb.query(query_vec, k)
        query_time = time.time() - start
        query_times.append(query_time)
        results.append(neighbors)
        
        if (i + 1) % 100 == 0:
            avg_time = np.mean(query_times[-100:])
            qps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"  Processed {i+1}/{len(X_test)} queries, QPS: {qps:.2f}")
    
    avg_query_time = np.mean(query_times)
    qps = 1.0 / avg_query_time if avg_query_time > 0 else 0
    
    metadata = {
        "build_time": build_time,
        "avg_query_time": avg_query_time,
        "qps": qps,
        "index_type": index_type,
        "index_params": index_params,
    }
    
    return results, qps, metadata


def benchmark_faiss(
    X_train: np.ndarray,
    X_test: np.ndarray,
    metric: str,
    index_type: str,
    index_params: Optional[dict],
    k: int
) -> Tuple[np.ndarray, float, Dict]:
    """
    Benchmark FAISS.
    
    Returns:
        (results, qps, metadata)
    """
    import faiss
    
    print(f"Building FAISS {index_type} index...")
    build_start = time.time()
    
    d = X_train.shape[1]
    
    if index_type == "flat":
        if metric == "euclidean":
            index = faiss.IndexFlatL2(d)
        else:
            index = faiss.IndexFlatIP(d)
            if metric == "angular":
                faiss.normalize_L2(X_train.astype('float32'))
                X_test_norm = X_test.copy().astype('float32')
                faiss.normalize_L2(X_test_norm)
                X_test = X_test_norm
    elif index_type == "ivf":
        nlist = index_params.get("nlist", 100) if index_params else 100
        quantizer = faiss.IndexFlatL2(d) if metric == "euclidean" else faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(X_train.astype('float32'))
    elif index_type == "hnsw":
        M = index_params.get("M", 16) if index_params else 16
        index = faiss.IndexHNSWFlat(d, M)
        if metric == "angular":
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(f"Unsupported FAISS index type: {index_type}")
    
    index.add(X_train.astype('float32'))
    build_time = time.time() - build_start
    
    # Run queries
    print("Running FAISS queries...")
    query_start = time.time()
    distances, indices = index.search(X_test.astype('float32'), k)
    query_time = time.time() - query_start
    
    n_queries = len(X_test)
    qps = n_queries / query_time if query_time > 0 else 0
    
    metadata = {
        "build_time": build_time,
        "query_time": query_time,
        "qps": qps,
        "index_type": index_type,
        "index_params": index_params,
    }
    
    return indices, qps, metadata


def run_comparison(
    dataset_name: str,
    metric: str = "euclidean",
    k: int = 10,
    neurondb_index: str = "hnsw",
    neurondb_params: Optional[dict] = None,
    faiss_index: str = "flat",
    faiss_params: Optional[dict] = None,
    connection_string: Optional[str] = None,
    output_dir: str = "./results",
    **db_kwargs
) -> Dict:
    """
    Run comparison between NeuronDB and FAISS.
    
    Returns:
        Dictionary with comparison results
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
    
    print(f"Dataset: {len(X_train)} training vectors, {len(X_test)} test queries, dim={X_train.shape[1]}")
    
    # Compute ground truth
    print("\nComputing ground truth with FAISS Flat...")
    ground_truth = compute_ground_truth(X_train, X_test, k, metric)
    
    # Benchmark NeuronDB
    print("\n" + "="*60)
    print("Benchmarking NeuronDB")
    print("="*60)
    neurondb_results, neurondb_qps, neurondb_meta = benchmark_neurondb(
        X_train, X_test, metric, neurondb_index, neurondb_params, k,
        connection_string, **db_kwargs
    )
    neurondb_recall = compute_recall(neurondb_results, ground_truth)
    
    # Benchmark FAISS
    print("\n" + "="*60)
    print("Benchmarking FAISS")
    print("="*60)
    faiss_results, faiss_qps, faiss_meta = benchmark_faiss(
        X_train, X_test, metric, faiss_index, faiss_params, k
    )
    faiss_recall = compute_recall(faiss_results.tolist(), ground_truth)
    
    # Compile results
    results = {
        "dataset": dataset_name,
        "metric": metric,
        "k": k,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dimension": X_train.shape[1],
        "neurondb": {
            "qps": neurondb_qps,
            "recall": neurondb_recall,
            **neurondb_meta
        },
        "faiss": {
            "qps": faiss_qps,
            "recall": faiss_recall,
            **faiss_meta
        },
        "ground_truth": {
            "source": "FAISS Flat (exact search)",
            "recall": 1.0  # Ground truth has perfect recall
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric}, k={k}")
    print(f"\nNeuronDB ({neurondb_index}):")
    print(f"  QPS: {neurondb_qps:.2f}")
    print(f"  Recall: {neurondb_recall:.4f}")
    print(f"  Build time: {neurondb_meta['build_time']:.2f}s")
    print(f"\nFAISS ({faiss_index}):")
    print(f"  QPS: {faiss_qps:.2f}")
    print(f"  Recall: {faiss_recall:.4f}")
    print(f"  Build time: {faiss_meta['build_time']:.2f}s")
    print(f"\nSpeedup: {faiss_qps/neurondb_qps:.2f}x" if neurondb_qps > 0 else "N/A")
    print(f"Recall ratio: {neurondb_recall/faiss_recall:.4f}" if faiss_recall > 0 else "N/A")
    print("="*60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"comparison_{dataset_name}_{metric}_{k}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Plot results
    try:
        plot_comparison(results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    return results


def plot_comparison(results: Dict, output_dir: str):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    
    dataset = results["dataset"]
    metric = results["metric"]
    k = results["k"]
    
    # Recall vs QPS plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    neurondb_qps = results["neurondb"]["qps"]
    neurondb_recall = results["neurondb"]["recall"]
    faiss_qps = results["faiss"]["qps"]
    faiss_recall = results["faiss"]["recall"]
    
    ax.scatter([neurondb_qps], [neurondb_recall], 
               s=200, label=f"NeuronDB ({results['neurondb']['index_type']})", 
               marker='o', color='blue')
    ax.scatter([faiss_qps], [faiss_recall], 
               s=200, label=f"FAISS ({results['faiss']['index_type']})", 
               marker='s', color='red')
    
    ax.set_xlabel('Queries Per Second (QPS)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(f'Recall vs QPS: {dataset} (k={k}, {metric})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"comparison_{dataset}_{metric}_{k}.png")
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NeuronDB against FAISS using ANN-Benchmarks"
    )
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean",
                       help="Dataset name")
    parser.add_argument("--metric", type=str, default="euclidean",
                       choices=["euclidean", "angular"],
                       help="Distance metric")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of neighbors")
    parser.add_argument("--neurondb-index", type=str, default="hnsw",
                       choices=["hnsw", "ivfflat", "none"],
                       help="NeuronDB index type")
    parser.add_argument("--neurondb-m", type=int, default=16,
                       help="NeuronDB HNSW m parameter")
    parser.add_argument("--neurondb-ef-construction", type=int, default=200,
                       help="NeuronDB HNSW ef_construction parameter")
    parser.add_argument("--faiss-index", type=str, default="flat",
                       choices=["flat", "ivf", "hnsw"],
                       help="FAISS index type")
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
    
    # Build index parameters
    neurondb_params = {}
    if args.neurondb_index == "hnsw":
        neurondb_params = {
            "m": args.neurondb_m,
            "ef_construction": args.neurondb_ef_construction
        }
    
    # Run comparison
    results = run_comparison(
        dataset_name=args.dataset,
        metric=args.metric,
        k=args.k,
        neurondb_index=args.neurondb_index,
        neurondb_params=neurondb_params,
        faiss_index=args.faiss_index,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )


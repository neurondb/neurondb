#!/usr/bin/env python3
"""
Main benchmark runner for NeuronDB ANN-Benchmarks compatibility tests.

This script orchestrates running multiple benchmarks with different
configurations and generates comprehensive reports.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import time

from compare_with_faiss import run_comparison


# Standard ANN-Benchmarks datasets
STANDARD_DATASETS = [
    "sift-128-euclidean",
    "glove-100-angular",
    "deep-image-96-angular",
    "gist-960-euclidean",
    "nytimes-256-angular",
    "lastfm-64-dot",
]

# Common index configurations
INDEX_CONFIGS = {
    "hnsw_fast": {
        "index_type": "hnsw",
        "params": {"m": 16, "ef_construction": 200}
    },
    "hnsw_accurate": {
        "index_type": "hnsw",
        "params": {"m": 32, "ef_construction": 400}
    },
    "ivfflat": {
        "index_type": "ivfflat",
        "params": {"lists": 100}
    },
    "none": {
        "index_type": "none",
        "params": {}
    }
}


def run_benchmark_suite(
    datasets: List[str],
    configs: List[str],
    k_values: List[int],
    output_dir: str,
    connection_string: str = None,
    **db_kwargs
) -> Dict:
    """
    Run a suite of benchmarks.
    
    Args:
        datasets: List of dataset names
        configs: List of configuration names
        k_values: List of k values to test
        output_dir: Output directory
        connection_string: Database connection string
        **db_kwargs: Database connection parameters
    
    Returns:
        Dictionary with all results
    """
    all_results = []
    total_benchmarks = len(datasets) * len(configs) * len(k_values)
    current = 0
    
    print(f"Running {total_benchmarks} benchmark configurations...")
    print("="*60)
    
    for dataset in datasets:
        # Extract metric from dataset name
        if "euclidean" in dataset:
            metric = "euclidean"
        elif "angular" in dataset:
            metric = "angular"
        elif "dot" in dataset:
            metric = "inner_product"
        else:
            metric = "euclidean"  # default
        
        for config_name in configs:
            if config_name not in INDEX_CONFIGS:
                print(f"Warning: Unknown config '{config_name}', skipping")
                continue
            
            config = INDEX_CONFIGS[config_name]
            
            for k in k_values:
                current += 1
                print(f"\n[{current}/{total_benchmarks}] Running: {dataset}, {config_name}, k={k}")
                print("-"*60)
                
                try:
                    result = run_comparison(
                        dataset_name=dataset,
                        metric=metric,
                        k=k,
                        neurondb_index=config["index_type"],
                        neurondb_params=config["params"],
                        faiss_index="flat",  # Always compare against FAISS Flat
                        connection_string=connection_string,
                        output_dir=output_dir,
                        **db_kwargs
                    )
                    
                    result["config_name"] = config_name
                    all_results.append(result)
                    
                    # Save intermediate results
                    summary_file = os.path.join(output_dir, "benchmark_summary.json")
                    with open(summary_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                except Exception as e:
                    print(f"Error running benchmark: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    return {
        "total_benchmarks": total_benchmarks,
        "completed": len(all_results),
        "results": all_results
    }


def generate_report(results: Dict, output_dir: str):
    """Generate a comprehensive benchmark report."""
    import json
    
    report = {
        "summary": {
            "total_benchmarks": results["total_benchmarks"],
            "completed": results["completed"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results_by_dataset": {},
        "results_by_config": {},
        "best_configs": {}
    }
    
    # Organize results
    for result in results["results"]:
        dataset = result["dataset"]
        config = result["config_name"]
        
        # By dataset
        if dataset not in report["results_by_dataset"]:
            report["results_by_dataset"][dataset] = []
        report["results_by_dataset"][dataset].append({
            "config": config,
            "k": result["k"],
            "neurondb_qps": result["neurondb"]["qps"],
            "neurondb_recall": result["neurondb"]["recall"],
            "faiss_qps": result["faiss"]["qps"],
            "faiss_recall": result["faiss"]["recall"],
        })
        
        # By config
        if config not in report["results_by_config"]:
            report["results_by_config"][config] = []
        report["results_by_config"][config].append({
            "dataset": dataset,
            "k": result["k"],
            "neurondb_qps": result["neurondb"]["qps"],
            "neurondb_recall": result["neurondb"]["recall"],
        })
    
    # Find best configs (highest recall at reasonable QPS)
    for dataset in report["results_by_dataset"]:
        dataset_results = report["results_by_dataset"][dataset]
        best = max(dataset_results, 
                   key=lambda x: x["neurondb_recall"] * (x["neurondb_qps"] / 1000.0))
        report["best_configs"][dataset] = best
    
    # Save report
    report_file = os.path.join(output_dir, "benchmark_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Report Summary")
    print("="*60)
    print(f"Total benchmarks: {report['summary']['total_benchmarks']}")
    print(f"Completed: {report['summary']['completed']}")
    print(f"\nBest configurations by dataset:")
    for dataset, best in report["best_configs"].items():
        print(f"  {dataset}:")
        print(f"    Config: {best['config']}, k={best['k']}")
        print(f"    Recall: {best['neurondb_recall']:.4f}, QPS: {best['neurondb_qps']:.2f}")
    print(f"\nFull report saved to: {report_file}")
    print("="*60)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NeuronDB ANN-Benchmarks compatibility suite"
    )
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["sift-128-euclidean"],
                       help="Datasets to benchmark")
    parser.add_argument("--configs", type=str, nargs="+",
                       default=["hnsw_fast", "hnsw_accurate"],
                       choices=list(INDEX_CONFIGS.keys()),
                       help="Index configurations to test")
    parser.add_argument("--k", type=int, nargs="+",
                       default=[10],
                       help="k values to test")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory")
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
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with minimal datasets")
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.datasets = ["sift-128-euclidean"]
        args.configs = ["hnsw_fast"]
        args.k = [10]
        print("Running in quick mode...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    results = run_benchmark_suite(
        datasets=args.datasets,
        configs=args.configs,
        k_values=args.k,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )
    
    # Generate report
    if results["results"]:
        generate_report(results, args.output_dir)
    else:
        print("No results to report.")




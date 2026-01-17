#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NeuronDB RAG Benchmark Suite                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  A comprehensive benchmark framework for evaluating NeuronDB's RAG           ║
║  (Retrieval-Augmented Generation) capabilities using MTEB, BEIR, and        ║
║  RAGAS frameworks.                                                           ║
║                                                                              ║
║  Features:                                                                   ║
║    • MTEB: Text embedding benchmarks across 56+ datasets                    ║
║    • BEIR: Zero-shot information retrieval evaluation                       ║
║    • RAGAS: End-to-end RAG pipeline quality assessment                      ║
║    • Automatic data preparation and database loading                        ║
║    • Real-time progress tracking with detailed metrics                      ║
║    • Professional CLI with comprehensive help documentation                 ║
║                                                                              ║
║  Benchmark Types:                                                            ║
║    • MTEB: Embedding quality (classification, clustering, retrieval)        ║
║    • BEIR: Retrieval performance (NDCG, MAP, Recall)                        ║
║    • RAGAS: RAG quality (faithfulness, relevancy, context precision)        ║
║                                                                              ║
║  Author: NeuronDB Team                                                       ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import traceback

# Check if running with --help or --version (don't require dependencies)
HELP_ARGS = {'--help', '-h', '--version', '--list-benchmarks'}
if not any(arg in HELP_ARGS for arg in sys.argv[1:]):
    try:
        import psycopg2
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}", file=sys.stderr)
        print("\nPlease install dependencies:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
else:
    # Dummy imports for help display
    psycopg2 = None
    tqdm = None

# Version information
__version__ = "2.1.0"
__author__ = "NeuronDB Team"

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

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

# Default database configuration
DEFAULT_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "neurondb",
    "user": "pge",
    "password": None,
}

# Benchmark types and their descriptions
BENCHMARK_TYPES = {
    "mteb": {
        "description": "Massive Text Embedding Benchmark - evaluate embedding quality",
        "tasks": ["classification", "clustering", "reranking", "retrieval", "sts", "summarization"],
        "module": "mteb_benchmark",
    },
    "beir": {
        "description": "Benchmarking IR - zero-shot retrieval evaluation",
        "datasets": ["msmarco", "nq", "scifact", "trec-covid", "hotpotqa"],
        "module": "beir_benchmark",
    },
    "ragas": {
        "description": "RAG Assessment - end-to-end RAG pipeline evaluation",
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "module": "ragas_benchmark",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print welcome banner."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║              NeuronDB RAG Benchmark Suite v{:<32}║".format(__version__))
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
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'─' * 80}{Colors.ENDC}\n")

def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print progress indicator."""
    percent = 100 * (current / float(total))
    filled = int(50 * current // total)
    bar = '█' * filled + '░' * (50 - filled)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()

# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Handles database connection and verification."""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Establish and verify database connection."""
        try:
            self.conn = psycopg2.connect(**{k: v for k, v in self.db_config.items() if v is not None})
            self.conn.autocommit = True
            
            # Verify NeuronDB extension
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'neurondb';")
                if not cur.fetchone():
                    print_status("Creating NeuronDB extension...", "warning")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS neurondb;")
            
            print_status("Database connection established", "success")
            return True
            
        except Exception as e:
            print_status(f"Database connection failed: {e}", "error")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def test_connection(self):
        """Test basic database operations."""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print_status(f"PostgreSQL version: {version[:50]}...", "info")
                
                # Test NeuronDB functions
                cur.execute("SELECT COUNT(*) FROM pg_proc WHERE proname LIKE 'neurondb%';")
                func_count = cur.fetchone()[0]
                print_status(f"NeuronDB functions available: {func_count}", "info")
            
            return True
            
        except Exception as e:
            print_status(f"Connection test failed: {e}", "error")
            return False

# ═══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK EXECUTION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Handles execution of different benchmark types."""
    
    def __init__(self, db_config: Dict, output_dir: str = "./results"):
        self.db_config = db_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_mteb(self, model_name: str, tasks: List[str] = None):
        """Run MTEB benchmarks."""
        print_section(f"Running MTEB Benchmark")
        print(f"  Model: {model_name}")
        print(f"  Tasks: {', '.join(tasks) if tasks else 'all'}\n")
        
        try:
            # Dynamically import MTEB benchmark module
            import mteb_benchmark
            
            results = mteb_benchmark.run_mteb_benchmark(
                model_name=model_name,
                tasks=tasks,
                output_dir=str(self.output_dir),
                **self.db_config
            )
            
            print_status("MTEB benchmark completed", "success")
            return results
            
        except ImportError:
            print_status("MTEB module not found. Install with: pip install mteb", "error")
            return None
        except Exception as e:
            print_status(f"MTEB benchmark failed: {e}", "error")
            traceback.print_exc()
            return None
    
    def run_beir(self, dataset: str, model_name: str):
        """Run BEIR benchmarks."""
        print_section(f"Running BEIR Benchmark")
        print(f"  Dataset: {dataset}")
        print(f"  Model: {model_name}\n")
        
        try:
            # Dynamically import BEIR benchmark module
            import beir_benchmark
            
            results = beir_benchmark.run_beir_benchmark(
                dataset=dataset,
                model_name=model_name,
                output_dir=str(self.output_dir),
                **self.db_config
            )
            
            print_status("BEIR benchmark completed", "success")
            return results
            
        except ImportError:
            print_status("BEIR module not found. Install with: pip install beir", "error")
            return None
        except Exception as e:
            print_status(f"BEIR benchmark failed: {e}", "error")
            traceback.print_exc()
            return None
    
    def run_ragas(self, dataset_path: str, model_name: str):
        """Run RAGAS benchmarks."""
        print_section(f"Running RAGAS Benchmark")
        print(f"  Dataset: {dataset_path}")
        print(f"  Model: {model_name}\n")
        
        if not os.path.exists(dataset_path):
            print_status(f"Dataset file not found: {dataset_path}", "error")
            return None
        
        try:
            # Dynamically import RAGAS benchmark module
            import ragas_benchmark
            
            results = ragas_benchmark.run_ragas_benchmark(
                dataset_path=dataset_path,
                model_name=model_name,
                output_dir=str(self.output_dir),
                **self.db_config
            )
            
            print_status("RAGAS benchmark completed", "success")
            return results
            
        except ImportError:
            print_status("RAGAS module not found. Install with: pip install ragas", "error")
            return None
        except Exception as e:
            print_status(f"RAGAS benchmark failed: {e}", "error")
            traceback.print_exc()
            return None
    
    def save_results(self, benchmark_name: str, results: Dict):
        """Save benchmark results."""
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_results_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print_status(f"Results saved to {filepath}", "success")
            
        except Exception as e:
            print_status(f"Failed to save results: {e}", "error")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RAGBenchmarkOrchestrator:
    """Main orchestrator for RAG benchmarks."""
    
    def __init__(self, args):
        self.args = args
        self.db_config = {
            "host": args.db_host,
            "port": args.db_port,
            "database": args.db_name,
            "user": args.db_user,
            "password": args.db_password,
        }
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self):
        """Prepare data for benchmarks."""
        print_banner()
        print_section("Data Preparation")
        
        benchmarks = self._parse_benchmarks()
        
        print("Data preparation steps:")
        if "mteb" in benchmarks:
            print("  • MTEB: Datasets will be auto-downloaded during benchmark execution")
        if "beir" in benchmarks:
            print("  • BEIR: Datasets will be auto-downloaded during benchmark execution")
        if "ragas" in benchmarks:
            if self.args.ragas_dataset:
                if os.path.exists(self.args.ragas_dataset):
                    print(f"  • RAGAS: Dataset found at {self.args.ragas_dataset}")
                else:
                    print(f"  • RAGAS: Dataset NOT found at {self.args.ragas_dataset}")
                    print_status("RAGAS dataset file missing", "error")
                    return False
            else:
                print("  • RAGAS: No dataset specified (use --ragas-dataset)")
        
        print_status("Data preparation completed", "success")
        return True
    
    def verify_database(self):
        """Verify database connection."""
        print_banner()
        print_section("Database Verification")
        
        db_manager = DatabaseManager(self.db_config)
        
        if not db_manager.connect():
            return False
        
        success = db_manager.test_connection()
        db_manager.disconnect()
        
        return success
    
    def run_benchmarks(self):
        """Execute all requested benchmarks."""
        print_banner()
        print_section("Benchmark Execution")
        
        benchmarks = self._parse_benchmarks()
        runner = BenchmarkRunner(self.db_config, str(self.output_dir))
        
        all_results = {}
        success_count = 0
        fail_count = 0
        
        # Run MTEB
        if "mteb" in benchmarks:
            try:
                results = runner.run_mteb(
                    model_name=self.args.model,
                    tasks=self.args.mteb_tasks.split(',') if self.args.mteb_tasks else None
                )
                if results:
                    all_results["mteb"] = results
                    runner.save_results("mteb", results)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print_status(f"MTEB execution failed: {e}", "error")
                fail_count += 1
        
        # Run BEIR
        if "beir" in benchmarks:
            if not self.args.beir_dataset:
                print_status("BEIR dataset not specified (use --beir-dataset)", "warning")
                fail_count += 1
            else:
                try:
                    results = runner.run_beir(
                        dataset=self.args.beir_dataset,
                        model_name=self.args.model
                    )
                    if results:
                        all_results["beir"] = results
                        runner.save_results("beir", results)
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print_status(f"BEIR execution failed: {e}", "error")
                    fail_count += 1
        
        # Run RAGAS
        if "ragas" in benchmarks:
            if not self.args.ragas_dataset:
                print_status("RAGAS dataset not specified (use --ragas-dataset)", "warning")
                fail_count += 1
            else:
                try:
                    results = runner.run_ragas(
                        dataset_path=self.args.ragas_dataset,
                        model_name=self.args.model
                    )
                    if results:
                        all_results["ragas"] = results
                        runner.save_results("ragas", results)
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print_status(f"RAGAS execution failed: {e}", "error")
                    fail_count += 1
        
        # Print summary
        print_section("Benchmark Summary")
        print(f"  Total benchmarks: {success_count + fail_count}")
        print(f"  {Colors.OKGREEN}Successful: {success_count}{Colors.ENDC}")
        print(f"  {Colors.FAIL}Failed: {fail_count}{Colors.ENDC}")
        
        return fail_count == 0
    
    def _parse_benchmarks(self) -> List[str]:
        """Parse which benchmarks to run."""
        if self.args.benchmarks.lower() == "all":
            return ["mteb", "beir", "ragas"]
        return [b.strip().lower() for b in self.args.benchmarks.split(',')]

# ═══════════════════════════════════════════════════════════════════════════════
#  COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def create_parser():
    """Create comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare and verify database connection
  %(prog)s --prepare --verify
  
  # Run MTEB benchmark
  %(prog)s --run --benchmarks mteb --model all-MiniLM-L6-v2
  
  # Run BEIR benchmark
  %(prog)s --run --benchmarks beir --beir-dataset msmarco --model all-MiniLM-L6-v2
  
  # Run RAGAS benchmark
  %(prog)s --run --benchmarks ragas --ragas-dataset ./data/ragas_test.json
  
  # Run all benchmarks
  %(prog)s --run --benchmarks all --beir-dataset msmarco --ragas-dataset ./data/ragas_test.json
  
  # Full pipeline
  %(prog)s --prepare --verify --run --benchmarks all

For more information, visit: https://github.com/neurondb/neurondb
        """
    )
    
    # Execution modes
    mode_group = parser.add_argument_group('Execution Modes')
    mode_group.add_argument('--prepare', action='store_true',
                           help='Prepare data and verify prerequisites')
    mode_group.add_argument('--verify', action='store_true',
                           help='Verify database connection and NeuronDB extension')
    mode_group.add_argument('--run', action='store_true',
                           help='Execute benchmarks')
    
    # Benchmark selection
    bench_group = parser.add_argument_group('Benchmark Selection')
    bench_group.add_argument('--benchmarks', type=str, default='all',
                            help='Comma-separated list of benchmarks: mteb,beir,ragas or "all" (default: all)')
    bench_group.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                            help='Embedding model name (default: all-MiniLM-L6-v2)')
    
    # MTEB options
    mteb_group = parser.add_argument_group('MTEB Options')
    mteb_group.add_argument('--mteb-tasks', type=str, default=None,
                           help='Comma-separated MTEB tasks (default: all)\n'
                                'Available: classification, clustering, reranking, retrieval, sts, summarization')
    
    # BEIR options
    beir_group = parser.add_argument_group('BEIR Options')
    beir_group.add_argument('--beir-dataset', type=str, default=None,
                           help='BEIR dataset name (required for BEIR benchmark)\n'
                                'Examples: msmarco, nq, scifact, trec-covid, hotpotqa')
    
    # RAGAS options
    ragas_group = parser.add_argument_group('RAGAS Options')
    ragas_group.add_argument('--ragas-dataset', type=str, default=None,
                            help='Path to RAGAS dataset JSON file (required for RAGAS benchmark)')
    
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
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output-dir', type=str, default='./results',
                             help='Output directory for results (default: ./results)')
    output_group.add_argument('--continue-on-error', action='store_true',
                             help='Continue execution even if individual benchmarks fail')
    
    # Information
    info_group = parser.add_argument_group('Information')
    info_group.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    info_group.add_argument('--list-benchmarks', action='store_true',
                           help='List all available benchmark types and exit')
    
    return parser

def list_benchmarks():
    """List available benchmark types."""
    print(f"\n{Colors.BOLD}Available RAG Benchmarks:{Colors.ENDC}\n")
    for name, info in BENCHMARK_TYPES.items():
        print(f"  {Colors.OKBLUE}{name.upper()}{Colors.ENDC}")
        print(f"    {info['description']}")
        if 'tasks' in info:
            print(f"    Tasks: {', '.join(info['tasks'])}")
        if 'datasets' in info:
            print(f"    Example datasets: {', '.join(info['datasets'])}")
        if 'metrics' in info:
            print(f"    Metrics: {', '.join(info['metrics'])}")
        print()

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle information requests
    if args.list_benchmarks:
        list_benchmarks()
        return 0
    
    # Validate execution mode
    if not (args.prepare or args.verify or args.run):
        parser.error("At least one mode (--prepare, --verify, --run) must be specified")
    
    # Create orchestrator
    orchestrator = RAGBenchmarkOrchestrator(args)
    
    # Execute requested operations
    try:
        if args.prepare:
            if not orchestrator.prepare_data():
                return 1
        
        if args.verify:
            if not orchestrator.verify_database():
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

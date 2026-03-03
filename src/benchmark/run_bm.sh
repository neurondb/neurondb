#!/bin/bash
#
# NeuronDB Benchmark Suite Runner
# Runs all benchmarks (vector, hybrid, RAG) with proper setup and error handling
#

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Get script directory
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_DIR"

# Print banner
echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              NeuronDB Benchmark Suite Runner                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Function to print status
print_status() {
    local status=$1
    local message=$2
    local timestamp=$(date +"%H:%M:%S")
    
    case $status in
        "success")
            echo -e "[${timestamp}] ${GREEN}✓${NC} ${message}"
            ;;
        "error")
            echo -e "[${timestamp}] ${RED}✗${NC} ${message}"
            ;;
        "warning")
            echo -e "[${timestamp}] ${YELLOW}⚠${NC} ${message}"
            ;;
        *)
            echo -e "[${timestamp}] ${BLUE}ℹ${NC} ${message}"
            ;;
    esac
}

# Function to setup and run a benchmark
run_benchmark() {
    local folder=$1
    local name=$2
    local prepare_cmd=$3
    local load_cmd=$4
    local run_cmd=$5
    
    echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $name Benchmark${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}\n"
    
    cd "$folder"
    
    # Setup virtual environment
    if [ ! -d "venv" ]; then
        print_status "info" "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install dependencies
    print_status "info" "Installing/updating dependencies..."
    pip install -q --upgrade pip 2>/dev/null || true
    if ! pip install -q -r requirements.txt 2>&1; then
        print_status "error" "Failed to install dependencies for $name"
        deactivate
        cd ..
        return 1
    fi
    print_status "success" "Dependencies installed"
    
    # Run prepare step
    if [ -n "$prepare_cmd" ]; then
        print_status "info" "Preparing data..."
        eval "$prepare_cmd" || print_status "warning" "Prepare step had issues (continuing...)"
    fi
    
    # Run load step
    if [ -n "$load_cmd" ]; then
        print_status "info" "Loading data..."
        eval "$load_cmd" || print_status "warning" "Load step had issues (continuing...)"
    fi
    
    # Run benchmark step
    if [ -n "$run_cmd" ]; then
        print_status "info" "Running benchmark..."
        eval "$run_cmd" || print_status "warning" "Run step had issues (continuing...)"
    fi
    
    deactivate
    cd ..
    print_status "success" "$name benchmark completed"
}

# Check PostgreSQL connection
print_status "info" "Checking PostgreSQL connection..."
if ! pg_isready -h localhost -p 5432 -U pge >/dev/null 2>&1; then
    print_status "error" "PostgreSQL is not running or not accessible"
    print_status "error" "Please start PostgreSQL and ensure 'pge' user can connect"
    exit 1
fi
print_status "success" "PostgreSQL is running"

# Run Vector Benchmark
run_benchmark "vector" "Vector" \
    "./run_bm.py --prepare --datasets sift-128-euclidean --continue-on-error" \
    "" \
    "./run_bm.py --run --datasets sift-128-euclidean --configs hnsw --k-values 10 --max-queries 50 --continue-on-error"

# Run Hybrid Benchmark
run_benchmark "hybrid" "Hybrid" \
    "./run_bm.py --prepare --datasets nfcorpus --continue-on-error" \
    "timeout 900 ./run_bm.py --load --datasets nfcorpus --model all-MiniLM-L6-v2 --index-config hnsw --batch-size 50 --continue-on-error" \
    "timeout 300 ./run_bm.py --run --datasets nfcorpus --vector-weights 0.7 --top-k 10 --continue-on-error"

# Run RAG Benchmark
run_benchmark "rag" "RAG" \
    "./run_bm.py --prepare --verify --continue-on-error" \
    "" \
    ""

# Generate consolidated report
print_status "info" "Generating consolidated benchmark report..."
if [ -f "generate_consolidated_report.py" ]; then
    python3 generate_consolidated_report.py || print_status "warning" "Failed to generate consolidated report"
else
    print_status "warning" "Consolidated report generator not found"
fi

# Final summary
echo -e "\n${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                    ALL BENCHMARKS EXECUTION COMPLETE                          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}\n"

print_status "info" "Results saved in:"
print_status "info" "  • Vector:  ./vector/results/"
print_status "info" "  • Hybrid:  ./hybrid/results/"
print_status "info" "  • RAG:     ./rag/results/"
print_status "info" "  • Consolidated: README.md (Benchmark Results section)"


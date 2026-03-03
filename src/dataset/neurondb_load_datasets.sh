#!/bin/bash
# ====================================================================
# NeurondB Dataset Loading Script
# ====================================================================
# Loads comprehensive datasets for testing all NeurondB features
# ====================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.1.0"

# Default values
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neurondb_load_datasets.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronDB Dataset Loading Script

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Loads comprehensive datasets for testing all NeuronDB features

Options:
    -v, --verbose    Enable verbose output
    -V, --version    Show version information
    -h, --help       Show this help message

Environment Variables:
    PGHOST       Database host (default: localhost)
    PGPORT       Database port (default: 5432)
    PGUSER       Database user (default: postgres)
    PGDATABASE   Database name (default: neurondb_test)

Examples:
    # Basic usage
    $SCRIPT_NAME

    # With verbose output
    $SCRIPT_NAME --verbose

EOF
			exit 0
			;;
		*)
			echo -e "${RED}Unknown option: $1${NC}" >&2
			echo "Use -h or --help for usage information" >&2
			exit 1
			;;
	esac
done

# Database configuration
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-postgres}"
export PGDATABASE="${PGDATABASE:-neurondb_test}"

if [ "$VERBOSE" = true ]; then
	echo "========================================"
	echo "NeuronDB Dataset Loading"
	echo "========================================"
	echo "Host: $PGHOST"
	echo "Port: $PGPORT"
	echo "Database: $PGDATABASE"
	echo "========================================"
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Step 1: Creating/Recreating Database"
echo "--------------------------------------"
python3 neurondb_dataset.py --recreate-db

echo ""
echo "Step 2: Loading MS MARCO Dataset (Document Retrieval)"
echo "------------------------------------------------------"
echo "MS MARCO: Large-scale information retrieval dataset"
echo "Usage: Testing semantic search, reranking, hybrid search"
python3 neurondb_dataset.py --load-msmarco --limit 10000

echo ""
echo "Step 3: Loading Wikipedia Embeddings (General Knowledge)"
echo "---------------------------------------------------------"
echo "Wikipedia: General knowledge articles with embeddings"
echo "Usage: Testing clustering, PCA, outlier detection"
python3 neurondb_dataset.py --load-wikipedia --limit 5000

echo ""
echo "Step 4: Loading HotpotQA Dataset (Question Answering)"
echo "------------------------------------------------------"
echo "HotpotQA: Multi-hop question answering dataset"
echo "Usage: Testing MMR, recall metrics, topic discovery"
python3 neurondb_dataset.py --load-hotpotqa --limit 3000

echo ""
echo "Step 5: Loading SIFT1M Dataset (Computer Vision)"
echo "-------------------------------------------------"
echo "SIFT: SIFT descriptors for image matching"
echo "Usage: Testing PQ, OPQ, high-dimensional vectors"
python3 neurondb_dataset.py --load-sift --limit 50000

echo ""
echo "Step 6: Loading Deep1B Sample (Large-Scale Vectors)"
echo "----------------------------------------------------"
echo "Deep1B: Deep learning embeddings at scale"
echo "Usage: Testing scalability, performance benchmarks"
python3 neurondb_dataset.py --load-deep1b --limit 20000

echo ""
echo "Step 7: Creating Synthetic Test Datasets"
echo "-----------------------------------------"
python3 neurondb_dataset.py --create-synthetic

echo ""
echo "Step 8: Creating Full-Text Search Indexes"
echo "------------------------------------------"
python3 neurondb_dataset.py --create-fts-indexes

echo ""
echo "Step 9: Generating Dataset Statistics"
echo "--------------------------------------"
python3 neurondb_dataset.py --show-stats

echo ""
echo "========================================"
echo "Dataset Loading Complete!"
echo "========================================"
echo ""
echo "Datasets loaded:"
echo "  - neurondb_datasets.msmarco_passages (~10K passages)"
echo "  - neurondb_datasets.wikipedia_articles (~5K articles)"
echo "  - neurondb_datasets.hotpotqa_questions (~3K questions)"
echo "  - neurondb_datasets.sift_vectors (~50K SIFT descriptors)"
echo "  - neurondb_datasets.deep1b_vectors (~20K embeddings)"
echo "  - neurondb_datasets.synthetic_* (various synthetic datasets)"
echo ""
echo "Ready for regression testing!"
echo "========================================"

deactivate

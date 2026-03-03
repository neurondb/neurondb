#!/bin/bash
# ====================================================================
# NeurondB Comprehensive Regression Testing with Real Datasets
# ====================================================================
# This script:
#   1. Sets up test database
#   2. Loads comprehensive datasets
#   3. Runs full regression suite
#   4. Reports results
# ====================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.1.0"

# Default values
VERBOSE=false

# Configuration
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-postgres}"
export PGDATABASE="${PGDATABASE:-neurondb_test}"
TEST_DB="neurondb_test"

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
			echo "neurondb_test_datasets.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronDB Comprehensive Regression Testing with Real Datasets

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    This script:
    1. Sets up test database
    2. Loads comprehensive datasets
    3. Runs full regression suite
    4. Reports results

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

if [ "$VERBOSE" = true ]; then
	echo "========================================"
	echo "NeuronDB Regression Testing with Datasets"
	echo "========================================"
	echo "PostgreSQL: $PGHOST:$PGPORT"
	echo "Test Database: $TEST_DB"
	echo "Project Root: $PROJECT_ROOT"
	echo "========================================"
	echo ""
fi

# Step 1: Ensure NeurondB is compiled
echo -e "${BLUE}Step 1: Building NeurondB Extension${NC}"
echo "----------------------------------------"
if [ ! -f "neurondb.so" ] && [ ! -f "neurondb.dylib" ]; then
    echo "Extension not built. Running 'make clean && make'..."
    make clean
    make
    echo -e "${GREEN}✓ Build complete${NC}"
else
    echo -e "${GREEN}✓ Extension already built${NC}"
fi
echo ""

# Step 2: Set up Python environment for dataset loading
echo -e "${BLUE}Step 2: Setting up Python Environment${NC}"
echo "----------------------------------------"
cd "$SCRIPT_DIR"
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Python environment ready${NC}"
echo ""

# Step 3: Create/Recreate test database
echo -e "${BLUE}Step 3: Setting up Test Database${NC}"
echo "----------------------------------------"
python3 neurondb_dataset.py --recreate-db --dbname "$TEST_DB"
echo ""

# Step 4: Install NeurondB extension in test database
echo -e "${BLUE}Step 4: Installing NeurondB Extension${NC}"
echo "----------------------------------------"
export PGDATABASE="$TEST_DB"
psql -v ON_ERROR_STOP=1 -f neurondb_setup_test_db.sql
echo -e "${GREEN}✓ Extension installed${NC}"
echo ""

# Step 5: Load datasets (with progress indicators)
echo -e "${BLUE}Step 5: Loading Comprehensive Datasets${NC}"
echo "----------------------------------------"
echo "This may take 10-30 minutes depending on network speed..."
echo ""

echo -e "${YELLOW}[1/6]${NC} Loading MS MARCO passages (document retrieval)..."
python3 neurondb_dataset.py --load-msmarco --limit 10000 --dbname "$TEST_DB"

echo -e "${YELLOW}[2/6]${NC} Loading Wikipedia embeddings (clustering, PCA)..."
python3 neurondb_dataset.py --load-wikipedia --limit 5000 --dbname "$TEST_DB"

echo -e "${YELLOW}[3/6]${NC} Loading HotpotQA (question answering, MMR)..."
python3 neurondb_dataset.py --load-hotpotqa --limit 3000 --dbname "$TEST_DB"

echo -e "${YELLOW}[4/6]${NC} Loading SIFT vectors (high-dim, quantization)..."
python3 neurondb_dataset.py --load-sift --limit 50000 --dbname "$TEST_DB"

echo -e "${YELLOW}[5/6]${NC} Loading Deep1B vectors (scalability)..."
python3 neurondb_dataset.py --load-deep1b --limit 20000 --dbname "$TEST_DB"

echo -e "${YELLOW}[6/6]${NC} Creating synthetic test datasets..."
python3 neurondb_dataset.py --create-synthetic --dbname "$TEST_DB"

echo ""
echo -e "${YELLOW}Creating full-text search indexes...${NC}"
python3 neurondb_dataset.py --create-fts-indexes --dbname "$TEST_DB"

echo ""
echo -e "${GREEN}✓ All datasets loaded${NC}"
echo ""

# Step 6: Show dataset statistics
echo -e "${BLUE}Step 6: Dataset Statistics${NC}"
echo "----------------------------------------"
python3 neurondb_dataset.py --show-stats --dbname "$TEST_DB"
echo ""

deactivate
cd "$PROJECT_ROOT"

# Step 7: Run regression tests
echo -e "${BLUE}Step 7: Running Regression Test Suite${NC}"
echo "----------------------------------------"
echo "Running all tests (including ML algorithms)..."
echo ""

export PGDATABASE="$TEST_DB"

# Run regression tests
if make installcheck; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    RESULT=0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check regression.diffs for details:"
    if [ -f "regression.diffs" ]; then
        echo ""
        cat regression.diffs
    fi
    RESULT=1
fi

# Step 8: Generate test report
echo ""
echo -e "${BLUE}Step 8: Test Summary${NC}"
echo "----------------------------------------"
if [ -f "regression.out" ]; then
    PASSED=$(grep -c "^ok" regression.out || true)
    FAILED=$(grep -c "^not ok" regression.out || true)
    echo "Tests Passed: $PASSED"
    echo "Tests Failed: $FAILED"
else
    echo "No test results found"
fi

# Show which tests were run
echo ""
echo "Tests executed:"
echo "  - Core types and operations (01-04)"
echo "  - Catalog and workers (05-06)"
echo "  - Data management (07-08)"
echo "  - GPU features (09-11)"
echo "  - Catalog validation (12)"
echo "  - ML: Clustering (K-Means, DBSCAN, GMM, Hierarchical) (13)"
echo "  - ML: Dimensionality (PCA, Whitening) (14)"
echo "  - ML: Quantization (PQ, OPQ) (15)"
echo "  - ML: Reranking (MMR, RRF, Ensemble) (16)"
echo "  - ML: Outliers (Z-score, IQR, Isolation Forest) (17)"
echo "  - ML: Metrics (Recall@K, Precision@K, F1, MRR) (18)"
echo "  - ML: Drift Detection (Centroid, Distribution, Temporal) (19)"
echo "  - ML: Hybrid Search & LTR (20)"
echo "  - ML: Analytics (KNN Graph, Quality, Topics, Histograms) (21)"
echo ""

# Cleanup option
echo -e "${YELLOW}Test database '$TEST_DB' is still running.${NC}"
echo "To remove it, run:"
echo "  dropdb $TEST_DB"
echo ""

exit $RESULT

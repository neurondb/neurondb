#!/bin/bash
# ====================================================================
# NeuronAgent Integration Verification
# ====================================================================
# Comprehensive NeuronAgent-NeuronDB Integration Verification Script
# Tests all integration points between NeuronAgent and NeuronDB
# ====================================================================

set -e

cd "$(dirname "$0")/.."
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.0.0"

# Default values
VERBOSE=false

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
DB_USER="${DB_USER:-neurondb}"
DB_NAME="${DB_NAME:-neurondb}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5433}"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-D|--database)
			DB_NAME="$2"
			shift 2
			;;
		-U|--user)
			DB_USER="$2"
			shift 2
			;;
		-H|--host)
			DB_HOST="$2"
			shift 2
			;;
		-p|--port)
			DB_PORT="$2"
			shift 2
			;;
		--server-url)
			SERVER_URL="$2"
			shift 2
			;;
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neuronagent_verify.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronAgent Integration Verification

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Comprehensive NeuronAgent-NeuronDB Integration Verification Script
    Tests all integration points between NeuronAgent and NeuronDB

Options:
    -D, --database DB      Database name (default: neurondb)
    -U, --user USER        Database user (default: neurondb)
    -H, --host HOST        Database host (default: localhost)
    -p, --port PORT        Database port (default: 5433)
    --server-url URL       NeuronAgent server URL (default: http://localhost:8080)
    -v, --verbose          Enable verbose output
    -V, --version          Show version information
    -h, --help             Show this help message

Environment Variables:
    DB_USER       Database user (default: neurondb)
    DB_NAME       Database name (default: neurondb)
    DB_HOST       Database host (default: localhost)
    DB_PORT       Database port (default: 5433)
    SERVER_URL    NeuronAgent server URL (default: http://localhost:8080)

Examples:
    # Basic usage
    $SCRIPT_NAME

    # Custom database and server
    $SCRIPT_NAME -D mydb --server-url http://localhost:9090

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
	echo "NeuronAgent Integration Verification"
	echo "========================================"
	echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
	echo "User: $DB_USER"
	echo "Server URL: $SERVER_URL"
	echo "========================================"
	echo ""
fi

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0
TESTS_WARNED=0

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

test_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((TESTS_WARNED++))
    ((TESTS_TOTAL++))
}

test_info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

test_section() {
    echo ""
    echo -e "${CYAN}$1${NC}"
    echo "$(echo "$1" | sed 's/./=/g')"
}

# Helper function to check if function exists
function_exists() {
    local func_name=$1
    if [ -n "$DB_USER" ]; then
        psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = '$func_name');" 2>/dev/null | xargs
    else
        psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = '$func_name');" 2>/dev/null | xargs
    fi
}

# Helper function to check if type exists
type_exists() {
    local type_name=$1
    if [ -n "$DB_USER" ]; then
        psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = '$type_name');" 2>/dev/null | xargs
    else
        psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = '$type_name');" 2>/dev/null | xargs
    fi
}

# Helper function to check if table exists
table_exists() {
    local schema=$1
    local table=$2
    if [ -n "$DB_USER" ]; then
        psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = '$schema' AND table_name = '$table');" 2>/dev/null | xargs
    else
        psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = '$schema' AND table_name = '$table');" 2>/dev/null | xargs
    fi
}

# Helper function to run psql commands
run_psql() {
    local query=$1
    if [ -n "$DB_USER" ]; then
        psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "$query" 2>/dev/null
    else
        psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c "$query" 2>/dev/null
    fi
}

# ============================================================================
# PHASE 0: Prerequisites
# ============================================================================
test_section "PHASE 0: Prerequisites and Environment Setup"

# Check server (optional - some tests can run without it)
test_info "Checking NeuronAgent server..."
if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    test_pass "Server is running on $SERVER_URL"
    SERVER_AVAILABLE=1
else
    test_warn "Server is not running on $SERVER_URL (some API tests will be skipped)"
    SERVER_AVAILABLE=0
fi

# Check database
test_info "Checking PostgreSQL database..."
DB_AVAILABLE=0
if [ -n "$DB_PASSWORD" ]; then
    export PGPASSWORD="$DB_PASSWORD"
fi

if psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1;" > /dev/null 2>&1; then
    test_pass "Database connection successful"
    DB_AVAILABLE=1
else
    test_warn "Database connection failed (trying alternative connection methods...)"
    # Try without password (trust auth) or with different user
    if psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1;" > /dev/null 2>&1; then
        test_pass "Database connection successful (without user specified)"
        DB_AVAILABLE=1
        DB_USER=""  # Clear user to use default
    else
        # Try with PGPASSWORD from environment
        if [ -n "$PGPASSWORD" ] && psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1;" > /dev/null 2>&1; then
            test_pass "Database connection successful (with PGPASSWORD)"
            DB_AVAILABLE=1
        else
            test_fail "Database connection failed. Please ensure:"
            echo ""
            echo "  Database is running and accessible"
            echo "  Connection details:"
            echo "    Host: $DB_HOST"
            echo "    Port: $DB_PORT"
            echo "    Database: $DB_NAME"
            echo "    User: $DB_USER"
            echo ""
            echo "  To start with Docker:"
            echo "    docker compose up -d neurondb"
            echo ""
            echo "  Or set environment variables:"
            echo "    export DB_HOST=localhost"
            echo "    export DB_PORT=5433"
            echo "    export DB_USER=neurondb"
            echo "    export DB_NAME=neurondb"
            echo "    export DB_PASSWORD=neurondb  # or PGPASSWORD"
            echo ""
            echo "  Then run: ./scripts/neuronagent_verify.sh"
            echo ""
            echo "  Note: Most verification tests require a running database."
            exit 1
        fi
    fi
fi

# Check NeuronDB extension
test_info "Checking NeuronDB extension..."
EXT_EXISTS=$(run_psql "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" | xargs)
if [ "$EXT_EXISTS" = "t" ]; then
    test_pass "NeuronDB extension installed"
else
    test_fail "NeuronDB extension not found"
    exit 1
fi

# Check vector type
test_info "Checking vector type..."
VECTOR_TYPE=$(type_exists "vector")
if [ "$VECTOR_TYPE" = "t" ]; then
    test_pass "Vector type available"
else
    test_fail "Vector type not found"
    exit 1
fi

# Check neurondb_vector type (for hierarchical memory)
test_info "Checking neurondb_vector type..."
NEURONDB_VECTOR_TYPE=$(type_exists "neurondb_vector")
if [ "$NEURONDB_VECTOR_TYPE" = "t" ]; then
    test_pass "neurondb_vector type available"
else
    test_warn "neurondb_vector type not found (hierarchical memory may not work)"
fi

# Check schema
test_info "Checking neurondb_agent schema..."
SCHEMA_EXISTS=$(run_psql "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb_agent');" | xargs)
if [ "$SCHEMA_EXISTS" = "t" ]; then
    test_pass "neurondb_agent schema exists"
else
    test_fail "neurondb_agent schema not found"
    exit 1
fi

# Generate API key if needed
if [ -z "$NEURONAGENT_API_KEY" ]; then
    test_info "Generating API key for testing..."
    if [ -f "cmd/generate-key/main.go" ]; then
        API_KEY_OUTPUT=$(go run cmd/generate-key/main.go \
            -org "neurondb-test" \
            -user "test-user" \
            -rate 1000 \
            -roles "user,admin" \
            -db-host "$DB_HOST" \
            -db-port "$DB_PORT" \
            -db-name "$DB_NAME" \
            -db-user "$DB_USER" 2>&1)
        
        if [ $? -eq 0 ]; then
            API_KEY=$(echo "$API_KEY_OUTPUT" | grep "^Key:" | sed 's/^Key: //' | tr -d '[:space:]' || \
                     echo "$API_KEY_OUTPUT" | tail -1 | tr -d '[:space:]')
            if [ -n "$API_KEY" ] && [ ${#API_KEY} -gt 20 ]; then
                test_pass "API key generated"
                export NEURONAGENT_API_KEY="$API_KEY"
            else
                test_warn "Could not extract API key, some API tests may fail"
            fi
        else
            test_warn "API key generation failed, some API tests may fail"
        fi
    else
        test_warn "API key generator not found, set NEURONAGENT_API_KEY environment variable"
    fi
fi

# ============================================================================
# PHASE 1: Embedding Generation Integration
# ============================================================================
test_section "PHASE 1: Embedding Generation Integration"

# Test neurondb_embed
test_info "Checking neurondb_embed function..."
if [ "$(function_exists 'neurondb_embed')" = "t" ]; then
    test_pass "neurondb_embed function exists"
    
    # Test embedding generation
    test_info "Testing embedding generation..."
    if [ -n "$DB_USER" ]; then
        EMBED_RESULT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT neurondb_embed('Test embedding generation', 'all-MiniLM-L6-v2')::text;" 2>&1)
    else
        EMBED_RESULT=$(psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
            "SELECT neurondb_embed('Test embedding generation', 'all-MiniLM-L6-v2')::text;" 2>&1)
    fi
    if echo "$EMBED_RESULT" | grep -qE "^\[[0-9\.,\s]+\]$" || echo "$EMBED_RESULT" | grep -q "vector"; then
        test_pass "Embedding generation successful"
    elif echo "$EMBED_RESULT" | grep -qi "error\|not found\|not available"; then
        test_warn "Embedding generation returned error (model may need configuration)"
    else
        test_pass "Embedding function callable"
    fi
else
    test_fail "neurondb_embed function not found"
fi

# Test neurondb_embed_batch
test_info "Checking neurondb_embed_batch function..."
if [ "$(function_exists 'neurondb_embed_batch')" = "t" ]; then
    test_pass "neurondb_embed_batch function exists"
else
    test_warn "neurondb_embed_batch function not found (batch embeddings will use fallback)"
fi

# ============================================================================
# PHASE 2: Vector Operations Integration
# ============================================================================
test_section "PHASE 2: Vector Operations Integration"

# Test vector type casting
test_info "Testing vector type casting..."
if [ -n "$DB_USER" ]; then
    VECTOR_CAST=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,2,3]'::vector(3);" 2>&1)
else
    VECTOR_CAST=$(psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,2,3]'::vector(3);" 2>&1)
fi
if echo "$VECTOR_CAST" | grep -q "\[1,2,3\]" || echo "$VECTOR_CAST" | grep -q "vector"; then
    test_pass "Vector type casting works"
else
    test_fail "Vector type casting failed"
fi

# Test cosine distance operator (<=>)
test_info "Testing cosine distance operator (<=>)..."
if [ -n "$DB_USER" ]; then
    COSINE_DIST=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3) AS distance;" 2>&1)
else
    COSINE_DIST=$(psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3) AS distance;" 2>&1)
fi
if echo "$COSINE_DIST" | grep -qE "^[0-9\.]+$" || echo "$COSINE_DIST" | grep -qE "^[[:space:]]*[0-9\.]+"; then
    test_pass "Cosine distance operator (<=>) works"
else
    test_fail "Cosine distance operator failed"
fi

# Test L2 distance operator (<->)
test_info "Testing L2 distance operator (<->)..."
if [ -n "$DB_USER" ]; then
    L2_DIST=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3) AS distance;" 2>&1)
else
    L2_DIST=$(psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,0,0]'::vector(3) <-> '[0,1,0]'::vector(3) AS distance;" 2>&1)
fi
if echo "$L2_DIST" | grep -qE "^[0-9\.]+$" || echo "$L2_DIST" | grep -qE "^[[:space:]]*[0-9\.]+"; then
    test_pass "L2 distance operator (<->) works"
else
    test_fail "L2 distance operator failed"
fi

# Test inner product operator (<#>)
test_info "Testing inner product operator (<#>)..."
if [ -n "$DB_USER" ]; then
    INNER_PROD=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,0,0]'::vector(3) <#> '[1,0,0]'::vector(3) AS product;" 2>&1)
else
    INNER_PROD=$(psql -d "$DB_NAME" -h "$DB_HOST" -p "$DB_PORT" -t -c \
        "SELECT '[1,0,0]'::vector(3) <#> '[1,0,0]'::vector(3) AS product;" 2>&1)
fi
if echo "$INNER_PROD" | grep -qE "^[0-9\.]+$" || echo "$INNER_PROD" | grep -qE "^[[:space:]]*[0-9\.]+"; then
    test_pass "Inner product operator (<#>) works"
else
    test_fail "Inner product operator failed"
fi

# ============================================================================
# PHASE 3: LLM Integration
# ============================================================================
test_section "PHASE 3: LLM Integration"

# Check LLM functions
test_info "Checking LLM functions..."
LLM_GENERATE=$(function_exists 'neurondb_llm_generate')
LLM_COMPLETE=$(function_exists 'neurondb_llm_complete')
LLM_STREAM=$(function_exists 'neurondb_llm_generate_stream')

if [ "$LLM_GENERATE" = "t" ]; then
    test_pass "neurondb_llm_generate function exists"
elif [ "$LLM_COMPLETE" = "t" ]; then
    test_pass "neurondb_llm_complete function exists (fallback)"
    test_warn "neurondb_llm_generate not found, using fallback"
else
    test_warn "LLM functions not found (LLM features may not be configured)"
fi

if [ "$LLM_STREAM" = "t" ]; then
    test_pass "neurondb_llm_generate_stream function exists"
else
    test_warn "neurondb_llm_generate_stream not found (streaming will use fallback)"
fi

# ============================================================================
# PHASE 4: RAG Pipeline Integration
# ============================================================================
test_section "PHASE 4: RAG Pipeline Integration"

# Check RAG functions
test_info "Checking RAG functions..."
RAG_CHUNK=$(function_exists 'neurondb_chunk_text')
RAG_RERANK=$(function_exists 'neurondb_rerank_results')
RAG_GENERATE=$(function_exists 'neurondb_generate_answer')

if [ "$RAG_CHUNK" = "t" ]; then
    test_pass "neurondb_chunk_text function exists"
else
    test_warn "neurondb_chunk_text function not found"
fi

if [ "$RAG_RERANK" = "t" ]; then
    test_pass "neurondb_rerank_results function exists"
else
    test_warn "neurondb_rerank_results function not found"
fi

if [ "$RAG_GENERATE" = "t" ]; then
    test_pass "neurondb_generate_answer function exists"
else
    test_warn "neurondb_generate_answer function not found"
fi

# ============================================================================
# PHASE 5: Hybrid Search Integration
# ============================================================================
test_section "PHASE 5: Hybrid Search Integration"

# Check hybrid search functions
test_info "Checking hybrid search functions..."
HYBRID_SEARCH=$(function_exists 'neurondb_hybrid_search')
RRF=$(function_exists 'neurondb_reciprocal_rank_fusion')
SEMANTIC_KEYWORD=$(function_exists 'neurondb_semantic_keyword_search')
MULTI_VECTOR=$(function_exists 'neurondb_multi_vector_search')

if [ "$HYBRID_SEARCH" = "t" ]; then
    test_pass "neurondb_hybrid_search function exists"
else
    test_warn "neurondb_hybrid_search function not found"
fi

if [ "$RRF" = "t" ]; then
    test_pass "neurondb_reciprocal_rank_fusion function exists"
else
    test_warn "neurondb_reciprocal_rank_fusion function not found"
fi

if [ "$SEMANTIC_KEYWORD" = "t" ]; then
    test_pass "neurondb_semantic_keyword_search function exists"
else
    test_warn "neurondb_semantic_keyword_search function not found"
fi

if [ "$MULTI_VECTOR" = "t" ]; then
    test_pass "neurondb_multi_vector_search function exists"
else
    test_warn "neurondb_multi_vector_search function not found"
fi

# ============================================================================
# PHASE 6: Reranking Integration
# ============================================================================
test_section "PHASE 6: Reranking Integration"

# Check reranking functions
test_info "Checking reranking functions..."
RERANK_CROSS=$(function_exists 'neurondb_rerank_cross_encoder')
RERANK_LLM=$(function_exists 'neurondb_rerank_llm')
RERANK_COLBERT=$(function_exists 'neurondb_rerank_colbert')
RERANK_ENSEMBLE=$(function_exists 'neurondb_rerank_ensemble')

if [ "$RERANK_CROSS" = "t" ]; then
    test_pass "neurondb_rerank_cross_encoder function exists"
else
    test_warn "neurondb_rerank_cross_encoder function not found"
fi

if [ "$RERANK_LLM" = "t" ]; then
    test_pass "neurondb_rerank_llm function exists"
else
    test_warn "neurondb_rerank_llm function not found"
fi

if [ "$RERANK_COLBERT" = "t" ]; then
    test_pass "neurondb_rerank_colbert function exists"
else
    test_warn "neurondb_rerank_colbert function not found"
fi

if [ "$RERANK_ENSEMBLE" = "t" ]; then
    test_pass "neurondb_rerank_ensemble function exists"
else
    test_warn "neurondb_rerank_ensemble function not found"
fi

# ============================================================================
# PHASE 7: ML Operations Integration
# ============================================================================
test_section "PHASE 7: ML Operations Integration"

# Check ML functions and tables
test_info "Checking ML functions..."
ML_TRAIN=$(function_exists 'neurondb.train')
ML_PREDICT=$(function_exists 'neurondb.predict')
ML_EVALUATE=$(function_exists 'neurondb.evaluate')

if [ "$ML_TRAIN" = "t" ]; then
    test_pass "neurondb.train function exists"
else
    test_warn "neurondb.train function not found"
fi

if [ "$ML_PREDICT" = "t" ]; then
    test_pass "neurondb.predict function exists"
else
    test_warn "neurondb.predict function not found"
fi

if [ "$ML_EVALUATE" = "t" ]; then
    test_pass "neurondb.evaluate function exists"
else
    test_warn "neurondb.evaluate function not found"
fi

# Check ML models table
test_info "Checking ML models table..."
ML_TABLE=$(table_exists "neurondb" "ml_models")
if [ "$ML_TABLE" = "t" ]; then
    test_pass "neurondb.ml_models table exists"
else
    test_warn "neurondb.ml_models table not found"
fi

# ============================================================================
# PHASE 8: Analytics Integration
# ============================================================================
test_section "PHASE 8: Analytics Integration"

# Check analytics functions
test_info "Checking analytics functions..."
ANALYTICS_CLUSTER=$(function_exists 'neurondb_cluster')
ANALYTICS_OUTLIER=$(function_exists 'neurondb_detect_outliers')
ANALYTICS_REDUCE=$(function_exists 'neurondb_reduce_dimensionality')
ANALYTICS_ANALYZE=$(function_exists 'neurondb_analyze_data')

if [ "$ANALYTICS_CLUSTER" = "t" ]; then
    test_pass "neurondb_cluster function exists"
else
    test_warn "neurondb_cluster function not found"
fi

if [ "$ANALYTICS_OUTLIER" = "t" ]; then
    test_pass "neurondb_detect_outliers function exists"
else
    test_warn "neurondb_detect_outliers function not found"
fi

if [ "$ANALYTICS_REDUCE" = "t" ]; then
    test_pass "neurondb_reduce_dimensionality function exists"
else
    test_warn "neurondb_reduce_dimensionality function not found"
fi

if [ "$ANALYTICS_ANALYZE" = "t" ]; then
    test_pass "neurondb_analyze_data function exists"
else
    test_warn "neurondb_analyze_data function not found"
fi

# ============================================================================
# PHASE 9: Memory Management Integration
# ============================================================================
test_section "PHASE 9: Memory Management Integration"

# Check memory tables
test_info "Checking memory tables..."
MEMORY_CHUNKS=$(table_exists "neurondb_agent" "memory_chunks")
MEMORY_STM=$(table_exists "neurondb_agent" "memory_stm")
MEMORY_MTM=$(table_exists "neurondb_agent" "memory_mtm")
MEMORY_LPM=$(table_exists "neurondb_agent" "memory_lpm")

if [ "$MEMORY_CHUNKS" = "t" ]; then
    test_pass "memory_chunks table exists"
    
    # Check embedding column type
    EMBED_COL_TYPE=$(run_psql "SELECT data_type FROM information_schema.columns WHERE table_schema = 'neurondb_agent' AND table_name = 'memory_chunks' AND column_name = 'embedding';" | xargs)
    if echo "$EMBED_COL_TYPE" | grep -qi "vector\|USER-DEFINED"; then
        test_pass "memory_chunks.embedding column is vector type"
    else
        test_fail "memory_chunks.embedding column type incorrect: $EMBED_COL_TYPE"
    fi
else
    test_fail "memory_chunks table not found"
fi

if [ "$MEMORY_STM" = "t" ]; then
    test_pass "memory_stm table exists"
else
    test_warn "memory_stm table not found (hierarchical memory not available)"
fi

if [ "$MEMORY_MTM" = "t" ]; then
    test_pass "memory_mtm table exists"
else
    test_warn "memory_mtm table not found (hierarchical memory not available)"
fi

if [ "$MEMORY_LPM" = "t" ]; then
    test_pass "memory_lpm table exists"
else
    test_warn "memory_lpm table not found (hierarchical memory not available)"
fi

# Check HNSW indexes
test_info "Checking HNSW indexes on memory tables..."
HNSW_CHUNKS=$(run_psql "SELECT indexname FROM pg_indexes WHERE schemaname = 'neurondb_agent' AND tablename = 'memory_chunks' AND indexname LIKE '%embedding%' LIMIT 1;" | xargs)
if [ -n "$HNSW_CHUNKS" ]; then
    test_pass "HNSW index exists on memory_chunks.embedding: $HNSW_CHUNKS"
else
    test_warn "HNSW index not found on memory_chunks.embedding (may be created automatically)"
fi

# ============================================================================
# PHASE 10: Database Schema Verification
# ============================================================================
test_section "PHASE 10: Database Schema Verification"

# Check all required tables
REQUIRED_TABLES=("agents" "sessions" "messages" "memory_chunks" "tools" "jobs" "api_keys" "schema_migrations")
for table in "${REQUIRED_TABLES[@]}"; do
    if [ "$(table_exists 'neurondb_agent' "$table")" = "t" ]; then
        test_pass "Table $table exists"
    else
        test_fail "Table $table missing"
    fi
done

# Check foreign key constraints
FK_COUNT=$(run_psql "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_schema = 'neurondb_agent' AND constraint_type = 'FOREIGN KEY';" | xargs)
if [ "$FK_COUNT" -gt 0 ]; then
    test_pass "Foreign key constraints exist ($FK_COUNT found)"
else
    test_fail "No foreign key constraints found"
fi

# Check indexes
INDEX_COUNT=$(run_psql "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'neurondb_agent';" | xargs)
if [ "$INDEX_COUNT" -gt 0 ]; then
    test_pass "Indexes exist ($INDEX_COUNT found)"
else
    test_warn "No indexes found"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
test_section "VERIFICATION SUMMARY"

echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo -e "${YELLOW}Warned: $TESTS_WARNED${NC}"
echo ""

# Calculate success rate
if [ $TESTS_TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(( TESTS_PASSED * 100 / TESTS_TOTAL ))
    echo "Success Rate: ${SUCCESS_RATE}%"
fi

if [ $TESTS_FAILED -eq 0 ]; then
    if [ $TESTS_WARNED -eq 0 ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
        echo ""
        echo "NeuronAgent-NeuronDB integration is fully functional!"
        exit 0
    else
        echo -e "${GREEN}✓ ALL CRITICAL TESTS PASSED!${NC}"
        echo -e "${YELLOW}⚠️  Some optional features have warnings (see above)${NC}"
        exit 0
    fi
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failures above."
    exit 1
fi


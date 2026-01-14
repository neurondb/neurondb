#!/bin/bash
#-------------------------------------------------------------------------
#
# neuronmcp-diagnose-embeddings.sh
#    NeuronMCP Embedding Diagnostics
#
# Diagnoses embedding configuration issues and tests embedding generation.
# Provides detailed diagnostics for embedding model configuration, connection
# testing, and embedding generation verification.
#
# Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
#
# IDENTIFICATION
#    NeuronMCP/scripts/neuronmcp-diagnose-embeddings.sh
#
#-------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME=$(basename "$0")

# Version
VERSION="2.0.0"

# Default values
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Database configuration
DB_HOST="${NEURONDB_HOST:-${DB_HOST:-localhost}}"
DB_PORT="${NEURONDB_PORT:-${DB_PORT:-5432}}"
DB_NAME="${NEURONDB_DATABASE:-${DB_NAME:-neurondb}}"
DB_USER="${NEURONDB_USER:-${DB_USER:-postgres}}"

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
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-V|--version)
			echo "neuronmcp_diagnose_embeddings.sh version $VERSION"
			exit 0
			;;
		-h|--help)
			cat << EOF
NeuronMCP Embedding Diagnostics

Usage:
    $SCRIPT_NAME [OPTIONS]

Description:
    Diagnoses embedding configuration issues and tests embedding generation.
    Checks API key configuration, tests embedding generation, and identifies
    why embeddings might be returning zeros.

Options:
    -D, --database DB    Database name (default: neurondb)
    -U, --user USER      Database user (default: postgres)
    -H, --host HOST      Database host (default: localhost)
    -p, --port PORT      Database port (default: 5432)
    -v, --verbose        Enable verbose output
    -V, --version        Show version information
    -h, --help           Show this help message

Environment Variables:
    NEURONDB_HOST / DB_HOST       Database host (default: localhost)
    NEURONDB_PORT / DB_PORT       Database port (default: 5432)
    NEURONDB_DATABASE / DB_NAME   Database name (default: neurondb)
    NEURONDB_USER / DB_USER       Database user (default: postgres)
    NEURONDB_PASSWORD / DB_PASSWORD   Database password

Examples:
    # Basic diagnostic
    $SCRIPT_NAME

    # Custom database
    $SCRIPT_NAME -D mydb -U myuser

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

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Function to check PostgreSQL connection
check_postgres() {
    print_section "Checking PostgreSQL Connection"
    
    if ! command -v psql &> /dev/null; then
        print_error "psql not found. Please install PostgreSQL client tools."
        return 1
    fi
    
    if [ -n "$NEURONDB_PASSWORD" ] || [ -n "$DB_PASSWORD" ]; then
        export PGPASSWORD="${NEURONDB_PASSWORD:-$DB_PASSWORD}"
    fi
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
        print_success "Connected to PostgreSQL: $DB_HOST:$DB_PORT/$DB_NAME"
        return 0
    else
        print_error "Cannot connect to PostgreSQL"
        print_info "Connection details: $DB_HOST:$DB_PORT/$DB_NAME as $DB_USER"
        return 1
    fi
}

# Function to check NeuronDB extension
check_neurondb_extension() {
    print_section "Checking NeuronDB Extension"
    
    local has_extension=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_extension WHERE extname = 'neurondb'" 2>/dev/null || echo "")
    
    if [ "$has_extension" = "1" ]; then
        print_success "NeuronDB extension is installed"
        
        # Get extension version
        local version=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SELECT extversion FROM pg_extension WHERE extname = 'neurondb'" 2>/dev/null || echo "unknown")
        print_info "  Version: $version"
        return 0
    else
        print_error "NeuronDB extension is not installed"
        print_info "Install with: CREATE EXTENSION neurondb;"
        return 1
    fi
}

# Function to check embedding configuration
check_embedding_config() {
    print_section "Checking Embedding Configuration"
    
    # Check API key
    local api_key=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_api_key', true)" 2>/dev/null || echo "")
    
    if [ -n "$api_key" ] && [ "$api_key" != "" ]; then
        # Mask the key for display
        local masked_key="${api_key:0:8}...${api_key: -4}"
        print_success "API key is configured: $masked_key"
    else
        print_error "API key is NOT configured"
        print_info "Set with: ALTER SYSTEM SET neurondb.llm_api_key = 'your-key';"
        print_info "Then reload: SELECT pg_reload_conf();"
    fi
    
    # Check provider
    local provider=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_provider', true)" 2>/dev/null || echo "huggingface")
    print_info "  Provider: ${provider:-huggingface (default)}"
    
    # Check endpoint
    local endpoint=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_endpoint', true)" 2>/dev/null || echo "https://api-inference.huggingface.co")
    print_info "  Endpoint: ${endpoint:-https://api-inference.huggingface.co (default)}"
    
    # Check model
    local model=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_model', true)" 2>/dev/null || echo "sentence-transformers/all-MiniLM-L6-v2")
    print_info "  Model: ${model:-sentence-transformers/all-MiniLM-L6-v2 (default)}"
    
    # Check timeout
    local timeout=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_timeout_ms', true)" 2>/dev/null || echo "15000")
    print_info "  Timeout: ${timeout:-15000 (default)} ms"
    
    # Check if config file has the setting
    print_info ""
    print_info "Checking postgresql.auto.conf..."
    local pgdata=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SHOW data_directory" 2>/dev/null || echo "")
    
    if [ -n "$pgdata" ] && [ -f "$pgdata/postgresql.auto.conf" ]; then
        if grep -q "neurondb.llm_api_key" "$pgdata/postgresql.auto.conf" 2>/dev/null; then
            print_success "API key found in postgresql.auto.conf"
            if [ "$VERBOSE" = true ]; then
                grep "neurondb.llm_api_key" "$pgdata/postgresql.auto.conf" | sed 's/=.*/=***/' | head -1
            fi
        else
            print_warning "API key NOT found in postgresql.auto.conf"
        fi
    else
        print_warning "Cannot access postgresql.auto.conf (may need to check manually)"
    fi
    
    return 0
}

# Function to test embedding generation
test_embedding_generation() {
    print_section "Testing Embedding Generation"
    
    print_info "Generating test embedding..."
    
    # Test single embedding
    local result=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT embed_text('test embedding')::text" 2>&1)
    
    if [ $? -ne 0 ]; then
        print_error "Failed to generate embedding"
        print_info "Error: $result"
        return 1
    fi
    
    # Check if result is all zeros
    local is_zero=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT CASE WHEN embed_text('test') = (SELECT array_agg(0.0::float)::vector(384) FROM generate_series(1, 384)) THEN 'true' ELSE 'false' END" 2>/dev/null || echo "unknown")
    
    if [ "$is_zero" = "true" ]; then
        print_error "Embeddings are returning ZEROS"
        print_info "This indicates the embedding API is not working properly."
        print_info ""
        print_info "Possible causes:"
        print_info "  1. API key is missing or invalid"
        print_info "  2. API endpoint is unreachable"
        print_info "  3. Model is not available"
        print_info "  4. Network connectivity issues"
        print_info "  5. Rate limiting or quota exceeded"
        return 1
    else
        print_success "Single embeddings are working correctly!"
        
        # Get embedding dimension
        local dim=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SELECT array_length(embed_text('test')::float[], 1)" 2>/dev/null || echo "unknown")
        print_info "  Embedding dimension: $dim"
        
        # Get a sample of the embedding values
        if [ "$VERBOSE" = true ]; then
            print_info "  Sample embedding values (first 5):"
            psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
                "SELECT array_to_string((embed_text('test')::float[])[1:5], ', ')" 2>/dev/null || echo "  (unable to display)"
        fi
    fi
    
    # Test batch embeddings
    print_info ""
    print_info "Testing batch embedding generation..."
    local batch_result=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT json_agg(embedding::text) FROM unnest(embed_text_batch(ARRAY['test1', 'test2'])) AS embedding" 2>&1)
    
    if [ $? -ne 0 ]; then
        print_error "Failed to generate batch embeddings"
        print_info "Error: $batch_result"
        return 1
    fi
    
    # Check if batch embeddings are zeros
    local batch_is_zero=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT CASE WHEN embed_text_batch(ARRAY['test'])[1] = (SELECT array_agg(0.0::float)::vector(384) FROM generate_series(1, 384)) THEN 'true' ELSE 'false' END" 2>/dev/null || echo "unknown")
    
    if [ "$batch_is_zero" = "true" ]; then
        print_error "Batch embeddings are returning ZEROS"
        return 1
    else
        print_success "Batch embeddings are working correctly!"
        local batch_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SELECT array_length(embed_text_batch(ARRAY['test1', 'test2']), 1)" 2>/dev/null || echo "unknown")
        print_info "  Batch size: $batch_count"
    fi
    
    return 0
}

# Function to check PostgreSQL logs for errors
check_logs() {
    print_section "Checking for Embedding Errors in Logs"
    
    local log_file=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SHOW log_directory" 2>/dev/null || echo "")
    local log_filename=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SHOW log_filename" 2>/dev/null || echo "postgresql-%Y-%m-%d.log")
    
    if [ -n "$log_file" ]; then
        local pgdata=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
            "SHOW data_directory" 2>/dev/null || echo "")
        
        if [ -n "$pgdata" ] && [ -d "$pgdata/$log_file" ]; then
            local latest_log=$(ls -t "$pgdata/$log_file"/postgresql-*.log 2>/dev/null | head -1)
            if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
                print_info "Checking log file: $latest_log"
                
                # Look for embedding-related errors
                local errors=$(grep -i "embed\|llm\|api.*key\|huggingface" "$latest_log" 2>/dev/null | tail -10 || echo "")
                
                if [ -n "$errors" ]; then
                    print_warning "Found potential errors in logs:"
                    echo "$errors" | while read -r line; do
                        echo "  $line"
                    done
                else
                    print_info "No embedding-related errors found in recent logs"
                fi
            else
                print_warning "Cannot access log files (may need manual check)"
            fi
        else
            print_warning "Cannot access log directory (may need manual check)"
        fi
    else
        print_warning "Logging may not be configured"
    fi
}

# Function to provide recommendations
provide_recommendations() {
    print_section "Recommendations"
    
    local api_key=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT current_setting('neurondb.llm_api_key', true)" 2>/dev/null || echo "")
    
    if [ -z "$api_key" ] || [ "$api_key" = "" ]; then
        print_info "1. Set the API key:"
        echo ""
        echo "   ALTER SYSTEM SET neurondb.llm_api_key = 'hf_your_token_here';"
        echo "   SELECT pg_reload_conf();"
        echo ""
        print_info "2. Get a Hugging Face token from:"
        echo "   https://huggingface.co/settings/tokens"
        echo ""
    else
        print_info "API key is set. If embeddings are still zeros:"
        echo ""
        print_info "1. Verify the API key is valid:"
        echo "   - Check it's not expired"
        echo "   - Verify it has the correct permissions"
        echo ""
        print_info "2. Test API connectivity:"
        echo "   - Check network connectivity to the endpoint"
        echo "   - Verify firewall rules allow outbound connections"
        echo ""
        print_info "3. Check for rate limiting:"
        echo "   - Free Hugging Face API has rate limits"
        echo "   - Consider upgrading to Pro account"
        echo ""
        print_info "4. Verify model availability:"
        echo "   - Check if the model exists on Hugging Face"
        echo "   - Some models may require authentication"
        echo ""
    fi
    
    print_info "5. Check PostgreSQL logs for detailed error messages:"
    echo "   - Look for 'embed', 'llm', or 'api' in error logs"
    echo "   - Enable DEBUG1 log level for more details"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo "NeuronMCP Embedding Diagnostics"
    echo "=========================================="
    echo ""
    
    # Check prerequisites
    if ! check_postgres; then
        exit 1
    fi
    
    if ! check_neurondb_extension; then
        exit 1
    fi
    
    # Check configuration
    check_embedding_config
    
    # Test embedding generation
    if ! test_embedding_generation; then
        check_logs
        provide_recommendations
        exit 1
    fi
    
    # Success
    print_section "Diagnostic Complete"
    print_success "Embedding configuration is working correctly!"
    echo ""
}

# Run main function
main "$@"


#!/bin/bash
#
# NeuronDB Storage Management Script
# Self-sufficient script for cloud storage validation and management
#
# Usage:
#   ./neurondb-storage.sh COMMAND [OPTIONS]
#
# Commands:
#   validate     Validate cloud storage backup configurations (S3/GCS/Azure)
#   test         Test cloud storage connectivity and operations
#
# This script is completely self-sufficient with no external dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors (inline - no external dependency)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false
DRY_RUN=false
CHART_DIR="${CHART_DIR:-helm/neurondb}"
RELEASE_NAME="${RELEASE_NAME:-neurondb-validation}"
NAMESPACE="${NAMESPACE:-neurondb-test}"

# Counters
PASSED=0
FAILED=0
WARNINGS=0


log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
    ((PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
    ((WARNINGS++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    ((FAILED++))
}

log_debug() {
    [[ "$VERBOSE" == "true" ]] && echo -e "${MAGENTA}[DEBUG]${NC} $*"
}

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Storage Management${NC}                            ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    local title="$1"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}${title}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_test() {
    local status="$1"
    local message="$2"
    
    case "$status" in
        PASS|pass)
            echo -e "${GREEN}✓${NC} $message"
            ((PASSED++))
            ;;
        FAIL|fail)
            echo -e "${RED}✗${NC} $message"
            ((FAILED++))
            ;;
        SKIP|skip)
            echo -e "${YELLOW}⊘${NC} $message"
            ((WARNINGS++))
            ;;
        *)
            echo -e "${CYAN}ℹ${NC} $message"
            ;;
    esac
}


show_help() {
    cat << EOF
${BOLD}NeuronDB Storage Management${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    validate     Validate cloud storage backup configurations (S3/GCS/Azure)
    test         Test cloud storage connectivity and operations

${BOLD}Validate Command:${NC}
    ${SCRIPT_NAME} validate [OPTIONS]
    
    Options:
        --chart-dir DIR       Chart directory (default: helm/neurondb)
        --release-name NAME   Release name for validation (default: neurondb-validation)
        --namespace NAME      Namespace for validation (default: neurondb-test)

${BOLD}Test Command:${NC}
    ${SCRIPT_NAME} test [OPTIONS]
    
    Options:
        --provider PROVIDER   Storage provider: s3, gcs, azure (default: all)
        --bucket BUCKET       Bucket/container name for testing
        --region REGION       Region for S3 testing

${BOLD}Global Options:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -V, --version           Show version information
    --dry-run               Preview changes without applying

${BOLD}Examples:${NC}
    # Validate cloud storage configurations
    ${SCRIPT_NAME} validate

    # Test S3 connectivity
    ${SCRIPT_NAME} test --provider s3 --bucket my-bucket --region us-east-1

    # Validate with custom chart directory
    ${SCRIPT_NAME} validate --chart-dir ./custom-helm/neurondb

EOF
}


validate_command() {
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --chart-dir)
                CHART_DIR="$2"
                shift 2
                ;;
            --release-name)
                RELEASE_NAME="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option for validate command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed"
        exit 1
    fi
    
    CHART_DIR="$PROJECT_ROOT/$CHART_DIR"
    
    if [ ! -d "$CHART_DIR" ]; then
        log_error "Chart directory not found: $CHART_DIR"
        exit 1
    fi
    
    print_header
    log_info "Chart Directory: $CHART_DIR"
    log_info "Release Name: $RELEASE_NAME"
    log_info "Namespace: $NAMESPACE"
    echo ""
    
    # Validate S3 config
    validate_s3_config
    
    # Validate GCS config
    validate_gcs_config
    
    # Validate Azure config
    validate_azure_config
    
    # Validate restore config
    validate_restore_config
    
    # Validate backup secret
    validate_backup_secret
    
    # Validate backup retention
    validate_backup_retention
    
    # Summary
    print_section "Validation Summary"
    echo -e "Total Checks: $((PASSED + FAILED + WARNINGS))"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    echo ""
    
    if [ "$FAILED" -gt 0 ]; then
        log_error "Validation FAILED with $FAILED error(s)"
        exit 1
    elif [ "$WARNINGS" -gt 0 ]; then
        log_warning "Validation completed with $WARNINGS warning(s)"
        exit 0
    else
        log_success "All validations PASSED"
        exit 0
    fi
}

validate_s3_config() {
    print_section "Validating S3 Backup Configuration"
    
    local rendered
    rendered=$(helm template "$RELEASE_NAME" "$CHART_DIR" 2>/dev/null || echo "")
    
    if [ -z "$rendered" ]; then
        print_test "FAIL" "Failed to render Helm templates"
        return
    fi
    
    if echo "$rendered" | grep -q "aws s3 cp"; then
        print_test "PASS" "S3 backup script found in templates"
    else
        print_test "FAIL" "S3 backup script not found"
    fi
    
    if echo "$rendered" | grep -q "AWS_ACCESS_KEY_ID\|AWS_DEFAULT_REGION"; then
        print_test "PASS" "AWS credential environment variables configured"
    else
        print_test "SKIP" "AWS credentials not configured (may use IAM roles)"
    fi
    
    if echo "$rendered" | grep -q "backup-credentials"; then
        print_test "PASS" "Backup credentials secret template found"
    else
        print_test "SKIP" "Backup credentials secret may not be required"
    fi
}

validate_gcs_config() {
    print_section "Validating GCS Backup Configuration"
    
    local rendered
    rendered=$(helm template "$RELEASE_NAME" "$CHART_DIR" 2>/dev/null || echo "")
    
    if [ -z "$rendered" ]; then
        print_test "FAIL" "Failed to render Helm templates"
        return
    fi
    
    if echo "$rendered" | grep -q "gsutil cp"; then
        print_test "PASS" "GCS backup script found in templates"
    else
        print_test "SKIP" "GCS backup script not found (may not be configured)"
    fi
    
    if echo "$rendered" | grep -q "GOOGLE_APPLICATION_CREDENTIALS"; then
        print_test "PASS" "GCS credentials environment variable configured"
    else
        print_test "SKIP" "GCS credentials not configured (may use workload identity)"
    fi
}

validate_azure_config() {
    print_section "Validating Azure Blob Storage Configuration"
    
    local rendered
    rendered=$(helm template "$RELEASE_NAME" "$CHART_DIR" 2>/dev/null || echo "")
    
    if [ -z "$rendered" ]; then
        print_test "FAIL" "Failed to render Helm templates"
        return
    fi
    
    if echo "$rendered" | grep -q "az storage blob"; then
        print_test "PASS" "Azure backup script found in templates"
    else
        print_test "SKIP" "Azure backup script not found (may not be configured)"
    fi
    
    if echo "$rendered" | grep -q "AZURE_CLIENT_ID\|AZURE_CLIENT_SECRET\|AZURE_TENANT_ID"; then
        print_test "PASS" "Azure credential environment variables configured"
    else
        print_test "SKIP" "Azure credentials not configured (may use managed identity)"
    fi
}

validate_restore_config() {
    print_section "Validating Restore Configuration"
    
    local rendered
    rendered=$(helm template "$RELEASE_NAME" "$CHART_DIR" \
        --set backup.restore.enabled=true \
        --set backup.restore.backupFile="test-backup.dump" 2>/dev/null || echo "")
    
    if echo "$rendered" | grep -qi "restore"; then
        print_test "PASS" "Restore job template found"
    else
        print_test "SKIP" "Restore job template not found (may not be configured)"
    fi
}

validate_backup_secret() {
    print_section "Validating Backup Credentials Secret"
    
    local rendered
    rendered=$(helm template "$RELEASE_NAME" "$CHART_DIR" \
        --set backup.s3.enabled=true \
        --set backup.s3.bucket="test-bucket" \
        --set backup.s3.region="us-east-1" 2>/dev/null || echo "")
    
    if echo "$rendered" | grep -q "backup-credentials"; then
        print_test "PASS" "Backup credentials secret template found"
    else
        print_test "SKIP" "Backup credentials secret may not be required"
    fi
}

validate_backup_retention() {
    print_section "Validating Backup Retention Policy"
    
    local rendered
    rendered=$(helm template "$RELEASE_NAME" "$CHART_DIR" \
        --set backup.enabled=true \
        --set backup.s3.enabled=true \
        --set backup.s3.bucket="test" \
        --set backup.s3.region="us-east-1" \
        --set backup.retention=30 2>/dev/null || echo "")
    
    if echo "$rendered" | grep -q "retention\|older than.*days"; then
        print_test "PASS" "Backup retention policy found in backup script"
    else
        print_test "SKIP" "Backup retention policy may not be implemented"
    fi
}


test_command() {
    local provider="all"
    local bucket=""
    local region=""
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --provider)
                provider="$2"
                shift 2
                ;;
            --bucket)
                bucket="$2"
                shift 2
                ;;
            --region)
                region="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option for test command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_header
    log_info "Testing cloud storage connectivity..."
    
    case "$provider" in
        s3|S3)
            test_s3 "$bucket" "$region"
            ;;
        gcs|GCS)
            test_gcs "$bucket"
            ;;
        azure|Azure)
            test_azure "$bucket"
            ;;
        all)
            log_info "Testing all providers..."
            test_s3 "$bucket" "$region"
            test_gcs "$bucket"
            test_azure "$bucket"
            ;;
        *)
            log_error "Invalid provider: $provider (must be s3, gcs, azure, or all)"
            exit 1
            ;;
    esac
}

test_s3() {
    local bucket="$1"
    local region="$2"
    
    print_section "Testing S3 Connectivity"
    
    if ! command -v aws &> /dev/null; then
        print_test "SKIP" "AWS CLI not installed"
        return
    fi
    
    if [ -z "$bucket" ]; then
        print_test "SKIP" "Bucket not specified (use --bucket)"
        return
    fi
    
    if aws s3 ls "s3://$bucket" --region "${region:-us-east-1}" &>/dev/null; then
        print_test "PASS" "S3 bucket accessible: $bucket"
    else
        print_test "FAIL" "S3 bucket not accessible: $bucket"
    fi
}

test_gcs() {
    local bucket="$1"
    
    print_section "Testing GCS Connectivity"
    
    if ! command -v gsutil &> /dev/null; then
        print_test "SKIP" "gsutil not installed"
        return
    fi
    
    if [ -z "$bucket" ]; then
        print_test "SKIP" "Bucket not specified (use --bucket)"
        return
    fi
    
    if gsutil ls "gs://$bucket" &>/dev/null; then
        print_test "PASS" "GCS bucket accessible: $bucket"
    else
        print_test "FAIL" "GCS bucket not accessible: $bucket"
    fi
}

test_azure() {
    local container="$1"
    
    print_section "Testing Azure Blob Storage Connectivity"
    
    if ! command -v az &> /dev/null; then
        print_test "SKIP" "Azure CLI not installed"
        return
    fi
    
    if [ -z "$container" ]; then
        print_test "SKIP" "Container not specified (use --bucket)"
        return
    fi
    
    if az storage blob list --container-name "$container" &>/dev/null; then
        print_test "PASS" "Azure container accessible: $container"
    else
        print_test "FAIL" "Azure container not accessible: $container"
    fi
}


parse_arguments() {
    # Check for help/version flags before command
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -V|--version)
                echo "${SCRIPT_NAME} version 2.0.0"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -V|--version)
                echo "${SCRIPT_NAME} version 2.0.0"
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
}


main() {
    parse_arguments "$@"
    
    case "$COMMAND" in
        validate)
            validate_command "$@"
            ;;
        test)
            test_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"






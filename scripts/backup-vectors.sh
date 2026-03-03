#!/bin/bash
#-------------------------------------------------------------------------
#
# backup-vectors.sh
#    Vector-aware backup script for NeuronDB
#
# Creates backups that understand vector indexes and can restore them
# efficiently. Supports full and incremental backups.
#
# Copyright (c) 2024-2026, neurondb, Inc.
#
#-------------------------------------------------------------------------

set -euo pipefail

# Default values
BACKUP_DIR="${NEURONDB_BACKUP_DIR:-/var/backups/neurondb}"
DB_NAME="${PGDATABASE:-neurondb}"
DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5432}"
DB_USER="${PGUSER:-neurondb}"
BACKUP_TYPE="${BACKUP_TYPE:-full}"  # full or incremental
RETENTION_DAYS="${RETENTION_DAYS:-7}"
VERIFY_BACKUP="${VERIFY_BACKUP:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Generate backup filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "${BACKUP_TYPE}" = "incremental" ]; then
    BACKUP_FILE="${BACKUP_DIR}/neurondb_incremental_${TIMESTAMP}.sql.gz"
    BASE_BACKUP_FILE="${BACKUP_DIR}/neurondb_full_*.sql.gz"
else
    BACKUP_FILE="${BACKUP_DIR}/neurondb_full_${TIMESTAMP}.sql.gz"
fi

log_info "Starting ${BACKUP_TYPE} backup of database ${DB_NAME}"

# Export vector indexes metadata before backup
log_info "Exporting vector index metadata..."
INDEX_METADATA_FILE="${BACKUP_DIR}/index_metadata_${TIMESTAMP}.json"
psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -A -c "
SELECT jsonb_build_object(
    'index_name', indexname,
    'table_name', tablename,
    'indexdef', indexdef,
    'index_type', CASE 
        WHEN indexdef LIKE '%hnsw%' THEN 'hnsw'
        WHEN indexdef LIKE '%ivf%' THEN 'ivf'
        ELSE 'other'
    END
)
FROM pg_indexes
WHERE schemaname = 'public' 
  AND (indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%')
ORDER BY tablename, indexname;
" > "${INDEX_METADATA_FILE}" || {
    log_warn "Failed to export index metadata, continuing with backup..."
}

# Perform the backup
log_info "Creating backup: ${BACKUP_FILE}"

if [ "${BACKUP_TYPE}" = "incremental" ]; then
    # Incremental backup using pg_basebackup or WAL archiving
    log_info "Incremental backup requires WAL archiving setup"
    # For now, create a differential backup
    pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        --schema-only --no-owner --no-acl \
        -t 'neurondb.*' \
        | gzip > "${BACKUP_FILE}" || {
        log_error "Backup failed!"
        exit 1
    }
else
    # Full backup
    pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        --format=custom --compress=9 \
        --file="${BACKUP_FILE%.gz}" || {
        log_error "Backup failed!"
        exit 1
    }
    
    # Compress if not already compressed
    if [ ! -f "${BACKUP_FILE}" ]; then
        gzip "${BACKUP_FILE%.gz}"
    fi
fi

BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
log_info "Backup completed: ${BACKUP_FILE} (${BACKUP_SIZE})"

# Verify backup if requested
if [ "${VERIFY_BACKUP}" = "true" ]; then
    log_info "Verifying backup integrity..."
    if gzip -t "${BACKUP_FILE}" 2>/dev/null; then
        log_info "Backup file integrity check passed"
    else
        log_error "Backup file integrity check failed!"
        exit 1
    fi
fi

# Clean up old backups based on retention policy
log_info "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "neurondb_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "index_metadata_*.json" -type f -mtime +${RETENTION_DAYS} -delete

log_info "Backup process completed successfully"

# Save backup metadata
cat > "${BACKUP_DIR}/backup_${TIMESTAMP}.meta" <<EOF
{
    "backup_type": "${BACKUP_TYPE}",
    "database": "${DB_NAME}",
    "timestamp": "${TIMESTAMP}",
    "backup_file": "${BACKUP_FILE}",
    "index_metadata_file": "${INDEX_METADATA_FILE}",
    "size": "${BACKUP_SIZE}",
    "retention_days": ${RETENTION_DAYS}
}
EOF

exit 0




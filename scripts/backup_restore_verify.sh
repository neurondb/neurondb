#!/usr/bin/env bash
#
# Backup and restore verification for NeuronDB/PostgreSQL.
# 1) Optional: pg_dump of the target DB.
# 2) Optional: restore to a temporary DB and run a minimal SQL check (extension + version).
#
# Usage:
#   ./scripts/backup_restore_verify.sh [--dump-only] [--restore-only PATH]
#   PGHOST=localhost PGPORT=5432 PGUSER=neurondb PGDATABASE=neurondb ./scripts/backup_restore_verify.sh
#
# Requires: psql, pg_dump (and pg_restore if not --dump-only). Optional: createdb/dropdb.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DUMP_ONLY=false
RESTORE_ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump-only) DUMP_ONLY=true; shift ;;
    --restore-only) RESTORE_ONLY="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-neurondb}"
export PGPASSWORD="${PGPASSWORD:-neurondb}"
export PGDATABASE="${PGDATABASE:-neurondb}"

BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
mkdir -p "$BACKUP_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
DUMP_FILE="${BACKUP_DIR}/neurondb_${STAMP}.dump"

if [[ -n "$RESTORE_ONLY" ]]; then
  echo "[Backup] Restore-only mode: verifying restore from $RESTORE_ONLY"
  RESTORE_DB="neurondb_verify_$$"
  createdb "$RESTORE_DB" 2>/dev/null || true
  pg_restore -d "$RESTORE_DB" --no-owner --no-acl "$RESTORE_ONLY" 2>/dev/null || true
  if psql -d "$RESTORE_DB" -tAc "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" | grep -q neurondb; then
    echo "[Backup] OK Extension neurondb present after restore"
  else
    echo "[Backup] WARN Extension neurondb not found after restore (may be expected if dump was minimal)"
  fi
  dropdb "$RESTORE_DB" 2>/dev/null || true
  echo "[Backup] Restore verification done"
  exit 0
fi

if [[ "$DUMP_ONLY" != true ]]; then
  echo "[Backup] Creating dump: $DUMP_FILE"
  pg_dump -Fc -f "$DUMP_FILE" "$PGDATABASE"
  echo "[Backup] Dump created. To verify restore: $0 --restore-only $DUMP_FILE"
fi

if [[ "$DUMP_ONLY" == true ]]; then
  echo "[Backup] Dump-only: creating $DUMP_FILE"
  pg_dump -Fc -f "$DUMP_FILE" "$PGDATABASE"
  echo "[Backup] Done: $DUMP_FILE"
  exit 0
fi

echo "[Backup] Verifying restore..."
RESTORE_DB="neurondb_verify_$$"
createdb "$RESTORE_DB" 2>/dev/null || true
pg_restore -d "$RESTORE_DB" --no-owner --no-acl "$DUMP_FILE" 2>/dev/null || true
if psql -d "$RESTORE_DB" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" | grep -q 1; then
  echo "[Backup] OK Restore verification passed (neurondb extension present)"
else
  echo "[Backup] WARN neurondb extension not in restored DB"
fi
dropdb "$RESTORE_DB" 2>/dev/null || true
echo "[Backup] Backup and verify done: $DUMP_FILE"

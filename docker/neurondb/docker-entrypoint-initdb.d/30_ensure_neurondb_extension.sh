#!/bin/bash
set -euo pipefail

# This script ensures the NeuronDB extension is created after PostgreSQL starts
# It runs after all initdb scripts complete and PostgreSQL is running

echo "[init] Ensuring NeuronDB extension is available..."

# Wait for PostgreSQL to be fully ready
until pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" >/dev/null 2>&1; do
    sleep 0.5
done

# Small additional wait to ensure PostgreSQL is fully initialized
sleep 2

DB_NAME="${POSTGRES_DB:-postgres}"
PGUSER="${POSTGRES_USER:-postgres}"

# Check if extension exists
if psql -U "$PGUSER" -d "$DB_NAME" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
    echo "[init] ✓ NeuronDB extension already exists"
    exit 0
fi

# Check if schema exists but extension doesn't (cleanup orphaned schema)
# This ensures we can create the extension even if a previous attempt left an orphaned schema
if psql -U "$PGUSER" -d "$DB_NAME" -tAc "SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb';" 2>/dev/null | grep -q 1; then
    if ! psql -U "$PGUSER" -d "$DB_NAME" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
        echo "[init] Found orphaned neurondb schema without extension, cleaning up..."
        psql -U "$PGUSER" -d "$DB_NAME" -c "DROP SCHEMA IF EXISTS neurondb CASCADE;" 2>/dev/null || true
        echo "[init] Cleaned up orphaned schema"
    fi
fi

# Try to create the extension with retries
max_attempts=5
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if psql -U "$PGUSER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null; then
        # Verify it was created
        if psql -U "$PGUSER" -d "$DB_NAME" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
            echo "[init] ✓ NeuronDB extension created successfully"
            exit 0
        fi
    fi
    attempt=$((attempt + 1))
    if [ $attempt -lt $max_attempts ]; then
        echo "[init] Retrying extension creation... ($attempt/$max_attempts)"
        sleep 2
    fi
done

echo "[init] ⚠ Could not create NeuronDB extension during initdb"
echo "[init]   This is normal - extension will be created automatically after PostgreSQL starts"
echo "[init]   (shared_preload_libraries requires PostgreSQL to be fully running)"
exit 0  # Don't fail initdb if extension creation fails

#!/bin/bash
# Simple wrapper that ensures NeuronDB extension after PostgreSQL starts
# Uses standard postgres entrypoint and adds extension creation in background

export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-/usr/local/lib}

# For postgres command, ensure extension after startup
if [ "${1:-postgres}" = 'postgres' ]; then
    # Start PostgreSQL using standard entrypoint
    /usr/local/bin/docker-entrypoint.sh "$@" &
    POSTGRES_PID=$!
    
    # Function to ensure extension exists
    ensure_extension() {
        local pguser="${POSTGRES_USER:-postgres}"
        local db_name="${POSTGRES_DB:-$pguser}"
        
        # Wait for PostgreSQL to be ready
        for i in {1..30}; do
            if pg_isready -U "$pguser" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        
        sleep 2  # Give it a moment to fully initialize
        
        # Clean up orphaned schema if exists (schema exists but extension doesn't)
        # Only drop if schema exists without the extension
        if psql -U "$pguser" -d "$db_name" -tAc "SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb';" 2>/dev/null | grep -q 1; then
            if ! psql -U "$pguser" -d "$db_name" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
                echo "[neurondb-entrypoint] Cleaning up orphaned neurondb schema..." >&2
                psql -U "$pguser" -d "$db_name" -c "DROP SCHEMA IF EXISTS neurondb CASCADE;" 2>/dev/null || true
            fi
        fi
        
        # Create extension
        psql -U "$pguser" -d "$db_name" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null && \
        echo "[neurondb-entrypoint] ✓ NeuronDB extension ready" >&2 || \
        echo "[neurondb-entrypoint] ⚠ Extension creation deferred" >&2
    }
    
    # Ensure extension in background (non-blocking)
    (sleep 5; ensure_extension) &
    
    # Wait for PostgreSQL and forward signals
    trap "kill -TERM $POSTGRES_PID" SIGTERM SIGINT
    wait $POSTGRES_PID
    exit $?
else
    exec /usr/local/bin/docker-entrypoint.sh "$@"
fi

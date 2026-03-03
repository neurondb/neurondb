#!/bin/bash
set -euo pipefail

echo "[init] Applying NeuronDB defaults"

# PostgreSQL 18+ uses different data directory structure
# During initdb, the config file location depends on PostgreSQL version
# Try multiple possible locations in order of likelihood

postgresql_conf=""

# First, try the standard location (works for most PostgreSQL versions)
if [ -n "${PGDATA:-}" ] && [ -f "${PGDATA}/postgresql.conf" ]; then
    postgresql_conf="${PGDATA}/postgresql.conf"
# PostgreSQL 18 uses /var/lib/postgresql/18/docker/postgresql.conf
elif [ -f "/var/lib/postgresql/18/docker/postgresql.conf" ]; then
    postgresql_conf="/var/lib/postgresql/18/docker/postgresql.conf"
# Try PGDATA/18/docker/postgresql.conf
elif [ -n "${PGDATA:-}" ] && [ -f "${PGDATA}/18/docker/postgresql.conf" ]; then
    postgresql_conf="${PGDATA}/18/docker/postgresql.conf"
# Try to find postgresql.conf anywhere in /var/lib/postgresql
elif postgresql_conf=$(find /var/lib/postgresql -name postgresql.conf -type f 2>/dev/null | head -1); then
    : # Found it
# Last resort: use PGDATA if set
elif [ -n "${PGDATA:-}" ]; then
    postgresql_conf="${PGDATA}/postgresql.conf"
else
    # Default PostgreSQL 18 location
    postgresql_conf="/var/lib/postgresql/18/docker/postgresql.conf"
fi

# If config file doesn't exist yet (during initdb), wait for it or try alternative locations
if [ ! -f "$postgresql_conf" ]; then
    echo "[INFO] postgresql.conf not found at $postgresql_conf, searching..."
    # Wait up to 5 seconds for config file to appear (initdb might still be running)
    for i in {1..10}; do
        if [ -f "$postgresql_conf" ]; then
            break
        fi
        # Try to find it in common locations
        if postgresql_conf=$(find /var/lib/postgresql -name postgresql.conf -type f 2>/dev/null | head -1); then
            if [ -n "$postgresql_conf" ] && [ -f "$postgresql_conf" ]; then
                echo "[INFO] Found postgresql.conf at $postgresql_conf"
                break
            fi
        fi
        sleep 0.5
    done
    
    # If still not found, try PGDATA location
    if [ ! -f "$postgresql_conf" ] && [ -n "${PGDATA:-}" ]; then
        postgresql_conf="${PGDATA}/postgresql.conf"
    fi
    
    # If still not found, we'll configure it later via ALTER SYSTEM
    if [ ! -f "$postgresql_conf" ]; then
        echo "[WARN] postgresql.conf not found, configuration will be applied after PostgreSQL starts"
        echo "[WARN] You may need to restart the container for NeuronDB to work properly"
        # Don't fail - we'll configure it after PostgreSQL starts
        exit 0
    fi
fi

# Compute mode parameter (0=cpu, 1=gpu, 2=auto, default=2)
# GPU backend type (0=cpu, 1=cuda, 2=rocm, 3=metal, default=0)
compute_mode="${NEURONDB_COMPUTE_MODE:-2}"
gpu_backend_type="${NEURONDB_GPU_BACKEND_TYPE:-0}"
automl_gpu="${NEURONDB_AUTOML_USE_GPU:-off}"

# Validate compute_mode (0=cpu, 1=gpu, 2=auto)
if [[ ! "${compute_mode}" =~ ^[0-2]$ ]]; then
    echo "[WARN] Invalid NEURONDB_COMPUTE_MODE=${compute_mode}, using default 2 (auto)"
    compute_mode=2
fi

# Validate gpu_backend_type (0=cpu, 1=cuda, 2=rocm, 3=metal)
if [[ ! "${gpu_backend_type}" =~ ^[0-3]$ ]]; then
    echo "[WARN] Invalid NEURONDB_GPU_BACKEND_TYPE=${gpu_backend_type}, using default 0 (cpu)"
    gpu_backend_type=0
fi

# Normalize accepted values
case "${automl_gpu,,}" in
    on|true|1) automl_gpu=on ;;
    *) automl_gpu=off ;;
esac

# Check if shared_preload_libraries is set (uncommented and active)
if ! grep -q "^shared_preload_libraries.*neurondb" "${postgresql_conf}"; then
    echo "[INFO] Configuring shared_preload_libraries in $postgresql_conf"
    # If shared_preload_libraries exists but is commented out, uncomment and set it
    if grep -q "^#shared_preload_libraries" "${postgresql_conf}"; then
        sed -i "s/^#shared_preload_libraries.*/shared_preload_libraries = 'neurondb'/" "${postgresql_conf}"
        echo "[INFO] Uncommented and set shared_preload_libraries = 'neurondb'"
    # If it exists but doesn't include neurondb, add neurondb to it
    elif grep -q "^shared_preload_libraries" "${postgresql_conf}"; then
        sed -i "s/^shared_preload_libraries.*/shared_preload_libraries = 'neurondb'/" "${postgresql_conf}"
        echo "[INFO] Updated shared_preload_libraries = 'neurondb'"
    # If it doesn't exist at all, add it
    else
        cat <<CONF >> "${postgresql_conf}"

# Added by NeuronDB docker image
shared_preload_libraries = 'neurondb'
CONF
        echo "[INFO] Added shared_preload_libraries = 'neurondb'"
    fi
else
    echo "[INFO] shared_preload_libraries already configured with neurondb"
fi

# Set compute_mode
if grep -q "^neurondb.compute_mode" "${postgresql_conf}"; then
    sed -i "s/^neurondb.compute_mode.*/neurondb.compute_mode = ${compute_mode}/g" "${postgresql_conf}"
else
    echo "neurondb.compute_mode = ${compute_mode}" >> "${postgresql_conf}"
fi

# Set gpu_backend_type
if grep -q "^neurondb.gpu_backend_type" "${postgresql_conf}"; then
    sed -i "s/^neurondb.gpu_backend_type.*/neurondb.gpu_backend_type = ${gpu_backend_type}/g" "${postgresql_conf}"
else
    echo "neurondb.gpu_backend_type = ${gpu_backend_type}" >> "${postgresql_conf}"
fi

# Comment out automl.use_gpu for CPU-only builds to avoid config errors
if grep -q "^neurondb.automl.use_gpu" "${postgresql_conf}"; then
    sed -i "s/^neurondb.automl.use_gpu.*/# neurondb.automl.use_gpu = ${automl_gpu}/g" "${postgresql_conf}"
fi


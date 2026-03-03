#!/bin/bash

# run_demo.sh - NeuronDB ML Demo Runner
# Usage: ./run_demo.sh [--db-size=100MB] [--gpu|--cpu]

set -e

# Default configuration
DB_SIZE="100MB"
GPU_MODE="gpu"  # default to GPU
PSQL="/usr/local/pgsql.18/bin/psql"
DB_NAME="postgres"
DB_USER="postgres"
DB_PORT="5434"
SQL_DIR="ML/sql"
OUT_DIR="ML/out"
PG_CTL="/usr/local/pgsql.18/bin/pg_ctl"
PG_DATA="${HOME}/neurondb_data"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --db-size=*)
            DB_SIZE="${arg#*=}"
            shift
            ;;
        --gpu)
            GPU_MODE="gpu"
            shift
            ;;
        --cpu)
            GPU_MODE="cpu"
            shift
            ;;
        --help)
            echo "NeuronDB ML Demo Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --db-size=SIZE    Dataset size (default: 100MB)"
            echo "  --gpu             Run with GPU acceleration (default)"
            echo "  --cpu             Run with CPU only"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --db-size=100MB --gpu"
            echo "  $0 --cpu"
            exit 0
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════════"
echo "  NeuronDB ML Demo Runner"
echo "════════════════════════════════════════════════════════════════════"
echo "Database Size: $DB_SIZE"
echo "GPU Mode: $GPU_MODE"
echo "SQL Directory: $SQL_DIR"
echo "Output Directory: $OUT_DIR"
echo ""

# Step 1: Kill and restart PostgreSQL
echo "Step 1: Restarting PostgreSQL..."
pkill -9 postgres 2>/dev/null || true
sleep 2
$PG_CTL -D $PG_DATA -l /tmp/neurondb_demo.log start -w
sleep 3
echo "✓ PostgreSQL restarted"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Step 2: Configure GPU/CPU mode
echo "Step 2: Configuring $GPU_MODE mode..."
if [ "$GPU_MODE" = "gpu" ]; then
    $PSQL -p $DB_PORT -U $DB_USER -d $DB_NAME -q <<EOF
SET neurondb.compute_mode = true;
SET neurondb.gpu_backend = 'metal';
EOF
    echo "✓ GPU mode enabled (Metal backend)"
else
    $PSQL -p $DB_PORT -U $DB_USER -d $DB_NAME -q <<EOF
SET neurondb.compute_mode = false;
EOF
    echo "✓ CPU-only mode enabled"
fi
echo ""

# Step 3: Run all SQL files in order
echo "Step 3: Running ML demos..."
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for sql_file in $(ls -1 $SQL_DIR/*.sql 2>/dev/null | sort -V); do
    filename=$(basename "$sql_file")
    output_file="$OUT_DIR/${filename%.sql}.out"
    
    echo "Running: $filename"
    
    # Run the SQL file and capture output
    if timeout 300 $PSQL -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$sql_file" > "$output_file" 2>&1; then
        echo "  ✓ Success → $output_file"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "  ⚠ Timeout → $output_file"
        else
            echo "  ✗ Failed → $output_file"
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

# Step 4: Run GPU performance tests
if [ "$GPU_MODE" = "gpu" ]; then
    echo "Step 4: Running GPU performance tests..."
    echo ""
    
    # Run extreme GPU test
    if [ -f "extreme_gpu_test.sql" ]; then
        echo "Running: extreme_gpu_test.sql"
        if timeout 1200 $PSQL -p $DB_PORT -U $DB_USER -d $DB_NAME -f "extreme_gpu_test.sql" > "$OUT_DIR/extreme_gpu_test.out" 2>&1; then
            echo "  ✓ Success → $OUT_DIR/extreme_gpu_test.out"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ Failed/Timeout → $OUT_DIR/extreme_gpu_test.out"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        echo ""
    fi
    
    # Run final CPU vs GPU comparison
    if [ -f "final_cpu_vs_gpu_5min.sql" ]; then
        echo "Running: final_cpu_vs_gpu_5min.sql"
        if timeout 600 $PSQL -p $DB_PORT -U $DB_USER -d $DB_NAME -f "final_cpu_vs_gpu_5min.sql" > "$OUT_DIR/final_cpu_vs_gpu.out" 2>&1; then
            echo "  ✓ Success → $OUT_DIR/final_cpu_vs_gpu.out"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ Failed/Timeout → $OUT_DIR/final_cpu_vs_gpu.out"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        echo ""
    fi
fi

# Step 5: Verification
echo "════════════════════════════════════════════════════════════════════"
echo "  Verification & Summary"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Total tests run: $((SUCCESS_COUNT + FAIL_COUNT))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Some tests failed. Check output files in $OUT_DIR"
fi
echo ""

# Show GPU status if in GPU mode
if [ "$GPU_MODE" = "gpu" ]; then
    echo "GPU Configuration:"
    $PSQL -p $DB_PORT -U $DB_USER -d $DB_NAME -t -q <<EOF
SELECT '  ' || name || ': ' || setting FROM pg_settings 
WHERE name LIKE 'neurondb.gpu%' AND name IN ('neurondb.compute_mode', 'neurondb.gpu_backend');
EOF
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Demo Complete!"
echo "════════════════════════════════════════════════════════════════════"
echo "Results saved to: $OUT_DIR"
echo ""

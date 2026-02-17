#!/usr/bin/env bash
# NeuronDB performance benchmark runner
# Run from repo root: ./NeuronDB/benchmark/run_benchmarks.sh
# Requires: psql with NeuronDB extension loaded, optionally PG connection env (PGHOST, PGPORT, PGUSER, PGDATABASE)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "NeuronDB benchmarks started at $(date), results under $RESULTS_DIR"

# Vector search throughput (simple L2 distance)
if [ -d "$SCRIPT_DIR/vector" ]; then
  for f in "$SCRIPT_DIR/vector"/*.sql; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .sql)
    echo "Running vector benchmark: $name"
    psql -v ON_ERROR_STOP=1 -f "$f" -o "$RESULTS_DIR/vector_${name}_${TS}.txt" 2>&1 || true
  done
fi

# ML training (if benchmark SQL exists)
if [ -d "$SCRIPT_DIR/../demo/ML" ]; then
  echo "ML benchmarks: run demo/ML scripts and record timing"
fi

echo "Benchmarks finished at $(date). Baseline: save $RESULTS_DIR for comparison."

#!/usr/bin/env bash
# Verify NeuronDB Docker ecosystem: containers running and health endpoints.
# Usage: ./scripts/verify-docker-ecosystem.sh [--verbose] [--skip-service SERVICE]
# Options:
#   --verbose         Show detailed output
#   --skip-service S  Skip checking service S (neurondb, neurondb-cuda, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERBOSE=false
SKIP_SERVICES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) VERBOSE=true; shift ;;
    --skip-service) SKIP_SERVICES+=("$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

skip() {
  local s="$1"
  for skip in "${SKIP_SERVICES[@]}"; do
    [[ "$skip" == "$s" ]] && return 0
  done
  return 1
}

cd "$ROOT"

# Ensure we have docker compose
if ! docker compose version >/dev/null 2>&1 && ! docker-compose version >/dev/null 2>&1; then
  echo "Error: docker compose (or docker-compose) not found." >&2
  exit 1
fi

COMPOSE="docker compose"
$COMPOSE version >/dev/null 2>&1 || COMPOSE="docker-compose"

echo "=== NeuronDB Docker ecosystem verification ==="
echo ""

# 1. Container status
echo "--- Container status ---"
$COMPOSE ps -a 2>/dev/null || true
echo ""

FAIL=0

# 2. NeuronDB (PostgreSQL + extension) - check default profile neurondb service
if ! skip neurondb; then
  if $COMPOSE exec -T neurondb-cpu psql -U neurondb -d neurondb -c "SELECT 1;" 2>/dev/null | grep -q 1; then
    echo "[OK] NeuronDB (PostgreSQL + extension) responding"
  elif $COMPOSE exec -T neurondb psql -U neurondb -d neurondb -c "SELECT 1;" 2>/dev/null | grep -q 1; then
    echo "[OK] NeuronDB (PostgreSQL + extension) responding"
  else
    echo "[FAIL] NeuronDB not responding (try: $COMPOSE --profile cpu up -d)"
    FAIL=1
  fi
  $VERBOSE && $COMPOSE exec -T neurondb-cpu psql -U neurondb -d neurondb -c "SELECT neurondb.version();" 2>/dev/null || $COMPOSE exec -T neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" 2>/dev/null || true
fi

# Done (agent/mcp/desktop are in separate repos)

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed. Run with --verbose for details."
  exit 1
fi

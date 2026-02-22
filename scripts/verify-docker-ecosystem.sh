#!/usr/bin/env bash
# Verify NeuronDB Docker ecosystem: containers running and health endpoints.
# Usage: ./scripts/verify-docker-ecosystem.sh [--verbose] [--skip-service SERVICE]
# Options:
#   --verbose         Show detailed output
#   --skip-service S  Skip checking service S (neurondb, neuronagent, neuronmcp, neurondesk-api, neurondesk-frontend)

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

# 2. NeuronDB (PostgreSQL + extension)
if ! skip neurondb; then
  if $COMPOSE exec -T neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" 2>/dev/null | grep -q .; then
    echo "[OK] NeuronDB (PostgreSQL + extension) responding"
  else
    echo "[FAIL] NeuronDB not responding (is neurondb container running? try: $COMPOSE --profile cpu ps)"
    FAIL=1
  fi
  $VERBOSE && $COMPOSE exec -T neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" 2>/dev/null || true
fi

# 3. NeuronAgent
if ! skip neuronagent; then
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://localhost:8080/health" 2>/dev/null || echo "000")
  if [[ "$CODE" == "200" ]]; then
    echo "[OK] NeuronAgent (http://localhost:8080/health)"
  else
    echo "[FAIL] NeuronAgent health returned $CODE (expected 200)"
    FAIL=1
  fi
  $VERBOSE && curl -s "http://localhost:8080/health" 2>/dev/null | head -5
fi

# 4. NeuronMCP (stdio; we only check container exists)
if ! skip neuronmcp; then
  if $COMPOSE ps neurondb-mcp 2>/dev/null | grep -q Up; then
    echo "[OK] NeuronMCP container running"
  else
    echo "[FAIL] NeuronMCP container not up"
    FAIL=1
  fi
fi

# 5. NeuronDesktop API
if ! skip neurondesk-api; then
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://localhost:8081/health" 2>/dev/null || echo "000")
  if [[ "$CODE" == "200" ]]; then
    echo "[OK] NeuronDesktop API (http://localhost:8081/health)"
  else
    echo "[FAIL] NeuronDesktop API health returned $CODE (expected 200)"
    FAIL=1
  fi
  $VERBOSE && curl -s "http://localhost:8081/health" 2>/dev/null | head -5
fi

# 6. NeuronDesktop Frontend
if ! skip neurondesk-frontend; then
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://localhost:3000" 2>/dev/null || echo "000")
  if [[ "$CODE" == "200" ]] || [[ "$CODE" == "304" ]]; then
    echo "[OK] NeuronDesktop Frontend (http://localhost:3000)"
  else
    echo "[FAIL] NeuronDesktop Frontend returned $CODE (expected 200/304)"
    FAIL=1
  fi
fi

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed. Run with --verbose for details."
  exit 1
fi

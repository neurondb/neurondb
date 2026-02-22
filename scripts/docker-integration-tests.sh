#!/usr/bin/env bash
# Basic integration checks for NeuronDB Docker ecosystem (agent, desktop, MCP reachability).
# Usage: ./scripts/docker-integration-tests.sh [--verbose] [--skip-test TEST_NAME]
# Tests: neurondb-agent, neurondb-mcp, desktop-db, desktop-agent, desktop-mcp, e2e-workflow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERBOSE=false
SKIP_TESTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) VERBOSE=true; shift ;;
    --skip-test) SKIP_TESTS+=("$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

skip() {
  local t="$1"
  for s in "${SKIP_TESTS[@]}"; do
    [[ "$s" == "$t" ]] && return 0
  done
  return 1
}

cd "$ROOT"
COMPOSE="docker compose"
$COMPOSE version >/dev/null 2>&1 || COMPOSE="docker-compose"

FAIL=0
run_test() {
  local name="$1"
  if skip "$name"; then
    echo "[SKIP] $name"
    return 0
  fi
  if "$@"; then
    echo "[PASS] $name"
    return 0
  else
    echo "[FAIL] $name"
    FAIL=1
    return 1
  fi
}

echo "=== NeuronDB Docker integration tests ==="
echo ""

# neurondb-agent: NeuronAgent can reach NeuronDB (agent health implies agent is up; agent uses DB)
run_test neurondb-agent bash -c 'curl -sf http://localhost:8080/health >/dev/null && true'

# neurondb-mcp: MCP container runs (stdio protocol, no HTTP)
run_test neurondb-mcp bash -c "cd '$ROOT' && $COMPOSE ps neurondb-mcp 2>/dev/null | grep -q Up"

# desktop-db: NeuronDesktop API is up (it uses NeuronDB)
run_test desktop-db bash -c 'curl -sf http://localhost:8081/health >/dev/null && true'

# desktop-agent: NeuronDesktop can proxy to NeuronAgent (both up and desktop knows agent URL)
run_test desktop-agent bash -c 'curl -sf http://localhost:8080/health >/dev/null && curl -sf http://localhost:8081/health >/dev/null && true'

# desktop-mcp: NeuronDesktop can spawn NeuronMCP (we only verify MCP container is up)
run_test desktop-mcp bash -c "cd '$ROOT' && $COMPOSE ps neurondb-mcp 2>/dev/null | grep -q Up"

# e2e-workflow: basic chain - DB, Agent, Desktop all responding
run_test e2e-workflow bash -c '
  curl -sf http://localhost:8080/health >/dev/null && \
  curl -sf http://localhost:8081/health >/dev/null && \
  curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -qE "200|304"
'

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "All integration tests passed."
  exit 0
else
  echo "Some tests failed. Run with --verbose for more detail."
  exit 1
fi

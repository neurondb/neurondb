#!/usr/bin/env bash
#
# E2E smoke test: start stack (optional), hit health and basic API endpoints.
# Usage:
#   ./scripts/e2e_smoke.sh [--no-start] [--compose-file FILE]
# --no-start: assume services are already up; only run HTTP checks.
# --compose-file: path to docker-compose file (default: docker-compose.yml).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

NO_START=false
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start) NO_START=true; shift ;;
    --compose-file) COMPOSE_FILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "[E2E] Using compose file: $COMPOSE_FILE"

if [[ "$NO_START" != true ]]; then
  echo "[E2E] Starting stack..."
  docker compose -f "$COMPOSE_FILE" up -d
  echo "[E2E] Waiting 15s for services..."
  sleep 15
fi

FAILED=0

# NeuronDB/Postgres (if exposed)
if command -v nc &>/dev/null; then
  if nc -z "${DB_HOST:-localhost}" "${DB_PORT:-5432}" 2>/dev/null; then
    echo "[E2E] OK Postgres reachable"
  else
    echo "[E2E] SKIP Postgres (not reachable or not exposed)"
  fi
fi

# NeuronAgent
AGENT_URL="${AGENT_URL:-http://localhost:8080}"
if curl -sf --connect-timeout 5 "${AGENT_URL}/health" >/dev/null; then
  echo "[E2E] OK NeuronAgent /health"
else
  echo "[E2E] FAIL NeuronAgent /health"; FAILED=1
fi

# NeuronDesktop API (if running)
DESKTOP_URL="${DESKTOP_API_URL:-http://localhost:3001}"
if curl -sf --connect-timeout 5 "${DESKTOP_URL}/health" >/dev/null 2>&1 || \
   curl -sf --connect-timeout 5 "${DESKTOP_URL}/api/v1/health" >/dev/null 2>&1; then
  echo "[E2E] OK NeuronDesktop health"
else
  echo "[E2E] SKIP NeuronDesktop (not reachable)"
fi

if [[ $FAILED -eq 0 ]]; then
  echo "[E2E] Smoke passed"
  exit 0
fi
echo "[E2E] Smoke failed"
exit 1

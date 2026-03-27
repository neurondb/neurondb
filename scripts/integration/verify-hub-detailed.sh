#!/usr/bin/env bash
# Detailed verification of NeuronHub: backend, gateway, optional frontend and dependencies.
# Usage: ./scripts/integration/verify-hub-detailed.sh [BACKEND_URL] [GATEWAY_URL]
# Defaults: BACKEND=http://localhost:8084, GATEWAY=http://localhost:8085
#
# Optional env:
#   HUB_FRONTEND_URL     - Frontend URL to check (default http://localhost:3001)
#   NEURONAGENT_URL      - If set and VERIFY_HUB_DEPS=1, also verify Agent /health
#   NEURONMCP_URL        - If set and VERIFY_HUB_DEPS=1, also verify MCP /health
#   VERIFY_HUB_DEPS=1     - Run dependency checks (Agent, optional MCP)
#   VERIFY_HUB_QUIET=1    - Only print failures and final summary

set -euo pipefail

BACKEND="${1:-${HUB_BACKEND_URL:-http://localhost:8084}}"
GATEWAY="${2:-${HUB_GATEWAY_URL:-http://localhost:8085}}"
FRONTEND="${HUB_FRONTEND_URL:-http://localhost:3001}"
BACKEND="${BACKEND%/}"
GATEWAY="${GATEWAY%/}"
FRONTEND="${FRONTEND%/}"

CONNECT_TIMEOUT=5
FAILED=0
PASSED=0

log() { [[ -z "${VERIFY_HUB_QUIET:-}" ]] && echo "$@"; }
section() { log ""; log "=== $* ==="; }

# curl: get HTTP code and body (without -f so we get body on 4xx/5xx)
# Usage: check_url "Label" "URL" [optional: 1 = required]
check_url() {
  local label="$1" url="$2" required="${3:-1}"
  local tmp code body
  tmp=$(mktemp)
  code=$(curl -s -w "%{http_code}" -o "$tmp" --connect-timeout "$CONNECT_TIMEOUT" "$url") || code="000"
  body=$(cat "$tmp" 2>/dev/null || true)
  rm -f "$tmp"
  if [[ "$code" == "200" ]]; then
    ((PASSED++)) || true
    log "PASS  $label  HTTP $code"
    [[ -z "${VERIFY_HUB_QUIET:-}" ]] && [[ -n "$body" ]] && log "      $body"
    return 0
  else
    ((FAILED++)) || true
    if [[ "$code" != "000" ]]; then
      log "FAIL  $label  HTTP $code"
      [[ -n "$body" ]] && log "      $body"
    else
      log "FAIL  $label  (connection failed or timeout)"
    fi
    [[ "$required" == "1" ]] && return 1
    return 0
  fi
}

section "NeuronHub Backend"
if ! check_url "GET $BACKEND/health" "$BACKEND/health" 1; then
  log "ERROR: Hub Backend is required. Start it (e.g. port 8084) and ensure DATABASE_URL is set."
  exit 1
fi

section "NeuronHub Gateway"
if ! check_url "GET $GATEWAY/health" "$GATEWAY/health" 0; then
  if ! check_url "GET $GATEWAY/readyz" "$GATEWAY/readyz" 1; then
    log "ERROR: Hub Gateway unreachable at $GATEWAY (tried /health and /readyz)."
    exit 1
  fi
fi
check_url "GET $GATEWAY/readyz" "$GATEWAY/readyz" 0

section "NeuronHub Frontend (optional)"
check_url "GET $FRONTEND" "$FRONTEND" 0

if [[ -n "${VERIFY_HUB_DEPS:-}" ]]; then
  section "Dependencies (VERIFY_HUB_DEPS=1)"
  if [[ -n "${NEURONAGENT_URL:-}" ]]; then
    check_url "NeuronAgent GET $NEURONAGENT_URL/health" "${NEURONAGENT_URL%/}/health" 0
  else
    log "SKIP  NeuronAgent (NEURONAGENT_URL not set)"
  fi
  if [[ -n "${NEURONMCP_URL:-}" ]]; then
    check_url "NeuronMCP GET $NEURONMCP_URL/health" "${NEURONMCP_URL%/}/health" 0
  else
    log "SKIP  NeuronMCP (NEURONMCP_URL not set)"
  fi
fi

section "Summary"
log "Passed: $PASSED  Failed: $FAILED"
if [[ "$FAILED" -gt 0 ]]; then
  exit 1
fi
log "verify-hub-detailed OK"

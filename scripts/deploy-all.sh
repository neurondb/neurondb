#!/bin/bash
#
# Deploy NeuronDB ecosystem (neurondb, neuron-cloud, neuron-hub) on a single
# machine via passwordless SSH. Uses Docker Compose with a shared network.
#
# Usage: ./deploy-all.sh TARGET_HOST [LOCAL_BASE_DIR]
#   TARGET_HOST       e.g. user@192.168.1.100 (passwordless SSH)
#   LOCAL_BASE_DIR    optional; parent dir containing neurondb, neuron-cloud, neuron-hub (default: parent of neurondb repo)
#
# Prerequisites: passwordless SSH to TARGET_HOST; Docker on remote (script can install).
# When neuron-hub is deployed: set HUB_POSTGRES_PASSWORD and HUB_JWT_SECRET (no hardcoded secrets).
# Ports: neurondb 5433 | neuron-cloud 5435,8083 | neuron-hub 5434,8084,8085,3001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURONDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_HOST="${1:-}"
LOCAL_BASE="${2:-$(dirname "$NEURONDB_ROOT")}"
REMOTE_BASE="$HOME/neurondb-platform"
SHARED_NETWORK="neurondb-platform-net"

# Optional: override sibling repo paths
NEURONDB_CLOUD_DIR="${NEURONDB_CLOUD_DIR:-$LOCAL_BASE/neuron-cloud}"
NEURON_HUB_DIR="${NEURON_HUB_DIR:-$LOCAL_BASE/neuron-hub}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

usage() {
  echo "Usage: $0 TARGET_HOST [LOCAL_BASE_DIR]"
  echo "  TARGET_HOST    e.g. user@192.168.1.100"
  echo "  LOCAL_BASE_DIR optional; default: parent of neurondb repo"
  exit 1
}

[[ -z "$TARGET_HOST" ]] && usage

# Run a command on the remote host
run_remote() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$TARGET_HOST" -- "$@"
}

# Copy content to remote file
remote_write() {
  local path="$1"
  shift
  run_remote "mkdir -p $(dirname "$path")"
  ssh "$TARGET_HOST" "cat > $path" "$@"
}

log "Target: $TARGET_HOST | Remote base: $REMOTE_BASE"
log "Local neurondb: $NEURONDB_ROOT"
log "Local neuron-cloud: $NEURONDB_CLOUD_DIR"
log "Local neuron-hub: $NEURON_HUB_DIR"

# --- Prerequisites: SSH and Docker on remote ---
log "Checking SSH access..."
if ! run_remote "true"; then
  err "Cannot SSH to $TARGET_HOST (passwordless SSH required)"
  exit 1
fi
ok "SSH OK"

log "Ensuring Docker and Docker Compose on remote..."
run_remote "command -v docker >/dev/null 2>&1 || (curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker \$USER 2>/dev/null; echo 'Docker installed; you may need to log out and back in for group')"
run_remote "docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 || (echo 'Docker Compose not found' && exit 1)"
ok "Docker ready"

# --- Create shared network ---
log "Creating shared Docker network: $SHARED_NETWORK"
run_remote "docker network inspect $SHARED_NETWORK >/dev/null 2>&1 || docker network create $SHARED_NETWORK"
ok "Network $SHARED_NETWORK ready"

# --- Sync repos to remote ---
log "Syncing neurondb to $TARGET_HOST:$REMOTE_BASE/neurondb..."
rsync -az --delete \
  --exclude '.git' --exclude '*.pyc' --exclude 'node_modules' --exclude '.next' \
  "$NEURONDB_ROOT/" "$TARGET_HOST:$REMOTE_BASE/neurondb/"
ok "neurondb synced"

if [[ -d "$NEURONDB_CLOUD_DIR" ]]; then
  log "Syncing neuron-cloud..."
  rsync -az --delete \
    --exclude '.git' --exclude '*.pyc' --exclude 'node_modules' \
    "$NEURONDB_CLOUD_DIR/" "$TARGET_HOST:$REMOTE_BASE/neuron-cloud/"
  ok "neuron-cloud synced"
else
  warn "neuron-cloud not found at $NEURONDB_CLOUD_DIR (skipping)"
fi

if [[ -d "$NEURON_HUB_DIR" ]]; then
  log "Syncing neuron-hub..."
  rsync -az --delete \
    --exclude '.git' --exclude '*.pyc' --exclude 'node_modules' --exclude '.next' \
    "$NEURON_HUB_DIR/" "$TARGET_HOST:$REMOTE_BASE/neuron-hub/"
  ok "neuron-hub synced"
else
  warn "neuron-hub not found at $NEURON_HUB_DIR (skipping)"
fi

# --- Neurondb: .env for ports and attach to shared network ---
# Security: no hardcoded passwords. Set POSTGRES_PASSWORD (and optionally POSTGRES_USER, POSTGRES_DB) before running.
: "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD for deploy (e.g. export POSTGRES_PASSWORD=your-secure-password)}"
log "Configuring neurondb (.env and network override)..."
run_remote "cat > $REMOTE_BASE/neurondb/.env << ENVEOF
POSTGRES_PORT=5433
POSTGRES_USER=${POSTGRES_USER:-neurondb}
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DB=${POSTGRES_DB:-neurondb}
ENVEOF"
run_remote "cat > $REMOTE_BASE/neurondb/docker-compose.platform.yml << 'YAMLEOF'
services:
  neurondb:
    networks:
      - neurondb-network
      - platform
  neurondb-cuda:
    networks:
      - neurondb-network
      - platform
  neurondb-rocm:
    networks:
      - neurondb-network
      - platform
  neurondb-metal:
    networks:
      - neurondb-network
      - platform

networks:
  neurondb-network:
    name: neurondb-network
    driver: bridge
  platform:
    name: $SHARED_NETWORK
    external: true
YAMLEOF"
run_remote "sed -i 's/\\\$SHARED_NETWORK/$SHARED_NETWORK/' $REMOTE_BASE/neurondb/docker-compose.platform.yml"
ok "neurondb configured"

# --- Deploy neurondb (CPU profile) ---
log "Deploying neurondb (CPU)..."
run_remote "cd $REMOTE_BASE/neurondb && docker compose -f docker-compose.yml -f docker-compose.platform.yml --profile cpu up -d --build"
ok "neurondb started"

# --- Neurondb-cloud (neuron-cloud): port overrides and shared network ---
if run_remote "test -d $REMOTE_BASE/neuron-cloud"; then
  log "Configuring neuron-cloud (ports 5435, 8083)..."
  run_remote "cat > $REMOTE_BASE/neuron-cloud/.env << 'ENVEOF'
# Control plane DB on host 5435 to avoid conflict with neurondb 5433
ENVEOF"
  run_remote "cat > $REMOTE_BASE/neuron-cloud/docker-compose.override.yml << 'YAMLEOF'
services:
  db:
    ports:
      - \"5435:5432\"
    networks:
      - default
      - platform
  migrate:
    networks:
      - default
      - platform
  gateway:
    ports:
      - \"8083:8080\"
    networks:
      - default
      - platform

networks:
  platform:
    name: $SHARED_NETWORK
    external: true
YAMLEOF"
  run_remote "sed -i 's/\\\$SHARED_NETWORK/$SHARED_NETWORK/' $REMOTE_BASE/neuron-cloud/docker-compose.override.yml"
  log "Deploying neuron-cloud..."
  run_remote "cd $REMOTE_BASE/neuron-cloud && docker compose up -d --build"
  ok "neuron-cloud started"
fi

# --- Neuron-hub: use repo docker/ compose + override (no hardcoded secrets) ---
if run_remote "test -d $REMOTE_BASE/neuron-hub"; then
  : "${HUB_POSTGRES_PASSWORD:?Set HUB_POSTGRES_PASSWORD for neuron-hub (e.g. export HUB_POSTGRES_PASSWORD=your-secure-password)}"
  : "${HUB_JWT_SECRET:?Set HUB_JWT_SECRET for neuron-hub (e.g. export HUB_JWT_SECRET=your-secure-secret)}"
  log "Setting up neuron-hub (docker/docker-compose.yml + override, env from .env)..."
  run_remote "cat > $REMOTE_BASE/neuron-hub/.env << ENVEOF
POSTGRES_USER=${HUB_POSTGRES_USER:-neuronhub}
POSTGRES_PASSWORD=$HUB_POSTGRES_PASSWORD
POSTGRES_DB=${HUB_POSTGRES_DB:-neuronhub}
JWT_SECRET=$HUB_JWT_SECRET
CORS_ORIGINS=\${CORS_ORIGINS:-http://localhost:3000}
ENVEOF"
  # Override: platform ports (8084, 8085, 3001), NEURONAGENT_URL, and shared network. Build from repo docker/Dockerfile.*.
  run_remote "cat > $REMOTE_BASE/neuron-hub/docker-compose.override.yml << 'YAMLEOF'
services:
  hub-db:
    ports:
      - \"5434:5432\"
    networks:
      - default
      - platform

  backend:
    ports:
      - \"8084:8084\"
    environment:
      PORT: \"8084\"
      NEURONAGENT_URL: \"http://neuronagent:8080\"
    networks:
      - default
      - platform
    healthcheck:
      test: [\"CMD-SHELL\", \"curl -sf http://localhost:8084/health || exit 1\"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s

  gateway:
    ports:
      - \"8085:8085\"
    environment:
      PORT: \"8085\"
      NEURONAGENT_URL: \"http://neuronagent:8080\"
      HUB_BACKEND_URL: \"http://backend:8084\"
    networks:
      - default
      - platform

  frontend:
    ports:
      - \"3001:3000\"
    environment:
      PORT: \"3000\"
      NEXT_PUBLIC_API_URL: \"http://localhost:8084\"
    networks:
      - default
      - platform

networks:
  platform:
    name: $SHARED_NETWORK
    external: true
YAMLEOF"
  run_remote "sed -i 's/\\\$SHARED_NETWORK/$SHARED_NETWORK/' $REMOTE_BASE/neuron-hub/docker-compose.override.yml"

  log "Deploying neuron-hub..."
  run_remote "cd $REMOTE_BASE/neuron-hub && docker compose -f docker/docker-compose.yml -f docker-compose.override.yml up -d hub-db"
  run_remote "sleep 5"
  log "Running Hub DB migrations (backend image runs migrate on startup; first up -d will apply them)..."
  run_remote "cd $REMOTE_BASE/neuron-hub && docker compose -f docker/docker-compose.yml -f docker-compose.override.yml up -d --build"
  ok "neuron-hub started"
fi

# --- Health checks and summary ---
log "Waiting for services..."
sleep 10

# Extract host for health checks (strip user@)
TARGET_HOST_ONLY="${TARGET_HOST#*@}"
[[ "$TARGET_HOST_ONLY" == "$TARGET_HOST" ]] && TARGET_HOST_ONLY="$TARGET_HOST"

log "Health check (on remote)..."
run_remote '(
  echo "Service                    | URL                          | Status"
  echo "---------------------------|------------------------------|--------"
  if test -d '"$REMOTE_BASE"'/neuron-cloud 2>/dev/null; then
    st=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:8083/health" 2>/dev/null || echo "---")
    [ "$st" = "200" ] && st="OK" || st="-"
    printf "%-26s | %-28s | %s\n" "NeuronDB Cloud Gateway" "http://127.0.0.1:8083" "$st"
  fi
  if test -d '"$REMOTE_BASE"'/neuron-hub 2>/dev/null; then
    for port in 8084 8085 3001; do
      case "$port" in 8084) name="Hub Backend";; 8085) name="Hub Gateway";; 3001) name="Hub Frontend";; *) name="Hub $port";; esac
      st=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$port/health" 2>/dev/null || echo "---")
      [ "$st" = "200" ] && st="OK" || st="-"
      printf "%-26s | %-28s | %s\n" "$name" "http://127.0.0.1:$port" "$st"
    done
  fi
)' 2>/dev/null || true

run_remote "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | head -30"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deployment summary (host: $TARGET_HOST)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  neurondb:"
echo "    NeuronDB (PostgreSQL)  : $TARGET_HOST:5433"
echo ""
if run_remote "test -d $REMOTE_BASE/neuron-cloud"; then
  echo "  neuron-cloud:"
  echo "    Control Plane DB      : $TARGET_HOST:5435"
  echo "    Gateway               : http://$TARGET_HOST:8083"
  echo ""
fi
if run_remote "test -d $REMOTE_BASE/neuron-hub"; then
  echo "  neuron-hub:"
  echo "    Hub DB                : $TARGET_HOST:5434"
  echo "    Backend               : http://$TARGET_HOST:8084"
  echo "    Gateway               : http://$TARGET_HOST:8085"
  echo "    Frontend              : http://$TARGET_HOST:3001"
  echo "    (Optional: set NEURONDB_CLOUD_API_URL to http://\$HOST:8083 and NEURONDB_CLOUD_API_KEY to use Cloud as backend.)"
  echo ""
fi
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
ok "Done."

#!/usr/bin/env bash
# Regenerate docs/assets/neurondb-demo.gif using VHS. See demos/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEMO_CONTAINER="neurondb-readme-demo"
DEMO_PORT="15433"
HUB_IMAGE="neurondb/neurondb:latest"
GIF_OUT="docs/assets/neurondb-demo.gif"
MAX_GIF_BYTES=$((10 * 1024 * 1024))

log() { printf '%s\n' "$*" >&2; }
die() { log "Error: $*"; exit 1; }

require_cmd() {
	local c="$1" msg="$2"
	command -v "$c" >/dev/null 2>&1 || die "$msg"
}

if ! command -v docker >/dev/null 2>&1; then
	die "Docker CLI not found. Install Docker: https://docs.docker.com/get-docker/"
fi

if ! command -v psql >/dev/null 2>&1; then
	die "psql not found. Install PostgreSQL client tools (e.g. postgresql-client on Debian/Ubuntu, libpq on macOS)."
fi

if ! command -v vhs >/dev/null 2>&1; then
	log "vhs is not installed. Install it to record the demo GIF:"
	log ""
	log "  macOS (Homebrew):"
	log "    brew install charmbracelet/tap/vhs"
	log ""
	log "  Linux and other platforms:"
	log "    https://github.com/charmbracelet/vhs"
	log ""
	exit 1
fi

if ! docker info >/dev/null 2>&1; then
	die "Docker daemon is not running or not accessible."
fi

mkdir -p docs/assets

resolve_image() {
	if docker pull -q "$HUB_IMAGE" 2>/dev/null; then
		printf '%s' "$HUB_IMAGE"
		return 0
	fi
	log "Note: docker pull ${HUB_IMAGE} failed (image may be unpublished). Building local CPU image from docker/neurondb/Dockerfile (see demos/README.md)."
	docker build -f docker/neurondb/Dockerfile -t "$HUB_IMAGE" "$ROOT"
	printf '%s' "$HUB_IMAGE"
}

RESOLVED_IMAGE="$(resolve_image)"

# Demo-only cleanup from a previous run (never touch user container "neurondb").
if docker inspect "$DEMO_CONTAINER" >/dev/null 2>&1; then
	log "Removing previous demo container ${DEMO_CONTAINER}..."
	docker rm -f "$DEMO_CONTAINER"
fi

log "Starting verification container ${DEMO_CONTAINER} on port ${DEMO_PORT}..."
bash "$ROOT/scripts/install-docker.sh" \
	--name "$DEMO_CONTAINER" \
	--port "$DEMO_PORT" \
	--image "$RESOLVED_IMAGE" \
	--quiet

wait_demo_ready() {
	local max=90 i=0
	while ((i < max)); do
		if docker exec "$DEMO_CONTAINER" pg_isready -U neurondb -d neurondb >/dev/null 2>&1; then
			return 0
		fi
		sleep 1
		((i += 1)) || true
	done
	die "PostgreSQL did not become ready in ${DEMO_CONTAINER} within ${max}s"
}

wait_demo_ready

log "Verifying CREATE EXTENSION and neurondb.version() inside ${DEMO_CONTAINER}..."
docker exec -e PGPASSWORD=neurondb "$DEMO_CONTAINER" \
	psql -v ON_ERROR_STOP=1 -U neurondb -d neurondb \
	-c "CREATE EXTENSION IF NOT EXISTS neurondb;" \
	-c "SELECT neurondb.version();" >/dev/null

log "Verifying sample vector query..."
docker exec -e PGPASSWORD=neurondb "$DEMO_CONTAINER" \
	psql -v ON_ERROR_STOP=1 -U neurondb -d neurondb \
	-c "SELECT ARRAY[1.0, 2.0, 3.0]::vector(3);" >/dev/null

docker rm -f "$DEMO_CONTAINER" >/dev/null

if docker inspect neurondb >/dev/null 2>&1; then
	die "A Docker container named 'neurondb' already exists. Remove or rename it before recording so the tape can match the README one-liner (default container name). Example: docker rm -f neurondb"
fi

if command -v nc >/dev/null 2>&1; then
	if nc -z 127.0.0.1 5433 2>/dev/null; then
		die "Something is already accepting connections on localhost:5433. Free the port or stop the conflicting service before recording."
	fi
elif command -v ss >/dev/null 2>&1; then
	if ss -tln | grep -qE '(:|\])5433\b'; then
		die "Port 5433 appears in use (ss). Free it before recording."
	fi
fi

log "Recording with VHS..."
vhs "demos/neurondb-demo.tape"

if [[ ! -f "$GIF_OUT" ]]; then
	die "Expected output missing: ${GIF_OUT}"
fi

sz="$(wc -c <"$GIF_OUT" | tr -d '[:space:]')"
if ((sz > MAX_GIF_BYTES)); then
	log "Warning: ${GIF_OUT} is $((sz / 1024 / 1024)) MB (target under 10 MB). Shorten the tape or reduce frame density in demos/neurondb-demo.tape."
else
	log "Wrote ${GIF_OUT} (${sz} bytes)"
fi

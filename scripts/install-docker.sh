#!/usr/bin/env bash
# End-user installer: run NeuronDB PostgreSQL (official image) with persistent storage.
# Usage: see --help. Idempotent: safe to re-run; does not destroy data unless --reset + NEURONDB_CONFIRM_RESET=yes.
set -euo pipefail

readonly DEFAULT_CONTAINER_NAME="neurondb"
readonly DEFAULT_HOST_PORT="5433"
readonly DEFAULT_USER="neurondb"
readonly DEFAULT_DB="neurondb"
readonly DEFAULT_PASSWORD="${NEURONDB_PASSWORD:-neurondb}"

CONTAINER_NAME="${NEURONDB_CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}"
HOST_PORT="${NEURONDB_PORT:-$DEFAULT_HOST_PORT}"
IMAGE="${NEURONDB_IMAGE:-neurondb/neurondb:latest}"
POSTGRES_PASSWORD_VALUE="${DEFAULT_PASSWORD}"
DO_RESET="false"
QUIET="false"

usage() {
	cat <<'USAGE'
Usage: install-docker.sh [options]

  One-command setup: pulls the official NeuronDB image (CPU quickstart by default),
  starts PostgreSQL with a persistent Docker volume, waits until the server is ready,
  then verifies the neurondb extension with SELECT neurondb.version().

Options:
  --name NAME       Container name (default: neurondb, or NEURONDB_CONTAINER_NAME)
  --port PORT       Host port mapped to PostgreSQL 5432 (default: 5433, or NEURONDB_PORT)
  --image IMAGE     Image to use (default: neurondb/neurondb:latest, or NEURONDB_IMAGE)
  --password PASS   POSTGRES_PASSWORD (default: neurondb, or NEURONDB_PASSWORD)
  --quiet           Less console output (quiet docker pull, minimal progress messages)
  --reset           Remove this container and its named volume (requires NEURONDB_CONFIRM_RESET=yes)
  -h, --help        Show this help

Environment:
  NEURONDB_CONFIRM_RESET=yes   Required with --reset to delete container and volume
  NEURONDB_CONTAINER_NAME      Default container name if --name is not passed
  NEURONDB_PORT                Default host port if --port is not passed
  NEURONDB_IMAGE               Default image if --image is not passed
  NEURONDB_PASSWORD            Default password if --password is not passed
  NEURONDB_DOCKER_GPUS         For CUDA images (name contains neurondb-cuda): passed as
                               docker --gpus (default: all). Set to no or 0 to skip GPU flags.

USAGE
}

log() { printf '%s\n' "$*" >&2; }
die() { log "Error: $*"; exit 1; }

require_docker() {
	command -v docker >/dev/null 2>&1 || die "Docker CLI not found. Install Docker: https://docs.docker.com/get-docker/"
}

volume_name_for_container() {
	local n="$1"
	printf 'neurondb-%s-pgdata' "$n"
}

# True when docker run should request GPU devices (CUDA images or explicit NEURONDB_DOCKER_GPUS).
gpu_run_args() {
	local img="$1"
	local g="${NEURONDB_DOCKER_GPUS:-}"
	if [[ "$g" == "no" || "$g" == "0" ]]; then
		return 1
	fi
	case "$img" in
	*neurondb-cuda*)
		printf '%s' "${g:-all}"
		return 0
		;;
	esac
	if [[ -n "$g" ]]; then
		printf '%s' "$g"
		return 0
	fi
	return 1
}

wait_for_postgres() {
	local name="$1" user="$2" db="$3"
	local max=60 i=0
	while ((i < max)); do
		if docker exec "$name" pg_isready -U "$user" -d "$db" >/dev/null 2>&1; then
			return 0
		fi
		sleep 1
		((i += 1)) || true
	done
	die "PostgreSQL did not become ready within ${max}s (container: $name)"
}

verify_extension() {
	local name="$1" user="$2" db="$3" password="$4" quiet="$5"
	if [[ "$quiet" == "true" ]]; then
		docker exec -e PGPASSWORD="$password" "$name" \
			psql -v ON_ERROR_STOP=1 -U "$user" -d "$db" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" \
			-c "SELECT neurondb.version();" >/dev/null
	else
		docker exec -e PGPASSWORD="$password" "$name" \
			psql -v ON_ERROR_STOP=1 -U "$user" -d "$db" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" \
			-c "SELECT neurondb.version();"
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	-h | --help)
		usage
		exit 0
		;;
	--name)
		[[ $# -ge 2 ]] || die "--name requires a value"
		CONTAINER_NAME="$2"
		shift 2
		;;
	--port)
		[[ $# -ge 2 ]] || die "--port requires a value"
		HOST_PORT="$2"
		shift 2
		;;
	--image)
		[[ $# -ge 2 ]] || die "--image requires a value"
		IMAGE="$2"
		shift 2
		;;
	--password)
		[[ $# -ge 2 ]] || die "--password requires a value"
		POSTGRES_PASSWORD_VALUE="$2"
		shift 2
		;;
	--quiet)
		QUIET="true"
		shift
		;;
	--reset)
		DO_RESET="true"
		shift
		;;
	*)
		die "Unknown option: $1 (use --help)"
		;;
	esac
done

require_docker

VOL_NAME="$(volume_name_for_container "$CONTAINER_NAME")"

if [[ "$DO_RESET" == "true" ]]; then
	if [[ "${NEURONDB_CONFIRM_RESET:-}" != "yes" ]]; then
		log "Refusing --reset: set environment variable NEURONDB_CONFIRM_RESET=yes to delete the"
		log "container (${CONTAINER_NAME}) and volume (${VOL_NAME}). This permanently removes local data."
		exit 1
	fi
	if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
		log "Removing container ${CONTAINER_NAME}..."
		docker rm -f "$CONTAINER_NAME"
	else
		log "No container named ${CONTAINER_NAME}."
	fi
	if docker volume inspect "$VOL_NAME" >/dev/null 2>&1; then
		log "Removing volume ${VOL_NAME}..."
		docker volume rm "$VOL_NAME"
	else
		log "No volume named ${VOL_NAME}."
	fi
	log "Reset complete. Run this script again without --reset to start a fresh instance."
	exit 0
fi

if ! docker info >/dev/null 2>&1; then
	die "Docker daemon is not running or not accessible."
fi

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
	state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
	if [[ "$state" != "running" ]]; then
		[[ "$QUIET" == "true" ]] || log "Starting existing container ${CONTAINER_NAME}..."
		docker start "$CONTAINER_NAME"
	fi
else
	if [[ "$QUIET" != "true" ]]; then
		log "Pulling image ${IMAGE}..."
	fi
	if [[ "$QUIET" == "true" ]]; then
		docker pull -q "$IMAGE"
	else
		docker pull "$IMAGE"
	fi
	[[ "$QUIET" == "true" ]] || log "Creating container ${CONTAINER_NAME} (volume ${VOL_NAME})..."
	if ! docker volume inspect "$VOL_NAME" >/dev/null 2>&1; then
		docker volume create "$VOL_NAME" >/dev/null
	fi
	GPU_RUN=( )
	gpu_spec=""
	if gpu_spec="$(gpu_run_args "$IMAGE")"; then
		GPU_RUN=( --gpus "$gpu_spec" )
	fi
	docker run -d \
		"${GPU_RUN[@]}" \
		--name "$CONTAINER_NAME" \
		-p "${HOST_PORT}:5432" \
		-e POSTGRES_USER="$DEFAULT_USER" \
		-e POSTGRES_PASSWORD="$POSTGRES_PASSWORD_VALUE" \
		-e POSTGRES_DB="$DEFAULT_DB" \
		-v "${VOL_NAME}:/var/lib/postgresql/data" \
		--restart unless-stopped \
		--init \
		"$IMAGE"
fi

[[ "$QUIET" == "true" ]] || log "Waiting for PostgreSQL to accept connections..."
wait_for_postgres "$CONTAINER_NAME" "$DEFAULT_USER" "$DEFAULT_DB"

[[ "$QUIET" == "true" ]] || log "Verifying NeuronDB extension..."
verify_extension "$CONTAINER_NAME" "$DEFAULT_USER" "$DEFAULT_DB" "$POSTGRES_PASSWORD_VALUE" "$QUIET"

ver_line="$(docker exec -e PGPASSWORD="$POSTGRES_PASSWORD_VALUE" "$CONTAINER_NAME" \
	psql -tAc "SELECT neurondb.version();" -U "$DEFAULT_USER" -d "$DEFAULT_DB" 2>/dev/null | head -1 | tr -d '\r')"

if [[ "$QUIET" == "true" ]]; then
	log "NeuronDB ready — localhost:${HOST_PORT} — ${ver_line:-ok}"
	log "psql \"postgresql://${DEFAULT_USER}:${POSTGRES_PASSWORD_VALUE}@localhost:${HOST_PORT}/${DEFAULT_DB}\""
else
	log ""
	log "NeuronDB is ready."
	log "  PostgreSQL:  localhost:${HOST_PORT}"
	log "  Database:    ${DEFAULT_DB}"
	log "  User:        ${DEFAULT_USER}"
	log "  Version:     ${ver_line:-ok}"
	log ""
	log "Connect with psql:"
	log "  psql \"postgresql://${DEFAULT_USER}:${POSTGRES_PASSWORD_VALUE}@localhost:${HOST_PORT}/${DEFAULT_DB}\""
	log ""
	log "Quick SQL check:"
	log "  CREATE EXTENSION IF NOT EXISTS neurondb;"
	log "  SELECT neurondb.version();"
fi

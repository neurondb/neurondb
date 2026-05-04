#!/usr/bin/env bash
# Maintainer: build NeuronDB Docker image (default: CUDA GPU), smoke-test, tag, push.
# Usage:
#   VERSION=3.1.0 ./scripts/docker-release.sh      # CUDA: primary PG — latest, pg17, pg17-3.1.0, 3.1.0
#   PG_MAJOR=18 ./scripts/docker-release.sh 3.1.0  # CUDA: only pg18, pg18-3.1.0
#   DOCKERFILE=docker/neurondb/Dockerfile HUB_IMAGE=neurondb/neurondb GHCR_IMAGE=ghcr.io/neurondb/neurondb ./scripts/docker-release.sh 3.1.0   # CPU
#   ./scripts/docker-release.sh v3.1.0
# Env:
#   DOCKERFILE (default docker/neurondb/Dockerfile.gpu.cuda) — set to docker/neurondb/Dockerfile for CPU builds
#   CUDA_VERSION (default 12.4.1) — used when building Dockerfile.gpu.cuda
#   DOCKERHUB_USERNAME, DOCKERHUB_TOKEN — required for push (docker login)
#   PG_MAJOR (default 17) — also 16 or 18; run per major to ship all lines
#   PRIMARY_PG_MAJOR (default 17) — :latest and :$VER tags only for this major (avoids overwrites)
#   HUB_IMAGE / GHCR_IMAGE — defaults neurondb/neurondb-cuda and ghcr.io/neurondb/neurondb-cuda
#   PUSH_GHCR=1 — also push GHCR (login: gh auth token | docker login ghcr.io -u USER --password-stdin)
# Never hardcode secrets in this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PG_MAJOR="${PG_MAJOR:-17}"
PRIMARY_PG_MAJOR="${PRIMARY_PG_MAJOR:-17}"
ONNX_VERSION="${ONNX_VERSION:-1.17.0}"
CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
DOCKERFILE="${DOCKERFILE:-docker/neurondb/Dockerfile.gpu.cuda}"
SMOKE_PORT="${SMOKE_PORT:-55433}"
SMOKE_NAME="${SMOKE_NAME:-neurondb-release-smoke-cuda}"
LOCAL_TAG="${LOCAL_TAG:-neurondb:local-release-cuda}"
HUB_IMAGE="${HUB_IMAGE:-neurondb/neurondb-cuda}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/neurondb/neurondb-cuda}"

RAW_VERSION="${1:-${VERSION:-}}"
[[ -n "$RAW_VERSION" ]] || {
	echo "Usage: $0 <version|vX.Y.Z>   or set VERSION=vX.Y.Z" >&2
	exit 1
}

strip_v() {
	local v="$1"
	v="${v#v}"
	printf '%s' "$v"
}

VER="$(strip_v "$RAW_VERSION")"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "==> Building ${LOCAL_TAG} (${DOCKERFILE}, PG_MAJOR=${PG_MAJOR}, VERSION=${VER})"
CUDA_ARGS=()
if [[ "${DOCKERFILE}" == *Dockerfile.gpu.cuda ]]; then
	CUDA_ARGS+=(--build-arg "CUDA_VERSION=${CUDA_VERSION}")
fi
docker build \
	-f "${DOCKERFILE}" \
	--build-arg "PG_MAJOR=${PG_MAJOR}" \
	"${CUDA_ARGS[@]}" \
	--build-arg "ONNX_VERSION=${ONNX_VERSION}" \
	--build-arg "VERSION=${VER}" \
	--build-arg "BUILD_DATE=${BUILD_DATE}" \
	--build-arg "VCS_REF=${VCS_REF}" \
	-t "${LOCAL_TAG}" \
	.

echo "==> Smoke test (${SMOKE_NAME} on localhost:${SMOKE_PORT})"
docker rm -f "${SMOKE_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${SMOKE_NAME}" \
	-p "${SMOKE_PORT}:5432" \
	-e POSTGRES_USER=neurondb \
	-e POSTGRES_PASSWORD=neurondb \
	-e POSTGRES_DB=neurondb \
	"${LOCAL_TAG}"

cleanup_smoke() {
	docker rm -f "${SMOKE_NAME}" >/dev/null 2>&1 || true
}
trap cleanup_smoke EXIT

for _ in $(seq 1 90); do
	if docker exec "${SMOKE_NAME}" pg_isready -U neurondb -d neurondb >/dev/null 2>&1; then
		break
	fi
	sleep 1
done

docker exec -e PGPASSWORD=neurondb "${SMOKE_NAME}" \
	psql -v ON_ERROR_STOP=1 -U neurondb -d neurondb \
	-c "CREATE EXTENSION IF NOT EXISTS neurondb;" \
	-c "SELECT neurondb.version();"

cleanup_smoke
trap - EXIT

TAG_LATEST="latest"
TAG_PG="pg${PG_MAJOR}"
TAG_PG_VER="pg${PG_MAJOR}-${VER}"
TAG_VER="${VER}"

echo "==> Tagging ${HUB_IMAGE}"
docker tag "${LOCAL_TAG}" "${HUB_IMAGE}:${TAG_PG}"
docker tag "${LOCAL_TAG}" "${HUB_IMAGE}:${TAG_PG_VER}"
if [[ "${PG_MAJOR}" == "${PRIMARY_PG_MAJOR}" ]]; then
	docker tag "${LOCAL_TAG}" "${HUB_IMAGE}:${TAG_LATEST}"
	docker tag "${LOCAL_TAG}" "${HUB_IMAGE}:${TAG_VER}"
fi

if [[ "${PUSH_GHCR:-0}" == "1" ]]; then
	echo "==> Tagging ${GHCR_IMAGE}"
	docker tag "${LOCAL_TAG}" "${GHCR_IMAGE}:${TAG_PG}"
	docker tag "${LOCAL_TAG}" "${GHCR_IMAGE}:${TAG_PG_VER}"
	if [[ "${PG_MAJOR}" == "${PRIMARY_PG_MAJOR}" ]]; then
		docker tag "${LOCAL_TAG}" "${GHCR_IMAGE}:${TAG_LATEST}"
		docker tag "${LOCAL_TAG}" "${GHCR_IMAGE}:${TAG_VER}"
	fi
fi

push_tags_dockerhub() {
	local img="$1"
	shift
	for t in "$@"; do
		docker push "${img}:${t}"
	done
}

if [[ -n "${DOCKERHUB_USERNAME:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then
	echo "==> Pushing to Docker Hub (${HUB_IMAGE})"
	echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin docker.io
	if [[ "${PG_MAJOR}" == "${PRIMARY_PG_MAJOR}" ]]; then
		push_tags_dockerhub "${HUB_IMAGE}" "${TAG_LATEST}" "${TAG_PG}" "${TAG_PG_VER}" "${TAG_VER}"
	else
		push_tags_dockerhub "${HUB_IMAGE}" "${TAG_PG}" "${TAG_PG_VER}"
	fi
else
	echo "Skipping Docker Hub push (set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN)."
fi

if [[ "${PUSH_GHCR:-0}" == "1" ]]; then
	echo "==> Pushing to GHCR (${GHCR_IMAGE}) — ensure you are logged in: docker login ghcr.io"
	if [[ "${PG_MAJOR}" == "${PRIMARY_PG_MAJOR}" ]]; then
		push_tags_dockerhub "${GHCR_IMAGE}" "${TAG_LATEST}" "${TAG_PG}" "${TAG_PG_VER}" "${TAG_VER}"
	else
		push_tags_dockerhub "${GHCR_IMAGE}" "${TAG_PG}" "${TAG_PG_VER}"
	fi
fi

echo "Done."

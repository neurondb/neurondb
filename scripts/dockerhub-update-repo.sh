#!/usr/bin/env bash
# Update Docker Hub repository description and full_description (README) via Hub API.
# Requires a Personal Access Token with Read, Write, Delete (Hub uses this for repo admin edits).
#
# Usage (from repo root):
#   export DOCKERHUB_USERNAME=youruser
#   export DOCKERHUB_TOKEN=dckr_pat_...
#   ./scripts/dockerhub-update-repo.sh
#
# Optional env:
#   DOCKERHUB_NAMESPACE   (default: neurondb)
#   DOCKERHUB_REPOSITORY  (default: neurondb-cuda — canonical image neurondb/neurondb-cuda)
#   DOCKERHUB_DESCRIPTION — short summary for Hub UI (~100 chars; default below)
#   DOCKERHUB_README_PATH — markdown for full_description (see defaults below)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DOCKERHUB_NAMESPACE="${DOCKERHUB_NAMESPACE:-neurondb}"
DOCKERHUB_REPOSITORY="${DOCKERHUB_REPOSITORY:-neurondb-cuda}"
DEFAULT_HUB_README="$REPO_ROOT/docker/README.DockerHub.md"
ALT_NEURONDB_README="$REPO_ROOT/docker/README.DockerHub.neurondb.md"
if [[ -n "${DOCKERHUB_README_PATH:-}" ]]; then
	README_PATH="$DOCKERHUB_README_PATH"
elif [[ "${DOCKERHUB_REPOSITORY}" == "neurondb" && -f "$ALT_NEURONDB_README" ]]; then
	README_PATH="$ALT_NEURONDB_README"
elif [[ -f "$DEFAULT_HUB_README" ]]; then
	README_PATH="$DEFAULT_HUB_README"
else
	README_PATH="$REPO_ROOT/README.md"
fi

[[ -n "${DOCKERHUB_USERNAME:-}" ]] || {
	echo "Set DOCKERHUB_USERNAME." >&2
	exit 1
}
[[ -n "${DOCKERHUB_TOKEN:-}" ]] || {
	echo "Set DOCKERHUB_TOKEN (PAT with Read, Write, Delete)." >&2
	exit 1
}
[[ -f "$README_PATH" ]] || {
	echo "README not found: $README_PATH" >&2
	exit 1
}

# Docker Hub “short description” — keep concise; CUDA + PG majors + arch + GPU
SHORT_DESC="${DOCKERHUB_DESCRIPTION:-CUDA PostgreSQL + NeuronDB: vectors and ML in SQL. PG 16-18, linux/amd64. NVIDIA GPU required.}"

LOGIN_JSON="$(jq -n --arg u "$DOCKERHUB_USERNAME" --arg p "$DOCKERHUB_TOKEN" '{username:$u,password:$p}')"
LOGIN_RESP="$(curl -fsS -X POST 'https://hub.docker.com/v2/users/login/' \
	-H 'Content-Type: application/json' \
	-d "$LOGIN_JSON")"

JWT="$(echo "$LOGIN_RESP" | jq -r '.token // empty')"
if [[ -z "$JWT" ]]; then
	echo "Docker Hub login failed:" >&2
	echo "$LOGIN_RESP" | jq . 2>/dev/null || echo "$LOGIN_RESP" >&2
	exit 1
fi

REPO_URL="https://hub.docker.com/v2/repositories/${DOCKERHUB_NAMESPACE}/${DOCKERHUB_REPOSITORY}/"
PREFLIGHT_OUT="$(mktemp)"
trap 'rm -f "$PREFLIGHT_OUT"' EXIT
PREFLIGHT_CODE="$(curl -sS -o "$PREFLIGHT_OUT" -w '%{http_code}' "$REPO_URL" \
	-H 'Accept: application/json' \
	-H "Authorization: JWT ${JWT}")"

if [[ "$PREFLIGHT_CODE" == "404" ]]; then
	cat >&2 <<EOF
Docker Hub repository not found: ${DOCKERHUB_NAMESPACE}/${DOCKERHUB_REPOSITORY}

  1. Create it: https://hub.docker.com/repositories/create
     — namespace: ${DOCKERHUB_NAMESPACE} (you need Owner or Repo Admin on that org)
     — name: ${DOCKERHUB_REPOSITORY}
  2. Or point this script at an existing repo:
       export DOCKERHUB_NAMESPACE=your-org-or-user
       export DOCKERHUB_REPOSITORY=your-repo-name

API response: $(cat "$PREFLIGHT_OUT" 2>/dev/null || true)
EOF
	rm -f "$PREFLIGHT_OUT"
	exit 1
fi
if [[ "$PREFLIGHT_CODE" != "200" ]]; then
	echo "Cannot read repository (HTTP ${PREFLIGHT_CODE}):" >&2
	cat "$PREFLIGHT_OUT" >&2 || true
	exit 1
fi
rm -f "$PREFLIGHT_OUT"
trap - EXIT

FULL_JSON="$(jq -Rs . "$README_PATH")"
BODY="$(jq -n --arg desc "$SHORT_DESC" --argjson full "$FULL_JSON" '{description:$desc, full_description:$full}')"

PATCH_URL="$REPO_URL"
PATCH_OUT="$(mktemp)"
trap 'rm -f "$PATCH_OUT"' EXIT
HTTP_CODE="$(curl -sS -o "$PATCH_OUT" -w '%{http_code}' -X PATCH "$PATCH_URL" \
	-H 'Content-Type: application/json' \
	-H 'Accept: application/json' \
	-H "Authorization: JWT ${JWT}" \
	-d "$BODY")"

if [[ "$HTTP_CODE" != "200" ]]; then
	echo "PATCH failed (HTTP $HTTP_CODE):" >&2
	cat "$PATCH_OUT" >&2 || true
	if [[ "$HTTP_CODE" == "403" ]]; then
		echo >&2
		echo "403: PAT needs Read, Write, Delete on Docker Hub, and your account needs permission to edit this repo." >&2
	fi
	if [[ "$HTTP_CODE" == "404" ]]; then
		echo >&2
		echo "404 on PATCH but GET succeeded: try again, or report to Docker Hub support (intermittent API issue)." >&2
	fi
	exit 1
fi
echo "Updated Docker Hub repo ${DOCKERHUB_NAMESPACE}/${DOCKERHUB_REPOSITORY} (description + README)."

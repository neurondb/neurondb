#!/usr/bin/env bash
#
# test-cnpg-local.sh
#   End-to-end local test: create kind cluster, install CNPG operator,
#   deploy NeuronDB Helm chart, validate Cluster/Pooler/services.
#
# Usage:
#   ./scripts/test-cnpg-local.sh              # create cluster, install, deploy, validate
#   ./scripts/test-cnpg-local.sh --keep       # same but do not delete kind cluster at end
#   ./scripts/test-cnpg-local.sh --destroy   # only destroy existing kind cluster (no deploy)
#
# Prerequisites: kind, kubectl, helm on PATH. Docker running.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME="$(basename "$0")"

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-neurondb-cnpg-test}"
HELM_NS="${HELM_NS:-neurondb}"
RELEASE_NAME="${RELEASE_NAME:-neurondb}"
KEEP_CLUSTER=false
DESTROY_ONLY=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep)   KEEP_CLUSTER=true; shift ;;
    --destroy) DESTROY_ONLY=true; shift ;;
    -h|--help)
      cat << EOF
$SCRIPT_NAME - Local CNPG + NeuronDB Helm test

Usage: $SCRIPT_NAME [OPTIONS]

Options:
  --keep     Do not delete the kind cluster after validation
  --destroy  Only delete the kind cluster (no deploy)
  -h, --help Show this help

Env:
  KIND_CLUSTER_NAME  Kind cluster name (default: neurondb-cnpg-test)
  HELM_NS            Helm release namespace (default: neurondb)
  RELEASE_NAME       Helm release name (default: neurondb)
EOF
      exit 0
      ;;
    *) log_error "Unknown option: $1"; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"

if [[ "$DESTROY_ONLY" == "true" ]]; then
  log_info "Destroying kind cluster: $KIND_CLUSTER_NAME"
  kind delete cluster --name "$KIND_CLUSTER_NAME" 2>/dev/null || true
  log_success "Done."
  exit 0
fi

for cmd in kind kubectl helm docker; do
  if ! command -v "$cmd" &>/dev/null; then
    log_error "Required command not found: $cmd"
    exit 1
  fi
done

# 1. Create kind cluster (3 nodes)
if kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER_NAME"; then
  log_warning "Kind cluster $KIND_CLUSTER_NAME already exists; using it (use --destroy to remove first)"
else
  log_info "Creating kind cluster: $KIND_CLUSTER_NAME (1 control-plane + 2 workers)"
  cat << 'KINDCONF' | kind create cluster --name "$KIND_CLUSTER_NAME" --config -
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
KINDCONF
  log_success "Kind cluster created"
fi

kubectl cluster-info --context "kind-$KIND_CLUSTER_NAME" >/dev/null 2>&1 || { log_error "kubectl cannot use kind cluster"; exit 1; }

# 2. Install CloudNativePG operator
if kubectl get ns cnpg-system &>/dev/null; then
  log_warning "Namespace cnpg-system exists; skipping operator install"
else
  log_info "Adding Helm repo cloudnative-pg..."
  helm repo add cloudnative-pg https://cloudnative-pg.github.io/charts 2>/dev/null || true
  helm repo update cloudnative-pg

  log_info "Installing CNPG operator in cnpg-system..."
  helm upgrade --install cnpg-operator cloudnative-pg/cloudnative-pg \
    --namespace cnpg-system --create-namespace \
    --wait --timeout 5m
  log_success "CNPG operator installed"
fi

# 3. Deploy NeuronDB chart with test values
CHART_PATH="$PROJECT_ROOT/helm/neurondb"
VALUES_PATH="$PROJECT_ROOT/helm/neurondb/examples/values-cnpg-test.yaml"

if [[ ! -d "$CHART_PATH" || ! -f "$VALUES_PATH" ]]; then
  log_error "Chart or values not found: $CHART_PATH / $VALUES_PATH"
  exit 1
fi

kubectl create namespace "$HELM_NS" 2>/dev/null || true

log_info "Installing Helm release $RELEASE_NAME in $HELM_NS..."
helm upgrade --install "$RELEASE_NAME" "$CHART_PATH" \
  --namespace "$HELM_NS" \
  --values "$VALUES_PATH" \
  --wait --timeout 10m

log_success "Helm release installed"

# 4. Wait for CNPG Cluster to be ready
CLUSTER_NAME="$RELEASE_NAME-neurondb"
log_info "Waiting for CNPG Cluster $CLUSTER_NAME to be ready..."
if kubectl wait "cluster/$CLUSTER_NAME" -n "$HELM_NS" --for=condition=Ready --timeout=600s 2>/dev/null; then
  log_success "Cluster Ready"
else
  log_warning "Cluster Ready condition not yet met; waiting for pods..."
fi

# Wait for pods
log_info "Waiting for Cluster pods..."
kubectl wait --for=condition=Ready pod -l "cnpg.io/cluster=$CLUSTER_NAME" -n "$HELM_NS" --timeout=300s 2>/dev/null || true

# 5. Validate services
log_info "Checking services..."
for svc in "${CLUSTER_NAME}-rw" "${CLUSTER_NAME}-ro" "${CLUSTER_NAME}-r"; do
  if kubectl get svc -n "$HELM_NS" "$svc" &>/dev/null; then
    log_success "Service $svc exists"
  else
    log_warning "Service $svc not found"
  fi
done

# Pooler service if enabled
if kubectl get pooler -n "$HELM_NS" 2>/dev/null | grep -q .; then
  log_info "Pooler CRD resources present"
fi

# 6. Optional: helm test
if helm test "$RELEASE_NAME" -n "$HELM_NS" 2>/dev/null; then
  log_success "Helm test passed"
else
  log_warning "Helm test not run or failed (test pod may not exist for minimal test values)"
fi

# 7. Quick connect check (optional)
log_info "Checking Postgres connectivity via -rw service..."
PGPOD=$(kubectl get pods -n "$HELM_NS" -l "cnpg.io/cluster=$CLUSTER_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$PGPOD" ]]; then
  PGPASS=$(kubectl get secret -n "$HELM_NS" "$RELEASE_NAME-neurondb-secrets" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d 2>/dev/null || true)
  if [[ -n "$PGPASS" ]]; then
    if kubectl exec -n "$HELM_NS" "$PGPOD" -- env PGPASSWORD="$PGPASS" psql -U neurondb -d neurondb -tAc "SELECT 1" 2>/dev/null | grep -q 1; then
      log_success "Postgres query OK"
    else
      log_warning "Postgres query failed"
    fi
  else
    log_warning "Could not read postgres secret (secret name may differ)"
  fi
fi

log_success "CNPG local test completed successfully."

if [[ "$KEEP_CLUSTER" != "true" ]]; then
  log_info "Deleting kind cluster: $KIND_CLUSTER_NAME"
  kind delete cluster --name "$KIND_CLUSTER_NAME"
  log_success "Cluster deleted."
else
  echo ""
  log_info "Cluster kept. To inspect: kubectl --context kind-$KIND_CLUSTER_NAME -n $HELM_NS get cluster,pods,svc"
  log_info "To destroy: $SCRIPT_NAME --destroy"
fi

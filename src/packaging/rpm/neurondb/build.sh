#!/bin/bash
#
# Build RPM package for NeuronDB PostgreSQL extension
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neurondb-{VERSION}-1.x86_64.rpm

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-2.0.0}
RELEASE=${RELEASE:-1}
ARCH=${ARCH:-x86_64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronDB RPM package version $VERSION"

# Check prerequisites
command -v rpmbuild >/dev/null 2>&1 || { echo "Error: rpmbuild not found. Install rpm-build." >&2; exit 1; }

# Create RPM build directories
RPM_DIR="$BUILD_DIR/rpm"
mkdir -p "$RPM_DIR"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

# Build NeuronDB extension
cd "$REPO_ROOT/NeuronDB"
make clean || true
GPU_BACKENDS=none make
GPU_BACKENDS=none make DESTDIR="$RPM_DIR/BUILDROOT/neurondb-${VERSION}-${RELEASE}.${ARCH}" install

# Create spec file
SPEC_FILE="$RPM_DIR/SPECS/neurondb.spec"
cat > "$SPEC_FILE" <<EOF
Name:           neurondb
Version:        ${VERSION}
Release:        ${RELEASE}%{?dist}
Summary:        PostgreSQL extension for vector search, ML algorithms, and RAG capabilities
License:        Proprietary
URL:            https://github.com/neurondb/neurondb
Source0:        %{name}-%{version}.tar.gz

%description
NeuronDB extends PostgreSQL with:
- Vector similarity search (HNSW, IVF indexes)
- 52+ ML algorithms (classification, regression, clustering)
- GPU acceleration (CUDA, ROCm, Metal)
- Embedding generation and RAG pipelines
- Hybrid search (vector + full-text)

%prep
%setup -q

%build
# Extension already built

%install
# Files installed to BUILDROOT

%files
%defattr(-,root,root,-)
/usr/lib/postgresql/*
/usr/share/postgresql/*
/usr/share/doc/neurondb/*

%changelog
* $(date +"%a %b %d %Y") NeuronDB <admin@neurondb.com> - ${VERSION}-${RELEASE}
- Initial package
EOF

# Build RPM
rpmbuild --define "_topdir $RPM_DIR" --define "_builddir $RPM_DIR/BUILD" \
    --define "_rpmdir $RPM_DIR/RPMS" -bb "$SPEC_FILE"

# Copy RPM to output directory
RPM_FILE=$(find "$RPM_DIR/RPMS" -name "*.rpm" | head -1)
cp "$RPM_FILE" "$REPO_ROOT/packaging/rpm/neurondb/"

echo "Package built: $(basename $RPM_FILE)"
ls -lh "$REPO_ROOT/packaging/rpm/neurondb/$(basename $RPM_FILE)"



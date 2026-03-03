#!/bin/bash
#
# Build DEB package for NeuronDB PostgreSQL extension
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neurondb_{VERSION}_amd64.deb

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-2.0.0}
# Strip 'v' prefix if present (Debian packages require versions to start with digit)
VERSION=${VERSION#v}
ARCH=${ARCH:-amd64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronDB DEB package version $VERSION"

# Check prerequisites
command -v dpkg-deb >/dev/null 2>&1 || { echo "Error: dpkg-deb not found. Install dpkg-dev." >&2; exit 1; }
command -v fakeroot >/dev/null 2>&1 || { echo "Error: fakeroot not found. Install fakeroot." >&2; exit 1; }

# Create package structure
PKG_DIR="$BUILD_DIR/neurondb_${VERSION}"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/lib/postgresql"
mkdir -p "$PKG_DIR/usr/share/postgresql"
mkdir -p "$PKG_DIR/usr/share/doc/neurondb"

# Build NeuronDB extension
cd "$REPO_ROOT/NeuronDB"
ONNX_RUNTIME_PATH="${ONNX_RUNTIME_PATH:-/usr/local/onnxruntime}" GPU_BACKENDS=none ./build.sh
make DESTDIR="$PKG_DIR" install

# Create control file
cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: neurondb
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: neurondb <admin@neurondb.com>
Description: PostgreSQL extension for vector search, ML algorithms, and RAG capabilities
 NeuronDB extends PostgreSQL with:
  - Vector similarity search (HNSW, IVF indexes)
  - 52+ ML algorithms (classification, regression, clustering)
  - GPU acceleration (CUDA, ROCm, Metal)
  - Embedding generation and RAG pipelines
  - Hybrid search (vector + full-text)
Depends: postgresql-common, libc6 (>= 2.17)
Section: database
Priority: optional
Homepage: https://github.com/neurondb/neurondb
EOF

# Create postinst script
cat > "$PKG_DIR/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
# PostgreSQL extension files are installed to versioned directories
# Users need to CREATE EXTENSION in their databases
echo "NeuronDB extension installed. Run 'CREATE EXTENSION neurondb;' in your database."
EOF
chmod +x "$PKG_DIR/DEBIAN/postinst"

# Create copyright file
cat > "$PKG_DIR/usr/share/doc/neurondb/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: NeuronDB
Source: https://github.com/neurondb/neurondb

Files: *
Copyright: 2024 NeuronDB
License: Proprietary
EOF

# Build DEB package
DEB_FILE="neurondb_${VERSION}_${ARCH}.deb"
fakeroot dpkg-deb --build "$PKG_DIR" "$REPO_ROOT/packaging/deb/neurondb/$DEB_FILE"

echo "Package built: $DEB_FILE"
ls -lh "$REPO_ROOT/packaging/deb/neurondb/$DEB_FILE"


# syntax=docker/dockerfile:1.6
#
# NeuronDB reference image (Metal GPU - Apple Silicon)
# Compiles the extension with Metal Performance Shaders support.
#
# Supports:
#   - PostgreSQL: 16, 17, 18 (via PG_MAJOR build arg)
#   - Architecture: arm64 (Apple Silicon only)
#   - OS: macOS (via custom base or buildx)
#
# Note: Metal support requires macOS host system. This Dockerfile is designed
# for use with Docker Desktop on macOS with Apple Silicon, or for building
# natively on macOS systems.
#
# Build args:
#   PG_MAJOR: PostgreSQL major version (16, 17, or 18, default: 17)
#   ONNX_VERSION: ONNX Runtime version (default: 1.17.0)
#
# Usage:
#   On macOS with Docker Desktop:
#     docker buildx build --platform linux/arm64 -f docker/Dockerfile.gpu.metal \
#       --build-arg PG_MAJOR=17 -t neurondb:17-metal .
#
#   For native macOS builds (outside Docker):
#     Use the standard build.sh script with GPU_MODE=metal

ARG PG_MAJOR=17
ARG ONNX_VERSION=1.17.0

# Use a base image that supports macOS/arm64 builds
# For Docker Desktop on macOS, we use a Debian base and install PostgreSQL
FROM debian:bookworm-slim AS builder

ARG PG_MAJOR
ARG ONNX_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PG_MAJOR=${PG_MAJOR} \
    ONNX_VERSION=${ONNX_VERSION} \
    ONNX_PATH=/usr/local/onnxruntime \
    PG_CONFIG=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        clang \
        cmake \
        curl \
        git \
        gnupg \
        libcurl4-openssl-dev \
        libssl-dev \
        pkg-config \
        python3 \
        python3-pip \
        software-properties-common \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Install PostgreSQL client/server headers
RUN install -d /usr/share/postgresql-common/pgdg && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.gpg && \
    sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.gpg] http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
    apt-get update && apt-get install -y --no-install-recommends \
        postgresql-${PG_MAJOR} \
        postgresql-server-dev-${PG_MAJOR} \
        postgresql-client-${PG_MAJOR} && \
    rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime (CPU build for Metal - Metal support is handled at runtime)
# Architecture-aware installation for arm64
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        arm64) onnx_pkg="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz" ;; \
        amd64) onnx_pkg="onnxruntime-linux-x64-${ONNX_VERSION}.tgz" ;; \
        *) echo "Unsupported architecture for ONNX Runtime: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${onnx_pkg}" -o /tmp/onnxruntime.tgz; \
    mkdir -p /tmp/onnxruntime; \
    tar -xzf /tmp/onnxruntime.tgz -C /tmp/onnxruntime --strip-components=1; \
    mv /tmp/onnxruntime ${ONNX_PATH}; \
    rm -f /tmp/onnxruntime.tgz

WORKDIR /build/Neurondb

COPY . .

# Build NeuronDB with Metal support
# Note: Metal framework is available on macOS host, but for Docker builds
# we compile with Metal support flags. Actual Metal runtime requires macOS.
# Generate configuration header first
RUN cd NeuronDB && \
    mkdir -p include && \
    cat > include/neurondb_config.h <<'EOF'
/*-------------------------------------------------------------------------
 * neurondb_config.h - Auto-generated configuration header
 * Generated for Docker Metal build
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_CONFIG_H
#define NEURONDB_CONFIG_H

#define NEURONDB_VERSION "1.0"
#define NEURONDB_BUILD_DATE "20260105"
#define NEURONDB_PLATFORM_MACOS 1

/* GPU support - Metal */
#define NDB_GPU_METAL 1

/* ONNX Runtime support */
#define HAVE_ONNX_RUNTIME 1
#define ONNX_RUNTIME_PATH "/usr/local/onnxruntime"

/* CPU fallback */
#define NEURONDB_CPU_FALLBACK 1

#endif /* NEURONDB_CONFIG_H */
EOF

# Build and install
RUN cd NeuronDB && \
    make clean && \
    make GPU_BACKENDS=metal ONNX_PATH=${ONNX_PATH} && \
    make install

# -----------------------------------------------------------------------------

FROM postgres:${PG_MAJOR}-bookworm

ARG PG_MAJOR
ARG ONNX_VERSION

ENV ONNX_PATH=/usr/local/onnxruntime \
    NEURONDB_HOME=/usr/local/share/neurondb \
    LANG=C.UTF-8

RUN mkdir -p ${NEURONDB_HOME} /docker-entrypoint-initdb.d

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        libssl3 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy runtime assets from builder
COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/neurondb*.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/neurondb* /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=builder /usr/local/onnxruntime ${ONNX_PATH}

COPY docker/neurondb/docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/

# Ship custom entrypoint wrapper (ensures extension exists after PostgreSQL starts)
COPY docker/neurondb/docker-entrypoint-neurondb.sh /usr/local/bin/docker-entrypoint-neurondb.sh
RUN chmod +x /usr/local/bin/docker-entrypoint-neurondb.sh && \
    chown root:root /usr/local/bin/docker-entrypoint-neurondb.sh

# Copy NeuronMCP SQL files for auto-setup
COPY NeuronMCP/sql/ /docker-entrypoint-initdb.d/neurondb_mcp/

# Copy NeuronAgent SQL files for auto-setup
COPY NeuronAgent/sql/ /docker-entrypoint-initdb.d/neurondb_agent/

VOLUME ["/var/lib/postgresql/data"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD pg_isready -U "${POSTGRES_USER:-postgres}"

# Default entrypoint inherited from postgres image
#
# Note: For full Metal GPU support, this container should be run on macOS
# with Docker Desktop, and Metal frameworks will be available through
# the host system integration.


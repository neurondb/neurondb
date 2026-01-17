# syntax=docker/dockerfile:1.6
#
# NeuronDB reference image (CPU)
# Multi-stage build that compiles the extension against PostgreSQL and bundles
# ONNX Runtime for HuggingFace integration.
#
# Supports:
#   - PostgreSQL: 16, 17, 18 (via PG_MAJOR build arg)
#   - Architectures: amd64, arm64
#   - OS: Debian Bookworm (via postgres base image)
#
# Build args:
#   PG_MAJOR: PostgreSQL major version (16, 17, or 18, default: 17)
#   ONNX_VERSION: ONNX Runtime version (default: 1.17.0)
#   VERSION: Application version (used in labels)
#   BUILD_DATE: Build date (ISO 8601 format)
#   VCS_REF: Git commit hash or version control reference

ARG PG_MAJOR=17
ARG ONNX_VERSION=1.17.0
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

FROM postgres:${PG_MAJOR}-bookworm AS builder

ARG PG_MAJOR
ARG ONNX_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PG_MAJOR=${PG_MAJOR} \
    ONNX_VERSION=${ONNX_VERSION} \
    ONNX_PATH=/usr/local/onnxruntime \
    PG_CONFIG=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        make \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        libssl-dev \
        libcurl4-openssl-dev \
        zlib1g-dev \
        python3 \
        python3-pip \
        postgresql-server-dev-${PG_MAJOR} && \
    rm -rf /var/lib/apt/lists/*

# Install ML libraries via pip (provides C libraries with headers)
RUN pip3 install --no-cache-dir --break-system-packages \
    xgboost \
    lightgbm \
    catboost

# Install ONNX Runtime (CPU build) matching architecture
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) onnx_pkg="onnxruntime-linux-x64-${ONNX_VERSION}.tgz" ;; \
        arm64) onnx_pkg="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz" ;; \
        *) echo "Unsupported architecture for ONNX Runtime: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${onnx_pkg}" -o /tmp/onnxruntime.tgz; \
    mkdir -p /tmp/onnxruntime; \
    tar -xzf /tmp/onnxruntime.tgz -C /tmp/onnxruntime --strip-components=1; \
    mv /tmp/onnxruntime ${ONNX_PATH}; \
    rm -f /tmp/onnxruntime.tgz

WORKDIR /build/Neurondb

COPY . .

# Build and install ONLY the NeuronDB PostgreSQL extension into the builder image.
# (Do not call the repo-root Makefile here: it builds the whole ecosystem and
# requires Go/Node which are intentionally not part of this image.)
# Generate configuration header before building
RUN cd NeuronDB && \
    mkdir -p include && \
    cat > include/neurondb_config.h <<'EOF'
/*-------------------------------------------------------------------------
 * neurondb_config.h - Auto-generated configuration header
 * Generated for Docker build
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_CONFIG_H
#define NEURONDB_CONFIG_H

#define NEURONDB_VERSION "1.0"
#define NEURONDB_BUILD_DATE "20260105"
#define NEURONDB_PLATFORM_LINUX 1

/* CPU-only build */
#define NEURONDB_CPU_FALLBACK 1

/* ONNX Runtime support */
#define HAVE_ONNX_RUNTIME 1
#define ONNX_RUNTIME_PATH "/usr/local/onnxruntime"

#endif /* NEURONDB_CONFIG_H */
EOF

# Build and install the extension
RUN cd NeuronDB && \
    GPU_BACKENDS=none make clean && \
    GPU_BACKENDS=none make && \
    # Ensure SQL file exists before installation (Makefile expects neurondb--1.0.sql)
    if [ ! -f neurondb--1.0.sql ]; then \
        if [ -f neurondb--1.0.sql.linux ]; then \
            cp neurondb--1.0.sql.linux neurondb--1.0.sql; \
        elif [ -f neurondb--2.0.sql ]; then \
            cp neurondb--2.0.sql neurondb--1.0.sql; \
        elif [ -f neurondb--2.0.sql.linux ]; then \
            cp neurondb--2.0.sql.linux neurondb--1.0.sql; \
        else \
            echo "Warning: No SQL file found, attempting to generate..."; \
            make neurondb--1.0.sql || echo "Could not generate SQL file"; \
        fi; \
    fi && \
    GPU_BACKENDS=none make install && \
    # After installation, check if the control file specifies version 2.0
    # and if neurondb--2.0.sql is missing, create it from neurondb--1.0.sql
    SHAREDIR=$(/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config --sharedir) && \
    DEFAULT_VERSION=$(grep -E "^default_version\s*=" "$SHAREDIR/extension/neurondb.control" | sed 's/.*=\s*\(.*\)/\1/' | tr -d "'\" " || echo "1.0") && \
    if [ "$DEFAULT_VERSION" = "2.0" ] && [ -f "$SHAREDIR/extension/neurondb--1.0.sql" ] && [ ! -f "$SHAREDIR/extension/neurondb--2.0.sql" ]; then \
      echo "Control file specifies version 2.0, creating neurondb--2.0.sql from 1.0..." && \
      cp "$SHAREDIR/extension/neurondb--1.0.sql" "$SHAREDIR/extension/neurondb--2.0.sql"; \
    fi

# -----------------------------------------------------------------------------

FROM postgres:${PG_MAJOR}-bookworm

ARG PG_MAJOR
ARG ONNX_VERSION
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Add metadata labels
LABEL org.opencontainers.image.title="NeuronDB" \
      org.opencontainers.image.description="PostgreSQL extension for vector search, ML algorithms, and RAG capabilities" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="neurondb" \
      org.opencontainers.image.licenses="Proprietary" \
      org.opencontainers.image.source="https://github.com/neurondb/NeurondB" \
      org.opencontainers.image.documentation="https://neurondb.ai/docs" \
      maintainer="neurondb <admin@neurondb.com>" \
      neurondb.postgresql.version="${PG_MAJOR}" \
      neurondb.onnx.version="${ONNX_VERSION}"

ENV ONNX_PATH=/usr/local/onnxruntime \
    NEURONDB_HOME=/usr/local/share/neurondb \
    LANG=C.UTF-8

RUN mkdir -p ${NEURONDB_HOME} /docker-entrypoint-initdb.d

# Runtime dependencies required by neurondb.so and ML libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        libssl3 \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy Python site-packages with ML libraries from builder
COPY --from=builder /usr/local/lib/python3.*/dist-packages/ /usr/local/lib/python3.11/dist-packages/
RUN ldconfig

# Copy runtime assets from builder
COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/neurondb*.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/neurondb* /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=builder /usr/local/onnxruntime ${ONNX_PATH}

# Ship helper init scripts
COPY dockers/neurondb/docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/

# Ship custom entrypoint wrapper (ensures extension exists after PostgreSQL starts)
COPY dockers/neurondb/docker-entrypoint-neurondb.sh /usr/local/bin/docker-entrypoint-neurondb.sh
RUN chmod +x /usr/local/bin/docker-entrypoint-neurondb.sh && \
    chown root:root /usr/local/bin/docker-entrypoint-neurondb.sh

# Copy NeuronMCP SQL files for auto-setup
RUN mkdir -p /docker-entrypoint-initdb.d/neurondb_mcp
COPY NeuronMCP/neuron-mcp.sql /docker-entrypoint-initdb.d/neurondb_mcp/neuron-mcp.sql

# Copy NeuronAgent migration files for auto-setup
COPY NeuronAgent/sql/ /docker-entrypoint-initdb.d/neurondb_agent/

# Volume declaration removed - let base postgres image handle it
# docker-compose.yml mounts the volume at the correct path for each PG version

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD pg_isready -U "${POSTGRES_USER:-postgres}"

# Use standard postgres entrypoint - init scripts handle extension creation
# The test script will also ensure extension exists if needed


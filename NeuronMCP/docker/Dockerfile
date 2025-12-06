# Multi-stage build for NeuronMCP
FROM golang:1.23-bookworm AS builder

WORKDIR /build

# Copy go mod files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o neurondb-mcp ./cmd/neurondb-mcp

# Final stage - use Debian slim for better compatibility
FROM debian:bookworm-slim

# Install runtime dependencies including Python3 for dataset loading
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    python3 \
    python3-pip \
    python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for dataset loading
RUN pip3 install --no-cache-dir --break-system-packages \
    datasets \
    psycopg2-binary

# Create non-root user for security with home directory
RUN groupadd -r neuronmcp && useradd -r -g neuronmcp -u 1000 -m -d /home/neuronmcp neuronmcp && \
    chmod 755 /home/neuronmcp && \
    chown -R neuronmcp:neuronmcp /home/neuronmcp && \
    mkdir -p /tmp/hf_cache /tmp/hf_cache/datasets && \
    chown -R neuronmcp:neuronmcp /tmp/hf_cache

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/neurondb-mcp .

# Change ownership to non-root user
RUN chown -R neuronmcp:neuronmcp /app

# Switch to non-root user
USER neuronmcp

# MCP protocol uses stdio, so we don't expose ports
# Keep stdin/stdout/stderr available for MCP communication

# Set entrypoint to ensure stdio is properly configured
ENTRYPOINT ["./neurondb-mcp"]


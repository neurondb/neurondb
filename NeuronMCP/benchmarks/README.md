# NeuronMCP Performance Benchmarks

This directory contains performance benchmarking tools and test suites for NeuronMCP.

## Overview

The benchmarking infrastructure provides:
- Tool call latency measurement (p50, p95, p99)
- Throughput testing (requests/second)
- Memory usage profiling
- Database connection pool efficiency testing
- Concurrent request handling tests
- Comparison with other MCP servers

## Structure

```
benchmarks/
├── README.md              # This file
├── latency/               # Latency benchmarks
├── throughput/           # Throughput benchmarks
├── memory/               # Memory profiling
├── concurrent/           # Concurrent request tests
├── comparison/           # Comparison with other MCP servers
└── tools/                # Benchmarking utilities
```

## Running Benchmarks

### Prerequisites

- NeuronMCP server running
- PostgreSQL with NeuronDB extension
- Go 1.23+

### Quick Start

```bash
# Run all benchmarks
go run benchmarks/main.go

# Run specific benchmark suite
go run benchmarks/latency/main.go
go run benchmarks/throughput/main.go
```

## Performance Targets

- **p95 latency**: < 10ms for simple tools
- **Throughput**: 10,000+ requests/second
- **Concurrent connections**: 10,000+
- **Memory usage**: < 500MB for 1000 concurrent connections

## Results

Benchmark results are stored in `benchmarks/results/` directory in JSON format.





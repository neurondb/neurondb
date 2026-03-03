# NeuronDB Build System

**Complete build system documentation for NeuronDB ecosystem.**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Makefile Structure](#makefile-structure)
- [Build Targets](#build-targets)
- [Platform-Specific Builds](#platform-specific-builds)
- [GPU Backend Compilation](#gpu-backend-compilation)
- [Dependency Management](#dependency-management)
- [Testing Infrastructure](#testing-infrastructure)

---

## Makefile Structure

### Main Makefile

**Location:** `/Makefile`

**Targets:**
- `all`: Build all components
- `neurondb`: Build NeuronDB extension
- `neuronagent`: Build NeuronAgent
- `neuronmcp`: Build NeuronMCP
- `neurondesktop`: Build NeuronDesktop
- `test`: Run tests
- `clean`: Clean build artifacts

---

## Build Targets

### NeuronDB Extension

**Build:**
```bash
cd NeuronDB
make
```

**Install:**
```bash
make install
```

**Test:**
```bash
make installcheck
```

### NeuronAgent

**Build:**
```bash
cd NeuronAgent
make build
```

**Run:**
```bash
make run
```

### NeuronMCP

**Build:**
```bash
cd NeuronMCP
make build
```

### NeuronDesktop

**Build:**
```bash
cd NeuronDesktop
npm install
npm run build
```

---

## Platform-Specific Builds

### macOS

**Requirements:**
- Xcode Command Line Tools
- PostgreSQL development headers

**Build:**
```bash
make PG_CONFIG=/usr/local/pgsql/bin/pg_config
```

### Linux

**Requirements:**
- gcc, make, cmake
- PostgreSQL development headers

**Build:**
```bash
make PG_CONFIG=/usr/pgsql-17/bin/pg_config
```

---

## GPU Backend Compilation

### CUDA

**Requirements:**
- CUDA Toolkit 12.2+
- cuDNN

**Build:**
```bash
make CUDA=1
```

### ROCm

**Requirements:**
- ROCm 5.7+

**Build:**
```bash
make ROCm=1
```

### Metal

**Requirements:**
- Apple Silicon
- macOS 13+

**Build:**
```bash
make Metal=1
```

---

## Dependency Management

### C Dependencies

**PostgreSQL:**
- Version: 16, 17, or 18
- Headers: `postgres.h`, `fmgr.h`

**ONNX Runtime:**
- Optional dependency
- Version: 1.17.0+

### Go Dependencies

**neuron-agent / neuron-mcp:** See those repositories for build and packaging.
- Go modules
- `go.mod` and `go.sum`

### Node.js Dependencies

**NeuronDesktop:**
- npm/yarn
- `package.json`

---

## Testing Infrastructure

### SQL Tests

**Location:** `NeuronDB/tests/sql/`

**Run:**
```bash
make installcheck
```

### Go Tests

**NeuronAgent:**
```bash
cd NeuronAgent
go test ./...
```

**NeuronMCP:**
```bash
cd NeuronMCP
go test ./...
```

### Frontend Tests

**NeuronDesktop:**
```bash
cd NeuronDesktop
npm test
```

---

## Related Documentation

- [Development Guide](development-guide.md)
- [Contributing Guide](../../CONTRIBUTING.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0




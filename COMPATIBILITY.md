# NeuronDB Compatibility Matrix

This document provides explicit compatibility information for NeuronDB components across PostgreSQL versions, operating systems, and GPU backends.

## PostgreSQL Version Compatibility

### Tested Versions

| PostgreSQL Version | Status | CI Tested | Security Updates | Notes |
|-------------------|--------|-----------|------------------|-------|
| 16.x | ✅ Supported | ✅ Yes (latest 16.x) | ✅ Yes | Full feature support, tested with latest 16.x in CI |
| 17.x | ✅ Supported | ✅ Yes (latest 17.x) | ✅ Yes | Full feature support, tested with latest 17.x in CI |
| 18.x | ✅ Supported | ✅ Yes (latest 18.x) | ✅ Yes | Full feature support, tested with latest 18.x in CI |

**CI Testing Details:**
- CI tests against latest minor version of each major release (e.g., `postgres:16-bookworm`, `postgres:17-bookworm`, `postgres:18-bookworm`)
- All minor versions within a major release are expected to work (no breaking changes in PostgreSQL minor releases)
- Specific minor versions are not individually tested in CI, but compatibility is maintained across all minor versions

### Support Policy

- **Current Release:** Full support with security updates and bug fixes
- **Previous Minor Release:** Security updates only (for 6 months after current release)
- **Older Versions:** No support, upgrade recommended
- **Pre-release/Beta:** No security support, use at own risk

### Version-Specific Notes

#### PostgreSQL 16.x
- Full feature support
- PL/pgSQL fallbacks for macOS dylib loader issues
- All vector and ML operations supported

#### PostgreSQL 17.x
- Full feature support
- PL/pgSQL fallbacks for macOS dylib loader issues
- Performance improvements in vector operations

#### PostgreSQL 18.x
- Full feature support
- Native C functions (dylib loader fixed in PG 18)
- Best performance characteristics

### Known Issues

| PostgreSQL Version | Issue | Workaround | Status |
|-------------------|-------|------------|--------|
| 16.x (macOS) | dylib loader limitations | Uses PL/pgSQL fallbacks | ✅ Resolved |
| 17.x (macOS) | dylib loader limitations | Uses PL/pgSQL fallbacks | ✅ Resolved |
| 18.x | None known | N/A | ✅ Clean |

## Operating System Compatibility

### Ubuntu

| Version | Status | Architecture | CI Tested | Notes |
|---------|--------|--------------|-----------|-------|
| 20.04 LTS | ✅ Supported | amd64, arm64 | ✅ Yes (ubuntu-20.04) | Full support |
| 22.04 LTS | ✅ Supported | amd64, arm64 | ✅ Yes (ubuntu-22.04) | Recommended, primary CI target |
| 24.04 LTS | ✅ Supported | amd64, arm64 | ⚠️ Partial | Latest LTS, tested when available |

**Build Tools:**
- GCC 9, 10, 11, 12 (supported; GCC 9, 11, 12 tested in CI)
- Clang 12, 13, 14, 15 (supported; Clang 14, 15 tested in CI)
- Make 4.2.1+

### Debian

| Version | Status | Architecture | CI Tested | Notes |
|---------|--------|--------------|-----------|-------|
| 11 (Bullseye) | ✅ Supported | amd64, arm64 | ⚠️ Partial | Full support, tested when available |
| 12 (Bookworm) | ✅ Supported | amd64, arm64 | ✅ Yes | Recommended, used in Docker images |

**Build Tools:**
- GCC 10, 11, 12 (supported; GCC 10, 12 tested in CI)
- Clang 12, 13, 14 (supported; tested when available)

### RHEL Family

| Distribution | Version | Status | Architecture | CI Tested | Notes |
|-------------|---------|--------|--------------|-----------|-------|
| Rocky Linux | 8 | ✅ Supported | amd64 | ⚠️ Manual | Full support, manual testing |
| Rocky Linux | 9 | ✅ Supported | amd64 | ⚠️ Manual | Recommended, manual testing |
| RHEL | 8 | ✅ Supported | amd64 | ⚠️ Manual | Full support, manual testing |
| RHEL | 9 | ✅ Supported | amd64 | ⚠️ Manual | Recommended, manual testing |
| CentOS Stream | 8 | ✅ Supported | amd64 | ⚠️ Manual | Full support, manual testing |
| CentOS Stream | 9 | ✅ Supported | amd64 | ⚠️ Manual | Recommended, manual testing |

**Build Tools:**
- GCC 8.5, 9, 10, 11 (supported)
- Clang 12, 13, 14 (supported)
- **Note:** RHEL-family distributions are supported but not automatically tested in CI. Manual testing recommended.

### macOS

| Version | Status | Architecture | CI Tested | Notes |
|---------|--------|--------------|-----------|-------|
| 11 (Big Sur) | ✅ Supported | Intel, Apple Silicon | ⚠️ Partial | Full support, tested when available |
| 12 (Monterey) | ✅ Supported | Intel, Apple Silicon | ⚠️ Partial | Full support, tested when available |
| 13 (Ventura) | ✅ Supported | Intel, Apple Silicon | ✅ Yes (macos-13) | Recommended, tested in CI |
| 14 (Sonoma) | ✅ Supported | Intel, Apple Silicon | ✅ Yes (macos-14) | Latest, tested in CI |

**Build Tools:**
- Clang 12, 13, 14, 15 (supported, tested in CI)
- Xcode Command Line Tools required
- Homebrew recommended for dependencies

**Architecture Notes:**
- **Intel (x86_64):** Full support, all features
- **Apple Silicon (arm64):** Full support, Metal GPU acceleration available

## GPU Backend Compatibility

### CUDA (NVIDIA)

| CUDA Version | Driver Version | Status | CI Tested | Features |
|--------------|----------------|--------|-----------|----------|
| 11.8 | 520.61.05+ | ✅ Supported | ⚠️ Manual | Full GPU acceleration, manual testing |
| 12.0 | 525.60.13+ | ✅ Supported | ⚠️ Manual | Full GPU acceleration, manual testing |
| 12.1 | 530.30.02+ | ✅ Supported | ⚠️ Manual | Full GPU acceleration, manual testing |
| 12.4 | 550.54.15+ | ✅ Supported | ⚠️ Manual | Recommended, latest, manual testing |

**Note:** CUDA builds are supported but not automatically tested in CI due to GPU hardware requirements. Manual testing recommended.

**Hardware Requirements:**
- NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- Minimum 4GB VRAM for vector operations
- 8GB+ VRAM recommended for large datasets

**Features:**
- Vector distance computation (L2, cosine, inner product)
- Vector quantization (int8, fp16, binary)
- ML training acceleration (selected algorithms)
- Index building acceleration

### ROCm (AMD)

| ROCm Version | Driver Version | Status | CI Tested | Features |
|--------------|----------------|--------|-----------|----------|
| 5.6 | 22.40+ | ✅ Supported | ⚠️ Manual | Full GPU acceleration, manual testing |
| 5.7 | 22.50+ | ✅ Supported | ⚠️ Manual | Full GPU acceleration, manual testing |
| 6.0 | 23.10+ | ✅ Supported | ⚠️ Manual | Recommended, latest, manual testing |

**Note:** ROCm builds are supported but not automatically tested in CI due to GPU hardware requirements. Manual testing recommended.

**Hardware Requirements:**
- AMD GPU with RDNA2, RDNA3, or CDNA architecture
- Minimum 4GB VRAM for vector operations
- 8GB+ VRAM recommended for large datasets

**Features:**
- Vector distance computation (L2, cosine, inner product)
- Vector quantization (int8, fp16, binary)
- ML training acceleration (selected algorithms)

**Known Limitations:**
- Some advanced ML algorithms may have limited ROCm support
- Check algorithm-specific documentation for ROCm compatibility

### Metal (Apple Silicon)

| macOS Version | Metal Version | Status | CI Tested | Features |
|---------------|---------------|--------|-----------|----------|
| 12+ | Metal 2.0+ | ✅ Supported | ⚠️ Partial | Full GPU acceleration, tested when available |
| 13+ | Metal 3.0+ | ✅ Supported | ✅ Yes (macos-13) | Enhanced features, tested in CI |
| 14+ | Metal 3.0+ | ✅ Supported | ✅ Yes (macos-14) | Latest features, tested in CI |

**Hardware Requirements:**
- Apple Silicon (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M3, M3 Pro, M3 Max)
- Unified memory architecture (no separate VRAM)
- 16GB+ unified memory recommended

**Features:**
- Vector distance computation (L2, cosine, inner product)
- Vector quantization (int8, fp16, binary)
- ML training acceleration (selected algorithms)
- Automatic memory management

**Known Limitations:**
- Intel Macs: Metal support limited, CPU fallback available
- Some advanced ML algorithms may have limited Metal support

## Feature Matrix by Backend

| Feature | CPU | CUDA | ROCm | Metal |
|---------|-----|------|------|-------|
| Vector L2 Distance | ✅ | ✅ | ✅ | ✅ |
| Vector Cosine Distance | ✅ | ✅ | ✅ | ✅ |
| Vector Inner Product | ✅ | ✅ | ✅ | ✅ |
| HNSW Index Building | ✅ | ✅ | ⚠️ Partial | ⚠️ Partial |
| IVF Index Building | ✅ | ✅ | ⚠️ Partial | ⚠️ Partial |
| Vector Quantization (int8) | ✅ | ✅ | ✅ | ✅ |
| Vector Quantization (fp16) | ✅ | ✅ | ✅ | ✅ |
| Vector Quantization (binary) | ✅ | ✅ | ✅ | ✅ |
| Random Forest Training | ✅ | ⚠️ Partial | ❌ | ❌ |
| Linear Regression | ✅ | ✅ | ✅ | ✅ |
| K-Means Clustering | ✅ | ✅ | ⚠️ Partial | ⚠️ Partial |
| Embedding Generation | ✅ | ❌ | ❌ | ❌ |

**Legend:**
- ✅ Full support
- ⚠️ Partial support (some limitations)
- ❌ Not supported (CPU fallback available)

## Component-Specific Compatibility

### NeuronDB Extension

**Requirements:**
- PostgreSQL 16, 17, or 18
- C compiler (GCC or Clang)
- Make
- PostgreSQL development headers

**Optional:**
- GPU drivers (CUDA, ROCm, or Metal)
- ML libraries (XGBoost, LightGBM, CatBoost)

### NeuronAgent

**Requirements:**
- Go 1.21, 1.22, or 1.23
- PostgreSQL 16+ with NeuronDB extension
- Network port (default: 8080)

**Tested Go Versions:**
- 1.21.x: ✅ Supported
- 1.22.x: ✅ Supported
- 1.23.x: ✅ Supported (recommended)

### NeuronMCP

**Requirements:**
- Go 1.21, 1.22, or 1.23
- PostgreSQL 16+ with NeuronDB extension
- MCP-compatible client

**Tested Go Versions:**
- 1.21.x: ✅ Supported
- 1.22.x: ✅ Supported
- 1.23.x: ✅ Supported (recommended)

### NeuronDesktop

**Requirements:**
- Go 1.21, 1.22, or 1.23 (backend)
- Node.js 18+ (frontend)
- PostgreSQL 16+ with NeuronDB extension
- Network ports (default: 3000, 8081)

**Tested Node.js Versions:**
- 18.x: ✅ Supported
- 20.x: ✅ Supported (recommended)
- 22.x: ✅ Supported

## Docker Image Compatibility

### Base Images

| Component | Base Image | PostgreSQL Versions | Architectures |
|-----------|------------|---------------------|---------------|
| NeuronDB **CUDA** (published default) | `Dockerfile.gpu.cuda`: `nvidia/cuda:*-devel` builder + `postgres:{16,17,18}-bookworm` runtime | **16, 17, 18** | **amd64** (registry) |
| NeuronDB CPU (local / optional) | `postgres:{16,17,18}-bookworm` | **16, 17, 18** | amd64, arm64 |
| NeuronDB ROCm (Dockerfile only) | `rocm/dev-ubuntu-22.04:5.7` + PostgreSQL dev packages | **16, 17, 18** | amd64 |
| NeuronDB Metal | `postgres:{16,17,18}-bookworm` | **16, 17, 18** | arm64 (Apple Silicon) |

**Note:** **Docker Hub / GHCR** ship **`neurondb/neurondb-cuda`** for **PostgreSQL 16, 17, and 18**. NeuronDB release comes from the git tag (e.g. **3.1.0**). **`latest`** and the bare **`<version>`** tag track **PostgreSQL 17** + that release; use **`pg16`/`pg16-<ver>`** or **`pg18`/`pg18-<ver>`** for other majors.

### Container Runtime Requirements

- **Docker:** 20.10+
- **Docker Compose:** 2.0+
- **Kubernetes:** 1.24+ (for K8s deployments)

## CI Test Matrix

The following combinations are tested in continuous integration:

### Build Matrix

- **PostgreSQL:** 16.x (latest), 17.x (latest), 18.x (latest) - tested with latest minor version of each major release
- **OS:** Ubuntu 20.04, 22.04; Debian 11, 12; macOS 13, 14
- **Compilers:** GCC 9, 10, 11, 12 (GCC 9, 11, 12 tested); Clang 14, 15 (tested)
- **Go:** 1.21, 1.22, 1.23
- **GPU:** CUDA 12.4 (manual testing), ROCm 6.0 (manual testing), Metal (macOS 13+ in CI)

### Integration Test Matrix

- **PostgreSQL:** 16, 17, 18
- **OS:** Ubuntu 22.04, Debian 12
- **Components:** NeuronDB + NeuronAgent + NeuronMCP

## Getting Help

If you encounter compatibility issues:

1. Check this matrix for your specific combination
2. Review [Troubleshooting Guide](NeuronDB/docs/troubleshooting.md)
3. Check [GitHub Issues](https://github.com/neurondb/NeurondB/issues)
4. Contact support@neurondb.ai

## Version History

- **2024-12:** Initial compatibility matrix published
- Updated with each release to reflect tested combinations


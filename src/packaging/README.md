# Packaging

This directory contains build scripts for creating DEB and RPM packages for NeuronDB components.

📚 **See [NAMING_STANDARDS.md](./NAMING_STANDARDS.md) for complete package naming conventions and standards.**

## Directory Structure

This repo only packages **NeuronDB** (extension). Agent and MCP packages are in their respective repos.

```
packaging/
├── deb/
│   └── neurondb/
│       └── build.sh
└── rpm/
    └── neurondb/
        └── build.sh
```

## Prerequisites

### For DEB packages (Ubuntu/Debian)

- `dpkg-dev`
- `fakeroot`
- `build-essential`
- `debhelper` (optional, for advanced packaging)

### For RPM packages (RHEL/CentOS/Rocky/Fedora)

- `rpm-build`
- `rpmdevtools`
- `make`
- `gcc`

### For NeuronDB packages

- PostgreSQL development headers (`postgresql-server-dev-{version}`)
- Build tools (make, gcc/clang, cmake)

### For NeuronAgent/NeuronMCP packages

See the **neuron-agent** and **neuron-mcp** repositories for their packaging (deb/rpm).

## Building Packages

### NeuronDB

```bash
cd packaging/deb/neurondb
VERSION=2.0.0 ./build.sh

# Output: neurondb_1.0.0_amd64.deb
```

### NeuronAgent / NeuronMCP

Build and package from the **neuron-agent** and **neuron-mcp** repositories. This repo only builds NeuronDB packages.

## Build Script Requirements

Each `build.sh` script should:

1. Accept `VERSION` environment variable (defaults to `2.0.0.beta` if not set)
2. Build the component from source (or use pre-built binaries)
3. Create package metadata (control files, spec files)
4. Generate the package file (`.deb` or `.rpm`)
5. Output the package file in the current directory

## Package Installation

### DEB packages

```bash
sudo dpkg -i neurondb_1.0.0_amd64.deb
sudo apt-get install -f  # Fix dependencies if needed
```

### RPM packages

```bash
sudo rpm -ivh neurondb-1.0.0-1.x86_64.rpm
```

## Integration with Docker

Package-based Docker builds use these scripts. See:
- [`docker/neurondb/Dockerfile.package`](../docker/neurondb/Dockerfile.package)
- [`Docs/deployment/package.md`](../Docs/deployment/package.md)

## Versioning

Package versions follow semantic versioning:
- Format: `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)
- Pre-release: `1.0.0-beta`, `1.0.0-rc1`
- Build metadata: `1.0.0+git.abc123`

## Architecture Support

Packages are built for:
- `amd64` (x86_64) - Default
- `arm64` (aarch64) - When available

Set architecture via environment variable or build script parameters.

## Related Documentation

- [Package Documentation](../Docs/deployment/package.md)
- [Installation Guide](../NeuronDB/INSTALL.md)
- [Ecosystem Setup Script](../scripts/ecosystem-setup.sh)


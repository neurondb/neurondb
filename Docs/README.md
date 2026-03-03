# NeuronDB Documentation

Welcome to the NeuronDB documentation. This is the main entry point for all documentation in the NeuronDB ecosystem.

## Full platform (neurondb + neuron-cloud + neuron-hub)

To deploy the **full platform** (core neurondb plus neuron-cloud and neuron-hub) on one host, use the unified deploy script and place the three repos as siblings. See **[Ecosystem Overview](ecosystem/README.md)** for product roles and **[Full platform deploy](ecosystem/README.md#full-platform-all-three)** for `./scripts/deploy-all.sh` usage.

## Quick Navigation

<details>
<summary><strong>🚀 Getting Started</strong></summary>

| Document | Description | Time | Difficulty |
|----------|-------------|------|------------|
| **[Simple Start Guide](getting-started/simple-start.md)** | Beginner-friendly setup | 10 min | ⭐ Easy |
| **[Quick Start Guide](../../QUICKSTART.md)** | Get all services running quickly | 5-10 min | ⭐ Easy |
| **[Architecture Overview](getting-started/architecture.md)** | Understand the system architecture | 15 min | ⭐⭐ Medium |
| **[Installation Guide](getting-started/installation.md)** | Installation instructions | 30+ min | ⭐⭐ Medium |
| **[Ecosystem Overview](ecosystem/README.md)** | neurondb vs cloud vs hub; full-platform deploy | 5 min | ⭐ Easy |

</details>

<details>
<summary><strong>📚 Documentation Indexes</strong></summary>

- **[Complete Documentation Index](documentation-index.md)** - Comprehensive index of all documentation
- **[Documentation Overview](documentation.md)** - Main documentation index

</details>

<details>
<summary><strong>📖 Reference Documentation</strong></summary>

| Document | Description |
|----------|-------------|
| **[API Reference](reference/api-reference.md)** | Complete API reference for all components |
| **[Data Types](reference/data-types.md)** | All data types with detailed specifications |
| **[Top Functions](reference/top_functions.md)** | Most commonly used functions |
| **[Glossary](reference/glossary.md)** | Terminology and definitions |

</details>

<details>
<summary><strong>🚢 Deployment</strong></summary>

| Document | Description | Use Case |
|----------|-------------|----------|
| **[Docker Deployment](deployment/docker.md)** | Component-specific Docker guide | Individual components |
| **[Docker Unified Guide](deployment/docker-unified.md)** | Unified Docker orchestration | Full stack |
| **[Docker Ecosystem](deployment/docker-ecosystem.md)** | Complete ecosystem setup | Production |
| **[Kubernetes/Helm](deployment/kubernetes-helm.md)** | Kubernetes deployment | Cloud/K8s |
| **[Production Installation](deployment/production-install.md)** | Production setup guide | Enterprise |

</details>

<details>
<summary><strong>⚙️ Operations</strong></summary>

- **[Troubleshooting](operations/troubleshooting.md)** - Comprehensive troubleshooting guide
- **[Observability Setup](operations/observability-setup.md)** - Monitoring and observability

</details>

<details>
<summary><strong>💻 Development</strong></summary>

- **[Development Guide](development/development-guide.md)** - Development procedures
- **[Build System](development/build-system.md)** - Build system documentation
- **[Documentation Structure](development/structure.md)** - How documentation is organized

</details>

<details>
<summary><strong>🔧 Internals</strong></summary>

- **[Architecture Documentation](internals/README.md)** - Internal architecture details
- **[Index Methods](internals/index-methods.md)** - Index implementation details
- **[Identity Integration](internals/identity-integration-guide.md)** - Identity system

</details>

## Documentation Structure

The documentation is organized into several main sections:

- **`getting-started/`** - Fastest path to a working setup
- **`reference/`** - Stable reference pages (APIs, types, functions)
- **`internals/`** - Deeper dives (architecture, optimization, production)
- **`deployment/`** - Deployment guides and configurations
- **`operations/`** - Operations and troubleshooting guides
- **`development/`** - Development guides and build system
- **`components/`** - Component-specific documentation
- **`ecosystem/`** - Ecosystem integration guides

## Component Documentation

Each component has its own documentation:

| Component | Documentation Path | Description |
|-----------|-------------------|-------------|
| **NeuronDB** | [`NeuronDB/docs/`](../../NeuronDB/docs/) | SQL API, configuration, ML algorithms |
| **NeuronAgent** | [neuron-agent repo](https://github.com/neurondb/neuron-agent) `docs/` | Agent runtime documentation |
| **NeuronMCP** | [neuron-mcp repo](https://github.com/neurondb/neuron-mcp) tools reference | MCP tools reference |
| **NeuronDesktop** | [neuron-desktop repo](https://github.com/neurondb/neuron-desktop) `docs/` | Desktop UI documentation |

## Contributing

See the [Contributing Guide](../../CONTRIBUTING.md) for information on how to contribute to the documentation.

---

<div align="center">

**Last Updated:** 2026-02-26  
**Documentation Version:** 2.0.0

[⬆ Back to Top](#neurondb-documentation)

</div>


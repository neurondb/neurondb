# Advanced Docs

This section is for deeper dives: internals, performance tuning, deployment patterns, and how to extend the ecosystem.

## Suggested topics to add here

- Postgres extension internals (types, operators, index AMs)
- GPU backends (CUDA/Metal/ROCm): build flags, fallbacks, debugging
- Planner and scan integration
- Security and multi-tenant patterns
- Observability and production hardening

## Where the code lives

- Extension internals: `NeuronDB/src/`
- Extension headers/APIs: `NeuronDB/include/`
- Agent service: `NeuronAgent/internal/`
- MCP server: `NeuronMCP/internal/`
- Desktop: `NeuronDesktop/`

# ⭐ Advanced Documentation

**Documentation for experienced users deploying to production**

> **Prerequisites:** Comfortable with Docker, databases, and system administration

---

## 📚 Advanced Topics

### Production Deployment
- [NeuronAgent Deployment](../../NeuronAgent/docs/deployment.md) - Production setup
- [NeuronDesktop Deployment](../../NeuronDesktop/docs/deployment.md) - Web UI deployment
- [Security Guide](../../SECURITY.md) - Security best practices
- [Enterprise Deployment Guide](../../ENTERPRISE_DEPLOYMENT_GUIDE.md) - Enterprise setup

### Performance & Scaling
- [Performance Tuning](../../README.md#performance) - Optimization strategies
- [GPU Acceleration](../../NeuronDB/docs/gpu/) - CUDA, ROCm, Metal
- [SIMD Optimization](../../NeuronDB/docs/performance/simd-optimization.md) - CPU optimization

### Architecture & Design
- [NeuronAgent Architecture](../../NeuronAgent/docs/architecture.md) - System design
- [Component Integration](../ecosystem/integration.md) - Integration patterns
- [Ecosystem Overview](../ecosystem/README.md) - How components work together

### API References
- [NeuronDB SQL API](../../NeuronDB/docs/sql-api.md) - 520+ SQL functions
- [NeuronAgent REST API](../../NeuronAgent/docs/api.md) - Complete REST API
- [NeuronMCP Tools](../../NeuronMCP/REGISTERED_TOOLS.md) - 100+ MCP tools
- [NeuronDesktop API](../../NeuronDesktop/docs/api.md) - Web UI API

### Development
- [Contributing Guide](../../CONTRIBUTING.md) - How to contribute
- [Testing Guide](../../NeuronAgent/TESTING.md) - Testing strategies
- [Function Stability Policy](../../NeuronDB/docs/function-stability.md) - API stability

---

## 🔧 For Developers

### Building from Source
- [NeuronDB Build](../../NeuronDB/INSTALL.md) - Compile extension
- [Component Build](../getting-started/installation.md#method-2-source-build) - Build each component

### Custom Integrations
- [API Integration](../ecosystem/integration.md) - Integration patterns
- [OpenAPI Spec](../../NeuronAgent/openapi/openapi.yaml) - API specification

---

## 📖 Reference Materials

See [../reference/](../reference/) for:
- Complete glossary
- API snapshots
- Documentation improvement notes

---

**Last updated:** 2025-12-31


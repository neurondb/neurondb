# 🔍 Internals

<div align="center">

**Deeper dives: internals, performance tuning, deployment patterns, and how to extend the ecosystem**

[![Internals](https://img.shields.io/badge/internals-advanced-orange)](.)
[![Architecture](https://img.shields.io/badge/architecture-documented-blue)](.)

</div>

---

## 📍 Where the Code Lives

| Component | Code Location | Description |
|-----------|---------------|-------------|
| **NeuronDB Extension** | `NeuronDB/src/` | Extension internals |
| **NeuronDB Headers** | `NeuronDB/include/` | Extension headers/APIs |
| **NeuronAgent Service** | `NeuronAgent/internal/` | Agent service internals |
| **NeuronMCP Server** | `NeuronMCP/internal/` | MCP server internals |
| **NeuronDesktop** | `NeuronDesktop/` | Desktop application |

---

## 📚 Suggested Reading (Code-Anchored)

### 🚢 Production Deployment

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **NeuronAgent** | `NeuronAgent/docs/deployment.md` | Production deployment |
| **NeuronDesktop** | `NeuronDesktop/docs/` | Desktop deployment |
| **Security** | `SECURITY.md` | Security overview |
| **Docker** | `dockers/README.md` and `docker-compose.yml` | Docker orchestration |

### ⚡ Performance & Scaling

| Topic | Documentation | Description |
|-------|---------------|-------------|
| **GPU Support** | `NeuronDB/docs/gpu/` | GPU acceleration |
| **Performance** | `NeuronDB/docs/performance/` | Performance optimization |

### 🏗️ Architecture & Design

| Topic | Documentation | Description |
|-------|---------------|-------------|
| **NeuronAgent** | `NeuronAgent/docs/` | Agent architecture |
| **Ecosystem** | `Docs/ecosystem/integration.md` | Integration patterns |

### 🔌 API References

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **NeuronDB SQL** | `NeuronDB/neurondb--1.0.sql` | Extension SQL definitions |
| **NeuronDB API** | `NeuronDB/docs/sql-api.md` | Generated API reference |
| **NeuronAgent** | `NeuronAgent/openapi/openapi.yaml` | OpenAPI 3.0 specification |
| **NeuronMCP** | `NeuronMCP/REGISTERED_TOOLS.md` | Tools reference |

### 💻 Development

| Topic | Documentation | Description |
|-------|---------------|-------------|
| **Contributing** | `CONTRIBUTING.md` | Contribution guidelines |
| **NeuronAgent Testing** | `NeuronAgent/TESTING.md` | Testing strategy |
| **NeuronDB Stability** | `NeuronDB/docs/function-stability.md` | API stability notes |

### 🔨 Building from Source

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **NeuronDB Build** | `NeuronDB/INSTALL.md` | Extension build |
| **Component Build** | `Docs/getting-started/installation.md#method-2-source-build` | Component builds |

### 🔗 Custom Integrations

| Topic | Documentation | Description |
|-------|---------------|-------------|
| **Integration Guide** | `Docs/ecosystem/integration.md` | Integration patterns |

---

## 📖 Internal Documentation

| Document | Description |
|----------|-------------|
| [NeuronAgent Architecture](neuronagent-architecture.md) | Agent runtime architecture |
| [NeuronDesktop Frontend](neurondesktop-frontend.md) | Frontend architecture |
| [Index Methods](index-methods.md) | Index implementation details |
| [Identity Integration](identity-integration-guide.md) | Identity system |
| [OIDC Session Security](oidc-session-security.md) | Security implementation |

---

## 🔗 Related Documentation

- **[Getting Started](../getting-started/README.md)** - Setup guides
- **[Components](../components/README.md)** - Component overviews
- **[Reference](../reference/README.md)** - API references

---

<div align="center">

[⬆ Back to Top](#-internals) · [📚 Main Documentation](../../documentation.md)

</div>

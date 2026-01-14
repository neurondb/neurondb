# 🌐 Ecosystem

<div align="center">

**How NeuronDB components integrate and work together**

[![Integration](https://img.shields.io/badge/integration-complete-brightgreen)](integration.md)
[![Components](https://img.shields.io/badge/components-4-blue)](../components/README.md)

</div>

---

## 🔗 Quick Links

| Document | Description |
|----------|-------------|
| [Integration Guide](integration.md) | Component integration patterns |
| [Docker Orchestration](../../dockers/README.md) | Docker deployment for all services |
| [Repository docker-compose.yml](../../docker-compose.yml) | Root orchestration file |

---

## 🏗️ Ecosystem Architecture

<details>
<summary><strong>📐 Component Integration Diagram</strong></summary>

```mermaid
graph TB
    subgraph "NeuronDB Ecosystem"
        DB[NeuronDB<br/>PostgreSQL Extension]
        AGENT[NeuronAgent<br/>REST/WebSocket]
        MCP[NeuronMCP<br/>MCP Protocol]
        DESKTOP[NeuronDesktop<br/>Web UI]
    end
    
    subgraph "External"
        CLI[CLI Tools]
        WEB[Web Browser]
        MCP_CLIENT[MCP Clients]
    end
    
    CLI -->|SQL| DB
    WEB -->|HTTP| DESKTOP
    MCP_CLIENT -->|JSON-RPC| MCP
    
    DESKTOP -->|HTTP| AGENT
    DESKTOP -->|SQL| DB
    AGENT -->|SQL| DB
    MCP -->|SQL| DB
    
    style DB fill:#e1f5ff
    style AGENT fill:#fff4e1
    style MCP fill:#e8f5e9
    style DESKTOP fill:#f3e5f5
```

</details>

---

## 🔄 Integration Patterns

<details>
<summary><strong>📡 Communication Patterns</strong></summary>

| Pattern | Components | Protocol | Use Case |
|---------|------------|----------|----------|
| **Direct SQL** | Any → NeuronDB | PostgreSQL | Direct database access |
| **REST API** | Client → NeuronAgent | HTTP | Agent management |
| **WebSocket** | Client → NeuronAgent | WebSocket | Streaming responses |
| **MCP Protocol** | MCP Client → NeuronMCP | JSON-RPC | Tool execution |
| **Web UI** | Browser → NeuronDesktop | HTTP | User interface |

</details>

---

## 🐳 Docker Orchestration

<details>
<summary><strong>🐋 Docker Compose Setup</strong></summary>

The root `docker-compose.yml` orchestrates all services:

```yaml
services:
  neurondb:      # PostgreSQL with NeuronDB extension
  neuronagent:   # REST/WebSocket API server
  neuronmcp:     # MCP protocol server
  neurondesk-api:      # Desktop API
  neurondesk-frontend: # Desktop UI
```

**See**: [Docker Guide](../../dockers/README.md)

</details>

---

## 📚 Related Documentation

- **[Components](../components/README.md)** - Individual component details
- **[Getting Started](../getting-started/README.md)** - Setup guides
- **[Deployment](../deployment/README.md)** - Production deployment

---

<div align="center">

[⬆ Back to Top](#-ecosystem) · [📚 Main Documentation](../../documentation.md)

</div>

# 🧩 Components

<div align="center">

**Four primary components that share the same PostgreSQL database**

[![Components](https://img.shields.io/badge/components-4-blue)](.)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)

</div>

---

## 📊 Component Summary

| Component | Code Location | Docker Service(s) | Default Port(s) | Primary Docs | Status |
|-----------|---------------|-------------------|-----------------|--------------|--------|
| **NeuronDB** | `NeuronDB/` | `neurondb`, `neurondb-cuda`, `neurondb-rocm`, `neurondb-metal` | 5433 (cpu), 5434 (cuda), 5435 (rocm), 5436 (metal) | [neurondb.md](neurondb.md) | ✅ Stable |
| **NeuronAgent** | `NeuronAgent/` | `neuronagent` (and GPU variants) | 8080 | [neuronagent.md](neuronagent.md) | ✅ Stable |
| **NeuronMCP** | `NeuronMCP/` | `neuronmcp` (and GPU variants) | stdio (container runs with tty/stdin) | [neuronmcp.md](neuronmcp.md) | ✅ Stable |
| **NeuronDesktop** | `NeuronDesktop/` | `neurondesk-api`, `neurondesk-frontend` | 8081 (api), 3000 (ui) | [neurondesktop.md](neurondesktop.md) | ✅ Stable |

---

## 🏗️ Architecture Overview

<details>
<summary><strong>📐 Component Relationships</strong></summary>

```mermaid
graph LR
    subgraph "Core"
        DB[NeuronDB<br/>PostgreSQL Extension]
    end
    
    subgraph "Services"
        AGENT[NeuronAgent<br/>Port 8080]
        MCP[NeuronMCP<br/>stdio]
        DESKTOP[NeuronDesktop<br/>Ports 8081, 3000]
    end
    
    AGENT -->|SQL| DB
    MCP -->|SQL| DB
    DESKTOP -->|SQL| DB
    DESKTOP -->|HTTP| AGENT
    
    style DB fill:#e1f5ff
    style AGENT fill:#fff4e1
    style MCP fill:#e8f5e9
    style DESKTOP fill:#f3e5f5
```

</details>

---

## 📝 Notes

> [!NOTE]
> **Canonical orchestration**: Repository root [`docker-compose.yml`](../../docker-compose.yml)
> 
> **Docker files by component**: `dockers/`

---

## 🔗 Component Documentation

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **NeuronDB** | [neurondb.md](neurondb.md) | PostgreSQL extension details |
| **NeuronAgent** | [neuronagent.md](neuronagent.md) | Agent runtime details |
| **NeuronMCP** | [neuronmcp.md](neuronmcp.md) | MCP server details |
| **NeuronDesktop** | [neurondesktop.md](neurondesktop.md) | Web UI details |

---

## 🚀 Quick Start by Component

<details>
<summary><strong>🐘 NeuronDB Only</strong></summary>

```bash
# Start just the database
docker compose up -d neurondb

# Connect and use
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb"
```

**See**: [NeuronDB Documentation](neurondb.md)

</details>

<details>
<summary><strong>🤖 NeuronAgent</strong></summary>

```bash
# Start database + agent
docker compose up -d neurondb neuronagent

# Test API
curl http://localhost:8080/health
```

**See**: [NeuronAgent Documentation](neuronagent.md)

</details>

<details>
<summary><strong>🔌 NeuronMCP</strong></summary>

```bash
# Start database + MCP server
docker compose up -d neurondb neuronmcp

# Use with MCP clients (Claude Desktop, etc.)
```

**See**: [NeuronMCP Documentation](neuronmcp.md)

</details>

<details>
<summary><strong>🖥️ NeuronDesktop</strong></summary>

```bash
# Start full stack
docker compose up -d

# Access web UI
open http://localhost:3000
```

**See**: [NeuronDesktop Documentation](neurondesktop.md)

</details>

---

## 🔄 Component Integration

<details>
<summary><strong>📡 How Components Communicate</strong></summary>

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| NeuronAgent | NeuronDB | SQL | Database operations |
| NeuronMCP | NeuronDB | SQL | Database operations |
| NeuronDesktop | NeuronDB | SQL | Database operations |
| NeuronDesktop | NeuronAgent | HTTP | Agent management |
| Clients | NeuronAgent | HTTP/WebSocket | Agent API |
| Clients | NeuronMCP | JSON-RPC (stdio) | MCP protocol |
| Clients | NeuronDesktop | HTTP | Web UI |

</details>

---

## 📚 Related Documentation

- **[Ecosystem Integration](../ecosystem/integration.md)** - How components work together
- **[Getting Started](../getting-started/README.md)** - Setup guides
- **[Deployment](../deployment/README.md)** - Production deployment

---

<div align="center">

[⬆ Back to Top](#-components) · [📚 Main Documentation](../../documentation.md)

</div>

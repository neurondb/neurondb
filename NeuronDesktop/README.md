# NeuronDesktop

<div align="center">

**Unified web interface for MCP servers, NeuronDB, and NeuronAgent**

[![Next.js](https://img.shields.io/badge/Next.js-14+-000000.svg)](https://nextjs.org/)
[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)](https://golang.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](https://github.com/neurondb/neurondb)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](../LICENSE)
[![Documentation](https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg)](https://www.neurondb.ai/docs/neurondesktop)

</div>

NeuronDesktop is a full-featured web application that provides a unified interface for managing and interacting with:
- **MCP Servers** - Model Context Protocol servers with tool inspection and testing
- **NeuronDB** - Vector database with semantic search and collection management
- **NeuronAgent** - AI agent runtime with session management

## Documentation

- **[Features](docs/features.md)** - Complete feature list and capabilities
- **[API Reference](docs/API.md)** - Complete REST API documentation
- **[Integration Guide](docs/INTEGRATION.md)** - Integration with NeuronMCP and NeuronAgent
- **[NeuronAgent Usage](docs/NEURONAGENT_USAGE.md)** - How to use NeuronAgent in NeuronDesktop
- **[NeuronMCP Setup](docs/NEURONMCP_SETUP.md)** - NeuronMCP setup and configuration
- **[Deployment](docs/DEPLOYMENT.md)** - Deployment and configuration

## Features

### 🎯 Core Features

- **Unified Interface** - Single dashboard for all NeuronDB ecosystem components
- **Real-time Communication** - WebSocket support for live updates
- **Markdown Rendering** - Beautiful formatting for AI responses with syntax highlighting
- **Secure Authentication** - API key-based authentication with rate limiting
- **Professional UI** - Modern, responsive design with smooth animations
- **Comprehensive Logging** - Request/response logging with detailed analytics
- **Metrics & Monitoring** - Built-in metrics collection and health checks

### 🔧 Technical Features

- **Modular Architecture** - Clean separation of concerns, easy to extend
- **Operational readiness** - Error handling, graceful shutdown, connection pooling
- **Docker Support** - Complete Docker Compose setup for easy deployment
- **Type Safety** - Full TypeScript frontend, strongly-typed Go backend
- **Validation** - Comprehensive input validation and SQL injection protection
- **Rate Limiting** - Configurable rate limits per API key

## Architecture

### System Architecture

```mermaid
graph TB
    subgraph FRONTEND["Frontend (Next.js)"]
        UI[React Components<br/>TypeScript]
        PAGES[App Router<br/>Pages & Routes]
        STATE[State Management<br/>Context + Zustand]
        WS_CLIENT[WebSocket Client<br/>Real-time Updates]
    end
    
    subgraph BACKEND["Backend API (Go)"]
        API[REST API<br/>Gorilla Mux]
        WS_SERVER[WebSocket Server<br/>Event Streaming]
        HANDLERS[HTTP Handlers<br/>Request Processing]
        MIDDLEWARE[Middleware<br/>Auth, CORS, Logging]
    end
    
    subgraph SERVICES["External Services"]
        MCP[NeuronMCP<br/>MCP Protocol<br/>stdio]
        DB[NeuronDB<br/>PostgreSQL<br/>Port 5433]
        AGENT[NeuronAgent<br/>REST API<br/>Port 8080]
    end
    
    subgraph INTEGRATION["Integration Layer"]
        MCP_PROXY[MCP Proxy<br/>stdio → HTTP]
        DB_CLIENT[NeuronDB Client<br/>SQL Queries]
        AGENT_CLIENT[Agent Client<br/>HTTP Client]
    end
    
    UI --> API
    UI --> WS_CLIENT
    WS_CLIENT --> WS_SERVER
    API --> HANDLERS
    HANDLERS --> MIDDLEWARE
    HANDLERS --> MCP_PROXY
    HANDLERS --> DB_CLIENT
    HANDLERS --> AGENT_CLIENT
    
    MCP_PROXY --> MCP
    DB_CLIENT --> DB
    AGENT_CLIENT --> AGENT
    
    style FRONTEND fill:#e3f2fd
    style BACKEND fill:#fff3e0
    style SERVICES fill:#e8f5e9
    style INTEGRATION fill:#f3e5f5
```

### Request Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Next.js Frontend
    participant API as NeuronDesktop API
    participant MCP as NeuronMCP
    participant DB as NeuronDB
    participant Agent as NeuronAgent
    
    User->>Frontend: Access Web UI
    Frontend->>API: GET /api/v1/profiles
    API-->>Frontend: Profile list
    
    User->>Frontend: Vector search
    Frontend->>API: POST /api/v1/profiles/{id}/neurondb/search
    API->>DB: Execute SQL query
    DB-->>API: Search results
    API-->>Frontend: JSON response
    Frontend-->>User: Display results
    
    User->>Frontend: MCP tool call
    Frontend->>API: POST /api/v1/profiles/{id}/mcp/tools/call
    API->>MCP: Proxy MCP request (stdio)
    MCP->>DB: Execute tool
    DB-->>MCP: Tool results
    MCP-->>API: MCP response
    API-->>Frontend: Tool output
    Frontend-->>User: Display result
    
    User->>Frontend: Agent chat
    Frontend->>API: POST /api/v1/agents/{id}/sessions/{id}/messages
    API->>Agent: Forward to NeuronAgent
    Agent->>DB: Vector search (memory)
    DB-->>Agent: Context
    Agent-->>API: Agent response
    API-->>Frontend: Stream response
    Frontend-->>User: Real-time updates
```

### UI Component Structure

```mermaid
graph TD
    subgraph PAGES["Pages"]
        DASHBOARD[Dashboard<br/>Overview & Stats]
        AGENTS[Agents<br/>Agent Management]
        MCP_PAGE[MCP<br/>Tool Testing]
        DB_PAGE[NeuronDB<br/>Vector Search]
        WORKFLOWS[Workflows<br/>DAG Visualization]
    end
    
    subgraph COMPONENTS["Components"]
        CHAT[ChatInterface<br/>Agent Conversations]
        SEARCH[SearchInterface<br/>Vector Search]
        TOOLS[ToolInspector<br/>MCP Tools]
        CHARTS[Charts<br/>Analytics]
        FORMS[Forms<br/>Configuration]
    end
    
    DASHBOARD --> CHARTS
    AGENTS --> CHAT
    MCP_PAGE --> TOOLS
    DB_PAGE --> SEARCH
    WORKFLOWS --> CHARTS
    
    style PAGES fill:#e3f2fd
    style COMPONENTS fill:#fff3e0
```

## 📑 Table of Contents

<details>
<summary><strong>Expand full table of contents</strong></summary>

- [Overview](#overview)
- [Documentation](#documentation)
- [Features](#features)
- [Architecture](#architecture)
  - [System Architecture](#system-architecture)
  - [Request Flow](#request-flow)
  - [UI Component Structure](#ui-component-structure)
- [Quick Start](#quick-start)
  - [Automated Setup](#automated-setup-recommended)
  - [Using Docker](#using-docker-recommended)
  - [Native Installation](#native-installation)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)
- [Roadmap](#roadmap)

</details>

---

## Quick Start

### Automated Setup (Recommended)

<details>
<summary><strong>📋 Setup Checklist</strong></summary>

- [ ] Docker and Docker Compose installed
- [ ] Ports 3000, 8081 available
- [ ] NeuronDB running (port 5433)
- [ ] NeuronAgent running (port 8080, optional)
- [ ] NeuronMCP binary available (optional)

</details>

The easiest way to get started is using the automated setup script:

```bash
# Clone the repository
git clone <repository-url>
cd NeuronDesktop

# Set database connection (optional - defaults for Docker Compose shown)
export DB_HOST=localhost
export DB_PORT=5433        # Docker Compose default port
export DB_NAME=neurondesk
export DB_USER=neurondb     # Docker Compose default user
export DB_PASSWORD=neurondb  # Docker Compose default password

# Run automated setup
./scripts/neurondesktop-setup.sh
```

This script will:
1. ✅ Check database connection
2. ✅ Run database migrations
3. ✅ Build NeuronMCP binary (if source available)
4. ✅ Auto-detect NeuronMCP binary location
5. ✅ Create default profile with NeuronMCP configured
6. ✅ Create sample NeuronAgent (if NeuronAgent is running)
7. ✅ Verify setup

**Environment Variables for Setup:**
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` - Database connection
- `NEURONMCP_BINARY_PATH` - Override NeuronMCP binary location
- `NEURONDB_HOST`, `NEURONDB_PORT`, `NEURONDB_DATABASE`, `NEURONDB_USER`, `NEURONDB_PASSWORD` - NeuronDB connection for MCP
- `NEURONAGENT_ENDPOINT` - NeuronAgent API endpoint (default: http://localhost:8080)
- `NEURONAGENT_API_KEY` - NeuronAgent API key (optional)

After setup, start the services:

```bash
# Start API server
cd api && go run cmd/server/main.go

# In another terminal, start frontend
cd frontend && npm run dev

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8081
```

### Using Docker (Recommended)

NeuronDesktop is integrated into the root-level Docker Compose configuration. From the repository root:

```bash
# Start all services (NeuronDB, NeuronAgent, NeuronMCP, and NeuronDesktop)
docker compose --profile default up -d

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8081
```

**Note:** 
- NeuronDesktop automatically uses the existing `neurondb` Postgres container (no separate Postgres needed)
- The `neurondesk` database is automatically created and initialized on first startup
- The default profile is automatically created on API startup
- To configure NeuronMCP, set the environment variables in the root `docker-compose.yml` or run the setup script before starting containers

**Standalone Docker Setup (Alternative):**

If you need to run NeuronDesktop independently, you can use the standalone `NeuronDesktop/docker-compose.yml`:

```bash
cd NeuronDesktop
docker-compose up -d
```

**Note:** The standalone setup uses its own Postgres container on port 5433, which may conflict with the root-level stack if both are running simultaneously.

#### Running as a Service

For systemd (Linux) or launchd (macOS), see [Service Management Guide](../../Docs/getting-started/installation-services.md).

### Native Installation

#### Automated Installation (Recommended)

Use the installation script for easy setup:

```bash
# From repository root
sudo ./scripts/install-neurondesktop.sh

# With system service enabled
sudo ./scripts/install-neurondesktop.sh --enable-service
```

**Note:** The installation script installs the API backend. For the frontend, see [Frontend Setup](#frontend-setup) below.

#### Manual Setup

##### Backend

```bash
cd api

# Set environment variables (Docker Compose defaults)
export DB_HOST=localhost
export DB_PORT=5433        # Docker Compose default port
export DB_USER=neurondb     # Docker Compose default user
export DB_PASSWORD=neurondb  # Docker Compose default password
export DB_NAME=neurondesk

# Initialize database
createdb neurondesk
psql -d neurondesk -f migrations/001_initial_schema.sql

# Run server
go run cmd/server/main.go
```

Or use the setup script:

```bash
cd NeuronDesktop
./scripts/neurondesktop-setup.sh
cd api
go run cmd/server/main.go
```

##### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## Project Structure

```
NeuronDesktop/
├── api/                      # Go backend
│   ├── cmd/server/          # Server entrypoint
│   ├── internal/
│   │   ├── mcp/             # MCP proxy client
│   │   ├── neurondb/        # NeuronDB Postgres client
│   │   ├── agent/           # NeuronAgent HTTP client
│   │   ├── auth/            # Authentication
│   │   ├── config/          # Configuration
│   │   ├── db/              # Database layer
│   │   ├── handlers/        # HTTP handlers
│   │   ├── logging/         # Logging
│   │   ├── middleware/      # HTTP middleware
│   │   ├── metrics/         # Metrics collection
│   │   └── utils/           # Utilities
│   ├── migrations/          # Database migrations
│   └── Dockerfile           # Docker image
├── frontend/                # Next.js frontend
│   ├── app/                # Next.js app router
│   ├── components/         # React components
│   ├── lib/                # Utilities and API clients
│   └── Dockerfile          # Docker image
├── docs/                   # Documentation
│   ├── API.md             # API documentation
│   └── deployment.md      # Deployment guide
└── docker-compose.yml      # Docker Compose configuration
```

## API Documentation

See [docs/API.md](docs/API.md) for complete API documentation.

### Key Endpoints

- `GET /health` - Health check
- `GET /api/v1/profiles` - List profiles
- `GET /api/v1/profiles/{id}/mcp/tools` - List MCP tools
- `POST /api/v1/profiles/{id}/neurondb/search` - Vector search
- `GET /api/v1/metrics` - Application metrics

## Configuration

### Default Profile

NeuronDesktop automatically creates a default profile on first startup with:
- **NeuronMCP Integration**: Auto-detected and configured
- **NeuronDB Connection**: Configured via environment variables
- **NeuronAgent Integration**: Optional, configured if endpoint is provided

The default profile is marked as `is_default = true` and is used when no specific profile is selected.

### Sample NeuronAgent

If NeuronAgent is running and accessible, the setup script will create a sample agent:
- **Name**: `sample-assistant`
- **Description**: General purpose assistant for answering questions and helping with tasks
- **Model**: `gpt-4` (configurable)
- **Tools**: `sql`, `http`
- **Config**: temperature: 0.7, max_tokens: 1000

You can customize the sample agent by setting:
- `SAMPLE_AGENT_NAME` - Agent name
- `SAMPLE_AGENT_MODEL` - Model to use
- `SAMPLE_AGENT_TOOLS` - Comma-separated list of tools

### Environment Variables

**Backend:**
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` - Database connection
- `SERVER_PORT` - Server port (default: 8081)
- `LOG_LEVEL` - Log level (debug, info, warn, error)
- `CORS_ALLOWED_ORIGINS` - CORS origins (comma-separated)
- `NEURONMCP_BINARY_PATH` - Override NeuronMCP binary location
- `NEURONDB_HOST`, `NEURONDB_PORT`, `NEURONDB_DATABASE`, `NEURONDB_USER`, `NEURONDB_PASSWORD` - NeuronDB connection for MCP
- `NEURONAGENT_ENDPOINT` - NeuronAgent API endpoint
- `NEURONAGENT_API_KEY` - NeuronAgent API key

**Frontend:**
- `NEXT_PUBLIC_API_URL` - Backend API URL

## Development

### Backend Development

```bash
cd api
go mod download
go run cmd/server/main.go
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend tests
cd api
go test ./...

# Frontend tests (when added)
cd frontend
npm test
```

## Deployment

See [docs/deployment.md](docs/deployment.md) for detailed deployment instructions.

### Deployment checklist

- [ ] Set strong database passwords
- [ ] Configure CORS allowed origins
- [ ] Enable HTTPS
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategy
- [ ] Review rate limits
- [ ] Set up log aggregation

## Security

- API key authentication required for all endpoints
- Rate limiting per API key
- SQL injection protection
- Input validation on all requests
- CORS configuration
- Secure password hashing (bcrypt)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

See [LICENSE](../LICENSE) file for license information.

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Report issues on GitHub
- **Email**: support@neurondb.ai

## Roadmap

- [ ] Multi-user support with organizations
- [ ] Advanced query builder for NeuronDB
- [ ] Real-time collaboration features
- [ ] Plugin system for custom integrations
- [ ] Advanced analytics dashboard
- [ ] Export/import functionality
- [ ] API documentation explorer
- [ ] Webhook support

---

<div align="center">

[⬆ Back to Top](#neurondesktop)

</div>

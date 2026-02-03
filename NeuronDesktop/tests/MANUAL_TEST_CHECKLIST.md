# NeuronDesktop Manual Testing Checklist

Use this checklist to systematically verify links, buttons, and features when testing NeuronDesktop manually.

## Pre-requisites

- [ ] API server running (default: http://localhost:8081)
- [ ] PostgreSQL with NeuronDesktop schema (for full functionality)
- [ ] NeuronDB extension (optional, for vector features)
- [ ] NeuronMCP (optional, for MCP Console)
- [ ] NeuronAgent (optional, for Agents)

## Authentication

### Login Page (`/login`)

| Element | Type | Expected Action |
|---------|------|-----------------|
| Username input | Form | Accept username |
| Password input | Form | Accept password (masked) |
| Sign In with Password | Button | Submit login, redirect to `/` |
| Continue with SSO | Button | Redirect to OIDC provider |
| Don't have an account? Sign up | Button | Toggle to signup mode |
| PostgreSQL Settings | Section | Host, Port, Database, User, Password |
| Test Connection | Button | Validate DB connection |

### Post-Login

- [ ] Redirect to `/` when setup complete
- [ ] Redirect to `/setup?new_user=true` when new user (signup)

---

## Sidebar Navigation

| Link | Destination | Notes |
|------|-------------|-------|
| Home | `/` | |
| Dashboard | `/dashboard` | |
| Factory | `/setup` | |
| Chat | `/chat` | |
| MCP Console | `/mcp` | |
| NeuronDB | `/neurondb` | |
| Agents | `/agents` | |
| Models | `/models` | |
| Monitoring | `/monitoring` | |
| Logs | `/logs` | |
| Settings | `/settings` | |

### Sidebar Features

- [ ] Search filter (filters nav items)
- [ ] Favorite toggle (star icon on hover)
- [ ] Recent items section (after navigation)
- [ ] Close button (mobile)
- [ ] Version display (2.0.0)

---

## Top Menu

| Element | Action |
|---------|--------|
| NeuronDesktop logo | Navigate to `/` |
| Home | Navigate to `/` |
| MCP Console | Navigate to `/mcp` |
| NeuronDB | Navigate to `/neurondb` |
| Agents | Navigate to `/agents` |
| Monitoring | Navigate to `/monitoring` |
| Settings | Navigate to `/settings` |
| Refresh | Reload page |
| Info / Help | Open dropdown |
| Logout | Clear auth, redirect to `/login` |

---

## Home Page (`/`)

### Feature Cards (6)

| Card | Link | Description |
|------|------|-------------|
| MCP Console | `/mcp` | Interact with MCP servers |
| NeuronDB Console | `/neurondb` | Search collections, manage vectors |
| Agents | `/agents` | Manage AI agents |
| Factory Console | `/setup` | Installation and monitoring |
| Monitoring | `/monitoring` | System metrics |
| Logs & Inspector | `/logs` | Request logs |

---

## Setup / Factory (`/setup`)

| Step | Elements | Actions |
|------|----------|---------|
| Welcome | Get Started | Next to PostgreSQL |
| PostgreSQL | Host, Port, Database, User, Password, Back, Next | Configure DSN |
| Create Profile | Profile Name, MCP Command, Agent Endpoint, Back, Create Profile | Create profile |
| Complete | Go to Dashboard | Finish setup |

---

## Dashboard (`/dashboard`)

- [ ] Profile selector
- [ ] System Metrics widget
- [ ] NeuronDB stats widget
- [ ] Health status indicators
- [ ] Auto-refresh (30s)

---

## MCP Console (`/mcp`)

- [ ] Profile selector
- [ ] Model selector
- [ ] Connect / Disconnect
- [ ] New chat / thread
- [ ] Message input
- [ ] Send button
- [ ] Thread list

### MCP Sub-routes

| Route | Purpose |
|-------|---------|
| `/mcp` | Main chat |
| `/mcp/tools` | Tools |
| `/mcp/datasets` | Datasets |
| `/mcp/resources` | Resources |
| `/mcp/graphql` | GraphQL |
| `/mcp/http` | HTTP |
| `/mcp/rest` | REST |

---

## NeuronDB (`/neurondb`)

- [ ] Profile selector
- [ ] Collection selector
- [ ] Tabs: Search, SQL, Collections
- [ ] Vector Search: query, limit, distance type, Search button
- [ ] SQL Editor: query input, Execute
- [ ] Collections list with refresh

### NeuronDB Sub-routes

| Route | Purpose |
|-------|---------|
| `/neurondb` | Main console |
| `/neurondb/analytics` | Analytics |
| `/neurondb/indexes` | Indexes |
| `/neurondb/vectors` | Vectors |
| `/neurondb/ml` | ML |

---

## Agents (`/agents`)

- [ ] Profile selector
- [ ] Create / New Agent link
- [ ] Agent list cards
- [ ] Agent detail: memory, retrieval, sessions

### Agent Sub-routes

| Route | Purpose |
|-------|---------|
| `/agents` | List |
| `/agents/create` | Create agent |
| `/agents/[id]` | Agent detail |
| `/agents/[id]/memory` | Memory |
| `/agents/[id]/retrieval` | Retrieval |
| `/agents/[id]/sessions` | Sessions |

---

## Models (`/models`)

- [ ] Profile selector
- [ ] Add Model
- [ ] Model config cards (Set Default, Delete)
- [ ] Add Model Configuration modal: Provider, Model, API Key, Create

---

## Settings (`/settings`)

### Sections

- [ ] Modules: NeuronDB, NeuronAgent, NeuronMCP config
- [ ] Appearance: Theme toggle
- [ ] Profiles: Add, Edit, Delete profiles

### Module Tests

- [ ] Test NeuronDB connection
- [ ] Test NeuronAgent connection
- [ ] Test MCP config

### Model Config

- [ ] Add model
- [ ] Set API key
- [ ] Set default
- [ ] Delete

---

## Monitoring (`/monitoring`)

- [ ] Metrics display
- [ ] Profile selector (if applicable)

---

## Logs (`/logs`)

- [ ] Request logs list
- [ ] Log detail / inspector
- [ ] Profile filter

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd/Ctrl+K | Open command palette |
| Cmd/Ctrl+P | Quick switch |
| Cmd/Ctrl+G | Go to |
| Cmd/Ctrl+B | Toggle sidebar |

---

## Command Palette (Cmd/Ctrl+K)

| Command | Action |
|---------|--------|
| Go to Dashboard | Navigate to `/dashboard` |
| Go to NeuronDB | Navigate to `/neurondb` |
| Go to MCP Console | Navigate to `/mcp` |
| Go to Agents | Navigate to `/agents` |
| Create New Agent | Navigate to `/agents/create` |
| Open Settings | Navigate to `/settings` |

---

## Global Search

- [ ] Opens with shortcut (if configured)
- [ ] Search results navigate on select

---

## Cross-cutting

- [ ] Theme toggle (dark/light)
- [ ] Profile selector (where shown)
- [ ] Logout clears session
- [ ] Unauthenticated redirect to `/login`
- [ ] Error toasts on API failures

# NeuronAgent Architecture

<div align="center">

**AI agent runtime system architecture and design**

[![Architecture](https://img.shields.io/badge/architecture-complete-brightgreen)](.)
[![Status](https://img.shields.io/badge/status-stable-blue)](.)

</div>

---

> [!NOTE]
> NeuronAgent integrates with the NeuronDB PostgreSQL extension. It provides agent capabilities including long-term memory, tool execution, planning, reflection, and advanced features.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuronAgent Service                       │
├─────────────────────────────────────────────────────────────┤
│  REST API Layer        │  WebSocket        │  Health/Metrics │
├─────────────────────────────────────────────────────────────┤
│  Agent Runtime Engine  │  Session Manager  │  Tool Registry  │
├─────────────────────────────────────────────────────────────┤
│  Memory Manager        │  Planner          │  Reflector      │
├─────────────────────────────────────────────────────────────┤
│  Hierarchical Memory   │  Event Stream     │  VFS            │
├─────────────────────────────────────────────────────────────┤
│  Collaboration         │  Async Tasks      │  Sub-Agents     │
├─────────────────────────────────────────────────────────────┤
│  Verification Agent    │  Multimodal Proc  │  Browser Driver │
├─────────────────────────────────────────────────────────────┤
│  Background Workers    │  Job Queue        │  Notifications  │
├─────────────────────────────────────────────────────────────┤
│              NeuronDB PostgreSQL Extension                   │
│  (Vector Search │ Embeddings │ LLM │ ML │ RAG)              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Database Layer (`internal/db/`)

Manages all database interactions with prepared statements and connection pooling.

**Key Components:**
- **Models**: Go structs representing database entities (Agent, Session, Message, Tool, MemoryChunk, etc.)
- **Connection**: Connection pool management with configurable limits and health checks
- **Queries**: All SQL queries use prepared statements for security and performance
- **Transactions**: Transaction management for complex operations
- **Migrations**: 20 migration files tracking schema evolution

**Database Schema:**
- `agents` - Agent configurations and metadata
- `sessions` - Conversation sessions with agents
- `messages` - Message history with tool call tracking
- `memory_chunks` - Vector-embedded long-term memory
- `tools` - Tool registry and configurations
- `webhooks` - Webhook configurations
- `budgets` - Cost tracking and limits
- `approval_requests` - Human-in-the-loop approvals
- `feedback` - User feedback on agent responses
- `plans` - Execution plans
- `reflections` - Agent reflection records
- `browser_sessions` - Browser automation sessions
- `collaboration_workspaces` - Shared workspaces
- `hierarchical_memory` - Three-tier memory structure
- `event_stream` - Event logging and processing
- `verification_rules` - Quality assurance rules
- `virtual_filesystem` - Virtual file storage
- `async_tasks` - Asynchronous task queue
- `sub_agents` - Sub-agent configurations
- `task_alerts` - Alert preferences and notifications

### Agent Runtime (`internal/agent/`)

Core execution engine orchestrating agent behavior.

**Runtime Components:**
- **Runtime**: Main execution engine with state machine
- **MemoryManager**: HNSW-based vector search for long-term memory
- **HierarchicalMemoryManager**: Three-tier memory architecture (working, episodic, semantic)
- **EventStreamManager**: Event logging with automatic summarization
- **LLMClient**: Integration with NeuronDB LLM functions
- **ContextLoader**: Context loading combining messages and memory
- **PromptBuilder**: Prompt construction with templating
- **Planner**: Task planning and decomposition
- **Reflector**: Self-reflection and improvement
- **VerificationAgent**: Quality assurance through configurable rules
- **VirtualFileSystem**: Hybrid storage (database + S3) with atomic operations
- **AsyncTaskExecutor**: Asynchronous task execution with notifications
- **SubAgentManager**: Sub-agent routing and delegation
- **TaskNotifier**: Alert and notification management
- **EnhancedMultimodalProcessor**: Image, audio, and code processing

**Data Flow:**
1. User sends message via API
2. Runtime loads agent and session
3. Context is loaded (recent messages + memory chunks from hierarchical memory)
4. Planner generates execution plan if needed
5. Prompt is built with context and system prompt
6. LLM generates response (via NeuronDB)
7. Tool calls are parsed and executed if needed
8. Verification agent checks output quality
9. Reflection occurs for improvement
10. Final response is generated
11. Messages and memory chunks are stored
12. Events are logged to event stream

### Tools System (`internal/tools/`)

Extensible tool registry with built-in and custom tools.

**Built-in Tools:**
- **SQL Tool**: Read-only SQL queries with comprehensive validation
- **HTTP Tool**: HTTP requests with URL allowlist
- **Code Tool**: Code execution with directory restrictions
- **Shell Tool**: Shell commands with command whitelist
- **Browser Tool**: Headless Chrome automation (chromedp)
- **Memory Tool**: Hierarchical memory management
- **Filesystem Tool**: Virtual filesystem operations
- **Collaboration Tool**: Workspace management
- **Multimodal Tool**: Image, audio, code processing
- **Vector Tool**: Vector operations via NeuronDB
- **RAG Tool**: Retrieval-augmented generation
- **ML Tool**: Machine learning operations
- **Analytics Tool**: Analytics and insights
- **Visualization Tool**: Data visualization generation
- **Hybrid Search Tool**: Hybrid dense-sparse search
- **Reranking Tool**: Result reranking

**Tool Registry:**
- Tool registration and discovery
- JSON Schema validation for arguments
- Tool execution with timeout and sandboxing
- Audit logging for all tool executions
- Permission checking for tools
- Analytics tracking

### API Layer (`internal/api/`)

REST API and WebSocket endpoints.

**API Handlers:**
- **handlers.go**: Core agent, session, message handlers
- **advanced_handlers.go**: Advanced agent features
- **batch_handlers.go**: Batch operations
- **budget_handlers.go**: Budget management
- **collaboration_handlers.go**: Workspace management
- **async_tasks_handlers.go**: Async task management
- **alert_preferences_handlers.go**: Alert configuration
- **memory_handlers.go**: Memory operations
- **plans_handlers.go**: Execution plans
- **reflections_handlers.go**: Reflection management
- **relationships_handlers.go**: Agent relationships
- **versions_handlers.go**: Agent versioning
- **webhooks_handlers.go**: Webhook management
- **humanloop_handlers.go**: Human-in-the-loop approvals

**Middleware:**
- **RequestIDMiddleware**: Request ID generation
- **SecurityHeadersMiddleware**: HTTP security headers
- **CORSMiddleware**: Cross-origin resource sharing
- **LoggingMiddleware**: Structured request logging
- **AuthMiddleware**: API key authentication and rate limiting

**WebSocket:**
- Streaming agent responses in real-time
- Session-based connections
- Message format standardization

### Authentication (`internal/auth/`)

Security and access control.

**Components:**
- **APIKeyManager**: API key generation and validation
- **Hasher**: Password hashing (Argon2id and bcrypt with cost 14)
- **Validator**: API key validation and verification
- **RateLimiter**: Per-key rate limiting
- **PrincipalManager**: User and organization management
- **Roles**: RBAC support
- **ToolPermissionChecker**: Tool-level permissions
- **DataPermissionChecker**: Data-level permissions
- **AuditLogger**: Security audit logging

### Background Jobs (`internal/jobs/`)

PostgreSQL-based job queue system.

**Components:**
- **Queue**: Job queue using PostgreSQL SKIP LOCKED
- **Worker**: Worker pool with graceful shutdown
- **Processor**: Job type processors
- **Scheduler**: Scheduled job execution
- **Retry**: Automatic retry with exponential backoff

**Background Workers:**
- **MemoryPromoter**: Promotes memory from working to episodic to semantic
- **VerifierWorker**: Processes verification requests
- **AsyncTaskWorker**: Executes async tasks
- **BrowserCleanup**: Cleans up expired browser sessions
- **SessionCleanup**: Cleans up expired sessions

### Browser Automation (`internal/browser/`)

Headless Chrome automation using chromedp.

**Components:**
- **Driver**: Chrome browser driver management
- **Config**: Configuration with 30+ parameters
- **Restore**: Session restoration from database
- **Cleanup**: Expired session cleanup worker

**Features:**
- Page navigation and interaction
- JavaScript execution
- Screenshot capture
- Cookie management
- Session persistence
- Viewport and user agent management

### Collaboration (`internal/collaboration/`)

Real-time workspace collaboration.

**Components:**
- **WorkspaceManager**: Workspace and participant management
- **PubSub**: Real-time pub/sub for workspace updates

**Features:**
- Multi-user workspaces
- Real-time updates via WebSocket
- Permission controls
- Participant management

### Multimodal Processing (`internal/multimodal/`)

Enhanced multimodal content processing.

**Features:**
- Image processing and analysis
- Audio transcription
- Code analysis
- Text extraction from images
- Content type detection

### Storage (`internal/storage/`)

Hybrid storage backend abstraction.

**Backends:**
- **DatabaseStorage**: PostgreSQL-based storage
- **S3Storage**: AWS S3 storage
- **Storage Interface**: Unified storage abstraction

### Notifications (`internal/notifications/`)

Multi-channel notification system.

**Channels:**
- **EmailService**: Email notifications
- **WebhookService**: Webhook notifications

### Metrics and Observability (`internal/metrics/`)

Monitoring and observability.

**Components:**
- **Prometheus**: Prometheus metrics export
- **Tracing**: Distributed tracing
- **Logging**: Structured logging with context
- **Advanced Metrics**: Custom metrics tracking

## Advanced Features

### Hierarchical Memory

Three-tier memory architecture:
1. **Working Memory**: Short-term, session-scoped
2. **Episodic Memory**: Medium-term, session and event-based
3. **Semantic Memory**: Long-term, semantic abstraction

Automatic promotion from working → episodic → semantic based on importance and usage patterns.

### Event Stream

Comprehensive event logging with:
- Event capture for all agent actions
- Automatic summarization using LLM
- Event querying and filtering
- Integration with memory system

### Verification Agent

Quality assurance through:
- Configurable verification rules
- Automatic output validation
- Quality scoring
- Improvement suggestions

### Virtual File System

Hybrid file storage with:
- Database-backed metadata
- S3 or database storage backend
- Atomic operations
- Snapshot support
- Directory structure management

### Async Tasks

Asynchronous execution with:
- Task queue management
- Status tracking
- Notification on completion
- Cancellation support
- Priority queuing

### Sub-Agents

Agent delegation and routing:
- Sub-agent configuration
- Task routing logic
- Parallel execution
- Result aggregation

### Browser Automation

Comprehensive browser automation:
- Headless Chrome via chromedp
- Session management and persistence
- Cookie and local storage handling
- Screenshot capture
- JavaScript execution
- Page interaction (clicks, forms, etc.)

## Security Architecture

### Authentication
- API key-based authentication with Argon2id/bcrypt hashing
- Per-key rate limiting
- Role-based access control (RBAC)
- Principal and organization management

### Authorization
- Tool-level permissions
- Data-level permissions
- Workspace access controls

### Input Validation
- JSON Schema validation for all inputs
- SQL injection prevention (read-only queries, keyword filtering)
- URL allowlist for HTTP tool
- Command whitelist for shell tool
- Directory restrictions for code tool

### Audit Logging
- Comprehensive audit trail
- Tool execution logging
- Security event tracking

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Referrer-Policy
- Permissions-Policy

## Performance Optimizations

### Connection Pooling
- Configurable database connection pool
- Connection health checks
- Idle connection management

### Caching
- Session caching
- Agent configuration caching
- Memory chunk caching

### Background Processing
- Asynchronous memory storage
- Background workers for cleanup
- Job queue for heavy operations

### Vector Search
- HNSW indexing for fast similarity search
- Efficient embedding storage
- Batch operations support

## Deployment Architecture

### Single Server
Standard deployment on a single server with all components.

### Distributed
Components can be distributed:
- API servers (stateless, horizontally scalable)
- Background workers (can scale independently)
- Database (PostgreSQL with NeuronDB)
- Storage backends (S3, database)

### Container Deployment
- Docker images for all components
- Docker Compose for local development
- Kubernetes-ready configuration
- Health checks and graceful shutdown

## Integration Points

### NeuronDB Extension
- Vector operations and search
- Embedding generation
- LLM function calls
- ML operations
- RAG pipelines

### External Services
- Email providers (SMTP)
- Webhook endpoints
- S3 storage
- Custom tool integrations

## Data Flow Diagrams

### Message Processing Flow

```
User Request → API Handler → Auth Middleware → Rate Limiter
    ↓
Runtime Execution
    ↓
Context Loading → Memory Search → Prompt Building
    ↓
LLM Generation (via NeuronDB)
    ↓
Tool Execution (if needed) → Verification → Reflection
    ↓
Response Generation → WebSocket Stream (if enabled)
    ↓
Memory Storage → Event Logging → Response Returned
```

### Tool Execution Flow

```
Tool Call Request → Tool Registry → Permission Check
    ↓
JSON Schema Validation → Tool Handler Execution
    ↓
Sandbox Execution (if applicable) → Result Collection
    ↓
Audit Logging → Result Return
```

## Configuration

Configuration is managed through:
- Environment variables (preferred for production)
- Configuration file (YAML format)
- Default values for optional settings

Key configuration areas:
- Database connection settings
- Server port and timeouts
- Logging level and format
- Feature flags for advanced features
- Tool-specific configurations
- Browser automation settings
- Storage backend configuration
- Notification settings

## Monitoring and Observability

### Metrics
- Prometheus metrics endpoint
- Custom metrics for agent operations
- Tool execution metrics
- Memory usage metrics
- API performance metrics

### Logging
- Structured JSON logging
- Request ID tracking
- Context-aware logging
- Log levels (debug, info, warn, error)

### Tracing
- Distributed tracing support
- Request flow tracking
- Performance profiling

### Health Checks
- Database health check
- Component health status
- Graceful degradation

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[NeuronAgent Component](../components/neuronagent.md)** | Component overview |
| **[NeuronAgent API Reference](../../NeuronAgent/docs/api-reference.md)** | API documentation |
| **[Agent Runtime Guide](../../NeuronAgent/docs/runtime.md)** | Runtime guide |
| **[Tool Registry](../../NeuronAgent/docs/tools.md)** | Tool documentation |

---

<div align="center">

[⬆ Back to Top](#neuronagent-architecture) · [📚 Internals Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

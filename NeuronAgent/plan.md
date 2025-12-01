# NeuronAgent Complete Implementation Plan

## Project Location

Build at: `/Users/ndbuser/neurondb/pge/NeuronDB/bin/NeuronAgent`

## Project Structure

```
NeuronAgent/
├── cmd/
│   └── agent-server/
│       └── main.go                    # Entry point with graceful shutdown
├── internal/
│   ├── api/                           # HTTP/WebSocket API layer
│   │   ├── handlers.go               # REST endpoint handlers (all CRUD operations)
│   │   ├── websocket.go              # WebSocket connection management with streaming
│   │   ├── middleware.go             # Auth, rate limiting, logging, CORS
│   │   ├── models.go                 # Request/response DTOs with JSON tags
│   │   ├── errors.go                 # API error types and formatting
│   │   └── streaming.go              # Streaming response helpers
│   ├── agent/                         # Agent runtime engine
│   │   ├── runtime.go                # Main agent execution loop (complete state machine)
│   │   ├── planner.go                # Multi-step planning logic with tool chaining
│   │   ├── memory.go                 # Memory retrieval (HNSW search) and storage
│   │   ├── prompt.go                 # Prompt construction with templating
│   │   ├── llm.go                    # LLM interaction via NeuronDB functions
│   │   ├── tool_parser.go            # Tool call parsing (OpenAI format + custom)
│   │   ├── context.go                # Context loading, compression, token counting
│   │   └── profiles.go               # Predefined agent profiles with configs
│   ├── db/                            # Database abstraction layer
│   │   ├── schema.go                 # Schema migrations with versioning
│   │   ├── queries.go                # All SQL queries with prepared statements
│   │   ├── connection.go              # Connection pool with health checks
│   │   ├── models.go                 # Database model structs with tags
│   │   ├── transactions.go           # Transaction helpers with retry logic
│   │   └── migrations.go             # Migration runner with rollback support
│   ├── tools/                         # Tool execution system
│   │   ├── registry.go                # Tool registration and discovery
│   │   ├── executor.go                # Tool execution dispatcher with timeout
│   │   ├── validator.go               # JSON Schema validation
│   │   ├── sandbox.go                 # Security sandboxing (chroot, resource limits)
│   │   ├── sql_tool.go                # SQL tool (read-only, EXPLAIN, schema introspection)
│   │   ├── http_tool.go               # HTTP webhook tool (allowlist, timeout, size limits)
│   │   ├── code_tool.go               # Code analysis tool (limited dirs, static checkers)
│   │   ├── shell_tool.go              # Shell command tool (optional, heavily restricted)
│   │   └── types.go                   # Tool interface definitions
│   ├── jobs/                          # Background job processing
│   │   ├── queue.go                   # PostgreSQL-based job queue (SKIP LOCKED)
│   │   ├── worker.go                  # Worker pool with graceful shutdown
│   │   ├── processor.go               # Job type processors (HTTP, SQL, shell)
│   │   ├── retry.go                   # Retry logic with exponential backoff
│   │   └── scheduler.go               # Job scheduling (cron-like)
│   ├── session/                        # Session management
│   │   ├── manager.go                 # Session CRUD operations
│   │   ├── cleanup.go                 # Inactive session cleanup (background goroutine)
│   │   └── cache.go                   # Session caching (optional in-memory cache)
│   ├── auth/                           # Authentication and authorization
│   │   ├── api_key.go                 # API key management (CRUD, hashing)
│   │   ├── hasher.go                  # Bcrypt hashing utilities
│   │   ├── validator.go               # Key validation and rate limiting
│   │   └── roles.go                   # Role-based access control (admin, user, read-only)
│   ├── config/                         # Configuration management
│   │   ├── config.go                  # Config struct with YAML/ENV support
│   │   ├── env.go                     # Environment variable parsing
│   │   └── defaults.go                # Default configuration values
│   ├── metrics/                        # Observability
│   │   ├── prometheus.go               # Prometheus metrics (all counters, histograms)
│   │   ├── logging.go                 # Structured logging (zerolog or logrus)
│   │   └── tracing.go                 # Request tracing (OpenTelemetry compatible)
│   └── utils/                         # Shared utilities
│       ├── json.go                    # JSON helpers (marshaling, validation)
│       ├── uuid.go                    # UUID generation and validation
│       ├── time.go                    # Time utilities (formatting, parsing)
│       └── validation.go               # Input validation (email, URL, etc.)
├── pkg/
│   └── neurondb/                      # NeuronDB integration utilities
│       ├── client.go                  # NeuronDB function wrappers
│       ├── types.go                   # Vector and NeuronDB types
│       ├── embedding.go                # Embedding generation helpers
│       ├── llm.go                     # LLM generation helpers (streaming support)
│       └── vector.go                  # Vector operations (distance, normalization)
├── migrations/
│   ├── 001_initial_schema.sql          # Initial schema creation
│   ├── 002_add_indexes.sql             # Performance indexes (HNSW, B-tree)
│   └── 003_add_triggers.sql            # Database triggers (updated_at, etc.)
├── scripts/
│   ├── setup_db.sh                    # Database setup script
│   ├── run_migrations.sh              # Migration runner
│   └── generate_api_keys.sh            # API key generator utility
├── docker/
│   ├── Dockerfile                     # Multi-stage Go build
│   ├── docker-compose.yml             # Full stack deployment
│   └── docker-compose.dev.yml         # Development setup
├── configs/
│   ├── config.yaml.example            # Example configuration
│   └── agent_profiles.yaml            # Default agent profiles
├── tests/
│   ├── integration/                   # Integration tests
│   ├── unit/                          # Unit tests
│   └── fixtures/                      # Test fixtures and mocks
├── docs/
│   ├── API.md                         # Complete API documentation
│   ├── ARCHITECTURE.md                # Architecture deep dive
│   ├── DEPLOYMENT.md                  # Deployment guide
│   └── TOOLS.md                       # Tool development guide
├── go.mod
├── go.sum
├── .gitignore
├── .env.example
├── Makefile
├── README.md
└── LICENSE
```

## Detailed Component Specifications

### 1. Database Schema (migrations/001_initial_schema.sql)

**Complete SQL schema with all constraints, indexes, and triggers:**

```sql
-- Schema: neurondb_agent
CREATE SCHEMA IF NOT EXISTS neurondb_agent;

-- Agents table: Agent profiles and configurations
CREATE TABLE neurondb_agent.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,  -- NeuronDB model identifier
    memory_table TEXT,          -- Optional per-agent memory table name
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',  -- temperature, max_tokens, top_p, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_model_name CHECK (model_name ~ '^[a-zA-Z0-9_-]+$'),
    CONSTRAINT valid_memory_table CHECK (memory_table IS NULL OR memory_table ~ '^[a-z][a-z0-9_]*$')
);

-- Sessions table: User conversation sessions
CREATE TABLE neurondb_agent.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    external_user_id TEXT,  -- Optional external user identifier
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_external_user_id CHECK (external_user_id IS NULL OR length(external_user_id) > 0)
);

-- Messages table: Conversation history
CREATE TABLE neurondb_agent.messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_name TEXT,  -- NULL unless role = 'tool'
    tool_call_id TEXT,  -- For associating tool calls with results
    token_count INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_tool_message CHECK (
        (role = 'tool' AND tool_name IS NOT NULL) OR
        (role != 'tool' AND tool_name IS NULL)
    )
);

-- Memory chunks table: Vector-embedded long-term memory
CREATE TABLE neurondb_agent.memory_chunks (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),  -- NeuronDB vector type, configurable dimension
    importance_score REAL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_embedding CHECK (embedding IS NOT NULL)
);

-- Tools table: Tool registry
CREATE TABLE neurondb_agent.tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    arg_schema JSONB NOT NULL,  -- JSON Schema for arguments
    handler_type TEXT NOT NULL CHECK (handler_type IN ('sql', 'http', 'code', 'shell', 'queue')),
    handler_config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_arg_schema CHECK (jsonb_typeof(arg_schema) = 'object')
);

-- Jobs table: Background job queue
CREATE TABLE neurondb_agent.jobs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    type TEXT NOT NULL CHECK (type IN ('http_call', 'sql_task', 'shell_task', 'custom')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    priority INT DEFAULT 0,
    payload JSONB NOT NULL,
    result JSONB,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- API keys table: Authentication
CREATE TABLE neurondb_agent.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT NOT NULL UNIQUE,  -- Bcrypt hash of API key
    key_prefix TEXT NOT NULL,  -- First 8 chars for identification
    organization_id TEXT,
    user_id TEXT,
    rate_limit_per_minute INT DEFAULT 60,
    roles TEXT[] DEFAULT '{user}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    CONSTRAINT valid_roles CHECK (array_length(roles, 1) > 0)
);

-- Indexes for performance
CREATE INDEX idx_sessions_agent_id ON neurondb_agent.sessions(agent_id);
CREATE INDEX idx_sessions_last_activity ON neurondb_agent.sessions(last_activity_at);
CREATE INDEX idx_messages_session_id ON neurondb_agent.messages(session_id, created_at DESC);
CREATE INDEX idx_messages_session_role ON neurondb_agent.messages(session_id, role);
CREATE INDEX idx_memory_chunks_agent_id ON neurondb_agent.memory_chunks(agent_id);
CREATE INDEX idx_memory_chunks_session_id ON neurondb_agent.memory_chunks(session_id);
CREATE INDEX idx_jobs_status_created ON neurondb_agent.jobs(status, created_at) WHERE status IN ('queued', 'running');
CREATE INDEX idx_jobs_agent_session ON neurondb_agent.jobs(agent_id, session_id);
CREATE INDEX idx_api_keys_prefix ON neurondb_agent.api_keys(key_prefix);

-- HNSW index on memory chunks embedding (NeuronDB)
CREATE INDEX idx_memory_chunks_embedding_hnsw ON neurondb_agent.memory_chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION neurondb_agent.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER agents_updated_at BEFORE UPDATE ON neurondb_agent.agents
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

CREATE TRIGGER tools_updated_at BEFORE UPDATE ON neurondb_agent.tools
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

CREATE TRIGGER jobs_updated_at BEFORE UPDATE ON neurondb_agent.jobs
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

-- Trigger for session last_activity_at
CREATE OR REPLACE FUNCTION neurondb_agent.update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE neurondb_agent.sessions
    SET last_activity_at = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER messages_session_activity AFTER INSERT ON neurondb_agent.messages
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_session_activity();
```

### 2. Database Layer (internal/db/)

**File: internal/db/models.go**

```go
package db

import (
    "time"
    "github.com/google/uuid"
    "github.com/lib/pq"
)

type Agent struct {
    ID           uuid.UUID       `db:"id"`
    Name         string          `db:"name"`
    Description  *string         `db:"description"`
    SystemPrompt string          `db:"system_prompt"`
    ModelName    string          `db:"model_name"`
    MemoryTable  *string         `db:"memory_table"`
    EnabledTools pq.StringArray  `db:"enabled_tools"`
    Config       map[string]interface{} `db:"config"`
    CreatedAt    time.Time       `db:"created_at"`
    UpdatedAt    time.Time       `db:"updated_at"`
}

type Session struct {
    ID              uuid.UUID  `db:"id"`
    AgentID         uuid.UUID  `db:"agent_id"`
    ExternalUserID  *string    `db:"external_user_id"`
    Metadata        map[string]interface{} `db:"metadata"`
    CreatedAt       time.Time  `db:"created_at"`
    LastActivityAt time.Time  `db:"last_activity_at"`
}

type Message struct {
    ID          int64      `db:"id"`
    SessionID   uuid.UUID  `db:"session_id"`
    Role        string     `db:"role"`
    Content     string     `db:"content"`
    ToolName    *string    `db:"tool_name"`
    ToolCallID  *string    `db:"tool_call_id"`
    TokenCount  *int       `db:"token_count"`
    Metadata    map[string]interface{} `db:"metadata"`
    CreatedAt   time.Time  `db:"created_at"`
}

type MemoryChunk struct {
    ID             int64      `db:"id"`
    AgentID        uuid.UUID  `db:"agent_id"`
    SessionID      *uuid.UUID `db:"session_id"`
    MessageID      *int64     `db:"message_id"`
    Content        string     `db:"content"`
    Embedding      []float32  `db:"embedding"`  // Will be converted to/from neurondb_vector
    ImportanceScore float64   `db:"importance_score"`
    Metadata       map[string]interface{} `db:"metadata"`
    CreatedAt      time.Time  `db:"created_at"`
}

type Tool struct {
    Name         string                 `db:"name"`
    Description  string                `db:"description"`
    ArgSchema    map[string]interface{} `db:"arg_schema"`
    HandlerType  string                 `db:"handler_type"`
    HandlerConfig map[string]interface{} `db:"handler_config"`
    Enabled      bool                   `db:"enabled"`
    CreatedAt    time.Time             `db:"created_at"`
    UpdatedAt    time.Time             `db:"updated_at"`
}

type Job struct {
    ID           int64                  `db:"id"`
    AgentID      *uuid.UUID             `db:"agent_id"`
    SessionID    *uuid.UUID             `db:"session_id"`
    Type         string                 `db:"type"`
    Status       string                 `db:"status"`
    Priority     int                    `db:"priority"`
    Payload      map[string]interface{} `db:"payload"`
    Result       map[string]interface{} `db:"result"`
    ErrorMessage *string                `db:"error_message"`
    RetryCount   int                    `db:"retry_count"`
    MaxRetries   int                    `db:"max_retries"`
    CreatedAt    time.Time             `db:"created_at"`
    UpdatedAt    time.Time             `db:"updated_at"`
    StartedAt    *time.Time            `db:"started_at"`
    CompletedAt   *time.Time            `db:"completed_at"`
}

type APIKey struct {
    ID               uuid.UUID  `db:"id"`
    KeyHash          string     `db:"key_hash"`
    KeyPrefix        string     `db:"key_prefix"`
    OrganizationID   *string    `db:"organization_id"`
    UserID           *string    `db:"user_id"`
    RateLimitPerMin  int        `db:"rate_limit_per_minute"`
    Roles            pq.StringArray `db:"roles"`
    Metadata         map[string]interface{} `db:"metadata"`
    CreatedAt        time.Time  `db:"created_at"`
    LastUsedAt       *time.Time `db:"last_used_at"`
    ExpiresAt        *time.Time `db:"expires_at"`
}
```

**File: internal/db/connection.go**

```go
package db

import (
    "context"
    "database/sql"
    "fmt"
    "time"
    _ "github.com/lib/pq"
    "github.com/jmoiron/sqlx"
)

type DB struct {
    *sqlx.DB
    poolConfig PoolConfig
}

type PoolConfig struct {
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
    ConnMaxIdleTime time.Duration
}

func NewDB(connStr string, poolConfig PoolConfig) (*DB, error) {
    db, err := sqlx.Connect("postgres", connStr)
    if err != nil {
        return nil, fmt.Errorf("failed to connect: %w", err)
    }
    
    db.SetMaxOpenConns(poolConfig.MaxOpenConns)
    db.SetMaxIdleConns(poolConfig.MaxIdleConns)
    db.SetConnMaxLifetime(poolConfig.ConnMaxLifetime)
    db.SetConnMaxIdleTime(poolConfig.ConnMaxIdleTime)
    
    return &DB{DB: db, poolConfig: poolConfig}, nil
}

func (d *DB) HealthCheck(ctx context.Context) error {
    var result int
    err := d.GetContext(ctx, &result, "SELECT 1")
    return err
}

func (d *DB) Close() error {
    return d.DB.Close()
}
```

**File: internal/db/queries.go**

```go
package db

import (
    "context"
    "github.com/google/uuid"
    "github.com/jmoiron/sqlx"
)

// Agent queries
const (
    createAgentQuery = `
        INSERT INTO neurondb_agent.agents 
        (name, description, system_prompt, model_name, memory_table, enabled_tools, config)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id, created_at, updated_at`
    
    getAgentByIDQuery = `SELECT * FROM neurondb_agent.agents WHERE id = $1`
    listAgentsQuery = `SELECT * FROM neurondb_agent.agents ORDER BY created_at DESC`
    updateAgentQuery = `
        UPDATE neurondb_agent.agents 
        SET name = $2, description = $3, system_prompt = $4, model_name = $5,
            memory_table = $6, enabled_tools = $7, config = $8
        WHERE id = $1
        RETURNING updated_at`
    deleteAgentQuery = `DELETE FROM neurondb_agent.agents WHERE id = $1`
)

// Session queries
const (
    createSessionQuery = `
        INSERT INTO neurondb_agent.sessions (agent_id, external_user_id, metadata)
        VALUES ($1, $2, $3)
        RETURNING id, created_at, last_activity_at`
    
    getSessionQuery = `SELECT * FROM neurondb_agent.sessions WHERE id = $1`
    listSessionsQuery = `
        SELECT * FROM neurondb_agent.sessions 
        WHERE agent_id = $1 
        ORDER BY last_activity_at DESC 
        LIMIT $2 OFFSET $3`
)

// Message queries
const (
    createMessageQuery = `
        INSERT INTO neurondb_agent.messages 
        (session_id, role, content, tool_name, tool_call_id, token_count, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id, created_at`
    
    getMessagesQuery = `
        SELECT * FROM neurondb_agent.messages 
        WHERE session_id = $1 
        ORDER BY created_at ASC 
        LIMIT $2 OFFSET $3`
    
    getRecentMessagesQuery = `
        SELECT * FROM neurondb_agent.messages 
        WHERE session_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2`
)

// Memory chunk queries
const (
    createMemoryChunkQuery = `
        INSERT INTO neurondb_agent.memory_chunks 
        (agent_id, session_id, message_id, content, embedding, importance_score, metadata)
        VALUES ($1, $2, $3, $4, $5::neurondb_vector, $6, $7)
        RETURNING id, created_at`
    
    searchMemoryQuery = `
        SELECT id, content, importance_score, metadata,
               1 - (embedding <=> $1::neurondb_vector) AS similarity
        FROM neurondb_agent.memory_chunks
        WHERE agent_id = $2
        ORDER BY embedding <=> $1::neurondb_vector
        LIMIT $3`
)

// NeuronDB function wrappers
const (
    embedTextQuery = `SELECT neurondb_embed($1, $2) AS embedding`
    llmGenerateQuery = `SELECT neurondb_llm_generate($1, $2, $3) AS output`
)

type Queries struct {
    db *sqlx.DB
}

func NewQueries(db *sqlx.DB) *Queries {
    return &Queries{db: db}
}

func (q *Queries) CreateAgent(ctx context.Context, agent *Agent) error {
    return q.db.GetContext(ctx, agent, createAgentQuery,
        agent.Name, agent.Description, agent.SystemPrompt, agent.ModelName,
        agent.MemoryTable, agent.EnabledTools, agent.Config)
}

func (q *Queries) GetAgentByID(ctx context.Context, id uuid.UUID) (*Agent, error) {
    var agent Agent
    err := q.db.GetContext(ctx, &agent, getAgentByIDQuery, id)
    return &agent, err
}

// ... (all other query methods with proper error handling)
```

### 3. Agent Runtime (internal/agent/)

**File: internal/agent/runtime.go**

```go
package agent

import (
    "context"
    "encoding/json"
    "fmt"
    "github.com/google/uuid"
    "github.com/neurondb/NeuronAgent/internal/db"
    "github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type Runtime struct {
    db        *db.DB
    queries   *db.Queries
    memory    *MemoryManager
    planner   *Planner
    prompt    *PromptBuilder
    llm       *LLMClient
    tools     *ToolRegistry
}

type ExecutionState struct {
    SessionID    uuid.UUID
    AgentID      uuid.UUID
    UserMessage  string
    Context      *Context
    LLMResponse  *LLMResponse
    ToolCalls    []ToolCall
    ToolResults  []ToolResult
    FinalAnswer  string
    TokensUsed   int
    Error        error
}

type LLMResponse struct {
    Content   string
    ToolCalls []ToolCall
    Usage     TokenUsage
}

type ToolCall struct {
    ID       string
    Name     string
    Arguments map[string]interface{}
}

type ToolResult struct {
    ToolCallID string
    Content    string
    Error      error
}

type TokenUsage struct {
    PromptTokens     int
    CompletionTokens int
    TotalTokens      int
}

func NewRuntime(db *db.DB, queries *db.Queries, tools *ToolRegistry) *Runtime {
    return &Runtime{
        db:      db,
        queries: queries,
        memory:  NewMemoryManager(db, queries),
        planner: NewPlanner(),
        prompt:  NewPromptBuilder(),
        llm:     NewLLMClient(db),
        tools:   tools,
    }
}

func (r *Runtime) Execute(ctx context.Context, sessionID uuid.UUID, userMessage string) (*ExecutionState, error) {
    state := &ExecutionState{
        SessionID:   sessionID,
        UserMessage: userMessage,
    }
    
    // Step 1: Load agent and session
    session, err := r.queries.GetSession(ctx, sessionID)
    if err != nil {
        return nil, fmt.Errorf("failed to get session: %w", err)
    }
    state.AgentID = session.AgentID
    
    agent, err := r.queries.GetAgentByID(ctx, session.AgentID)
    if err != nil {
        return nil, fmt.Errorf("failed to get agent: %w", err)
    }
    
    // Step 2: Load context (recent messages + memory)
    context, err := r.loadContext(ctx, sessionID, agent.ID, userMessage)
    if err != nil {
        return nil, fmt.Errorf("failed to load context: %w", err)
    }
    state.Context = context
    
    // Step 3: Build prompt
    prompt, err := r.prompt.Build(agent, context, userMessage)
    if err != nil {
        return nil, fmt.Errorf("failed to build prompt: %w", err)
    }
    
    // Step 4: Call LLM via NeuronDB
    llmResponse, err := r.llm.Generate(ctx, agent.ModelName, prompt, agent.Config)
    if err != nil {
        return nil, fmt.Errorf("LLM generation failed: %w", err)
    }
    state.LLMResponse = llmResponse
    
    // Step 5: Parse tool calls
    if len(llmResponse.ToolCalls) > 0 {
        state.ToolCalls = llmResponse.ToolCalls
        
        // Step 6: Execute tools
        toolResults, err := r.executeTools(ctx, agent, llmResponse.ToolCalls)
        if err != nil {
            return nil, fmt.Errorf("tool execution failed: %w", err)
        }
        state.ToolResults = toolResults
        
        // Step 7: Call LLM again with tool results
        finalPrompt, err := r.prompt.BuildWithToolResults(agent, context, userMessage, llmResponse, toolResults)
        if err != nil {
            return nil, fmt.Errorf("failed to build final prompt: %w", err)
        }
        
        finalResponse, err := r.llm.Generate(ctx, agent.ModelName, finalPrompt, agent.Config)
        if err != nil {
            return nil, fmt.Errorf("final LLM generation failed: %w", err)
        }
        state.FinalAnswer = finalResponse.Content
        state.TokensUsed = finalResponse.Usage.TotalTokens
    } else {
        state.FinalAnswer = llmResponse.Content
        state.TokensUsed = llmResponse.Usage.TotalTokens
    }
    
    // Step 8: Store messages
    if err := r.storeMessages(ctx, sessionID, userMessage, state.FinalAnswer, state.ToolCalls, state.ToolResults); err != nil {
        return nil, fmt.Errorf("failed to store messages: %w", err)
    }
    
    // Step 9: Store memory chunks (async, non-blocking)
    go r.memory.StoreChunks(context.Background(), agent.ID, sessionID, state.FinalAnswer, state.ToolResults)
    
    return state, nil
}

func (r *Runtime) loadContext(ctx context.Context, sessionID uuid.UUID, agentID uuid.UUID, userMessage string) (*Context, error) {
    // Load recent messages (last 20)
    messages, err := r.queries.GetRecentMessages(ctx, sessionID, 20)
    if err != nil {
        return nil, err
    }
    
    // Compute embedding for user message
    embedding, err := r.llm.Embed(ctx, "all-MiniLM-L6-v2", userMessage)
    if err != nil {
        return nil, err
    }
    
    // Search memory chunks (top 5)
    memoryChunks, err := r.memory.Retrieve(ctx, agentID, embedding, 5)
    if err != nil {
        return nil, err
    }
    
    return &Context{
        Messages:     messages,
        MemoryChunks: memoryChunks,
    }, nil
}

func (r *Runtime) executeTools(ctx context.Context, agent *db.Agent, toolCalls []ToolCall) ([]ToolResult, error) {
    results := make([]ToolResult, 0, len(toolCalls))
    
    for _, call := range toolCalls {
        // Get tool from registry
        tool, err := r.tools.Get(call.Name)
        if err != nil {
            results = append(results, ToolResult{
                ToolCallID: call.ID,
                Error:      err,
            })
            continue
        }
        
        // Check if tool is enabled for this agent
        if !contains(agent.EnabledTools, call.Name) {
            results = append(results, ToolResult{
                ToolCallID: call.ID,
                Error:      fmt.Errorf("tool %s not enabled for agent", call.Name),
            })
            continue
        }
        
        // Execute tool
        result, err := r.tools.Execute(ctx, tool, call.Arguments)
        results = append(results, ToolResult{
            ToolCallID: call.ID,
            Content:   result,
            Error:      err,
        })
    }
    
    return results, nil
}

func (r *Runtime) storeMessages(ctx context.Context, sessionID uuid.UUID, userMsg, assistantMsg string, toolCalls []ToolCall, toolResults []ToolResult) error {
    // Store user message
    if _, err := r.queries.CreateMessage(ctx, &db.Message{
        SessionID: sessionID,
        Role:      "user",
        Content:   userMsg,
    }); err != nil {
        return err
    }
    
    // Store tool calls as messages
    for _, call := range toolCalls {
        callJSON, _ := json.Marshal(call.Arguments)
        if _, err := r.queries.CreateMessage(ctx, &db.Message{
            SessionID:  sessionID,
            Role:       "assistant",
            Content:    fmt.Sprintf("Tool call: %s", call.Name),
            ToolCallID: &call.ID,
            Metadata:   map[string]interface{}{"tool_call": call},
        }); err != nil {
            return err
        }
    }
    
    // Store tool results
    for _, result := range toolResults {
        if _, err := r.queries.CreateMessage(ctx, &db.Message{
            SessionID:  sessionID,
            Role:       "tool",
            Content:    result.Content,
            ToolName:   &result.ToolCallID,
            ToolCallID: &result.ToolCallID,
        }); err != nil {
            return err
        }
    }
    
    // Store assistant message
    if _, err := r.queries.CreateMessage(ctx, &db.Message{
        SessionID: sessionID,
        Role:      "assistant",
        Content:   assistantMsg,
    }); err != nil {
        return err
    }
    
    return nil
}
```

**File: internal/agent/memory.go**

```go
package agent

import (
    "context"
    "github.com/google/uuid"
    "github.com/neurondb/NeuronAgent/internal/db"
)

type MemoryManager struct {
    db      *db.DB
    queries *db.Queries
    llm     *LLMClient
}

type MemoryChunk struct {
    ID             int64
    Content        string
    ImportanceScore float64
    Similarity     float64
    Metadata       map[string]interface{}
}

func NewMemoryManager(db *db.DB, queries *db.Queries) *MemoryManager {
    return &MemoryManager{
        db:      db,
        queries: queries,
        llm:     NewLLMClient(db),
    }
}

func (m *MemoryManager) Retrieve(ctx context.Context, agentID uuid.UUID, queryEmbedding []float32, topK int) ([]MemoryChunk, error) {
    chunks, err := m.queries.SearchMemory(ctx, agentID, queryEmbedding, topK)
    if err != nil {
        return nil, err
    }
    
    result := make([]MemoryChunk, len(chunks))
    for i, chunk := range chunks {
        result[i] = MemoryChunk{
            ID:             chunk.ID,
            Content:        chunk.Content,
            ImportanceScore: chunk.ImportanceScore,
            Similarity:     chunk.Similarity,
            Metadata:       chunk.Metadata,
        }
    }
    
    return result, nil
}

func (m *MemoryManager) StoreChunks(ctx context.Context, agentID, sessionID uuid.UUID, content string, toolResults []ToolResult) {
    // Compute importance score (heuristic: length, user flags, etc.)
    importance := m.computeImportance(content, toolResults)
    
    // Only store if importance > threshold
    if importance < 0.3 {
        return
    }
    
    // Compute embedding
    embedding, err := m.llm.Embed(ctx, "all-MiniLM-L6-v2", content)
    if err != nil {
        return // Log error but don't fail
    }
    
    // Store chunk
    _, err = m.queries.CreateMemoryChunk(ctx, &db.MemoryChunk{
        AgentID:        agentID,
        SessionID:       &sessionID,
        Content:        content,
        Embedding:      embedding,
        ImportanceScore: importance,
    })
    if err != nil {
        // Log error
        return
    }
}

func (m *MemoryManager) computeImportance(content string, toolResults []ToolResult) float64 {
    score := 0.5 // Base score
    
    // Increase score based on content length (longer = more important)
    if len(content) > 500 {
        score += 0.2
    } else if len(content) > 200 {
        score += 0.1
    }
    
    // Increase score if tool results present (actionable information)
    if len(toolResults) > 0 {
        score += 0.2
    }
    
    // Cap at 1.0
    if score > 1.0 {
        score = 1.0
    }
    
    return score
}
```

### 4. API Layer (internal/api/)

**File: internal/api/models.go**

```go

package api

import "github.com/google/uuid"

// Request DTOs

type CreateAgentRequest struct {

Name         string   `json:"name" validate:"required,min=1,max=100"`

Description  *string  `json:"description"`

SystemPrompt string   `json:"system_prompt" validate:"required,min=10"`

ModelName    string   `json:"model_name" validate:"required"`

MemoryTable  *string  `json:"memory_table"`

EnabledTools []string `json:"enabled_tools"`

Config       map[string]interface{} `json:"config"`

}

type CreateSessionRequest struct {

AgentID        uuid.UUID              `json:"agent_id" validate:"required"`

ExternalUserID *string                 `json:"external_user_id"`

Metadata       map[string]interface{} `json:"metadata"`

}

type SendMessageRequest struct {

Role    string                 `json:"role" validate:"required,oneof=user system"`

Content string                 `json:"content" validate:"required,min=1"`

Stream  bool                   `json:"stream"`

Metadata map[string]interface{} `json:"metadata"`

}

// Response DTOs

type AgentResponse struct {

ID           uuid.UUID              `json:"id"`

Name         string                 `json:"name"`

Description  *string                `json:"description"`

SystemPrompt string                 `json:"system_prompt"`

ModelName    string                 `json:"model_name"`

MemoryTable  *string                `json:"memory_table"`

EnabledTools []string               `json:"enabled_tools"`

Config       map[string]interface{} `json:"config"`

CreatedAt    string                 `json:"created_at"`

UpdatedAt    string                 `json:"updated_at"`

}

type SessionResponse struct {

ID             uuid.UUID              `json:"id"`

AgentID        uuid.UUID              `json:"agent_id"`

ExternalUserID *string                `json:"external_user_id"`

Metadata       map[string]interface{} `json:"metadata"`

CreatedAt      string                 `json:"created_at"`

LastActivityAt string                `json:"last_activity_at"`

}

type MessageResponse struct {

ID         int64                  `json:"id"`

SessionID  uuid.UUID              `json:"session_id"`

Role       string                 `json:"role"`

Content    string                 `json:"content"`

ToolName   *string                `json:"tool_name"`

ToolCall

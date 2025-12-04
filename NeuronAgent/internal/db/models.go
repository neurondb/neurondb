package db

import (
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

type Agent struct {
	ID           uuid.UUID              `db:"id"`
	Name         string                 `db:"name"`
	Description  *string                `db:"description"`
	SystemPrompt string                 `db:"system_prompt"`
	ModelName    string                 `db:"model_name"`
	MemoryTable  *string                `db:"memory_table"`
	EnabledTools pq.StringArray         `db:"enabled_tools"`
	Config       JSONBMap               `db:"config"`
	CreatedAt    time.Time              `db:"created_at"`
	UpdatedAt    time.Time              `db:"updated_at"`
}

type Session struct {
	ID             uuid.UUID              `db:"id"`
	AgentID        uuid.UUID              `db:"agent_id"`
	ExternalUserID *string                `db:"external_user_id"`
	Metadata       JSONBMap               `db:"metadata"`
	CreatedAt      time.Time              `db:"created_at"`
	LastActivityAt time.Time              `db:"last_activity_at"`
}

type Message struct {
	ID         int64                  `db:"id"`
	SessionID  uuid.UUID              `db:"session_id"`
	Role       string                 `db:"role"`
	Content    string                 `db:"content"`
	ToolName   *string                `db:"tool_name"`
	ToolCallID *string                `db:"tool_call_id"`
	TokenCount *int                   `db:"token_count"`
	Metadata   map[string]interface{} `db:"metadata"`
	CreatedAt  time.Time              `db:"created_at"`
}

type MemoryChunk struct {
	ID              int64                  `db:"id"`
	AgentID         uuid.UUID              `db:"agent_id"`
	SessionID       *uuid.UUID             `db:"session_id"`
	MessageID       *int64                 `db:"message_id"`
	Content         string                 `db:"content"`
	Embedding       []float32              `db:"embedding"` // Will be converted to/from neurondb_vector
	ImportanceScore float64                `db:"importance_score"`
	Metadata        JSONBMap               `db:"metadata"`
	CreatedAt       time.Time              `db:"created_at"`
}

// MemoryChunkWithSimilarity includes similarity score from vector search
type MemoryChunkWithSimilarity struct {
	MemoryChunk
	Similarity float64 `db:"similarity"`
}

type Tool struct {
	Name          string                 `db:"name"`
	Description   string                 `db:"description"`
	ArgSchema     JSONBMap               `db:"arg_schema"`
	HandlerType   string                 `db:"handler_type"`
	HandlerConfig JSONBMap               `db:"handler_config"`
	Enabled       bool                   `db:"enabled"`
	CreatedAt     time.Time              `db:"created_at"`
	UpdatedAt     time.Time              `db:"updated_at"`
}

type Job struct {
	ID           int64                  `db:"id"`
	AgentID      *uuid.UUID             `db:"agent_id"`
	SessionID    *uuid.UUID             `db:"session_id"`
	Type         string                 `db:"type"`
	Status       string                 `db:"status"`
	Priority     int                    `db:"priority"`
	Payload      JSONBMap               `db:"payload"`
	Result       JSONBMap               `db:"result"`
	ErrorMessage *string                `db:"error_message"`
	RetryCount   int                    `db:"retry_count"`
	MaxRetries   int                    `db:"max_retries"`
	CreatedAt    time.Time              `db:"created_at"`
	UpdatedAt    time.Time              `db:"updated_at"`
	StartedAt    *time.Time             `db:"started_at"`
	CompletedAt  *time.Time             `db:"completed_at"`
}

type APIKey struct {
	ID              uuid.UUID              `db:"id"`
	KeyHash         string                 `db:"key_hash"`
	KeyPrefix       string                 `db:"key_prefix"`
	OrganizationID  *string                `db:"organization_id"`
	UserID          *string                `db:"user_id"`
	RateLimitPerMin int                    `db:"rate_limit_per_minute"`
	Roles           pq.StringArray         `db:"roles"`
	Metadata        JSONBMap               `db:"metadata"`
	CreatedAt       time.Time              `db:"created_at"`
	LastUsedAt      *time.Time             `db:"last_used_at"`
	ExpiresAt       *time.Time             `db:"expires_at"`
}

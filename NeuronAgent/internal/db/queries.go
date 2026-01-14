/*-------------------------------------------------------------------------
 *
 * queries.go
 *    Database queries for NeuronAgent
 *
 * Provides database query functions for agents, sessions, messages,
 * memory chunks, tools, jobs, and API keys.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
	"github.com/neurondb/NeuronAgent/internal/utils"
)

/* Agent queries */
const (
	createAgentQuery = `
		INSERT INTO neurondb_agent.agents 
		(name, description, system_prompt, model_name, memory_table, enabled_tools, config)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
		RETURNING id, created_at, updated_at`

	getAgentByIDQuery = `SELECT * FROM neurondb_agent.agents WHERE id = $1`

	listAgentsQuery = `SELECT * FROM neurondb_agent.agents ORDER BY created_at DESC`

	listAgentsWithFilterQuery = `
		SELECT * FROM neurondb_agent.agents 
		WHERE ($1::text IS NULL OR name ILIKE $1 OR description ILIKE $1)
		ORDER BY created_at DESC`

	updateAgentQuery = `
		UPDATE neurondb_agent.agents 
		SET name = $2, description = $3, system_prompt = $4, model_name = $5,
			memory_table = $6, enabled_tools = $7, config = $8::jsonb
		WHERE id = $1
		RETURNING updated_at`

	deleteAgentQuery = `DELETE FROM neurondb_agent.agents WHERE id = $1`
)

/* Session queries */
const (
	createSessionQuery = `
		INSERT INTO neurondb_agent.sessions (agent_id, external_user_id, metadata)
		VALUES ($1, $2, $3::jsonb)
		RETURNING id, created_at, last_activity_at`

	getSessionQuery = `SELECT * FROM neurondb_agent.sessions WHERE id = $1`

	listSessionsQuery = `
		SELECT * FROM neurondb_agent.sessions 
		WHERE agent_id = $1 
		ORDER BY last_activity_at DESC 
		LIMIT $2 OFFSET $3`

	listSessionsWithFilterQuery = `
		SELECT * FROM neurondb_agent.sessions 
		WHERE agent_id = $1 
		AND ($2::text IS NULL OR external_user_id = $2)
		AND ($3::timestamptz IS NULL OR created_at >= $3)
		AND ($4::timestamptz IS NULL OR created_at <= $4)
		ORDER BY last_activity_at DESC 
		LIMIT $5 OFFSET $6`

	updateSessionQuery = `
		UPDATE neurondb_agent.sessions 
		SET external_user_id = $2, metadata = $3::jsonb, last_activity_at = NOW()
		WHERE id = $1
		RETURNING id, agent_id, external_user_id, metadata, created_at, last_activity_at`

	deleteSessionQuery = `DELETE FROM neurondb_agent.sessions WHERE id = $1`
)

/* Message queries */
const (
	createMessageQuery = `
		INSERT INTO neurondb_agent.messages 
		(session_id, role, content, tool_name, tool_call_id, token_count, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
		RETURNING id, created_at`

	getMessagesQuery = `
		SELECT * FROM neurondb_agent.messages 
		WHERE session_id = $1 
		ORDER BY created_at ASC 
		LIMIT $2 OFFSET $3`

	getMessagesWithFilterQuery = `
		SELECT * FROM neurondb_agent.messages 
		WHERE session_id = $1 
		AND ($2::text IS NULL OR role = $2)
		AND ($3::text IS NULL OR content ILIKE $3)
		AND ($4::timestamptz IS NULL OR created_at >= $4)
		AND ($5::timestamptz IS NULL OR created_at <= $5)
		ORDER BY created_at ASC 
		LIMIT $6 OFFSET $7`

	getRecentMessagesQuery = `
		SELECT * FROM neurondb_agent.messages 
		WHERE session_id = $1 
		ORDER BY created_at DESC 
		LIMIT $2`

	getMessageByIDQuery = `SELECT * FROM neurondb_agent.messages WHERE id = $1`

	updateMessageQuery = `
		UPDATE neurondb_agent.messages 
		SET content = $2, metadata = $3::jsonb
		WHERE id = $1
		RETURNING id, session_id, role, content, tool_name, tool_call_id, token_count, metadata, created_at`

	deleteMessageQuery = `DELETE FROM neurondb_agent.messages WHERE id = $1`
)

/* Memory chunk queries */
const (
	createMemoryChunkQuery = `
		INSERT INTO neurondb_agent.memory_chunks 
		(agent_id, session_id, message_id, content, embedding, importance_score, metadata)
		VALUES ($1, $2, $3, $4, $5::vector, $6, $7::jsonb)
		RETURNING id, created_at`

	searchMemoryQuery = `
		SELECT id, agent_id, session_id, message_id, content, importance_score, metadata, created_at,
			   1 - (embedding <=> $1::vector) AS similarity
		FROM neurondb_agent.memory_chunks
		WHERE agent_id = $2
		ORDER BY embedding <=> $1::vector
		LIMIT $3`

	getMemoryChunkQuery = `SELECT id, agent_id, session_id, message_id, content, embedding, importance_score, metadata, created_at FROM neurondb_agent.memory_chunks WHERE id = $1`

	listMemoryChunksQuery = `
		SELECT id, agent_id, session_id, message_id, content, embedding, importance_score, metadata, created_at
		FROM neurondb_agent.memory_chunks
		WHERE agent_id = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3`

	deleteMemoryChunkQuery = `DELETE FROM neurondb_agent.memory_chunks WHERE id = $1`
)

/* Tool queries */
const (
	createToolQuery = `
		INSERT INTO neurondb_agent.tools 
		(name, description, arg_schema, handler_type, handler_config, enabled)
		VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6)
		RETURNING created_at, updated_at`

	getToolQuery = `SELECT * FROM neurondb_agent.tools WHERE name = $1`

	listToolsQuery = `SELECT * FROM neurondb_agent.tools WHERE enabled = true ORDER BY name`

	listToolsWithFilterQuery = `
		SELECT * FROM neurondb_agent.tools 
		WHERE ($1::boolean IS NULL OR enabled = $1)
		AND ($2::text IS NULL OR name ILIKE $2 OR description ILIKE $2)
		AND ($3::text IS NULL OR handler_type = $3)
		ORDER BY name`

	updateToolQuery = `
		UPDATE neurondb_agent.tools 
		SET description = $2, arg_schema = $3::jsonb, handler_type = $4, 
			handler_config = $5::jsonb, enabled = $6
		WHERE name = $1
		RETURNING updated_at`

	deleteToolQuery = `DELETE FROM neurondb_agent.tools WHERE name = $1`
)

/* Job queries */
const (
	createJobQuery = `
		INSERT INTO neurondb_agent.jobs 
		(agent_id, session_id, type, status, priority, payload, max_retries)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
		RETURNING id, created_at, updated_at`

	getJobQuery = `SELECT * FROM neurondb_agent.jobs WHERE id = $1`

	claimJobQuery = `
		UPDATE neurondb_agent.jobs 
		SET status = 'running', started_at = NOW(), updated_at = NOW()
		WHERE id = (
			SELECT id FROM neurondb_agent.jobs
			WHERE status = 'queued'
			ORDER BY priority DESC, created_at ASC
			LIMIT 1
			FOR UPDATE SKIP LOCKED
		)
		RETURNING id, agent_id, session_id, type, status, priority, payload, 
		          result, error_message, retry_count, max_retries, 
		          created_at, updated_at, started_at, completed_at`

	updateJobQuery = `
		UPDATE neurondb_agent.jobs 
		SET status = $2, result = $3::jsonb, error_message = $4, 
			retry_count = $5, completed_at = $6, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	listJobsQuery = `
		SELECT * FROM neurondb_agent.jobs 
		WHERE ($1::uuid IS NULL OR agent_id = $1)
		AND ($2::uuid IS NULL OR session_id = $2)
		ORDER BY created_at DESC 
		LIMIT $3 OFFSET $4`
)

/* API Key queries */
const (
	createAPIKeyQuery = `
		INSERT INTO neurondb_agent.api_keys 
		(key_hash, key_prefix, organization_id, user_id, principal_id, rate_limit_per_minute, roles, metadata, expires_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
		RETURNING id, created_at`

	getAPIKeyByPrefixQuery = `SELECT id, key_hash, key_prefix, organization_id, user_id, principal_id, rate_limit_per_minute, roles, metadata, created_at, last_used_at, expires_at FROM neurondb_agent.api_keys WHERE key_prefix = $1`

	getAPIKeyByIDQuery = `SELECT * FROM neurondb_agent.api_keys WHERE id = $1`

	listAPIKeysQuery = `
		SELECT * FROM neurondb_agent.api_keys 
		WHERE ($1::text IS NULL OR organization_id = $1)
		ORDER BY created_at DESC`

	updateAPIKeyLastUsedQuery = `
		UPDATE neurondb_agent.api_keys 
		SET last_used_at = NOW()
		WHERE id = $1`

	deleteAPIKeyQuery = `DELETE FROM neurondb_agent.api_keys WHERE id = $1`
)

/* NeuronDB function wrappers */
const (
	embedTextQuery   = `SELECT neurondb_embed($1, $2) AS embedding`
	llmGenerateQuery = `SELECT neurondb_llm_generate($1, $2, $3) AS output`
)

type Queries struct {
	DB       *sqlx.DB
	connInfo func() string
}

/* GetDB returns the database connection (for compatibility) */
func (q *Queries) GetDB() *sqlx.DB {
	return q.DB
}

func NewQueries(db *sqlx.DB) *Queries {
	return &Queries{
		DB: db,
		connInfo: func() string {
			return "unknown database connection"
		},
	}
}

/* SetConnInfoFunc sets a function to retrieve connection info for error messages */
func (q *Queries) SetConnInfoFunc(fn func() string) {
	q.connInfo = fn
}

/* getConnInfoString returns connection info string */
func (q *Queries) getConnInfoString() string {
	if q.connInfo != nil {
		return q.connInfo()
	}
	return "unknown database connection"
}

/* formatQueryError formats a detailed query error message */
func (q *Queries) formatQueryError(operation string, query string, paramCount int, table string, err error) error {
	queryContext := utils.FormatQueryContext(query, paramCount, operation, table)
	connInfo := q.getConnInfoString()
	return fmt.Errorf("query execution failed on %s: %s, error=%w", connInfo, queryContext, err)
}

/* Agent methods */
func (q *Queries) CreateAgent(ctx context.Context, agent *Agent) error {
	params := []interface{}{agent.Name, agent.Description, agent.SystemPrompt, agent.ModelName,
		agent.MemoryTable, agent.EnabledTools, agent.Config}
	err := q.DB.GetContext(ctx, agent, createAgentQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createAgentQuery, len(params), "neurondb_agent.agents", err)
	}
	return nil
}

func (q *Queries) GetAgentByID(ctx context.Context, id uuid.UUID) (*Agent, error) {
	var agent Agent
	err := q.DB.GetContext(ctx, &agent, getAgentByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("agent not found on %s: query='%s', agent_id='%s', table='neurondb_agent.agents', error=%w",
			q.getConnInfoString(), getAgentByIDQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getAgentByIDQuery, 1, "neurondb_agent.agents", err)
	}
	return &agent, nil
}

func (q *Queries) ListAgents(ctx context.Context) ([]Agent, error) {
	var agents []Agent
	err := q.DB.SelectContext(ctx, &agents, listAgentsQuery)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listAgentsQuery, 0, "neurondb_agent.agents", err)
	}
	return agents, nil
}

func (q *Queries) ListAgentsWithFilter(ctx context.Context, search *string) ([]Agent, error) {
	var agents []Agent
	var searchPattern *string
	if search != nil && *search != "" {
		pattern := "%" + *search + "%"
		searchPattern = &pattern
	}
	err := q.DB.SelectContext(ctx, &agents, listAgentsWithFilterQuery, searchPattern)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listAgentsWithFilterQuery, 1, "neurondb_agent.agents", err)
	}
	return agents, nil
}

func (q *Queries) UpdateAgent(ctx context.Context, agent *Agent) error {
	params := []interface{}{agent.ID, agent.Name, agent.Description, agent.SystemPrompt, agent.ModelName,
		agent.MemoryTable, agent.EnabledTools, agent.Config}
	err := q.DB.GetContext(ctx, agent, updateAgentQuery, params...)
	if err != nil {
		return q.formatQueryError("UPDATE", updateAgentQuery, len(params), "neurondb_agent.agents", err)
	}
	return nil
}

func (q *Queries) DeleteAgent(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteAgentQuery, id)
	if err != nil {
		return q.formatQueryError("DELETE", deleteAgentQuery, 1, "neurondb_agent.agents", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', agent_id='%s', table='neurondb_agent.agents', error=%w",
			q.getConnInfoString(), deleteAgentQuery, id.String(), err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("agent not found on %s: query='%s', agent_id='%s', table='neurondb_agent.agents', rows_affected=0",
			q.getConnInfoString(), deleteAgentQuery, id.String())
	}
	return nil
}

/* Session methods */
func (q *Queries) CreateSession(ctx context.Context, session *Session) error {
	params := []interface{}{session.AgentID, session.ExternalUserID, session.Metadata}
	err := q.DB.GetContext(ctx, session, createSessionQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createSessionQuery, len(params), "neurondb_agent.sessions", err)
	}
	return nil
}

func (q *Queries) GetSession(ctx context.Context, id uuid.UUID) (*Session, error) {
	var session Session
	err := q.DB.GetContext(ctx, &session, getSessionQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("session not found on %s: query='%s', session_id='%s', table='neurondb_agent.sessions', error=%w",
			q.getConnInfoString(), getSessionQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getSessionQuery, 1, "neurondb_agent.sessions", err)
	}
	return &session, nil
}

func (q *Queries) ListSessions(ctx context.Context, agentID uuid.UUID, limit, offset int) ([]Session, error) {
	var sessions []Session
	params := []interface{}{agentID, limit, offset}
	err := q.DB.SelectContext(ctx, &sessions, listSessionsQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listSessionsQuery, len(params), "neurondb_agent.sessions", err)
	}
	return sessions, nil
}

func (q *Queries) ListSessionsWithFilter(ctx context.Context, agentID uuid.UUID, externalUserID *string, startDate, endDate *string, limit, offset int) ([]Session, error) {
	var sessions []Session
	var startDateParsed, endDateParsed interface{}
	if startDate != nil && *startDate != "" {
		startDateParsed = *startDate
	}
	if endDate != nil && *endDate != "" {
		endDateParsed = *endDate
	}
	params := []interface{}{agentID, externalUserID, startDateParsed, endDateParsed, limit, offset}
	err := q.DB.SelectContext(ctx, &sessions, listSessionsWithFilterQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listSessionsWithFilterQuery, len(params), "neurondb_agent.sessions", err)
	}
	return sessions, nil
}

func (q *Queries) UpdateSession(ctx context.Context, session *Session) error {
	params := []interface{}{session.ID, session.ExternalUserID, session.Metadata}
	err := q.DB.GetContext(ctx, session, updateSessionQuery, params...)
	if err == sql.ErrNoRows {
		return fmt.Errorf("session not found on %s: query='%s', session_id='%s', table='neurondb_agent.sessions', error=%w",
			q.getConnInfoString(), updateSessionQuery, session.ID.String(), err)
	}
	if err != nil {
		return q.formatQueryError("UPDATE", updateSessionQuery, len(params), "neurondb_agent.sessions", err)
	}
	return nil
}

func (q *Queries) DeleteSession(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteSessionQuery, id)
	if err != nil {
		return q.formatQueryError("DELETE", deleteSessionQuery, 1, "neurondb_agent.sessions", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', session_id='%s', table='neurondb_agent.sessions', error=%w",
			q.getConnInfoString(), deleteSessionQuery, id.String(), err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("session not found on %s: query='%s', session_id='%s', table='neurondb_agent.sessions', rows_affected=0",
			q.getConnInfoString(), deleteSessionQuery, id.String())
	}
	return nil
}

/* Message methods */
func (q *Queries) CreateMessage(ctx context.Context, message *Message) (*Message, error) {
	params := []interface{}{message.SessionID, message.Role, message.Content, message.ToolName,
		message.ToolCallID, message.TokenCount, message.Metadata}
	err := q.DB.GetContext(ctx, message, createMessageQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("INSERT", createMessageQuery, len(params), "neurondb_agent.messages", err)
	}
	return message, nil
}

func (q *Queries) GetMessages(ctx context.Context, sessionID uuid.UUID, limit, offset int) ([]Message, error) {
	var messages []Message
	params := []interface{}{sessionID, limit, offset}
	err := q.DB.SelectContext(ctx, &messages, getMessagesQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", getMessagesQuery, len(params), "neurondb_agent.messages", err)
	}
	return messages, nil
}

func (q *Queries) GetMessagesWithFilter(ctx context.Context, sessionID uuid.UUID, role, contentSearch *string, startDate, endDate *string, limit, offset int) ([]Message, error) {
	var messages []Message
	var contentPattern *string
	if contentSearch != nil && *contentSearch != "" {
		pattern := "%" + *contentSearch + "%"
		contentPattern = &pattern
	}
	var startDateParsed, endDateParsed interface{}
	if startDate != nil && *startDate != "" {
		startDateParsed = *startDate
	}
	if endDate != nil && *endDate != "" {
		endDateParsed = *endDate
	}
	params := []interface{}{sessionID, role, contentPattern, startDateParsed, endDateParsed, limit, offset}
	err := q.DB.SelectContext(ctx, &messages, getMessagesWithFilterQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", getMessagesWithFilterQuery, len(params), "neurondb_agent.messages", err)
	}
	return messages, nil
}

func (q *Queries) GetMessageByID(ctx context.Context, id int64) (*Message, error) {
	var message Message
	err := q.DB.GetContext(ctx, &message, getMessageByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("message not found on %s: query='%s', message_id=%d, table='neurondb_agent.messages', error=%w",
			q.getConnInfoString(), getMessageByIDQuery, id, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getMessageByIDQuery, 1, "neurondb_agent.messages", err)
	}
	return &message, nil
}

func (q *Queries) UpdateMessage(ctx context.Context, message *Message) error {
	params := []interface{}{message.ID, message.Content, message.Metadata}
	err := q.DB.GetContext(ctx, message, updateMessageQuery, params...)
	if err == sql.ErrNoRows {
		return fmt.Errorf("message not found on %s: query='%s', message_id=%d, table='neurondb_agent.messages', error=%w",
			q.getConnInfoString(), updateMessageQuery, message.ID, err)
	}
	if err != nil {
		return q.formatQueryError("UPDATE", updateMessageQuery, len(params), "neurondb_agent.messages", err)
	}
	return nil
}

func (q *Queries) DeleteMessage(ctx context.Context, id int64) error {
	result, err := q.DB.ExecContext(ctx, deleteMessageQuery, id)
	if err != nil {
		return q.formatQueryError("DELETE", deleteMessageQuery, 1, "neurondb_agent.messages", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', message_id=%d, table='neurondb_agent.messages', error=%w",
			q.getConnInfoString(), deleteMessageQuery, id, err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("message not found on %s: query='%s', message_id=%d, table='neurondb_agent.messages', rows_affected=0",
			q.getConnInfoString(), deleteMessageQuery, id)
	}
	return nil
}

func (q *Queries) GetRecentMessages(ctx context.Context, sessionID uuid.UUID, limit int) ([]Message, error) {
	var messages []Message
	params := []interface{}{sessionID, limit}
	err := q.DB.SelectContext(ctx, &messages, getRecentMessagesQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", getRecentMessagesQuery, len(params), "neurondb_agent.messages", err)
	}
	return messages, nil
}

/* Memory chunk methods */
func (q *Queries) CreateMemoryChunk(ctx context.Context, chunk *MemoryChunk) (*MemoryChunk, error) {
	embeddingStr := formatVector(chunk.Embedding)
	params := []interface{}{chunk.AgentID, chunk.SessionID, chunk.MessageID, chunk.Content,
		embeddingStr, chunk.ImportanceScore, chunk.Metadata}
	err := q.DB.GetContext(ctx, chunk, createMemoryChunkQuery, params...)
	if err != nil {
		embeddingDim := len(chunk.Embedding)
		return nil, fmt.Errorf("memory chunk creation failed on %s: query='%s', params_count=%d, agent_id='%s', session_id='%s', content_length=%d, embedding_dimension=%d, importance_score=%.2f, table='neurondb_agent.memory_chunks', error=%w",
			q.getConnInfoString(), createMemoryChunkQuery, len(params), chunk.AgentID.String(),
			utils.SanitizeValue(chunk.SessionID), len(chunk.Content), embeddingDim, chunk.ImportanceScore, err)
	}
	return chunk, nil
}

func (q *Queries) GetMemoryChunk(ctx context.Context, id int64) (*MemoryChunk, error) {
	var chunk MemoryChunk
	err := q.DB.GetContext(ctx, &chunk, getMemoryChunkQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("memory chunk not found on %s: query='%s', chunk_id=%d, table='neurondb_agent.memory_chunks', error=%w",
			q.getConnInfoString(), getMemoryChunkQuery, id, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getMemoryChunkQuery, 1, "neurondb_agent.memory_chunks", err)
	}
	return &chunk, nil
}

func (q *Queries) ListMemoryChunks(ctx context.Context, agentID uuid.UUID, limit, offset int) ([]MemoryChunk, error) {
	var chunks []MemoryChunk
	params := []interface{}{agentID, limit, offset}
	err := q.DB.SelectContext(ctx, &chunks, listMemoryChunksQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listMemoryChunksQuery, len(params), "neurondb_agent.memory_chunks", err)
	}
	return chunks, nil
}

func (q *Queries) DeleteMemoryChunk(ctx context.Context, id int64) error {
	result, err := q.DB.ExecContext(ctx, deleteMemoryChunkQuery, id)
	if err != nil {
		return q.formatQueryError("DELETE", deleteMemoryChunkQuery, 1, "neurondb_agent.memory_chunks", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', chunk_id=%d, table='neurondb_agent.memory_chunks', error=%w",
			q.getConnInfoString(), deleteMemoryChunkQuery, id, err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("memory chunk not found on %s: query='%s', chunk_id=%d, table='neurondb_agent.memory_chunks', rows_affected=0",
			q.getConnInfoString(), deleteMemoryChunkQuery, id)
	}
	return nil
}

func (q *Queries) SearchMemory(ctx context.Context, agentID uuid.UUID, queryEmbedding []float32, topK int) ([]MemoryChunkWithSimilarity, error) {
	embeddingStr := formatVector(queryEmbedding)
	var chunks []MemoryChunkWithSimilarity
	params := []interface{}{embeddingStr, agentID, topK}
	err := q.DB.SelectContext(ctx, &chunks, searchMemoryQuery, params...)
	if err != nil {
		embeddingDim := len(queryEmbedding)
		return nil, fmt.Errorf("memory search failed on %s: query='%s', params_count=%d, agent_id='%s', query_embedding_dimension=%d, top_k=%d, table='neurondb_agent.memory_chunks', error=%w",
			q.getConnInfoString(), searchMemoryQuery, len(params), agentID.String(), embeddingDim, topK, err)
	}
	return chunks, nil
}

/* Tool methods */
func (q *Queries) CreateTool(ctx context.Context, tool *Tool) error {
	params := []interface{}{tool.Name, tool.Description, tool.ArgSchema, tool.HandlerType,
		tool.HandlerConfig, tool.Enabled}
	err := q.DB.GetContext(ctx, tool, createToolQuery, params...)
	if err != nil {
		return fmt.Errorf("tool creation failed on %s: query='%s', params_count=%d, tool_name='%s', handler_type='%s', enabled=%v, table='neurondb_agent.tools', error=%w",
			q.getConnInfoString(), createToolQuery, len(params), tool.Name, tool.HandlerType, tool.Enabled, err)
	}
	return nil
}

func (q *Queries) GetTool(ctx context.Context, name string) (*Tool, error) {
	var tool Tool
	err := q.DB.GetContext(ctx, &tool, getToolQuery, name)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("tool not found on %s: query='%s', tool_name='%s', table='neurondb_agent.tools', error=%w",
			q.getConnInfoString(), getToolQuery, name, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getToolQuery, 1, "neurondb_agent.tools", err)
	}
	return &tool, nil
}

func (q *Queries) ListTools(ctx context.Context) ([]Tool, error) {
	var tools []Tool
	err := q.DB.SelectContext(ctx, &tools, listToolsQuery)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listToolsQuery, 0, "neurondb_agent.tools", err)
	}
	return tools, nil
}

func (q *Queries) ListToolsWithFilter(ctx context.Context, enabled *bool, search, handlerType *string) ([]Tool, error) {
	var tools []Tool
	var searchPattern *string
	if search != nil && *search != "" {
		pattern := "%" + *search + "%"
		searchPattern = &pattern
	}
	params := []interface{}{enabled, searchPattern, handlerType}
	err := q.DB.SelectContext(ctx, &tools, listToolsWithFilterQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listToolsWithFilterQuery, len(params), "neurondb_agent.tools", err)
	}
	return tools, nil
}

func (q *Queries) UpdateTool(ctx context.Context, tool *Tool) error {
	params := []interface{}{tool.Name, tool.Description, tool.ArgSchema, tool.HandlerType,
		tool.HandlerConfig, tool.Enabled}
	err := q.DB.GetContext(ctx, tool, updateToolQuery, params...)
	if err != nil {
		return fmt.Errorf("tool update failed on %s: query='%s', params_count=%d, tool_name='%s', handler_type='%s', enabled=%v, table='neurondb_agent.tools', error=%w",
			q.getConnInfoString(), updateToolQuery, len(params), tool.Name, tool.HandlerType, tool.Enabled, err)
	}
	return nil
}

func (q *Queries) DeleteTool(ctx context.Context, name string) error {
	result, err := q.DB.ExecContext(ctx, deleteToolQuery, name)
	if err != nil {
		return q.formatQueryError("DELETE", deleteToolQuery, 1, "neurondb_agent.tools", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', tool_name='%s', table='neurondb_agent.tools', error=%w",
			q.getConnInfoString(), deleteToolQuery, name, err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("tool not found on %s: query='%s', tool_name='%s', table='neurondb_agent.tools', rows_affected=0",
			q.getConnInfoString(), deleteToolQuery, name)
	}
	return nil
}

/* Job methods */
func (q *Queries) CreateJob(ctx context.Context, job *Job) (*Job, error) {
	params := []interface{}{job.AgentID, job.SessionID, job.Type, job.Status, job.Priority,
		job.Payload, job.MaxRetries}
	err := q.DB.GetContext(ctx, job, createJobQuery, params...)
	if err != nil {
		agentIDStr := utils.SanitizeValue(job.AgentID)
		sessionIDStr := utils.SanitizeValue(job.SessionID)
		return nil, fmt.Errorf("job creation failed on %s: query='%s', params_count=%d, job_type='%s', status='%s', priority=%d, agent_id=%s, session_id=%s, max_retries=%d, table='neurondb_agent.jobs', error=%w",
			q.getConnInfoString(), createJobQuery, len(params), job.Type, job.Status, job.Priority,
			agentIDStr, sessionIDStr, job.MaxRetries, err)
	}
	return job, nil
}

func (q *Queries) GetJob(ctx context.Context, id int64) (*Job, error) {
	var job Job
	err := q.DB.GetContext(ctx, &job, getJobQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("job not found on %s: query='%s', job_id=%d, table='neurondb_agent.jobs', error=%w",
			q.getConnInfoString(), getJobQuery, id, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getJobQuery, 1, "neurondb_agent.jobs", err)
	}
	return &job, nil
}

func (q *Queries) ClaimJob(ctx context.Context) (*Job, error) {
	var job Job
	err := q.DB.GetContext(ctx, &job, claimJobQuery)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, q.formatQueryError("UPDATE", claimJobQuery, 0, "neurondb_agent.jobs", err)
	}
	return &job, nil
}

func (q *Queries) UpdateJob(ctx context.Context, id int64, status string, result map[string]interface{}, errorMsg *string, retryCount int, completedAt *sql.NullTime) error {
	var completedAtVal interface{}
	if completedAt != nil && completedAt.Valid {
		completedAtVal = completedAt.Time
	} else {
		completedAtVal = nil
	}
	params := []interface{}{id, status, result, errorMsg, retryCount, completedAtVal}
	_, err := q.DB.ExecContext(ctx, updateJobQuery, params...)
	if err != nil {
		errorMsgStr := utils.SanitizeValue(errorMsg)
		return fmt.Errorf("job update failed on %s: query='%s', params_count=%d, job_id=%d, status='%s', retry_count=%d, error_message=%s, table='neurondb_agent.jobs', error=%w",
			q.getConnInfoString(), updateJobQuery, len(params), id, status, retryCount, errorMsgStr, err)
	}
	return nil
}

func (q *Queries) ListJobs(ctx context.Context, agentID *uuid.UUID, sessionID *uuid.UUID, limit, offset int) ([]Job, error) {
	var jobs []Job
	params := []interface{}{agentID, sessionID, limit, offset}
	err := q.DB.SelectContext(ctx, &jobs, listJobsQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listJobsQuery, len(params), "neurondb_agent.jobs", err)
	}
	return jobs, nil
}

/* API Key methods */
func (q *Queries) CreateAPIKey(ctx context.Context, apiKey *APIKey) error {
	metadataValue, err := apiKey.Metadata.Value()
	if err != nil {
		return fmt.Errorf("failed to convert metadata: %w", err)
	}

	params := []interface{}{apiKey.KeyHash, apiKey.KeyPrefix, apiKey.OrganizationID, apiKey.UserID,
		apiKey.PrincipalID, apiKey.RateLimitPerMin, apiKey.Roles, metadataValue, apiKey.ExpiresAt}
	err = q.DB.GetContext(ctx, apiKey, createAPIKeyQuery, params...)
	if err != nil {
		return fmt.Errorf("API key creation failed on %s: query='%s', params_count=%d, key_prefix='%s', organization_id=%s, user_id=%s, principal_id=%s, rate_limit_per_min=%d, table='neurondb_agent.api_keys', error=%w",
			q.getConnInfoString(), createAPIKeyQuery, len(params), apiKey.KeyPrefix,
			utils.SanitizeValue(apiKey.OrganizationID), utils.SanitizeValue(apiKey.UserID), utils.SanitizeValue(apiKey.PrincipalID), apiKey.RateLimitPerMin, err)
	}
	return nil
}

func (q *Queries) GetAPIKeyByPrefix(ctx context.Context, prefix string) (*APIKey, error) {
	var apiKey APIKey
	err := q.DB.GetContext(ctx, &apiKey, getAPIKeyByPrefixQuery, prefix)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("API key not found on %s: query='%s', key_prefix='%s', table='neurondb_agent.api_keys', error=%w",
				q.getConnInfoString(), getAPIKeyByPrefixQuery, prefix, err)
		}
		return nil, fmt.Errorf("API key lookup failed on %s: query='%s', key_prefix='%s', error=%w (error_type=%T)",
			q.getConnInfoString(), getAPIKeyByPrefixQuery, prefix, err, err)
	}
	return &apiKey, nil
}

func (q *Queries) GetAPIKeyByID(ctx context.Context, id uuid.UUID) (*APIKey, error) {
	var apiKey APIKey
	err := q.DB.GetContext(ctx, &apiKey, getAPIKeyByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("API key not found on %s: query='%s', key_id='%s', table='neurondb_agent.api_keys', error=%w",
			q.getConnInfoString(), getAPIKeyByIDQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getAPIKeyByIDQuery, 1, "neurondb_agent.api_keys", err)
	}
	return &apiKey, nil
}

func (q *Queries) ListAPIKeys(ctx context.Context, organizationID *string) ([]APIKey, error) {
	var keys []APIKey
	err := q.DB.SelectContext(ctx, &keys, listAPIKeysQuery, organizationID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listAPIKeysQuery, 1, "neurondb_agent.api_keys", err)
	}
	return keys, nil
}

func (q *Queries) UpdateAPIKeyLastUsed(ctx context.Context, id uuid.UUID) error {
	_, err := q.DB.ExecContext(ctx, updateAPIKeyLastUsedQuery, id)
	if err != nil {
		return q.formatQueryError("UPDATE", updateAPIKeyLastUsedQuery, 1, "neurondb_agent.api_keys", err)
	}
	return nil
}

func (q *Queries) DeleteAPIKey(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteAPIKeyQuery, id)
	if err != nil {
		return q.formatQueryError("DELETE", deleteAPIKeyQuery, 1, "neurondb_agent.api_keys", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', key_id='%s', table='neurondb_agent.api_keys', error=%w",
			q.getConnInfoString(), deleteAPIKeyQuery, id.String(), err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("API key not found on %s: query='%s', key_id='%s', table='neurondb_agent.api_keys', rows_affected=0",
			q.getConnInfoString(), deleteAPIKeyQuery, id.String())
	}
	return nil
}

/* formatVector formats vector for PostgreSQL */
func formatVector(vec []float32) string {
	if len(vec) == 0 {
		return "[]"
	}
	result := "["
	for i, v := range vec {
		if i > 0 {
			result += ","
		}
		result += fmt.Sprintf("%.6f", v)
	}
	result += "]"
	return result
}

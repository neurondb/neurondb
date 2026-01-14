/*-------------------------------------------------------------------------
 *
 * registry.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/registry.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/collaboration"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* Context key types (duplicated from agent to avoid import cycle) */
type contextKey string

const (
	agentIDContextKey   contextKey = "agent_id"
	sessionIDContextKey contextKey = "session_id"
)

/* GetAgentIDFromContext gets agent ID from context (duplicated from agent to avoid import cycle) */
func GetAgentIDFromContext(ctx context.Context) (uuid.UUID, bool) {
	agentID, ok := ctx.Value(agentIDContextKey).(uuid.UUID)
	return agentID, ok
}

/* GetSessionIDFromContext gets session ID from context (duplicated from agent to avoid import cycle) */
func GetSessionIDFromContext(ctx context.Context) (uuid.UUID, bool) {
	sessionID, ok := ctx.Value(sessionIDContextKey).(uuid.UUID)
	return sessionID, ok
}

/* GetPrincipalFromContext gets principal from context (duplicated from agent to avoid import cycle) */
func GetPrincipalFromContext(ctx context.Context) *db.Principal {
	principal, ok := ctx.Value(principalContextKey).(*db.Principal)
	if !ok {
		return nil
	}
	return principal
}

const (
	principalContextKey contextKey = "principal"
)

/* Registry manages tool registration and execution */
type Registry struct {
	queries     *db.Queries
	db          *db.DB
	handlers    map[string]ToolHandler
	auditLogger *auth.AuditLogger
	browserTool *BrowserTool
	mu          sync.RWMutex
}

/* NewRegistry creates a new tool registry */
func NewRegistry(queries *db.Queries, database *db.DB) *Registry {
	registry := &Registry{
		queries:     queries,
		db:          database,
		handlers:    make(map[string]ToolHandler),
		auditLogger: auth.NewAuditLogger(queries),
	}

	/* Register built-in handlers */
	sqlTool := NewSQLTool(queries)
	sqlTool.db = database
	registry.RegisterHandler("sql", sqlTool)
	registry.RegisterHandler("http", NewHTTPTool())
	registry.RegisterHandler("code", NewCodeTool())
	registry.RegisterHandler("shell", NewShellTool())
	browserTool := NewBrowserTool()
	browserTool.SetDB(queries, database)
	registry.RegisterHandler("browser", browserTool)
	registry.browserTool = browserTool /* Store for cleanup */
	registry.RegisterHandler("visualization", NewVisualizationTool())

	return registry
}

/* NewRegistryWithVFS creates a registry with virtual file system support */
func NewRegistryWithVFS(queries *db.Queries, database *db.DB, vfs interface{}) *Registry {
	registry := NewRegistry(queries, database)

	/* Register filesystem tool if VFS is provided */
	if vfsInstance, ok := vfs.(VirtualFileSystemInterface); ok {
		registry.RegisterHandler("filesystem", NewFileSystemTool(vfsInstance))
	}

	return registry
}

/* NewRegistryWithAllFeatures creates a registry with all new tools */
func NewRegistryWithAllFeatures(queries *db.Queries, database *db.DB, vfs interface{}, hierMemory interface{}, workspace interface{}) *Registry {
	registry := NewRegistry(queries, database)

	/* Register filesystem tool if VFS is provided */
	if vfsInstance, ok := vfs.(VirtualFileSystemInterface); ok {
		registry.RegisterHandler("filesystem", NewFileSystemTool(vfsInstance))
	}

	/* Register memory tool if hierarchical memory is provided */
	if hierMemoryInstance, ok := hierMemory.(HierarchicalMemoryInterface); ok {
		registry.RegisterHandler("memory", NewMemoryTool(hierMemoryInstance))
	}

	/* Register collaboration tool if workspace is provided */
	if workspaceInstance, ok := workspace.(*collaboration.WorkspaceManager); ok {
		registry.RegisterHandler("collaboration", NewCollaborationTool(workspaceInstance))
	}

	/* Register multimodal tool */
	registry.RegisterHandler("multimodal", NewMultimodalTool())

	return registry
}

/* NewRegistryWithNeuronDB creates a new tool registry with NeuronDB clients */
func NewRegistryWithNeuronDB(queries *db.Queries, database *db.DB, neurondbClient interface{}) *Registry {
	registry := NewRegistry(queries, database)

	/* Register NeuronDB tool handlers if client is provided */
	if client, ok := neurondbClient.(*neurondb.Client); ok {
		if client.ML != nil {
			registry.RegisterHandler("ml", NewMLTool(client.ML))
		}
		if client.Vector != nil {
			registry.RegisterHandler("vector", NewVectorTool(client.Vector))
		}
		if client.RAG != nil {
			registry.RegisterHandler("rag", NewRAGTool(client.RAG))
		}
		if client.Analytics != nil {
			registry.RegisterHandler("analytics", NewAnalyticsTool(client.Analytics))
		}
		if client.HybridSearch != nil {
			registry.RegisterHandler("hybrid_search", NewHybridSearchTool(client.HybridSearch))
		}
		if client.Reranking != nil {
			registry.RegisterHandler("reranking", NewRerankingTool(client.Reranking))
		}
	}

	return registry
}

/* RegisterHandler registers a tool handler for a specific handler type */
func (r *Registry) RegisterHandler(handlerType string, handler ToolHandler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.handlers[handlerType] = handler
}

/* Get retrieves a tool from the database */
/* Implements agent.ToolRegistry interface */
func (r *Registry) Get(ctx context.Context, name string) (*db.Tool, error) {
	/* Validate input */
	if name == "" {
		return nil, fmt.Errorf("tool retrieval failed: tool_name_empty=true")
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return nil, fmt.Errorf("tool retrieval cancelled: tool_name='%s', context_error=%w", name, ctx.Err())
	}

	tool, err := r.queries.GetTool(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("tool retrieval failed: tool_name='%s', error=%w", name, err)
	}
	return tool, nil
}

/* Execute executes a tool with the given arguments */
/* Implements agent.ToolRegistry interface */
func (r *Registry) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	return r.ExecuteTool(ctx, tool, args)
}

/* ExecuteTool executes a tool with the given arguments (internal method) */
func (r *Registry) ExecuteTool(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	/* Validate inputs */
	if tool == nil {
		return "", fmt.Errorf("tool execution failed: tool_nil=true")
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return "", fmt.Errorf("tool execution cancelled: tool_name='%s', context_error=%w", tool.Name, ctx.Err())
	}

	if !tool.Enabled {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', enabled=false, args_count=%d, arg_keys=[%v]",
			tool.Name, tool.HandlerType, len(args), argKeys)
	}

	/* Validate arguments */
	if err := ValidateArgs(args, tool.ArgSchema); err != nil {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool validation failed: tool_name='%s', handler_type='%s', args_count=%d, arg_keys=[%v], validation_error='%v'",
			tool.Name, tool.HandlerType, len(args), argKeys, err)
	}

	/* Get handler */
	r.mu.RLock()
	handler, exists := r.handlers[tool.HandlerType]
	r.mu.RUnlock()

	if !exists {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		availableHandlers := make([]string, 0, len(r.handlers))
		for k := range r.handlers {
			availableHandlers = append(availableHandlers, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', handler_not_found=true, args_count=%d, arg_keys=[%v], available_handlers=[%v]",
			tool.Name, tool.HandlerType, len(args), argKeys, availableHandlers)
	}

	/* Check context cancellation before execution */
	if ctx.Err() != nil {
		return "", fmt.Errorf("tool execution cancelled before handler execution: tool_name='%s', handler_type='%s', context_error=%w",
			tool.Name, tool.HandlerType, ctx.Err())
	}

	/* Execute tool */
	result, err := handler.Execute(ctx, tool, args)

	/* Audit log tool execution */
	agentID, agentOK := GetAgentIDFromContext(ctx)
	if !agentOK {
		/* Agent ID not available in context - use zero UUID */
		agentID = uuid.Nil
	}
	sessionID, sessionOK := GetSessionIDFromContext(ctx)
	if !sessionOK {
		/* Session ID not available in context - use zero UUID */
		sessionID = uuid.Nil
	}

	outputs := make(map[string]interface{})
	if err == nil {
		outputs["result"] = result
		outputs["success"] = true
	} else {
		outputs["error"] = err.Error()
		outputs["success"] = false
	}

	/* Log audit trail (async, don't block on errors) */
	go func() {
		/* Recover from any panics in background goroutine */
		defer func() {
			if r := recover(); r != nil {
				/* Log panic but don't crash the application */
				/* Note: metrics package may not be available in this context */
			}
		}()

		bgCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		var agentIDPtr, sessionIDPtr *uuid.UUID
		if agentID != uuid.Nil {
			agentIDPtr = &agentID
		}
		if sessionID != uuid.Nil {
			sessionIDPtr = &sessionID
		}

		if auditErr := r.auditLogger.LogToolCall(bgCtx, nil, nil, agentIDPtr, sessionIDPtr, tool.Name, args, outputs); auditErr != nil {
			/* Log audit failure but don't block tool execution */
			/* Note: In production, consider using a metrics/logging system that's available here */
		}
	}()

	if err != nil {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', args_count=%d, arg_keys=[%v], error=%w",
			tool.Name, tool.HandlerType, len(args), argKeys, err)
	}
	return result, nil
}

/* ListTools returns all enabled tools */
func (r *Registry) ListTools(ctx context.Context) ([]db.Tool, error) {
	return r.queries.ListTools(ctx)
}

/* Cleanup shuts down browser contexts and other cleanup tasks */
func (r *Registry) Cleanup() {
	if r.browserTool != nil {
		r.browserTool.Cleanup()
	}
}

/* GetBrowserTool returns the browser tool (for cleanup worker) */
func (r *Registry) GetBrowserTool() *BrowserTool {
	return r.browserTool
}

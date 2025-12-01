package tools

import (
	"context"
	"fmt"
	"sync"

	"github.com/neurondb/NeuronAgent/internal/db"
)

// Registry manages tool registration and execution
type Registry struct {
	queries  *db.Queries
	db       *db.DB
	handlers map[string]ToolHandler
	mu       sync.RWMutex
}

// NewRegistry creates a new tool registry
func NewRegistry(queries *db.Queries, database *db.DB) *Registry {
	registry := &Registry{
		queries:  queries,
		db:       database,
		handlers: make(map[string]ToolHandler),
	}

	// Register built-in handlers
	sqlTool := NewSQLTool(queries)
	sqlTool.db = database
	registry.RegisterHandler("sql", sqlTool)
	registry.RegisterHandler("http", NewHTTPTool())
	registry.RegisterHandler("code", NewCodeTool())
	registry.RegisterHandler("shell", NewShellTool())

	return registry
}

// RegisterHandler registers a tool handler for a specific handler type
func (r *Registry) RegisterHandler(handlerType string, handler ToolHandler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.handlers[handlerType] = handler
}

// Get retrieves a tool from the database
// Implements agent.ToolRegistry interface
func (r *Registry) Get(name string) (*db.Tool, error) {
	return r.queries.GetTool(context.Background(), name)
}

// Execute executes a tool with the given arguments
// Implements agent.ToolRegistry interface
func (r *Registry) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	return r.ExecuteTool(ctx, tool, args)
}

// ExecuteTool executes a tool with the given arguments (internal method)
func (r *Registry) ExecuteTool(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	if !tool.Enabled {
		return "", fmt.Errorf("tool %s is disabled", tool.Name)
	}

	// Validate arguments
	if err := ValidateArgs(args, tool.ArgSchema); err != nil {
		return "", fmt.Errorf("validation failed: %w", err)
	}

	// Get handler
	r.mu.RLock()
	handler, exists := r.handlers[tool.HandlerType]
	r.mu.RUnlock()

	if !exists {
		return "", fmt.Errorf("no handler registered for type: %s", tool.HandlerType)
	}

	// Execute tool
	return handler.Execute(ctx, tool, args)
}

// ListTools returns all enabled tools
func (r *Registry) ListTools(ctx context.Context) ([]db.Tool, error) {
	return r.queries.ListTools(ctx)
}


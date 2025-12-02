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
	tool, err := r.queries.GetTool(context.Background(), name)
	if err != nil {
		return nil, fmt.Errorf("tool retrieval failed: tool_name='%s', error=%w", name, err)
	}
	return tool, nil
}

// Execute executes a tool with the given arguments
// Implements agent.ToolRegistry interface
func (r *Registry) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	return r.ExecuteTool(ctx, tool, args)
}

// ExecuteTool executes a tool with the given arguments (internal method)
func (r *Registry) ExecuteTool(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	if !tool.Enabled {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', enabled=false, args_count=%d, arg_keys=[%v]",
			tool.Name, tool.HandlerType, len(args), argKeys)
	}

	// Validate arguments
	if err := ValidateArgs(args, tool.ArgSchema); err != nil {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool validation failed: tool_name='%s', handler_type='%s', args_count=%d, arg_keys=[%v], validation_error='%v'",
			tool.Name, tool.HandlerType, len(args), argKeys, err)
	}

	// Get handler
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

	// Execute tool
	result, err := handler.Execute(ctx, tool, args)
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

// ListTools returns all enabled tools
func (r *Registry) ListTools(ctx context.Context) ([]db.Tool, error) {
	return r.queries.ListTools(ctx)
}


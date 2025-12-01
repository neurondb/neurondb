package tools

import (
	"fmt"
	"sync"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// ToolDefinition represents a tool's definition for MCP
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// ToolRegistry manages tool registration and execution
type ToolRegistry struct {
	tools      map[string]Tool
	definitions map[string]ToolDefinition
	mu         sync.RWMutex
	db         *database.Database
	logger     *logging.Logger
}

// NewToolRegistry creates a new tool registry
func NewToolRegistry(db *database.Database, logger *logging.Logger) *ToolRegistry {
	return &ToolRegistry{
		tools:       make(map[string]Tool),
		definitions: make(map[string]ToolDefinition),
		db:          db,
		logger:      logger,
	}
}

// Register registers a tool
func (r *ToolRegistry) Register(tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	definition := ToolDefinition{
		Name:        tool.Name(),
		Description: tool.Description(),
		InputSchema: tool.InputSchema(),
	}

	r.tools[tool.Name()] = tool
	r.definitions[tool.Name()] = definition
	r.logger.Debug(fmt.Sprintf("Registered tool: %s", tool.Name()), nil)
}

// RegisterAll registers multiple tools
func (r *ToolRegistry) RegisterAll(tools []Tool) {
	for _, tool := range tools {
		r.Register(tool)
	}
}

// GetTool retrieves a tool by name
func (r *ToolRegistry) GetTool(name string) Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tools[name]
}

// GetDefinition retrieves a tool definition by name
func (r *ToolRegistry) GetDefinition(name string) (ToolDefinition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	def, exists := r.definitions[name]
	return def, exists
}

// GetAllDefinitions returns all tool definitions
func (r *ToolRegistry) GetAllDefinitions() []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	definitions := make([]ToolDefinition, 0, len(r.definitions))
	for _, def := range r.definitions {
		definitions = append(definitions, def)
	}
	return definitions
}

// GetAllToolNames returns all registered tool names
func (r *ToolRegistry) GetAllToolNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

// HasTool checks if a tool exists
func (r *ToolRegistry) HasTool(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, exists := r.tools[name]
	return exists
}

// Unregister removes a tool
func (r *ToolRegistry) Unregister(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	removed := false
	if _, exists := r.tools[name]; exists {
		delete(r.tools, name)
		delete(r.definitions, name)
		removed = true
		r.logger.Debug(fmt.Sprintf("Unregistered tool: %s", name), nil)
	}
	return removed
}

// Clear removes all tools
func (r *ToolRegistry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools = make(map[string]Tool)
	r.definitions = make(map[string]ToolDefinition)
}

// GetCount returns the number of registered tools
func (r *ToolRegistry) GetCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}


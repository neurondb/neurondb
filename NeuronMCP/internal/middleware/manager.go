package middleware

import (
	"context"
	"sync"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

// Manager manages middleware registration and execution
type Manager struct {
	middlewares []Middleware
	mu          sync.RWMutex
	logger      *logging.Logger
}

// NewManager creates a new middleware manager
func NewManager(logger *logging.Logger) *Manager {
	return &Manager{
		middlewares: make([]Middleware, 0),
		logger:      logger,
	}
}

// Register registers a middleware
func (m *Manager) Register(middleware Middleware) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.middlewares = append(m.middlewares, middleware)
	m.logger.Debug("Registered middleware", map[string]interface{}{
		"name":  middleware.Name(),
		"order": middleware.Order(),
	})
}

// Execute executes the middleware chain
func (m *Manager) Execute(ctx context.Context, req *MCPRequest, handler Handler) (*MCPResponse, error) {
	m.mu.RLock()
	chain := NewChain(m.middlewares)
	m.mu.RUnlock()
	return chain.Execute(ctx, req, handler)
}

// GetAll returns all registered middlewares
func (m *Manager) GetAll() []Middleware {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return append([]Middleware(nil), m.middlewares...)
}


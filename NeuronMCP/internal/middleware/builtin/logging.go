package builtin

import (
	"context"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

// LoggingMiddleware logs requests and responses
type LoggingMiddleware struct {
	logger              *logging.Logger
	enableRequestLogging  bool
	enableResponseLogging bool
}

// NewLoggingMiddleware creates a new logging middleware
func NewLoggingMiddleware(logger *logging.Logger, enableRequest, enableResponse bool) *LoggingMiddleware {
	return &LoggingMiddleware{
		logger:                logger,
		enableRequestLogging:  enableRequest,
		enableResponseLogging: enableResponse,
	}
}

// Name returns the middleware name
func (m *LoggingMiddleware) Name() string {
	return "logging"
}

// Order returns the execution order
func (m *LoggingMiddleware) Order() int {
	return 2
}

// Enabled returns whether the middleware is enabled
func (m *LoggingMiddleware) Enabled() bool {
	return true
}

// Execute executes the middleware
func (m *LoggingMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	start := time.Now()

	if m.enableRequestLogging {
		m.logger.Info("Request", map[string]interface{}{
			"method":   req.Method,
			"params":   req.Params,
			"metadata": req.Metadata,
		})
	}

	resp, err := next(ctx)
	duration := time.Since(start)

	if err != nil {
		m.logger.Error("Request failed", err, map[string]interface{}{
			"method":   req.Method,
			"duration": duration,
			"params":   req.Params,
		})
		return nil, err
	}

	if m.enableResponseLogging {
		m.logger.Info("Response", map[string]interface{}{
			"method":   req.Method,
			"duration": duration,
			"success":  !resp.IsError,
			"metadata": resp.Metadata,
		})
	}

	return resp, nil
}


/*-------------------------------------------------------------------------
 *
 * correlation.go
 *    Request correlation ID middleware for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/correlation.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* CorrelationIDKey is the context key for correlation ID */
type CorrelationIDKey struct{}

/* CorrelationMiddleware adds correlation IDs to requests */
type CorrelationMiddleware struct {
	counter atomic.Uint64
	logger  *logging.Logger
}

/* NewCorrelationMiddleware creates a new correlation middleware */
func NewCorrelationMiddleware(logger *logging.Logger) *CorrelationMiddleware {
	return &CorrelationMiddleware{
		logger: logger,
	}
}

/* Name returns the middleware name */
func (m *CorrelationMiddleware) Name() string {
	return "correlation"
}

/* Order returns the middleware order */
func (m *CorrelationMiddleware) Order() int {
	return -1 /* Run first, before other middleware */
}

/* Enabled returns whether the middleware is enabled */
func (m *CorrelationMiddleware) Enabled() bool {
	return true /* Always enabled */
}

/* Execute adds correlation ID to request context */
func (m *CorrelationMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	/* Generate correlation ID */
	correlationID := m.generateCorrelationID()
	
	/* Add to context */
	ctx = context.WithValue(ctx, CorrelationIDKey{}, correlationID)
	
	/* Add to request metadata if available */
	if req.Metadata == nil {
		req.Metadata = make(map[string]interface{})
	}
	req.Metadata["correlationId"] = correlationID
	
	/* Log request with correlation ID */
	if m.logger != nil {
		m.logger.Debug("Request received", map[string]interface{}{
			"correlationId": correlationID,
			"method":        req.Method,
		})
	}
	
	/* Execute next middleware */
	resp, err := next(ctx, req)
	
	/* Add correlation ID to response metadata */
	if resp != nil {
		if resp.Metadata == nil {
			resp.Metadata = make(map[string]interface{})
		}
		resp.Metadata["correlationId"] = correlationID
	}
	
	return resp, err
}

/* generateCorrelationID generates a unique correlation ID */
func (m *CorrelationMiddleware) generateCorrelationID() string {
	counter := m.counter.Add(1)
	timestamp := uint64(time.Now().UnixNano())
	return fmt.Sprintf("req-%d-%d", timestamp, counter)
}


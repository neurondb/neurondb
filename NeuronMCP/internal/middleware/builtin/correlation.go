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

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/observability"
)

/* CorrelationMiddleware adds correlation IDs (request IDs) to requests */
type CorrelationMiddleware struct {
	logger *logging.Logger
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

/* Execute adds request ID (correlation ID) to request context */
func (m *CorrelationMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	/* Generate or get request ID */
	ctx, reqID := observability.GetOrCreateRequestID(ctx)
	requestIDStr := reqID.String()
	
	/* Add to request metadata if available */
	if req.Metadata == nil {
		req.Metadata = make(map[string]interface{})
	}
	req.Metadata["request_id"] = requestIDStr
	req.Metadata["correlationId"] = requestIDStr /* Keep for backward compatibility */
	
	/* Log request with request ID */
	if m.logger != nil {
		m.logger.Debug("Request received", map[string]interface{}{
			"request_id": requestIDStr,
			"method":     req.Method,
		})
	}
	
	/* Execute next middleware */
	resp, err := next(ctx, req)
	
	/* Add request ID to response metadata */
	if resp != nil {
		if resp.Metadata == nil {
			resp.Metadata = make(map[string]interface{})
		}
		resp.Metadata["request_id"] = requestIDStr
		resp.Metadata["correlationId"] = requestIDStr /* Keep for backward compatibility */
	}
	
	return resp, err
}


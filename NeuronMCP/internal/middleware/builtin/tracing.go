/*-------------------------------------------------------------------------
 *
 * tracing.go
 *    OpenTelemetry tracing middleware for NeuronMCP
 *
 * Provides distributed tracing for tool calls with DB timing tracking
 * and request ID correlation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/tracing.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/observability"
)

/* TracingMiddleware provides distributed tracing */
type TracingMiddleware struct {
	tracer        *observability.Tracer
	dbTiming      *observability.DBTimingTracker
	logger        *logging.Logger
	enabled       bool
}

/* NewTracingMiddleware creates a new tracing middleware */
func NewTracingMiddleware(tracer *observability.Tracer, dbTiming *observability.DBTimingTracker, logger *logging.Logger, enabled bool) *TracingMiddleware {
	return &TracingMiddleware{
		tracer:   tracer,
		dbTiming: dbTiming,
		logger:   logger,
		enabled:  enabled,
	}
}

/* Name returns the middleware name */
func (m *TracingMiddleware) Name() string {
	return "tracing"
}

/* Order returns the execution order - runs early to capture all operations */
func (m *TracingMiddleware) Order() int {
	return 2
}

/* Enabled returns whether the middleware is enabled */
func (m *TracingMiddleware) Enabled() bool {
	return m.enabled && m.tracer != nil
}

/* Execute executes the tracing middleware */
func (m *TracingMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if !m.Enabled() {
		return next(ctx, req)
	}

	/* Create span name from method */
	spanName := req.Method
	if req.Method == "tools/call" {
		if toolName, ok := req.Params["name"].(string); ok {
			spanName = "tool." + toolName
		}
	}

	/* Start span */
	ctx, spanID := m.tracer.StartSpanWithRequestID(ctx, spanName)
	startTime := time.Now()

	/* Add request attributes */
	m.tracer.AddSpanAttribute(spanID, "method", req.Method)
	if req.Method == "tools/call" {
		if toolName, ok := req.Params["name"].(string); ok {
			m.tracer.AddSpanAttribute(spanID, "tool_name", toolName)
		}
	}

	/* Execute next middleware */
	resp, err := next(ctx, req)

	/* Calculate duration */
	duration := time.Since(startTime)

	/* Add response attributes */
	if resp != nil {
		m.tracer.AddSpanAttribute(spanID, "success", !resp.IsError)
		if resp.IsError {
			m.tracer.SetSpanStatus(spanID, "error")
		} else {
			m.tracer.SetSpanStatus(spanID, "ok")
		}
	} else if err != nil {
		m.tracer.SetSpanStatus(spanID, "error")
		m.tracer.AddSpanAttribute(spanID, "error", err.Error())
	} else {
		m.tracer.SetSpanStatus(spanID, "ok")
	}

	/* Add duration */
	m.tracer.AddSpanAttribute(spanID, "duration_ms", duration.Milliseconds())

	/* End span */
	m.tracer.EndSpan(spanID)

	/* Log if enabled */
	if m.logger != nil {
		m.logger.Debug("Trace completed", map[string]interface{}{
			"span_name": spanName,
			"duration_ms": duration.Milliseconds(),
			"success": err == nil && (resp == nil || !resp.IsError),
		})
	}

	return resp, err
}




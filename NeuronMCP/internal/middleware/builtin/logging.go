/*-------------------------------------------------------------------------
 *
 * logging.go
 *    Logging middleware for NeuronMCP
 *
 * Provides request and response logging middleware for MCP requests
 * with configurable request/response logging and structured output.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/logging.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* Keys to redact in logs (case-insensitive) */
var sensitiveLogKeys = map[string]bool{
	"apikey": true, "api_key": true, "token": true, "password": true,
	"passwd": true, "secret": true, "authorization": true, "x-api-key": true,
	"bearer": true, "access_token": true, "refresh_token": true,
}

/* LoggingMiddleware logs requests and responses */
type LoggingMiddleware struct {
	logger                *logging.Logger
	enableRequestLogging  bool
	enableResponseLogging bool
}

/* NewLoggingMiddleware creates a new logging middleware */
func NewLoggingMiddleware(logger *logging.Logger, enableRequest, enableResponse bool) *LoggingMiddleware {
	return &LoggingMiddleware{
		logger:                logger,
		enableRequestLogging:  enableRequest,
		enableResponseLogging: enableResponse,
	}
}

/* Name returns the middleware name */
func (m *LoggingMiddleware) Name() string {
	return "logging"
}

/* Order returns the execution order */
func (m *LoggingMiddleware) Order() int {
	return 2
}

/* Enabled returns whether the middleware is enabled */
func (m *LoggingMiddleware) Enabled() bool {
	return true
}

/* sanitizeForLogging redacts sensitive keys from maps before logging */
func sanitizeForLogging(in map[string]interface{}) map[string]interface{} {
	if in == nil {
		return nil
	}
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		keyLower := strings.ToLower(strings.TrimSpace(k))
		if sensitiveLogKeys[keyLower] {
			out[k] = "[REDACTED]"
			continue
		}
		/* Redact "query" to avoid logging raw SQL (may contain sensitive data) */
		if keyLower == "query" {
			if _, ok := v.(string); ok {
				out[k] = "[REDACTED]"
				continue
			}
		}
		/* Recurse into nested maps (e.g. arguments) */
		if m, ok := v.(map[string]interface{}); ok {
			out[k] = sanitizeForLogging(m)
			continue
		}
		out[k] = v
	}
	return out
}

/* Execute executes the middleware */
func (m *LoggingMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	start := time.Now()

	if m.enableRequestLogging {
		m.logger.Info("Request", map[string]interface{}{
			"method":   req.Method,
			"params":   sanitizeForLogging(req.Params),
			"metadata": sanitizeForLogging(req.Metadata),
		})
	}

	resp, err := next(ctx, req)
	duration := time.Since(start)

	if err != nil {
		m.logger.Error("Request failed", err, map[string]interface{}{
			"method":   req.Method,
			"duration": duration,
			"params":   sanitizeForLogging(req.Params),
		})
		return nil, err
	}

	if m.enableResponseLogging {
		m.logger.Info("Response", map[string]interface{}{
			"method":   req.Method,
			"duration": duration,
			"success":  !resp.IsError,
			"metadata": sanitizeForLogging(resp.Metadata),
		})
	}

	return resp, nil
}


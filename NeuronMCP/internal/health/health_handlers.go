/*-------------------------------------------------------------------------
 *
 * health_handlers.go
 *    Health check MCP handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/health/health_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package health

import (
	"context"
	"encoding/json"
)

/* HandleHealthCheck handles the health/check request */
func (c *Checker) HandleHealthCheck(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return c.Check(ctx), nil
}













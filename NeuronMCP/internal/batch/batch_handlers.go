/*-------------------------------------------------------------------------
 *
 * batch_handlers.go
 *    Batch MCP handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/batch/batch_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package batch

import (
	"context"
	"encoding/json"
	"fmt"
)

/* HandleCallBatch handles the tools/call_batch request */
func (p *Processor) HandleCallBatch(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req BatchRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse tools/call_batch request: %w", err)
	}

	if len(req.Tools) == 0 {
		return nil, fmt.Errorf("tools array is required and cannot be empty")
	}

	return p.ProcessBatch(ctx, req)
}













/*-------------------------------------------------------------------------
 *
 * progress_handlers.go
 *    Progress MCP handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/progress/progress_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package progress

import (
	"context"
	"encoding/json"
	"fmt"
)

/* GetProgressRequest represents a progress/get request */
type GetProgressRequest struct {
	ID string `json:"id"`
}

/* HandleGetProgress handles the progress/get request */
func (t *Tracker) HandleGetProgress(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req GetProgressRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse progress/get request: %w", err)
	}

	if req.ID == "" {
		return nil, fmt.Errorf("progress ID is required")
	}

	status, err := t.Get(req.ID)
	if err != nil {
		return nil, err
	}

	return status, nil
}













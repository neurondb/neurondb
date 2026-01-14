/*-------------------------------------------------------------------------
 *
 * batch.go
 *    Batch operations for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/batch/batch.go
 *
 *-------------------------------------------------------------------------
 */

package batch

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/progress"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

const (
	DefaultBatchTimeout = 300 * time.Second /* 5 minutes default timeout for batch operations */
	DefaultMaxBatchSize = 100                /* Default maximum batch size */
)

/* BatchRequest represents a batch tool call request */
type BatchRequest struct {
	Tools      []ToolCall            `json:"tools"`
	Transaction bool                 `json:"transaction,omitempty"`
}

/* ToolCall represents a single tool call in a batch */
type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

/* BatchResult represents a batch operation result */
type BatchResult struct {
	Results []ToolResult            `json:"results"`
	Success bool                    `json:"success"`
	Error   string                  `json:"error,omitempty"`
}

/* ToolResult represents a single tool result */
type ToolResult struct {
	Tool    string                 `json:"tool"`
	Success bool                   `json:"success"`
	Data    interface{}            `json:"data,omitempty"`
	Error   string                 `json:"error,omitempty"`
}

/* Processor processes batch operations */
type Processor struct {
	db           *database.Database
	toolRegistry *tools.ToolRegistry
	logger       *logging.Logger
	progress     *progress.Tracker
}

/* NewProcessor creates a new batch processor */
func NewProcessor(db *database.Database, toolRegistry *tools.ToolRegistry, logger *logging.Logger) *Processor {
	return &Processor{
		db:           db,
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* SetProgressTracker sets the progress tracker for batch operations */
func (p *Processor) SetProgressTracker(tracker *progress.Tracker) {
	p.progress = tracker
}

/* ProcessBatch processes a batch of tool calls */
func (p *Processor) ProcessBatch(ctx context.Context, req BatchRequest) (*BatchResult, error) {
	/* Nil checks */
	if p == nil {
		return nil, fmt.Errorf("batch processor instance is nil")
	}
	if p.db == nil {
		return nil, fmt.Errorf("batch processor database instance is nil")
	}
	if p.toolRegistry == nil {
		return nil, fmt.Errorf("batch processor tool registry is not initialized")
	}
	if p.logger == nil {
		return nil, fmt.Errorf("batch processor logger is not initialized")
	}

	/* Validate batch size */
	if len(req.Tools) == 0 {
		return &BatchResult{
			Results: []ToolResult{},
			Success: true,
		}, nil
	}

	if len(req.Tools) > DefaultMaxBatchSize {
		return nil, fmt.Errorf("batch processing: batch size exceeds maximum of %d tools: received %d tools, max_allowed=%d", DefaultMaxBatchSize, len(req.Tools), DefaultMaxBatchSize)
	}

	/* Add timeout context */
	batchCtx, cancel := context.WithTimeout(ctx, DefaultBatchTimeout)
	defer cancel()

	/* Generate progress ID if progress tracking is enabled */
	progressID := ""
	if p.progress != nil {
		progressID = fmt.Sprintf("batch_%d", time.Now().UnixNano())
		_, err := p.progress.Start(progressID, fmt.Sprintf("Processing batch of %d tools", len(req.Tools)))
		if err != nil {
			if p.logger != nil {
				p.logger.Warn("Failed to start progress tracking", map[string]interface{}{
					"error": err.Error(),
				})
			}
			/* Continue without progress tracking */
			progressID = ""
		}
	}

	results := make([]ToolResult, 0, len(req.Tools))

	if req.Transaction {
		/* NOTE: Transaction support is currently limited. Tools execute using the connection pool,
		 * not within the transaction context. This means that if a tool fails mid-batch,
		 * previous tools' changes may not be rolled back. Full transaction support would require
		 * modifying the Tool interface to accept a transaction context, which is a breaking change.
		 * For now, we start a transaction but tools execute outside it. The transaction is used
		 * only for commit/rollback signaling.
		 */
		if p.logger != nil {
			p.logger.Warn("Transaction mode requested but tools execute outside transaction context", map[string]interface{}{
				"tool_count": len(req.Tools),
				"note":       "Full transaction support requires Tool interface changes",
			})
		}

		/* Start transaction */
		tx, err := p.db.Begin(batchCtx)
		if err != nil {
			return nil, fmt.Errorf("batch processing: failed to start transaction: tool_count=%d, error=%w", len(req.Tools), err)
		}
		
		/* Track if we need to rollback */
		shouldRollback := false
		defer func() {
			if shouldRollback {
				if rollbackErr := tx.Rollback(batchCtx); rollbackErr != nil {
					if p.logger != nil {
						p.logger.Warn("Failed to rollback transaction", map[string]interface{}{
							"error": rollbackErr.Error(),
						})
					}
				}
			}
		}()

		/* Execute all tools in transaction */
		for i, toolCall := range req.Tools {
			/* Update progress */
			if progressID != "" && p.progress != nil {
				progress := float64(i) / float64(len(req.Tools))
				_ = p.progress.Update(progressID, progress, fmt.Sprintf("Processing tool %d/%d: %s", i+1, len(req.Tools), toolCall.Name))
			}

			/* Check context timeout */
			if batchCtx.Err() != nil {
				shouldRollback = true
				if progressID != "" && p.progress != nil {
					_ = p.progress.Fail(progressID, fmt.Errorf("timeout after %v", DefaultBatchTimeout))
				}
				return &BatchResult{
					Results: results,
					Success: false,
					Error:   fmt.Sprintf("batch processing: timeout after %v: processed %d/%d tools, transaction rolled back", DefaultBatchTimeout, len(results), len(req.Tools)),
				}, nil
			}

			if toolCall.Name == "" {
				shouldRollback = true
				if progressID != "" && p.progress != nil {
					_ = p.progress.Fail(progressID, fmt.Errorf("tool at index %d has empty name", i))
				}
				return &BatchResult{
					Results: results,
					Success: false,
					Error:   fmt.Sprintf("batch processing: tool at index %d has empty name, processed %d/%d tools, transaction rolled back", i, len(results), len(req.Tools)),
				}, nil
			}

			result := p.executeTool(batchCtx, toolCall, i, len(req.Tools))
			results = append(results, result)
			
			/* If any tool fails and transaction is enabled, stop and rollback */
			if !result.Success {
				shouldRollback = true
				if progressID != "" && p.progress != nil {
					_ = p.progress.Fail(progressID, fmt.Errorf("tool '%s' failed at index %d", toolCall.Name, i))
				}
				return &BatchResult{
					Results: results,
					Success: false,
					Error:   fmt.Sprintf("batch processing: tool '%s' failed at index %d, processed %d/%d tools, transaction rolled back", toolCall.Name, i, len(results), len(req.Tools)),
				}, nil
			}
		}

		/* Commit transaction */
		if err := tx.Commit(batchCtx); err != nil {
			/* Rollback on commit failure */
			if rollbackErr := tx.Rollback(batchCtx); rollbackErr != nil {
				if p.logger != nil {
					p.logger.Warn("Failed to rollback transaction after commit failure", map[string]interface{}{
						"commit_error": err.Error(),
						"rollback_error": rollbackErr.Error(),
					})
				}
			}
			if progressID != "" && p.progress != nil {
				_ = p.progress.Fail(progressID, err)
			}
			return nil, fmt.Errorf("batch processing: failed to commit transaction: tool_count=%d, processed_count=%d, error=%w", len(req.Tools), len(results), err)
		}

		/* Mark progress as complete */
		if progressID != "" && p.progress != nil {
			_ = p.progress.Complete(progressID, fmt.Sprintf("Successfully processed %d tools", len(req.Tools)))
		}
	} else {
		/* Execute tools without transaction */
		for i, toolCall := range req.Tools {
			/* Update progress */
			if progressID != "" && p.progress != nil {
				progress := float64(i) / float64(len(req.Tools))
				_ = p.progress.Update(progressID, progress, fmt.Sprintf("Processing tool %d/%d: %s", i+1, len(req.Tools), toolCall.Name))
			}

			/* Check context timeout */
			if batchCtx.Err() != nil {
				if progressID != "" && p.progress != nil {
					_ = p.progress.Fail(progressID, fmt.Errorf("timeout after %v", DefaultBatchTimeout))
				}
				return &BatchResult{
					Results: results,
					Success: false,
					Error:   fmt.Sprintf("batch processing: timeout after %v: processed %d/%d tools", DefaultBatchTimeout, len(results), len(req.Tools)),
				}, nil
			}

			if toolCall.Name == "" {
				results = append(results, ToolResult{
					Tool:    fmt.Sprintf("tool_%d", i),
					Success: false,
					Error:   fmt.Sprintf("batch processing: tool at index %d has empty name, processed %d/%d tools", i, len(results), len(req.Tools)),
				})
				continue
			}

			result := p.executeTool(batchCtx, toolCall, i, len(req.Tools))
			results = append(results, result)
		}
	}

	/* Check if all succeeded */
	allSuccess := true
	for _, result := range results {
		if !result.Success {
			allSuccess = false
			break
		}
	}

	/* Mark progress as complete if not already done (for non-transactional case) */
	if progressID != "" && p.progress != nil && allSuccess {
		_ = p.progress.Complete(progressID, fmt.Sprintf("Successfully processed %d tools", len(req.Tools)))
	} else if progressID != "" && p.progress != nil && !allSuccess {
		_ = p.progress.Fail(progressID, fmt.Errorf("some tools failed"))
	}

	return &BatchResult{
		Results: results,
		Success: allSuccess,
	}, nil
}

/* executeTool executes a single tool */
func (p *Processor) executeTool(ctx context.Context, toolCall ToolCall, index int, total int) ToolResult {
	if p.toolRegistry == nil {
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   fmt.Sprintf("batch processing: tool registry is not initialized: tool='%s', index=%d/%d", toolCall.Name, index+1, total),
		}
	}

	tool := p.toolRegistry.GetTool(toolCall.Name)
	if tool == nil {
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   fmt.Sprintf("batch processing: tool not found: tool_name='%s', index=%d/%d", toolCall.Name, index+1, total),
		}
	}

	/* Validate arguments is not nil */
	if toolCall.Arguments == nil {
		toolCall.Arguments = make(map[string]interface{})
	}

	result, err := tool.Execute(ctx, toolCall.Arguments)
	if err != nil {
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   fmt.Sprintf("batch processing: tool execution error: tool='%s', index=%d/%d, error=%v", toolCall.Name, index+1, total, err),
		}
	}

	if result == nil {
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   fmt.Sprintf("batch processing: tool returned nil result: tool='%s', index=%d/%d", toolCall.Name, index+1, total),
		}
	}

	if !result.Success {
		errorMsg := "Unknown error"
		if result.Error != nil {
			errorMsg = result.Error.Message
		}
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   fmt.Sprintf("batch processing: tool execution failed: tool='%s', index=%d/%d, error='%s'", toolCall.Name, index+1, total, errorMsg),
		}
	}

	return ToolResult{
		Tool:    toolCall.Name,
		Success: true,
		Data:    result.Data,
	}
}


/*-------------------------------------------------------------------------
 *
 * pagination.go
 *    Pagination support for MCP
 *
 * Provides pagination with continuation tokens for large result sets.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/pagination.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"
)

/* PaginationParams represents pagination parameters in requests */
type PaginationParams struct {
	Limit            int    `json:"limit,omitempty"`
	Cursor           string `json:"cursor,omitempty"`
	ContinuationToken string `json:"continuation_token,omitempty"`
}

/* PaginatedResponse represents a paginated response */
type PaginatedResponse struct {
	Items            []interface{}     `json:"items"`
	HasMore          bool              `json:"has_more"`
	NextCursor       string            `json:"next_cursor,omitempty"`
	ContinuationToken string            `json:"continuation_token,omitempty"`
	TotalCount       *int              `json:"total_count,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

/* CursorData represents the data encoded in a cursor */
type CursorData struct {
	Offset    int       `json:"offset"`
	Timestamp time.Time `json:"timestamp"`
	Params    map[string]interface{} `json:"params,omitempty"`
}

/* EncodeCursor encodes pagination data into a cursor token */
func EncodeCursor(offset int, params map[string]interface{}) (string, error) {
	data := CursorData{
		Offset:    offset,
		Timestamp: time.Now(),
		Params:    params,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal cursor data: %w", err)
	}

	return base64.URLEncoding.EncodeToString(jsonData), nil
}

/* DecodeCursor decodes a cursor token into pagination data */
func DecodeCursor(cursor string) (*CursorData, error) {
	if cursor == "" {
		return nil, fmt.Errorf("empty cursor")
	}

	data, err := base64.URLEncoding.DecodeString(cursor)
	if err != nil {
		return nil, fmt.Errorf("failed to decode cursor: %w", err)
	}

	var cursorData CursorData
	if err := json.Unmarshal(data, &cursorData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cursor data: %w", err)
	}

	return &cursorData, nil
}

/* DefaultLimit is the default page size */
const DefaultLimit = 50

/* MaxLimit is the maximum allowed page size */
const MaxLimit = 1000

/* ValidateLimit validates and normalizes a limit value */
func ValidateLimit(limit int) int {
	if limit <= 0 {
		return DefaultLimit
	}
	if limit > MaxLimit {
		return MaxLimit
	}
	return limit
}













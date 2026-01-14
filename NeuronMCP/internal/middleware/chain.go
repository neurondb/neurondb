/*-------------------------------------------------------------------------
 *
 * chain.go
 *    Middleware chain execution
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/chain.go
 *
 *-------------------------------------------------------------------------
 */

package middleware

import (
	"context"
	"sort"
)

/* Chain executes middleware in order */
type Chain struct {
	middlewares []Middleware
}

/* NewChain creates a new middleware chain */
func NewChain(middlewares []Middleware) *Chain {
  /* Sort by order */
	sorted := make([]Middleware, len(middlewares))
	copy(sorted, middlewares)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Order() < sorted[j].Order()
	})

	return &Chain{middlewares: sorted}
}

/* Execute executes the middleware chain */
func (c *Chain) Execute(ctx context.Context, req *MCPRequest, finalHandler Handler) (*MCPResponse, error) {
  /* Filter enabled middlewares */
	enabled := make([]Middleware, 0)
	for _, mw := range c.middlewares {
		if mw.Enabled() {
			enabled = append(enabled, mw)
		}
	}

  /* Build chain */
	index := 0
	var next Handler
	next = func(ctx context.Context, r *MCPRequest) (*MCPResponse, error) {
		if index >= len(enabled) {
			return finalHandler(ctx, r)
		}
		mw := enabled[index]
		index++
		return mw.Execute(ctx, r, next)
	}

	return next(ctx, req)
}

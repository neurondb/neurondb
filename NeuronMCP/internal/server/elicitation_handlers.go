/*-------------------------------------------------------------------------
 *
 * elicitation_handlers.go
 *    Elicitation MCP handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/elicitation_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/elicitation"
)

/* setupElicitationHandlers sets up elicitation-related MCP handlers */
func (s *Server) setupElicitationHandlers() {
	if s.elicitation == nil {
		return /* Elicitation not enabled */
	}

	/* Register handlers */
	s.mcpServer.SetHandler("prompts/request", s.handleRequestPrompt)
	s.mcpServer.SetHandler("prompts/respond", s.handleRespondPrompt)
}

/* handleRequestPrompt handles the prompts/request request */
func (s *Server) handleRequestPrompt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s.elicitation == nil {
		return nil, fmt.Errorf("elicitation is not enabled")
	}

	startTime := time.Now()
	method := "prompts/request"

	/* Track request */
	if s.metricsCollector != nil {
		s.metricsCollector.IncrementRequest(method)
		defer func() {
			s.metricsCollector.AddDuration(time.Since(startTime))
		}()
	}

	/* Call HandleRequestPrompt on elicitation manager */
	/* The import is needed for type checking of s.elicitation field */
	var _ *elicitation.Manager = s.elicitation
	return s.elicitation.HandleRequestPrompt(ctx, params)
}

/* handleRespondPrompt handles the prompts/respond request */
func (s *Server) handleRespondPrompt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s.elicitation == nil {
		return nil, fmt.Errorf("elicitation is not enabled")
	}

	startTime := time.Now()
	method := "prompts/respond"

	/* Track request */
	if s.metricsCollector != nil {
		s.metricsCollector.IncrementRequest(method)
		defer func() {
			s.metricsCollector.AddDuration(time.Since(startTime))
		}()
	}

	return s.elicitation.HandleRespondPrompt(ctx, params)
}

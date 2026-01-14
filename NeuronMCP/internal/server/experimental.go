/*-------------------------------------------------------------------------
 *
 * experimental.go
 *    Experimental features support for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/experimental.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
)

/* ExperimentalHandler is a function that handles experimental requests */
type ExperimentalHandler func(ctx context.Context, params json.RawMessage) (interface{}, error)

/* experimentalHandlers stores experimental method handlers */
var experimentalHandlers = make(map[string]ExperimentalHandler)

/* RegisterExperimentalHandler registers a handler for an experimental method */
/* Experimental methods should be prefixed with "x-neurondb/" */
func RegisterExperimentalHandler(method string, handler ExperimentalHandler) {
	experimentalHandlers[method] = handler
}

/* setupExperimentalHandlers sets up experimental method handlers */
func (s *Server) setupExperimentalHandlers() {
	/* Register experimental handlers */
	for method, handler := range experimentalHandlers {
		s.mcpServer.SetHandler(method, func(ctx context.Context, params json.RawMessage) (interface{}, error) {
			return handler(ctx, params)
		})
	}
}

/* IsExperimentalMethod checks if a method is experimental */
func IsExperimentalMethod(method string) bool {
	return len(method) > 11 && method[:11] == "x-neurondb/"
}


/*-------------------------------------------------------------------------
 *
 * graphql.go
 *    GraphQL endpoint for NeuronMCP
 *
 * Provides flexible GraphQL querying interface.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/api/graphql.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/neurondb/NeuronMCP/internal/tools"
)

/* GraphQLEndpoint provides GraphQL endpoint */
type GraphQLEndpoint struct {
	toolRegistry *tools.ToolRegistry
}

/* NewGraphQLEndpoint creates a new GraphQL endpoint */
func NewGraphQLEndpoint(toolRegistry *tools.ToolRegistry) *GraphQLEndpoint {
	return &GraphQLEndpoint{
		toolRegistry: toolRegistry,
	}
}

/* HandleRequest handles GraphQL request */
func (gql *GraphQLEndpoint) HandleRequest(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Query     string                 `json:"query"`
		Variables map[string]interface{} `json:"variables"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	/* Parse and execute GraphQL query */
	result := gql.executeQuery(r.Context(), request.Query, request.Variables)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

/* executeQuery executes GraphQL query */
func (gql *GraphQLEndpoint) executeQuery(ctx context.Context, query string, variables map[string]interface{}) map[string]interface{} {
	/* Simple GraphQL implementation - would use proper GraphQL library in production */
	/* For now, return a basic response */
	return map[string]interface{}{
		"data": map[string]interface{}{
			"tools": gql.getTools(),
		},
	}
}

/* getTools gets tools list */
func (gql *GraphQLEndpoint) getTools() []map[string]interface{} {
	definitions := gql.toolRegistry.GetAllDefinitions()
	tools := []map[string]interface{}{}

	for _, def := range definitions {
		tools = append(tools, map[string]interface{}{
			"name":        def.Name,
			"description": def.Description,
		})
	}

	return tools
}

/* RegisterRoutes registers GraphQL routes */
func (gql *GraphQLEndpoint) RegisterRoutes(mux *http.ServeMux, path string) {
	mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			gql.HandleRequest(w, r)
		} else {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
}


/*-------------------------------------------------------------------------
 *
 * rest_wrapper.go
 *    REST API Wrapper for NeuronMCP
 *
 * Provides HTTP/JSON interface for MCP tools.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/api/rest_wrapper.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/tools"
)

/* RESTWrapper provides REST API wrapper for MCP tools */
type RESTWrapper struct {
	toolRegistry *tools.ToolRegistry
	basePath     string
}

/* NewRESTWrapper creates a new REST wrapper */
func NewRESTWrapper(toolRegistry *tools.ToolRegistry, basePath string) *RESTWrapper {
	return &RESTWrapper{
		toolRegistry: toolRegistry,
		basePath:     basePath,
	}
}

/* HandleRequest handles HTTP request */
func (rw *RESTWrapper) HandleRequest(w http.ResponseWriter, r *http.Request) {
	/* Parse path */
	path := strings.TrimPrefix(r.URL.Path, rw.basePath)
	pathParts := strings.Split(strings.Trim(path, "/"), "/")

	if len(pathParts) < 2 || pathParts[0] != "tools" {
		http.Error(w, "Invalid path", http.StatusBadRequest)
		return
	}

	toolName := pathParts[1]
	method := r.Method

	switch method {
	case "GET":
		rw.handleGet(w, r, toolName)
	case "POST":
		rw.handlePost(w, r, toolName)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

/* handleGet handles GET request */
func (rw *RESTWrapper) handleGet(w http.ResponseWriter, r *http.Request, toolName string) {
	/* Get tool info */
	tool := rw.toolRegistry.GetTool(toolName)
	if tool == nil {
		http.Error(w, "Tool not found", http.StatusNotFound)
		return
	}

	response := map[string]interface{}{
		"name":        tool.Name(),
		"description": tool.Description(),
		"input_schema": tool.InputSchema(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

/* handlePost handles POST request */
func (rw *RESTWrapper) handlePost(w http.ResponseWriter, r *http.Request, toolName string) {
	/* Parse request body */
	var requestBody map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	/* Get tool */
	tool := rw.toolRegistry.GetTool(toolName)
	if tool == nil {
		http.Error(w, "Tool not found", http.StatusNotFound)
		return
	}

	/* Execute tool */
	ctx := r.Context()
	arguments, _ := requestBody["arguments"].(map[string]interface{})
	if arguments == nil {
		arguments = requestBody
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	/* Return result */
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

/* RegisterRoutes registers REST routes */
func (rw *RESTWrapper) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc(rw.basePath+"/tools/", func(w http.ResponseWriter, r *http.Request) {
		rw.HandleRequest(w, r)
	})

	mux.HandleFunc(rw.basePath+"/tools", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			/* List all tools */
			definitions := rw.toolRegistry.GetAllDefinitions()
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"tools": definitions,
			})
		} else {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
}


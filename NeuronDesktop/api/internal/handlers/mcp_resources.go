package handlers

import (
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* MCPResourcesHandlers handles MCP resource-related endpoints */
type MCPResourcesHandlers struct {
	queries    *db.Queries
	mcpManager *MCPManager
}

/* NewMCPResourcesHandlers creates new MCP resources handlers */
func NewMCPResourcesHandlers(queries *db.Queries, mcpManager *MCPManager) *MCPResourcesHandlers {
	return &MCPResourcesHandlers{
		queries:    queries,
		mcpManager: mcpManager,
	}
}

/* ListResources lists MCP resources from the MCP server */
func (h *MCPResourcesHandlers) ListResources(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	resp, err := client.ListResources()
	if err != nil {
		WriteError(w, r, http.StatusBadGateway, err, nil)
		return
	}
	resources := make([]map[string]interface{}, 0, len(resp.Resources))
	for _, res := range resp.Resources {
		resources = append(resources, map[string]interface{}{
			"uri":         res.URI,
			"name":        res.Name,
			"description": res.Description,
			"mimeType":    res.MimeType,
		})
	}
	WriteSuccess(w, map[string]interface{}{"resources": resources}, http.StatusOK)
}

/* GetResource gets a specific resource from the MCP server */
func (h *MCPResourcesHandlers) GetResource(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	resourceURI := vars["uri"]

	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	resp, err := client.ReadResource(resourceURI)
	if err != nil {
		WriteError(w, r, http.StatusBadGateway, err, nil)
		return
	}
	var contents []map[string]interface{}
	for _, c := range resp.Contents {
		contents = append(contents, map[string]interface{}{
			"uri":      c.URI,
			"mimeType": c.MimeType,
			"text":     c.Text,
		})
	}
	resource := map[string]interface{}{
		"uri":      resourceURI,
		"contents": contents,
	}
	WriteSuccess(w, resource, http.StatusOK)
}

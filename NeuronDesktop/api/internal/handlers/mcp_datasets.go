package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* MCPDatasetsHandlers handles MCP dataset loading endpoints */
type MCPDatasetsHandlers struct {
	queries    *db.Queries
	mcpManager *MCPManager
}

/* NewMCPDatasetsHandlers creates new MCP datasets handlers */
func NewMCPDatasetsHandlers(queries *db.Queries, mcpManager *MCPManager) *MCPDatasetsHandlers {
	return &MCPDatasetsHandlers{
		queries:    queries,
		mcpManager: mcpManager,
	}
}

/* LoadDataset loads a dataset from various sources */
func (h *MCPDatasetsHandlers) LoadDataset(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req struct {
		SourceType    string `json:"source_type"`    // huggingface, url, github, s3, local
		SourcePath    string `json:"source_path"`
		Format        string `json:"format,omitempty"` // csv, json, jsonl, parquet
		AutoEmbed     bool   `json:"auto_embed,omitempty"`
		EmbeddingModel string `json:"embedding_model,omitempty"`
		CreateIndexes bool   `json:"create_indexes,omitempty"`
		Limit         int    `json:"limit,omitempty"`
		Split         string `json:"split,omitempty"` // for HuggingFace
		SchemaName    string `json:"schema_name,omitempty"`
		TableName     string `json:"table_name,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	/* Call load_dataset tool via MCP */
	toolArgs := map[string]interface{}{
		"source_type": req.SourceType,
		"source_path": req.SourcePath,
	}
	if req.Format != "" {
		toolArgs["format"] = req.Format
	}
	if req.AutoEmbed {
		toolArgs["auto_embed"] = true
		if req.EmbeddingModel != "" {
			toolArgs["embedding_model"] = req.EmbeddingModel
		}
	}
	if req.CreateIndexes {
		toolArgs["create_indexes"] = true
	}
	if req.Limit > 0 {
		toolArgs["limit"] = req.Limit
	}
	if req.Split != "" {
		toolArgs["split"] = req.Split
	}
	if req.SchemaName != "" {
		toolArgs["schema_name"] = req.SchemaName
	}
	if req.TableName != "" {
		toolArgs["table_name"] = req.TableName
	}

	result, err := client.CallTool("load_dataset", toolArgs)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, result, http.StatusOK)
}

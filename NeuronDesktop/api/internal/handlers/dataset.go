package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

/* DatasetHandlers handles dataset ingestion and management */
type DatasetHandlers struct {
	queries *db.Queries
	store   IngestJobStore
}

/* defaultIngestStore is used when no store is provided so GetIngestStatus/ListIngestJobs work */
var defaultIngestStore = NewMemoryIngestStore()

/* NewDatasetHandlers creates new dataset handlers. If store is nil, a default in-memory store is used. */
func NewDatasetHandlers(queries *db.Queries, store IngestJobStore) *DatasetHandlers {
	if store == nil {
		store = defaultIngestStore
	}
	return &DatasetHandlers{queries: queries, store: store}
}

/* IngestRequest represents a dataset ingestion request */
type IngestRequest struct {
	SourceType   string                 `json:"source_type"` // "file", "url", "s3", "github", "huggingface"
	SourcePath   string                 `json:"source_path"`
	Format       string                 `json:"format,omitempty"` // "csv", "json", "jsonl", "parquet"
	TableName    string                 `json:"table_name,omitempty"`
	SchemaName   string                 `json:"schema_name,omitempty"`
	AutoEmbed    bool                   `json:"auto_embed,omitempty"`
	EmbeddingModel string               `json:"embedding_model,omitempty"`
	CreateIndex  bool                   `json:"create_index,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

/* IngestResponse represents the ingestion response */
type IngestResponse struct {
	JobID        string    `json:"job_id"`
	Status       string    `json:"status"`
	TableName    string    `json:"table_name,omitempty"`
	RowsIngested int64     `json:"rows_ingested,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
}

/* IngestDataset ingests a dataset from various sources */
func (h *DatasetHandlers) IngestDataset(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate request
	if req.SourceType == "" || req.SourcePath == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("source_type and source_path are required"), nil)
		return
	}

	// Get profile for database connection
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	// Create NeuronDB client
	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create NeuronDB client: %w", err), nil)
		return
	}

	jobID := fmt.Sprintf("ingest_%d", time.Now().UnixNano())
	h.store.SetQueued(profileID, jobID, req.TableName)

	go func() {
		ctx := context.Background()
		h.store.SetRunning(profileID, jobID)
		query := fmt.Sprintf(`
			SELECT neurondb_mcp_tool_call(
				'load_dataset',
				jsonb_build_object(
					'source_type', %s,
					'source_path', %s,
					'format', %s,
					'schema_name', %s,
					'table_name', %s,
					'auto_embed', %s,
					'embedding_model', %s,
					'create_indexes', %s
				)
			)
		`,
			quoteSQLString(req.SourceType),
			quoteSQLString(req.SourcePath),
			quoteSQLString(req.Format),
			quoteSQLString(req.SchemaName),
			quoteSQLString(req.TableName),
			fmt.Sprintf("%t", req.AutoEmbed),
			quoteSQLString(req.EmbeddingModel),
			fmt.Sprintf("%t", req.CreateIndex),
		)
		result, err := client.ExecuteSQLFull(ctx, query)
		if err != nil {
			h.store.SetFailed(profileID, jobID, err)
			return
		}
		var rowsIngested int64
		if res, ok := result.([]map[string]interface{}); ok {
			rowsIngested = int64(len(res))
		}
		h.store.SetCompleted(profileID, jobID, rowsIngested)
	}()

	response := IngestResponse{
		JobID:     jobID,
		Status:    "queued",
		TableName: req.TableName,
		CreatedAt: time.Now(),
	}

	WriteSuccess(w, response, http.StatusAccepted)
}

/* GetIngestStatus gets the status of an ingestion job from the store */
func (h *DatasetHandlers) GetIngestStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	jobID := vars["job_id"]

	j, ok := h.store.Get(profileID, jobID)
	if !ok {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("job not found: %s", jobID), nil)
		return
	}
	response := map[string]interface{}{
		"job_id":         j.JobID,
		"status":         j.Status,
		"progress":       j.Progress,
		"rows_ingested":  j.RowsIngested,
		"table_name":     j.TableName,
		"error":          j.Error,
		"created_at":    j.CreatedAt,
		"updated_at":    j.UpdatedAt,
	}
	WriteSuccess(w, response, http.StatusOK)
}

/* ListIngestJobs lists all ingestion jobs for the profile from the store */
func (h *DatasetHandlers) ListIngestJobs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	jobs := h.store.List(profileID)
	out := make([]map[string]interface{}, 0, len(jobs))
	for _, j := range jobs {
		out = append(out, map[string]interface{}{
			"job_id":        j.JobID,
			"status":       j.Status,
			"progress":     j.Progress,
			"rows_ingested": j.RowsIngested,
			"table_name":   j.TableName,
			"created_at":   j.CreatedAt,
			"updated_at":   j.UpdatedAt,
		})
	}
	WriteSuccess(w, out, http.StatusOK)
}

func quoteSQLString(s string) string {
	if s == "" {
		return "NULL"
	}
	return fmt.Sprintf("'%s'", s)
}


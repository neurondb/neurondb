/*-------------------------------------------------------------------------
 *
 * vfs_handlers.go
 *    Virtual Filesystem API handlers for NeuronAgent
 *
 * Provides REST API endpoints for virtual filesystem operations including
 * file creation, reading, writing, deletion, listing, copying, and moving.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/vfs_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/gorilla/mux"
	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* VFSHandlers handles virtual filesystem API requests */
type VFSHandlers struct {
	queries *db.Queries
	runtime *agent.Runtime
}

/* NewVFSHandlers creates new VFS handlers */
func NewVFSHandlers(queries *db.Queries, runtime *agent.Runtime) *VFSHandlers {
	return &VFSHandlers{
		queries: queries,
		runtime: runtime,
	}
}

/* CreateFileRequest represents a request to create a file */
type CreateFileRequest struct {
	SessionID *string `json:"session_id,omitempty"`
	Path      string  `json:"path"`
	Content   string  `json:"content"` // Base64 encoded or plain text
	MimeType  string  `json:"mime_type,omitempty"`
	Encoding  string  `json:"encoding,omitempty"` // "base64" or "text"
}

/* FileResponse represents a file in API responses */
type FileResponse struct {
	ID             string    `json:"id"`
	AgentID        string    `json:"agent_id"`
	SessionID      *string   `json:"session_id,omitempty"`
	Path           string    `json:"path"`
	Content        string    `json:"content,omitempty"` // Base64 encoded if binary
	MimeType       string    `json:"mime_type"`
	Size           int64     `json:"size"`
	Compressed     bool      `json:"compressed"`
	StorageBackend string    `json:"storage_backend"`
	CreatedAt      string    `json:"created_at"`
	UpdatedAt      string    `json:"updated_at"`
}

/* ListFilesResponse represents a file listing */
type ListFilesResponse struct {
	Path  string         `json:"path"`
	Files []FileResponse `json:"files"`
}

/* CreateFile creates a new file in the virtual filesystem */
func (h *VFSHandlers) CreateFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 100 * 1024 * 1024 // 100MB max
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateFileRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Validate required fields */
	if req.Path == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	if req.Content == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "content is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Parse session ID if provided */
	var sessionID uuid.UUID
	if req.SessionID != nil {
		parsedSessionID, err := uuid.Parse(*req.SessionID)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session_id format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
			return
		}
		sessionID = parsedSessionID
	}

	/* Decode content based on encoding */
	var contentBytes []byte
	if req.Encoding == "base64" {
		contentBytes, err = base64.StdEncoding.DecodeString(req.Content)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid base64 content", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
			return
		}
	} else {
		contentBytes = []byte(req.Content)
	}

	/* Set default mime type */
	mimeType := req.MimeType
	if mimeType == "" {
		mimeType = "text/plain"
	}

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Create file */
	fileID, err := vfs.CreateFile(r.Context(), agentID, sessionID, req.Path, contentBytes, mimeType)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create file", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{
		"id":      fileID.String(),
		"path":    req.Path,
		"agent_id": agentID.String(),
	})
}

/* ReadFile reads a file from the virtual filesystem */
func (h *VFSHandlers) ReadFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get path from path variable or query parameter */
	path := ""
	if pathVar, ok := vars["path"]; ok && pathVar != "" {
		path = pathVar
	}
	if path == "" {
		path = r.URL.Query().Get("path")
	}
	if path == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	/* Clean and normalize path */
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	/* Get encoding preference */
	encoding := r.URL.Query().Get("encoding") // "base64" or "text"

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Read file */
	file, err := vfs.ReadFile(r.Context(), agentID, path)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "file not found", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Encode content based on mime type and preference */
	var contentStr string
	isBinary := !strings.HasPrefix(file.MimeType, "text/") && !strings.HasPrefix(file.MimeType, "application/json")
	if isBinary || encoding == "base64" {
		contentStr = base64.StdEncoding.EncodeToString(file.Content)
	} else {
		contentStr = string(file.Content)
	}

	sessionIDStr := ""
	if file.SessionID != nil {
		sessionIDStr = file.SessionID.String()
	}

	response := FileResponse{
		ID:             file.ID.String(),
		AgentID:        file.AgentID.String(),
		Path:           file.Path,
		MimeType:       file.MimeType,
		Size:           file.Size,
		Compressed:     file.Compressed,
		StorageBackend: file.StorageBackend,
		CreatedAt:      file.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
		UpdatedAt:      file.UpdatedAt.Format("2006-01-02T15:04:05Z07:00"),
	}

	if sessionIDStr != "" {
		response.SessionID = &sessionIDStr
	}
	if encoding == "base64" || isBinary {
		response.Content = contentStr
	} else {
		response.Content = contentStr
	}

	respondJSON(w, http.StatusOK, response)
}

/* WriteFile writes or updates file content */
func (h *VFSHandlers) WriteFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get path from path variable or query parameter */
	path := ""
	if pathVar, ok := vars["path"]; ok && pathVar != "" {
		path = pathVar
	}
	if path == "" {
		path = r.URL.Query().Get("path")
	}
	if path == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	/* Clean and normalize path */
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	/* Validate request body */
	const maxBodySize = 100 * 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Content  string `json:"content"`
		Encoding string `json:"encoding,omitempty"` // "base64" or "text"
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	if req.Content == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "content is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Decode content */
	var contentBytes []byte
	if req.Encoding == "base64" {
		contentBytes, err = base64.StdEncoding.DecodeString(req.Content)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid base64 content", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
			return
		}
	} else {
		contentBytes = []byte(req.Content)
	}

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Write file */
	err = vfs.WriteFile(r.Context(), agentID, path, contentBytes)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to write file", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* DeleteFile deletes a file from the virtual filesystem */
func (h *VFSHandlers) DeleteFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get path from path variable or query parameter */
	path := ""
	if pathVar, ok := vars["path"]; ok && pathVar != "" {
		path = pathVar
	}
	if path == "" {
		path = r.URL.Query().Get("path")
	}
	if path == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	/* Clean and normalize path */
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Delete file */
	err = vfs.DeleteFile(r.Context(), agentID, path)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "file not found", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* ListFiles lists files in a directory */
func (h *VFSHandlers) ListFiles(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get directory path */
	dirPath := r.URL.Query().Get("path")
	if dirPath == "" {
		dirPath = "/"
	}

	/* Get session ID if provided */
	var sessionID *uuid.UUID
	if sessionIDStr := r.URL.Query().Get("session_id"); sessionIDStr != "" {
		parsedSessionID, err := uuid.Parse(sessionIDStr)
		if err == nil {
			sessionID = &parsedSessionID
		}
	}

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* List files */
	files, err := vfs.ListFiles(r.Context(), agentID, sessionID, dirPath)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list files", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Convert to response format */
	fileResponses := make([]FileResponse, len(files))
	for i, file := range files {
		sessionIDStr := ""
		if file.SessionID != nil {
			sessionIDStr = file.SessionID.String()
		}

		fileResponses[i] = FileResponse{
			ID:             file.ID.String(),
			AgentID:        file.AgentID.String(),
			Path:           file.Path,
			MimeType:       file.MimeType,
			Size:           file.Size,
			Compressed:     file.Compressed,
			StorageBackend: file.StorageBackend,
			CreatedAt:      file.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
			UpdatedAt:      file.UpdatedAt.Format("2006-01-02T15:04:05Z07:00"),
		}

		if sessionIDStr != "" {
			fileResponses[i].SessionID = &sessionIDStr
		}
	}

	respondJSON(w, http.StatusOK, ListFilesResponse{
		Path:  dirPath,
		Files: fileResponses,
	})
}

/* CopyFileRequest represents a request to copy a file */
type CopyFileRequest struct {
	DestinationPath string `json:"destination_path"`
}

/* CopyFile copies a file to a new path */
func (h *VFSHandlers) CopyFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get source path */
	srcPath := vars["path"]
	if srcPath == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CopyFileRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	if req.DestinationPath == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "destination_path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Copy file */
	err = vfs.CopyFile(r.Context(), agentID, srcPath, req.DestinationPath)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to copy file", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"source_path":      srcPath,
		"destination_path": req.DestinationPath,
	})
}

/* MoveFileRequest represents a request to move a file */
type MoveFileRequest struct {
	DestinationPath string `json:"destination_path"`
}

/* MoveFile moves a file to a new path */
func (h *VFSHandlers) MoveFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get source path */
	srcPath := vars["path"]
	if srcPath == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req MoveFileRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	if req.DestinationPath == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "destination_path is required", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Get VFS from runtime */
	vfs := h.runtime.VFS()
	if vfs == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "virtual filesystem not available", nil, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	/* Move file */
	err = vfs.MoveFile(r.Context(), agentID, srcPath, req.DestinationPath)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to move file", err, requestID, r.URL.Path, r.Method, "vfs", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"source_path":      srcPath,
		"destination_path": req.DestinationPath,
	})
}


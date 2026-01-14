/*-------------------------------------------------------------------------
 *
 * filesystem_tool.go
 *    File system tool for agent scratchpad operations
 *
 * Provides agent access to virtual file system for creating, reading,
 * writing, and managing files in the agent scratchpad.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/filesystem_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* VirtualFileSystemInterface defines the interface for virtual file system operations */
/* This interface is used to avoid import cycles between tools and agent packages */
type VirtualFileSystemInterface interface {
	CreateFile(ctx context.Context, agentID, sessionID uuid.UUID, path string, content []byte, mimeType string) (uuid.UUID, error)
	ReadFile(ctx context.Context, agentID uuid.UUID, path string) (*VirtualFile, error)
	WriteFile(ctx context.Context, agentID uuid.UUID, path string, content []byte) error
	DeleteFile(ctx context.Context, agentID uuid.UUID, path string) error
	ListFiles(ctx context.Context, agentID uuid.UUID, sessionID *uuid.UUID, dirPath string) ([]VirtualFile, error)
	CopyFile(ctx context.Context, agentID uuid.UUID, srcPath, dstPath string) error
	MoveFile(ctx context.Context, agentID uuid.UUID, srcPath, dstPath string) error
}

/* VirtualFile represents a file in the virtual file system */
type VirtualFile struct {
	ID             uuid.UUID
	AgentID        uuid.UUID
	SessionID      *uuid.UUID
	Path           string
	Content        []byte
	MimeType       string
	Size           int64
	Compressed     bool
	StorageBackend string
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

/* FileSystemTool provides file system operations for agents */
type FileSystemTool struct {
	vfs VirtualFileSystemInterface
}

/* NewFileSystemTool creates a new file system tool */
func NewFileSystemTool(vfs VirtualFileSystemInterface) *FileSystemTool {
	return &FileSystemTool{
		vfs: vfs,
	}
}

/* Execute executes a file system operation */
func (t *FileSystemTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		return "", fmt.Errorf("filesystem tool requires action parameter")
	}

	agentIDStr, ok := args["agent_id"].(string)
	if !ok {
		return "", fmt.Errorf("filesystem tool requires agent_id parameter")
	}

	agentID, err := uuid.Parse(agentIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid agent_id: %w", err)
	}

	var sessionID *uuid.UUID
	if sid, ok := args["session_id"].(string); ok && sid != "" {
		parsed, err := uuid.Parse(sid)
		if err == nil {
			sessionID = &parsed
		}
	}

	switch action {
	case "create_file":
		return t.createFile(ctx, agentID, sessionID, args)
	case "read_file":
		return t.readFile(ctx, agentID, args)
	case "write_file":
		return t.writeFile(ctx, agentID, args)
	case "delete_file":
		return t.deleteFile(ctx, agentID, args)
	case "list_files":
		return t.listFiles(ctx, agentID, sessionID, args)
	case "search_files":
		return t.searchFiles(ctx, agentID, args)
	default:
		return "", fmt.Errorf("unknown filesystem action: %s", action)
	}
}

/* createFile creates a new file */
func (t *FileSystemTool) createFile(ctx context.Context, agentID uuid.UUID, sessionID *uuid.UUID, args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("create_file requires path parameter")
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("create_file requires content parameter")
	}

	mimeType := "text/plain"
	if mt, ok := args["mime_type"].(string); ok {
		mimeType = mt
	}

	var sessionUUID uuid.UUID
	if sessionID != nil {
		sessionUUID = *sessionID
	} else {
		sessionUUID = uuid.Nil
	}

	fileID, err := t.vfs.CreateFile(ctx, agentID, sessionUUID, path, []byte(content), mimeType)
	if err != nil {
		return "", fmt.Errorf("file creation failed: %w", err)
	}

	result := map[string]interface{}{
		"action":  "create_file",
		"file_id": fileID.String(),
		"path":    path,
		"status":  "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* readFile reads file content */
func (t *FileSystemTool) readFile(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("read_file requires path parameter")
	}

	file, err := t.vfs.ReadFile(ctx, agentID, path)
	if err != nil {
		return "", fmt.Errorf("file read failed: %w", err)
	}

	result := map[string]interface{}{
		"action":    "read_file",
		"path":      file.Path,
		"content":   string(file.Content),
		"mime_type": file.MimeType,
		"size":      file.Size,
		"status":    "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* writeFile writes or updates file content */
func (t *FileSystemTool) writeFile(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("write_file requires path parameter")
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("write_file requires content parameter")
	}

	err := t.vfs.WriteFile(ctx, agentID, path, []byte(content))
	if err != nil {
		return "", fmt.Errorf("file write failed: %w", err)
	}

	result := map[string]interface{}{
		"action": "write_file",
		"path":   path,
		"status": "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* deleteFile deletes a file */
func (t *FileSystemTool) deleteFile(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("delete_file requires path parameter")
	}

	err := t.vfs.DeleteFile(ctx, agentID, path)
	if err != nil {
		return "", fmt.Errorf("file deletion failed: %w", err)
	}

	result := map[string]interface{}{
		"action": "delete_file",
		"path":   path,
		"status": "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* listFiles lists files in a directory */
func (t *FileSystemTool) listFiles(ctx context.Context, agentID uuid.UUID, sessionID *uuid.UUID, args map[string]interface{}) (string, error) {
	dirPath := "/"
	if dp, ok := args["dir_path"].(string); ok {
		dirPath = dp
	}

	files, err := t.vfs.ListFiles(ctx, agentID, sessionID, dirPath)
	if err != nil {
		return "", fmt.Errorf("file listing failed: %w", err)
	}

	fileList := make([]map[string]interface{}, len(files))
	for i, file := range files {
		fileList[i] = map[string]interface{}{
			"path":       file.Path,
			"mime_type":  file.MimeType,
			"size":       file.Size,
			"created_at": file.CreatedAt,
		}
	}

	result := map[string]interface{}{
		"action":     "list_files",
		"dir_path":   dirPath,
		"files":      fileList,
		"file_count": len(fileList),
		"status":     "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* searchFiles searches files by name or content */
func (t *FileSystemTool) searchFiles(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("search_files requires query parameter")
	}

	/* List all files and filter */
	files, err := t.vfs.ListFiles(ctx, agentID, nil, "/")
	if err != nil {
		return "", fmt.Errorf("file search failed: %w", err)
	}

	/* Simple name-based search */
	matchingFiles := make([]map[string]interface{}, 0)
	for _, file := range files {
		if strings.Contains(strings.ToLower(file.Path), strings.ToLower(query)) {
			matchingFiles = append(matchingFiles, map[string]interface{}{
				"path":      file.Path,
				"mime_type": file.MimeType,
				"size":      file.Size,
			})
		}
	}

	result := map[string]interface{}{
		"action":      "search_files",
		"query":       query,
		"files":       matchingFiles,
		"match_count": len(matchingFiles),
		"status":      "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* Validate validates tool arguments */
func (t *FileSystemTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	action, ok := args["action"].(string)
	if !ok {
		return fmt.Errorf("action parameter required")
	}

	validActions := map[string]bool{
		"create_file":  true,
		"read_file":    true,
		"write_file":   true,
		"delete_file":  true,
		"list_files":   true,
		"search_files": true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return nil
}

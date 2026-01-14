/*-------------------------------------------------------------------------
 *
 * virtual_fs.go
 *    Virtual file system for agent scratchpad operations
 *
 * Provides persistent file storage for agents to externalize memory and
 * store large data without overloading context windows. Supports both
 * database storage for small files and object storage for large files.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/virtual_fs.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* VirtualFileSystem manages virtual file system operations */
type VirtualFileSystem struct {
	queries *db.Queries
	storage StorageBackend
	maxSize int64
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

/* StorageBackend interface for different storage implementations */
type StorageBackend interface {
	Store(ctx context.Context, key string, content []byte) error
	Retrieve(ctx context.Context, key string) ([]byte, error)
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
}

/* NewVirtualFileSystem creates a new virtual file system */
func NewVirtualFileSystem(queries *db.Queries, storage StorageBackend, maxSize int64) *VirtualFileSystem {
	return &VirtualFileSystem{
		queries: queries,
		storage: storage,
		maxSize: maxSize,
	}
}

/* CreateFile creates a new file in the virtual file system */
func (v *VirtualFileSystem) CreateFile(ctx context.Context, agentID, sessionID uuid.UUID, path string, content []byte, mimeType string) (uuid.UUID, error) {
	/* Validate path */
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	/* Validate size */
	if int64(len(content)) > v.maxSize {
		return uuid.Nil, fmt.Errorf("file size %d exceeds maximum %d", len(content), v.maxSize)
	}

	/* Determine storage backend */
	storageBackend := "database"
	var contentBytes []byte = content
	var s3Key *string
	compressed := false

	if int64(len(content)) > 1024*1024 {
		/* Use S3 for files > 1MB */
		storageBackend = "s3"
		key := fmt.Sprintf("agents/%s/%s", agentID.String(), path)
		s3Key = &key

		err := v.storage.Store(ctx, key, content)
		if err != nil {
			return uuid.Nil, fmt.Errorf("S3 storage failed: error=%w", err)
		}
		contentBytes = nil
	} else {
		/* Compress text files */
		if strings.HasPrefix(mimeType, "text/") {
			compressed, contentBytes = v.compressContent(content)
		}
	}

	/* Insert file record */
	query := `INSERT INTO neurondb_agent.virtual_files
		(agent_id, session_id, path, content, content_s3_key, mime_type, size, compressed, storage_backend)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		RETURNING id, created_at, updated_at`

	type FileResult struct {
		ID        uuid.UUID `db:"id"`
		CreatedAt time.Time `db:"created_at"`
		UpdatedAt time.Time `db:"updated_at"`
	}

	var result FileResult
	err := v.queries.GetDB().GetContext(ctx, &result, query, agentID, sessionID, path, contentBytes, s3Key, mimeType, len(content), compressed, storageBackend)
	if err != nil {
		return uuid.Nil, fmt.Errorf("file creation failed: path=%s, error=%w", path, err)
	}

	/* Log access */
	v.logAccess(ctx, result.ID, agentID, "create")

	return result.ID, nil
}

/* ReadFile reads file content from the virtual file system */
func (v *VirtualFileSystem) ReadFile(ctx context.Context, agentID uuid.UUID, path string) (*VirtualFile, error) {
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	query := `SELECT id, agent_id, session_id, path, content, content_s3_key, mime_type, size, compressed, storage_backend, created_at, updated_at
		FROM neurondb_agent.virtual_files
		WHERE agent_id = $1 AND path = $2`

	type FileRow struct {
		ID             uuid.UUID  `db:"id"`
		AgentID        uuid.UUID  `db:"agent_id"`
		SessionID      *uuid.UUID `db:"session_id"`
		Path           string     `db:"path"`
		Content        []byte     `db:"content"`
		ContentS3Key   *string    `db:"content_s3_key"`
		MimeType       string     `db:"mime_type"`
		Size           int64      `db:"size"`
		Compressed     bool       `db:"compressed"`
		StorageBackend string     `db:"storage_backend"`
		CreatedAt      time.Time  `db:"created_at"`
		UpdatedAt      time.Time  `db:"updated_at"`
	}

	var row FileRow
	err := v.queries.GetDB().GetContext(ctx, &row, query, agentID, path)
	if err != nil {
		return nil, fmt.Errorf("file read failed: path=%s, error=%w", path, err)
	}

	/* Retrieve content based on storage backend */
	var content []byte
	if row.StorageBackend == "s3" && row.ContentS3Key != nil {
		content, err = v.storage.Retrieve(ctx, *row.ContentS3Key)
		if err != nil {
			return nil, fmt.Errorf("S3 retrieval failed: key=%s, error=%w", *row.ContentS3Key, err)
		}
	} else {
		content = row.Content
	}

	/* Decompress if needed */
	if row.Compressed {
		content, err = v.decompressContent(content)
		if err != nil {
			return nil, fmt.Errorf("decompression failed: error=%w", err)
		}
	}

	/* Log access */
	v.logAccess(ctx, row.ID, agentID, "read")

	file := &VirtualFile{
		ID:             row.ID,
		AgentID:        row.AgentID,
		SessionID:      row.SessionID,
		Path:           row.Path,
		Content:        content,
		MimeType:       row.MimeType,
		Size:           row.Size,
		Compressed:     row.Compressed,
		StorageBackend: row.StorageBackend,
		CreatedAt:      row.CreatedAt,
		UpdatedAt:      row.UpdatedAt,
	}

	return file, nil
}

/* WriteFile writes or updates file content */
func (v *VirtualFileSystem) WriteFile(ctx context.Context, agentID uuid.UUID, path string, content []byte) error {
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	/* Check if file exists */
	existing, err := v.ReadFile(ctx, agentID, path)
	if err != nil {
		/* File doesn't exist, create it */
		_, err = v.CreateFile(ctx, agentID, uuid.Nil, path, content, "text/plain")
		return err
	}

	/* Update existing file */
	if int64(len(content)) > v.maxSize {
		return fmt.Errorf("file size %d exceeds maximum %d", len(content), v.maxSize)
	}

	storageBackend := "database"
	var contentBytes []byte = content
	var s3Key *string
	compressed := false

	if int64(len(content)) > 1024*1024 {
		storageBackend = "s3"
		key := fmt.Sprintf("agents/%s/%s", agentID.String(), path)
		s3Key = &key

		err := v.storage.Store(ctx, key, content)
		if err != nil {
			return fmt.Errorf("S3 storage failed: error=%w", err)
		}

		/* Delete old S3 object if different */
		if existing.StorageBackend == "s3" && existing.Path != path {
			v.storage.Delete(ctx, existing.Path)
		}

		contentBytes = nil
	} else {
		if strings.HasPrefix(existing.MimeType, "text/") {
			compressed, contentBytes = v.compressContent(content)
		}
	}

	/* Update file record */
	updateQuery := `UPDATE neurondb_agent.virtual_files
		SET content = $1, content_s3_key = $2, size = $3, compressed = $4, storage_backend = $5, updated_at = NOW()
		WHERE id = $6`

	_, err = v.queries.GetDB().ExecContext(ctx, updateQuery, contentBytes, s3Key, len(content), compressed, storageBackend, existing.ID)
	if err != nil {
		return fmt.Errorf("file update failed: path=%s, error=%w", path, err)
	}

	/* Log access */
	v.logAccess(ctx, existing.ID, agentID, "write")

	return nil
}

/* DeleteFile deletes a file from the virtual file system */
func (v *VirtualFileSystem) DeleteFile(ctx context.Context, agentID uuid.UUID, path string) error {
	path = filepath.Clean(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	/* Get file info before deletion */
	file, err := v.ReadFile(ctx, agentID, path)
	if err != nil {
		return err
	}

	/* Delete from S3 if applicable */
	if file.StorageBackend == "s3" && file.Path != "" {
		v.storage.Delete(ctx, file.Path)
	}

	/* Delete file record */
	deleteQuery := `DELETE FROM neurondb_agent.virtual_files WHERE agent_id = $1 AND path = $2`
	_, err = v.queries.GetDB().ExecContext(ctx, deleteQuery, agentID, path)
	if err != nil {
		return fmt.Errorf("file deletion failed: path=%s, error=%w", path, err)
	}

	/* Log access */
	v.logAccess(ctx, file.ID, agentID, "delete")

	return nil
}

/* ListFiles lists files in a directory */
func (v *VirtualFileSystem) ListFiles(ctx context.Context, agentID uuid.UUID, sessionID *uuid.UUID, dirPath string) ([]VirtualFile, error) {
	dirPath = filepath.Clean(dirPath)
	if !strings.HasPrefix(dirPath, "/") {
		dirPath = "/" + dirPath
	}
	if !strings.HasSuffix(dirPath, "/") {
		dirPath = dirPath + "/"
	}

	query := `SELECT id, agent_id, session_id, path, mime_type, size, storage_backend, created_at, updated_at
		FROM neurondb_agent.virtual_files
		WHERE agent_id = $1 AND path LIKE $2`

	if sessionID != nil {
		query += ` AND (session_id = $3 OR session_id IS NULL)`
	}

	query += ` ORDER BY path`

	type FileRow struct {
		ID             uuid.UUID  `db:"id"`
		AgentID        uuid.UUID  `db:"agent_id"`
		SessionID      *uuid.UUID `db:"session_id"`
		Path           string     `db:"path"`
		MimeType       string     `db:"mime_type"`
		Size           int64      `db:"size"`
		StorageBackend string     `db:"storage_backend"`
		CreatedAt      time.Time  `db:"created_at"`
		UpdatedAt      time.Time  `db:"updated_at"`
	}

	var rows []FileRow
	var err error
	if sessionID != nil {
		err = v.queries.GetDB().SelectContext(ctx, &rows, query, agentID, dirPath+"%", sessionID)
	} else {
		err = v.queries.GetDB().SelectContext(ctx, &rows, query, agentID, dirPath+"%")
	}

	if err != nil {
		return nil, fmt.Errorf("file listing failed: dir_path=%s, error=%w", dirPath, err)
	}

	files := make([]VirtualFile, len(rows))
	for i, row := range rows {
		files[i] = VirtualFile{
			ID:             row.ID,
			AgentID:        row.AgentID,
			SessionID:      row.SessionID,
			Path:           row.Path,
			MimeType:       row.MimeType,
			Size:           row.Size,
			StorageBackend: row.StorageBackend,
			CreatedAt:      row.CreatedAt,
			UpdatedAt:      row.UpdatedAt,
		}
	}

	return files, nil
}

/* CopyFile copies a file to a new path */
func (v *VirtualFileSystem) CopyFile(ctx context.Context, agentID uuid.UUID, srcPath, dstPath string) error {
	srcFile, err := v.ReadFile(ctx, agentID, srcPath)
	if err != nil {
		return fmt.Errorf("source file read failed: error=%w", err)
	}

	var sessionID uuid.UUID
	if srcFile.SessionID != nil {
		sessionID = *srcFile.SessionID
	}

	_, err = v.CreateFile(ctx, agentID, sessionID, dstPath, srcFile.Content, srcFile.MimeType)
	if err != nil {
		return fmt.Errorf("destination file creation failed: error=%w", err)
	}

	return nil
}

/* MoveFile moves a file to a new path */
func (v *VirtualFileSystem) MoveFile(ctx context.Context, agentID uuid.UUID, srcPath, dstPath string) error {
	err := v.CopyFile(ctx, agentID, srcPath, dstPath)
	if err != nil {
		return err
	}

	err = v.DeleteFile(ctx, agentID, srcPath)
	if err != nil {
		return fmt.Errorf("source file deletion failed after copy: error=%w", err)
	}

	return nil
}

/* compressContent compresses content using gzip */
func (v *VirtualFileSystem) compressContent(content []byte) (bool, []byte) {
	var buf strings.Builder
	writer := gzip.NewWriter(&buf)
	writer.Write(content)
	writer.Close()

	compressed := []byte(buf.String())
	if len(compressed) < len(content) {
		return true, compressed
	}

	return false, content
}

/* decompressContent decompresses gzip-compressed content */
func (v *VirtualFileSystem) decompressContent(content []byte) ([]byte, error) {
	reader, err := gzip.NewReader(strings.NewReader(string(content)))
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	decompressed, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	return decompressed, nil
}

/* logAccess logs file access for audit */
func (v *VirtualFileSystem) logAccess(ctx context.Context, fileID, agentID uuid.UUID, operation string) {
	query := `INSERT INTO neurondb_agent.file_access_log (file_id, agent_id, operation)
		VALUES ($1, $2, $3)`

	_, err := v.queries.GetDB().ExecContext(ctx, query, fileID, agentID, operation)
	/* Log access errors are non-fatal - silently ignore */
	_ = err
}

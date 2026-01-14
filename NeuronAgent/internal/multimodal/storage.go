/*-------------------------------------------------------------------------
 *
 * storage.go
 *    Media file storage
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/multimodal/storage.go
 *
 *-------------------------------------------------------------------------
 */

package multimodal

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

/* MediaStorage manages media file storage */
type MediaStorage struct {
	basePath string
}

/* NewMediaStorage creates a new media storage */
func NewMediaStorage(basePath string) (*MediaStorage, error) {
	if basePath == "" {
		basePath = "/tmp/neuronagent/media"
	}

	/* Create base directory */
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create media storage directory: %w", err)
	}

	return &MediaStorage{
		basePath: basePath,
	}, nil
}

/* Store stores a media file */
func (ms *MediaStorage) Store(ctx context.Context, file *MediaFile) (string, error) {
	if file.ID == "" {
		file.ID = generateFileID()
	}

	/* Create subdirectory by type */
	filePath := filepath.Join(ms.basePath, string(file.Type), file.ID)

	/* Create type directory */
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	/* Write file */
	if err := os.WriteFile(filePath, file.Data, 0644); err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	return filePath, nil
}

/* Retrieve retrieves a media file */
func (ms *MediaStorage) Retrieve(ctx context.Context, mediaType MediaType, fileID string) (*MediaFile, error) {
	filePath := filepath.Join(ms.basePath, string(mediaType), fileID)

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("file not found: %s", fileID)
		}
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	return &MediaFile{
		ID:         fileID,
		Type:       mediaType,
		Size:       fileInfo.Size(),
		Data:       data,
		UploadedAt: fileInfo.ModTime(),
	}, nil
}

/* Delete deletes a media file */
func (ms *MediaStorage) Delete(ctx context.Context, mediaType MediaType, fileID string) error {
	filePath := filepath.Join(ms.basePath, string(mediaType), fileID)

	if err := os.Remove(filePath); err != nil {
		if os.IsNotExist(err) {
			return nil /* Already deleted */
		}
		return fmt.Errorf("failed to delete file: %w", err)
	}

	return nil
}

/* generateFileID generates a unique file ID */
func generateFileID() string {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		/* Fallback to timestamp-based ID */
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(bytes)
}

/*-------------------------------------------------------------------------
 *
 * gitlab.go
 *    GitLab connector implementation
 *
 * Provides GitLab API integration for reading repositories and files.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/connectors/gitlab.go
 *
 *-------------------------------------------------------------------------
 */

package connectors

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

/* GitLabConnector implements ReadConnector for GitLab */
type GitLabConnector struct {
	client   *http.Client
	endpoint string
	token    string
}

/* NewGitLabConnector creates a new GitLab connector */
func NewGitLabConnector(config Config) (*GitLabConnector, error) {
	endpoint := "https://gitlab.com/api/v4"
	if config.Endpoint != "" {
		endpoint = config.Endpoint
	}

	return &GitLabConnector{
		client:   &http.Client{},
		endpoint: strings.TrimSuffix(endpoint, "/"),
		token:    config.Token,
	}, nil
}

/* Type returns the connector type */
func (g *GitLabConnector) Type() string {
	return "gitlab"
}

/* Connect establishes connection */
func (g *GitLabConnector) Connect(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", g.endpoint+"/user", nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if g.token != "" {
		req.Header.Set("PRIVATE-TOKEN", g.token)
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to GitLab: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GitLab connection failed with status %d", resp.StatusCode)
	}

	return nil
}

/* Close closes the connection */
func (g *GitLabConnector) Close() error {
	return nil
}

/* Health checks connection health */
func (g *GitLabConnector) Health(ctx context.Context) error {
	return g.Connect(ctx)
}

/* Read reads a file from GitLab repository */
func (g *GitLabConnector) Read(ctx context.Context, path string) (io.Reader, error) {
	/* Path format: project_id/path/to/file */
	parts := strings.SplitN(path, "/", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid GitLab path format: expected project_id/path")
	}

	projectID, filePath := parts[0], parts[1]
	url := fmt.Sprintf("%s/projects/%s/repository/files/%s/raw", g.endpoint, projectID, filePath)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if g.token != "" {
		req.Header.Set("PRIVATE-TOKEN", g.token)
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to read from GitLab: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("GitLab read failed with status %d", resp.StatusCode)
	}

	return resp.Body, nil
}

/* List lists files in a GitLab repository */
func (g *GitLabConnector) List(ctx context.Context, path string) ([]string, error) {
	parts := strings.SplitN(path, "/", 2)
	if len(parts) < 1 {
		return nil, fmt.Errorf("invalid GitLab path format: expected project_id[/path]")
	}

	projectID := parts[0]
	dirPath := ""
	if len(parts) > 1 {
		dirPath = parts[1]
	}

	url := fmt.Sprintf("%s/projects/%s/repository/tree", g.endpoint, projectID)
	if dirPath != "" {
		url += "?path=" + dirPath
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if g.token != "" {
		req.Header.Set("PRIVATE-TOKEN", g.token)
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to list from GitLab: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitLab list failed with status %d", resp.StatusCode)
	}

	/* Parse JSON response and extract file paths */
	var treeItems []struct {
		ID   string `json:"id"`
		Name string `json:"name"`
		Type string `json:"type"`
		Path string `json:"path"`
		Mode string `json:"mode"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&treeItems); err != nil {
		return nil, fmt.Errorf("failed to parse GitLab API response: %w", err)
	}

	/* Extract file paths */
	result := make([]string, 0, len(treeItems))
	for _, item := range treeItems {
		if item.Type == "blob" {
			/* It's a file */
			result = append(result, item.Path)
		} else if item.Type == "tree" {
			/* It's a directory, include with trailing slash */
			result = append(result, item.Path+"/")
		}
	}

	return result, nil
}

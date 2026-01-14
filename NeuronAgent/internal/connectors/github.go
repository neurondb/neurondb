/*-------------------------------------------------------------------------
 *
 * github.go
 *    GitHub connector implementation
 *
 * Provides GitHub API integration for reading repositories, files, and issues.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/connectors/github.go
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

/* GitHubConnector implements ReadConnector for GitHub */
type GitHubConnector struct {
	client   *http.Client
	endpoint string
	token    string
}

/* NewGitHubConnector creates a new GitHub connector */
func NewGitHubConnector(config Config) (*GitHubConnector, error) {
	endpoint := "https://api.github.com"
	if config.Endpoint != "" {
		endpoint = config.Endpoint
	}

	return &GitHubConnector{
		client:   &http.Client{},
		endpoint: strings.TrimSuffix(endpoint, "/"),
		token:    config.Token,
	}, nil
}

/* Type returns the connector type */
func (g *GitHubConnector) Type() string {
	return "github"
}

/* Connect establishes connection */
func (g *GitHubConnector) Connect(ctx context.Context) error {
	/* Test connection by making a simple API call */
	req, err := http.NewRequestWithContext(ctx, "GET", g.endpoint+"/user", nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if g.token != "" {
		req.Header.Set("Authorization", "token "+g.token)
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to GitHub: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GitHub connection failed with status %d", resp.StatusCode)
	}

	return nil
}

/* Close closes the connection */
func (g *GitHubConnector) Close() error {
	return nil
}

/* Health checks connection health */
func (g *GitHubConnector) Health(ctx context.Context) error {
	return g.Connect(ctx)
}

/* Read reads a file from GitHub repository */
func (g *GitHubConnector) Read(ctx context.Context, path string) (io.Reader, error) {
	/* Path format: owner/repo/path/to/file */
	parts := strings.SplitN(path, "/", 3)
	if len(parts) < 3 {
		return nil, fmt.Errorf("invalid GitHub path format: expected owner/repo/path")
	}

	owner, repo, filePath := parts[0], parts[1], parts[2]
	url := fmt.Sprintf("%s/repos/%s/%s/contents/%s", g.endpoint, owner, repo, filePath)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if g.token != "" {
		req.Header.Set("Authorization", "token "+g.token)
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to read from GitHub: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("GitHub read failed with status %d", resp.StatusCode)
	}

	return resp.Body, nil
}

/* List lists files in a GitHub repository */
func (g *GitHubConnector) List(ctx context.Context, path string) ([]string, error) {
	/* Path format: owner/repo/path/to/dir */
	parts := strings.SplitN(path, "/", 3)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid GitHub path format: expected owner/repo[/path]")
	}

	owner, repo := parts[0], parts[1]
	dirPath := ""
	if len(parts) > 2 {
		dirPath = parts[2]
	}

	url := fmt.Sprintf("%s/repos/%s/%s/contents/%s", g.endpoint, owner, repo, dirPath)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if g.token != "" {
		req.Header.Set("Authorization", "token "+g.token)
	}

	resp, err := g.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to list from GitHub: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub list failed with status %d", resp.StatusCode)
	}

	/* Parse JSON response and extract file paths */
	var contents []struct {
		Name        string `json:"name"`
		Path        string `json:"path"`
		Type        string `json:"type"`
		Size        int64  `json:"size"`
		DownloadURL string `json:"download_url"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&contents); err != nil {
		return nil, fmt.Errorf("failed to parse GitHub API response: %w", err)
	}

	/* Extract file paths */
	result := make([]string, 0, len(contents))
	for _, item := range contents {
		if item.Type == "file" {
			result = append(result, item.Path)
		} else if item.Type == "dir" {
			/* For directories, include the path with a trailing slash to indicate it's a directory */
			result = append(result, item.Path+"/")
		}
	}

	/* Handle pagination if needed */
	/* GitHub API uses Link header for pagination, but for simplicity we return what we got */
	/* If more results are needed, the caller can make additional requests with different paths */

	return result, nil
}

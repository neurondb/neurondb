/*-------------------------------------------------------------------------
 *
 * interface.go
 *    Connector framework interface
 *
 * Provides interface for connectors (GitHub, GitLab, Slack, email, S3, web crawler).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/connectors/interface.go
 *
 *-------------------------------------------------------------------------
 */

package connectors

import (
	"context"
	"fmt"
	"io"
)

/* Connector defines the interface for connectors */
type Connector interface {
	/* Type returns the connector type */
	Type() string

	/* Connect establishes connection */
	Connect(ctx context.Context) error

	/* Close closes the connection */
	Close() error

	/* Health checks connection health */
	Health(ctx context.Context) error
}

/* ReadConnector defines interface for read-only connectors */
type ReadConnector interface {
	Connector

	/* Read reads data from the source */
	Read(ctx context.Context, path string) (io.Reader, error)

	/* List lists items in the source */
	List(ctx context.Context, path string) ([]string, error)
}

/* WriteConnector defines interface for write-only connectors */
type WriteConnector interface {
	Connector

	/* Write writes data to the destination */
	Write(ctx context.Context, path string, data io.Reader) error
}

/* ReadWriteConnector defines interface for read-write connectors */
type ReadWriteConnector interface {
	ReadConnector
	WriteConnector
}

/* Config defines connector configuration */
type Config struct {
	Type     string                 `json:"type"`     // "github", "gitlab", "slack", "email", "s3", "web"
	Endpoint string                 `json:"endpoint"` // API endpoint
	Token    string                 `json:"token"`    // Authentication token
	Metadata map[string]interface{} `json:"metadata"` // Additional config
}

/* NewConnector creates a new connector based on config */
func NewConnector(config Config) (Connector, error) {
	switch config.Type {
	case "github":
		return NewGitHubConnector(config)
	case "gitlab":
		return NewGitLabConnector(config)
	case "slack":
		return NewSlackConnector(config)
	case "email":
		return nil, fmt.Errorf("email connector not yet implemented")
	case "s3":
		return NewS3Connector(config)
	case "web":
		return nil, fmt.Errorf("web crawler connector not yet implemented")
	default:
		return nil, fmt.Errorf("unknown connector type: %s", config.Type)
	}
}

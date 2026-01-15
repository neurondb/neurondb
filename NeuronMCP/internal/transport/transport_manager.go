/*-------------------------------------------------------------------------
 *
 * transport_manager.go
 *    Multi-transport coordinator for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/transport/transport_manager.go
 *
 *-------------------------------------------------------------------------
 */

package transport

import (
	"context"
	"fmt"
	"net/http"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* TransportType represents a transport type */
type TransportType string

const (
	TransportStdio TransportType = "stdio"
	TransportHTTP  TransportType = "http"
	TransportSSE   TransportType = "sse"
)

/* Manager manages multiple transports */
type Manager struct {
	transports        map[TransportType]interface{}
	mcpServer         *mcp.Server
	prometheusHandler http.Handler
}

/* NewManager creates a new transport manager */
func NewManager(mcpServer *mcp.Server) *Manager {
	return &Manager{
		transports: make(map[TransportType]interface{}),
		mcpServer:  mcpServer,
	}
}

/* SetPrometheusHandler sets the Prometheus metrics handler */
func (m *Manager) SetPrometheusHandler(handler http.Handler) {
	m.prometheusHandler = handler
}

/* StartHTTP starts the HTTP transport */
/* Note: This method uses the old signature. For full middleware support, create HTTPTransport directly with server instance */
func (m *Manager) StartHTTP(addr string) error {
	/* Create with minimal parameters - middleware and server instance should be set separately if needed */
	httpTransport := NewHTTPTransport(addr, m.mcpServer, nil, nil, m.prometheusHandler)
	m.transports[TransportHTTP] = httpTransport
	return httpTransport.Start()
}

/* Shutdown shuts down all transports */
func (m *Manager) Shutdown(ctx context.Context) error {
	for transportType, transport := range m.transports {
		switch transportType {
		case TransportHTTP:
			if httpTransport, ok := transport.(*HTTPTransport); ok {
				if err := httpTransport.Shutdown(ctx); err != nil {
					return fmt.Errorf("failed to shutdown HTTP transport: %w", err)
				}
			}
		}
	}
	return nil
}













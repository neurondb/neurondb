/*-------------------------------------------------------------------------
 *
 * server.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/server.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

/* HandlerFunc is a function that handles an MCP request */
type HandlerFunc func(ctx context.Context, params json.RawMessage) (interface{}, error)

/* Server is an MCP protocol server */
type Server struct {
	transport *StdioTransport
	handlers  map[string]HandlerFunc
	info      ServerInfo
	caps      ServerCapabilities
	maxRequestSize int64 /* Maximum request size in bytes */
}

/* NewServer creates a new MCP server */
func NewServer(name, version string) *Server {
	return NewServerWithMaxRequestSize(name, version, 0) /* Default: unlimited */
}

/* NewServerWithMaxRequestSize creates a new MCP server with max request size */
func NewServerWithMaxRequestSize(name, version string, maxRequestSize int64) *Server {
	return &Server{
		transport:      NewStdioTransportWithMaxSize(maxRequestSize),
		handlers:       make(map[string]HandlerFunc),
		maxRequestSize: maxRequestSize,
		info: ServerInfo{
			Name:    name,
			Version: version,
		},
		caps: ServerCapabilities{
			Tools: ToolsCapability{
				ListChanged: false,
			},
			Resources: ResourcesCapability{
				Subscribe:   false,
				ListChanged: false,
			},
			Prompts:   make(map[string]interface{}),
			Sampling:  make(map[string]interface{}),
		},
	}
}

/* SetHandler registers a handler for a method */
func (s *Server) SetHandler(method string, handler HandlerFunc) {
	s.handlers[method] = handler
}

/* SetCapabilities sets server capabilities */
func (s *Server) SetCapabilities(caps ServerCapabilities) {
	s.caps = caps
}

/* HandleInitialize handles the initialize request */
func (s *Server) HandleInitialize(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req InitializeRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse initialize request: %w", err)
	}

	return InitializeResponse{
		ProtocolVersion: ProtocolVersion,
		Capabilities:    s.caps,
		ServerInfo:      s.info,
	}, nil
}

/* Run starts the server and processes requests */
func (s *Server) Run(ctx context.Context) error {
  /* Register initialize handler */
	s.SetHandler("initialize", s.HandleInitialize)
	
	var initializedSent bool

	for {
		select {
		case <-ctx.Done():
    /* Context cancelled - exit gracefully */
			return ctx.Err()
		default:
    /* Read next message - this will block until a message arrives or EOF */
			req, err := s.transport.ReadMessage()
			if err != nil {
     /* Check for EOF - this means stdin closed (client disconnected) */
				if err == io.EOF {
      /* Client disconnected - exit gracefully */
					return nil
				}
     /* Check if error message contains EOF */
				errStr := err.Error()
				if errStr == "EOF" || strings.Contains(errStr, "EOF") {
      /* Client disconnected - exit gracefully */
					return nil
				}
				
     /* For any other error, log it but CONTINUE running */
     /* The server MUST stay alive and wait for the next message */
     /* Errors like "missing Content-Length header" can happen if there's */
     /* partial input or the client is still connected but hasn't sent a complete message yet */
     /* DO NOT exit on these errors - only exit on EOF */
				s.transport.WriteError(fmt.Errorf("ReadMessage error (server continuing, will retry): %w", err))
				
     /* CRITICAL: Continue the loop - server MUST stay alive */
     /* Only exit on EOF (client disconnect) or context cancellation */
     /* This ensures the server doesn't exit prematurely */
				continue
			}

    /* Handle initialize request - respond with server capabilities */
			if req.Method == "initialize" && !initializedSent {
				resp := s.handleRequest(ctx, req)
				
     /* CRITICAL: ALWAYS send response for initialize request immediately */
				if !IsNotification(req) {
      /* Send the initialize response - client will then send notifications/initialized */
					if err := s.transport.WriteMessage(resp); err != nil {
						s.transport.WriteError(fmt.Errorf("CRITICAL: failed to write initialize response: %w", err))
					}
					
      /* Note: Client sends notifications/initialized notification, not the server */
      /* Per MCP spec: client -> server: initialize request, server -> client: initialize response, */
      /* then client -> server: notifications/initialized notification */
					if resp.Error == nil {
						initializedSent = true
					} else {
       /* Even if there was an error, mark as initialized to prevent retry loops */
						initializedSent = true
					}
				}
     /* Continue loop to wait for next message (including notifications/initialized from client) */
			} else {
     /* Handle other requests */
				resp := s.handleRequest(ctx, req)
				
     /* Only send response if it's a request (has ID), not a notification */
				if !IsNotification(req) {
					if err := s.transport.WriteMessage(resp); err != nil {
						s.transport.WriteError(err)
						continue
					}
				}
			}
		}
	}
}

/* HandleRequest handles an MCP request (exported for HTTP transport) */
func (s *Server) HandleRequest(ctx context.Context, req *JSONRPCRequest) *JSONRPCResponse {
	return s.handleRequest(ctx, req)
}

func (s *Server) handleRequest(ctx context.Context, req *JSONRPCRequest) *JSONRPCResponse {
  /* Validate request */
	if err := ValidateRequest(req); err != nil {
		return CreateErrorResponse(req.ID, ErrCodeInvalidRequest, err.Error(), nil)
	}

  /* Find handler */
	handler, exists := s.handlers[req.Method]
	if !exists {
		return CreateErrorResponse(req.ID, ErrCodeMethodNotFound,
			fmt.Sprintf("method not found: %s", req.Method), nil)
	}

  /* Execute handler */
	result, err := handler(ctx, req.Params)
	if err != nil {
		return CreateErrorResponse(req.ID, ErrCodeInternalError, err.Error(), nil)
	}

	return CreateResponse(req.ID, result)
}


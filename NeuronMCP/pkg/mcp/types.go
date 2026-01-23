/*-------------------------------------------------------------------------
 *
 * types.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/types.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import "encoding/json"

/* JSON-RPC 2.0 message types */
type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  interface{}     `json:"result,omitempty"`
	Error   *JSONRPCError   `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

/* MCP Request types */
type ListToolsRequest struct {
	Method string `json:"method"`
}

type CallToolRequest struct {
	Name           string                 `json:"name"`
	Arguments      map[string]interface{} `json:"arguments,omitempty"`
	DryRun         bool                   `json:"dryRun,omitempty"`
	IdempotencyKey string                 `json:"idempotencyKey,omitempty"`
	RequireConfirm bool                   `json:"requireConfirm,omitempty"`
}

type ListResourcesRequest struct {
	Method string `json:"method"`
}

type ReadResourceRequest struct {
	URI string `json:"uri"`
}

type SubscribeResourceRequest struct {
	URI    string `json:"uri"`
	Filter *string `json:"filter,omitempty"` /* Optional filter pattern */
}

type UnsubscribeResourceRequest struct {
	SubscriptionID string `json:"subscriptionId"`
}

type SubscribeResourceResponse struct {
	SubscriptionID string `json:"subscriptionId"`
}

type ResourceUpdateNotification struct {
	SubscriptionID string                 `json:"subscriptionId"`
	URI            string                 `json:"uri"`
	Type           string                 `json:"type"` /* created, updated, deleted */
	Content        interface{}            `json:"content,omitempty"`
}

/* MCP Response types */
type ToolDefinition struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]interface{} `json:"inputSchema"`
	OutputSchema map[string]interface{} `json:"outputSchema,omitempty"`
	Version      string                 `json:"version,omitempty"`
	Deprecated   bool                   `json:"deprecated,omitempty"`
	Deprecation  *DeprecationInfo       `json:"deprecation,omitempty"`
}

/* DeprecationInfo provides information about tool deprecation */
type DeprecationInfo struct {
	Message        string `json:"message,omitempty"`
	DeprecatedAt   string `json:"deprecatedAt,omitempty"`
	SunsetAt       string `json:"sunsetAt,omitempty"`
	Replacement    string `json:"replacement,omitempty"`
	MigrationGuide string `json:"migrationGuide,omitempty"`
}

type ListToolsResponse struct {
	Tools []ToolDefinition `json:"tools"`
}

type ToolResult struct {
	Content  []ContentBlock `json:"content"`
	IsError  bool           `json:"isError,omitempty"`
	Metadata interface{}    `json:"metadata,omitempty"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ResourceDefinition struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mimeType"`
}

type ListResourcesResponse struct {
	Resources []ResourceDefinition `json:"resources"`
}

type ReadResourceResponse struct {
	Contents []ResourceContent `json:"contents"`
}

type ResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType"`
	Text     string `json:"text"`
}

/* Server info */
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type ServerCapabilities struct {
	Tools        ToolsCapability              `json:"tools,omitempty"`
	Resources    ResourcesCapability          `json:"resources,omitempty"`
	Prompts      map[string]interface{}       `json:"prompts,omitempty"`
	Sampling     map[string]interface{}       `json:"sampling,omitempty"`
	Elicitation  *ElicitationCapability       `json:"elicitation,omitempty"`
	Completions  *CompletionsCapability       `json:"completions,omitempty"`
	Experimental map[string]interface{}       `json:"experimental,omitempty"`
}

type ToolsCapability struct {
	ListChanged bool `json:"listChanged"` /* Remove omitempty - Claude Desktop needs to see this field */
}

type ResourcesCapability struct {
	Subscribe   bool `json:"subscribe"`   /* Remove omitempty - Claude Desktop needs to see this field */
	ListChanged bool `json:"listChanged"` /* Remove omitempty - Claude Desktop needs to see this field */
}

type InitializeRequest struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities,omitempty"`
	ClientInfo      map[string]interface{} `json:"clientInfo,omitempty"`
}

type ElicitationCapability struct {
	Enabled bool `json:"enabled"`
}

type CompletionsCapability struct {
	Enabled bool `json:"enabled"`
}

type RequestPromptRequest struct {
	Message  string                 `json:"message"`
	Type     string                 `json:"type,omitempty"`
	Options  []string               `json:"options,omitempty"`
	Default  interface{}            `json:"default,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type RespondPromptRequest struct {
	RequestID string      `json:"requestId"`
	Value     interface{} `json:"value"`
}

type InitializeResponse struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
}

/* Completion request types */
type CompletionRequest struct {
	Ref       CompletionReference `json:"ref"`
	Argument  CompletionArgument  `json:"argument"`
	Context   map[string]interface{} `json:"context,omitempty"`
}

type CompletionReference struct {
	Type string `json:"type"` /* ref/prompt or ref/resource */
	Name string `json:"name,omitempty"` /* Prompt name for ref/prompt */
}

type CompletionArgument struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type CompletionResponse struct {
	Completion CompletionResult `json:"completion"`
}

type CompletionResult struct {
	Values  []string `json:"values"`
	Total   int      `json:"total"`
	HasMore bool     `json:"hasMore"`
}


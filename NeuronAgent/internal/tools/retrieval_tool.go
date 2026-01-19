/*-------------------------------------------------------------------------
 *
 * retrieval_tool.go
 *    Retrieval tool for agent-controlled knowledge retrieval
 *
 * Provides agent with ability to decide when and where to retrieve
 * information from different knowledge sources (vector DB, web, APIs).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/retrieval_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* RetrievalInterface defines the interface for retrieval operations */
/* This interface is used to avoid import cycles between tools and agent packages */
type RetrievalInterface interface {
	RetrieveFromVectorDB(ctx context.Context, agentID uuid.UUID, query string, topK int) ([]map[string]interface{}, error)
	ShouldRetrieve(ctx context.Context, query string, context string) (bool, float64, string, error)
	CheckRelevance(ctx context.Context, query string, existingContext []string) (float64, bool, error)
}

/* KnowledgeRouterInterface defines the interface for knowledge source routing */
type KnowledgeRouterInterface interface {
	RouteQuery(ctx context.Context, query string) ([]string, map[string]float64, error)
}

/* RetrievalTool provides agent-controlled retrieval operations */
type RetrievalTool struct {
	retrieval RetrievalInterface
	router    KnowledgeRouterInterface
	webSearch WebSearchInterface
	httpTool  *HTTPTool
}

/* WebSearchInterface defines the interface for web search operations */
type WebSearchInterface interface {
	Search(ctx context.Context, query string, maxResults int) ([]map[string]interface{}, error)
}

/* NewRetrievalTool creates a new retrieval tool */
func NewRetrievalTool(retrieval RetrievalInterface, router KnowledgeRouterInterface, webSearch WebSearchInterface, httpTool *HTTPTool) *RetrievalTool {
	return &RetrievalTool{
		retrieval: retrieval,
		router:    router,
		webSearch: webSearch,
		httpTool:  httpTool,
	}
}

/* Execute executes a retrieval operation */
func (t *RetrievalTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		return "", fmt.Errorf("retrieval tool requires action parameter")
	}

	agentIDStr, ok := args["agent_id"].(string)
	if !ok {
		return "", fmt.Errorf("retrieval tool requires agent_id parameter")
	}

	agentID, err := uuid.Parse(agentIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid agent_id: %w", err)
	}

	switch action {
	case "should_retrieve":
		return t.shouldRetrieve(ctx, agentID, args)
	case "retrieve_from_vector_db":
		return t.retrieveFromVectorDB(ctx, agentID, args)
	case "retrieve_from_web":
		return t.retrieveFromWeb(ctx, agentID, args)
	case "retrieve_from_api":
		return t.retrieveFromAPI(ctx, agentID, args)
	case "check_relevance":
		return t.checkRelevance(ctx, agentID, args)
	case "route_query":
		return t.routeQuery(ctx, agentID, args)
	default:
		return "", fmt.Errorf("unknown retrieval action: %s", action)
	}
}

/* shouldRetrieve determines if retrieval is needed using LLM-based decision */
func (t *RetrievalTool) shouldRetrieve(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("should_retrieve requires query parameter")
	}

	contextStr := ""
	if c, ok := args["context"].(string); ok {
		contextStr = c
	}

	if t.retrieval == nil {
		return "", fmt.Errorf("retrieval interface not available")
	}

	shouldRetrieve, confidence, reason, err := t.retrieval.ShouldRetrieve(ctx, query, contextStr)
	if err != nil {
		return "", fmt.Errorf("retrieval decision failed: %w", err)
	}

	result := map[string]interface{}{
		"action":         "should_retrieve",
		"should_retrieve": shouldRetrieve,
		"confidence":     confidence,
		"reason":         reason,
		"query":          query,
		"status":         "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* retrieveFromVectorDB retrieves from vector database */
func (t *RetrievalTool) retrieveFromVectorDB(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("retrieve_from_vector_db requires query parameter")
	}

	topK := 5
	if k, ok := args["top_k"].(float64); ok {
		topK = int(k)
	}

	if t.retrieval == nil {
		return "", fmt.Errorf("retrieval interface not available")
	}

	results, err := t.retrieval.RetrieveFromVectorDB(ctx, agentID, query, topK)
	if err != nil {
		return "", fmt.Errorf("vector DB retrieval failed: %w", err)
	}

	result := map[string]interface{}{
		"action":      "retrieve_from_vector_db",
		"source":      "vector_db",
		"query":       query,
		"top_k":       topK,
		"results":     results,
		"result_count": len(results),
		"status":      "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* retrieveFromWeb retrieves from web search */
func (t *RetrievalTool) retrieveFromWeb(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("retrieve_from_web requires query parameter")
	}

	if query == "" {
		return "", fmt.Errorf("query cannot be empty")
	}

	maxResults := 5
	if mr, ok := args["max_results"].(float64); ok {
		maxResults = int(mr)
		if maxResults < 1 {
			maxResults = 1
		}
		if maxResults > 20 {
			maxResults = 20 /* Cap at reasonable limit */
		}
	}

	var results []map[string]interface{}
	var err error
	sourceUsed := "unknown"

	/* Try web search tool first */
	if t.webSearch != nil {
		webResults, webErr := t.webSearch.Search(ctx, query, maxResults)
		if webErr == nil && len(webResults) > 0 {
			results = webResults
			sourceUsed = "web_search_tool"
		} else if webErr != nil {
			err = webErr
		}
	}

	/* Fallback to HTTP tool if web search not available or failed */
	if len(results) == 0 && t.httpTool != nil {
		/* Try multiple fallback sources */
		fallbackSources := []struct {
			name string
			url  string
		}{
			{
				name: "duckduckgo_api",
				url:  fmt.Sprintf("https://api.duckduckgo.com/?q=%s&format=json&no_html=1&skip_disambig=1", strings.ReplaceAll(query, " ", "+")),
			},
		}

		for _, fallback := range fallbackSources {
			httpResult, httpErr := t.httpTool.Execute(ctx, &db.Tool{Name: "http"}, map[string]interface{}{
				"url":    fallback.url,
				"method": "GET",
				"headers": map[string]interface{}{
					"User-Agent": "Mozilla/5.0 (compatible; NeuronAgent/1.0)",
				},
			})
			if httpErr == nil {
				var apiResponse map[string]interface{}
				if json.Unmarshal([]byte(httpResult), &apiResponse) == nil {
					if body, ok := apiResponse["body"].(string); ok {
						var ddgResponse map[string]interface{}
						if json.Unmarshal([]byte(body), &ddgResponse) == nil {
							if abstract, ok := ddgResponse["Abstract"].(string); ok && abstract != "" {
								results = []map[string]interface{}{
									{
										"title":   getStringValue(ddgResponse, "Heading"),
										"content": abstract,
										"url":     getStringValue(ddgResponse, "AbstractURL"),
										"source":  fallback.name,
									},
								}
								sourceUsed = fallback.name
								err = nil
								break
							}
						}
					}
				}
			}
		}
	}

	/* If still no results, return error with helpful message */
	if len(results) == 0 {
		if err != nil {
			return "", fmt.Errorf("web retrieval failed from all sources: %w", err)
		}
		return "", fmt.Errorf("web retrieval returned no results for query: %s", query)
	}

	result := map[string]interface{}{
		"action":       "retrieve_from_web",
		"source":       sourceUsed,
		"query":        query,
		"max_results":  maxResults,
		"results":      results,
		"result_count": len(results),
		"status":       "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* getStringValue safely extracts string value from map */
func getStringValue(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

/* retrieveFromAPI retrieves from external API */
func (t *RetrievalTool) retrieveFromAPI(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	apiURL, ok := args["api_url"].(string)
	if !ok {
		return "", fmt.Errorf("retrieve_from_api requires api_url parameter")
	}

	method := "GET"
	if m, ok := args["method"].(string); ok {
		method = m
	}

	headers := make(map[string]interface{})
	if h, ok := args["headers"].(map[string]interface{}); ok {
		headers = h
	}

	body := ""
	if b, ok := args["body"].(string); ok {
		body = b
	}

	if t.httpTool == nil {
		return "", fmt.Errorf("HTTP tool not available for API retrieval")
	}

	httpArgs := map[string]interface{}{
		"url":    apiURL,
		"method": method,
	}
	if len(headers) > 0 {
		httpArgs["headers"] = headers
	}
	if body != "" {
		httpArgs["body"] = body
	}

	httpResult, err := t.httpTool.Execute(ctx, &db.Tool{Name: "http"}, httpArgs)
	if err != nil {
		return "", fmt.Errorf("API retrieval failed: %w", err)
	}

	var apiResponse map[string]interface{}
	if err := json.Unmarshal([]byte(httpResult), &apiResponse); err != nil {
		/* If not JSON, return as plain text */
		apiResponse = map[string]interface{}{
			"body": httpResult,
		}
	}

	result := map[string]interface{}{
		"action":     "retrieve_from_api",
		"source":     "api",
		"api_url":    apiURL,
		"method":     method,
		"response":   apiResponse,
		"status":     "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* checkRelevance checks if retrieval is likely to be relevant */
func (t *RetrievalTool) checkRelevance(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("check_relevance requires query parameter")
	}

	existingContext := make([]string, 0)
	if ctxList, ok := args["existing_context"].([]interface{}); ok {
		for _, item := range ctxList {
			if str, ok := item.(string); ok {
				existingContext = append(existingContext, str)
			}
		}
	}

	if t.retrieval == nil {
		return "", fmt.Errorf("retrieval interface not available")
	}

	relevanceScore, isRelevant, err := t.retrieval.CheckRelevance(ctx, query, existingContext)
	if err != nil {
		return "", fmt.Errorf("relevance check failed: %w", err)
	}

	result := map[string]interface{}{
		"action":          "check_relevance",
		"query":           query,
		"relevance_score": relevanceScore,
		"is_relevant":     isRelevant,
		"recommendation":  "retrieve",
		"status":          "success",
	}

	if !isRelevant {
		result["recommendation"] = "skip"
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* routeQuery routes query to appropriate knowledge sources */
func (t *RetrievalTool) routeQuery(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("route_query requires query parameter")
	}

	if t.router == nil {
		return "", fmt.Errorf("knowledge router not available")
	}

	sources, scores, err := t.router.RouteQuery(ctx, query)
	if err != nil {
		return "", fmt.Errorf("query routing failed: %w", err)
	}

	result := map[string]interface{}{
		"action":        "route_query",
		"query":         query,
		"recommended_sources": sources,
		"source_scores": scores,
		"status":        "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* Validate validates tool arguments */
func (t *RetrievalTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	action, ok := args["action"].(string)
	if !ok {
		return fmt.Errorf("action parameter required")
	}

	validActions := map[string]bool{
		"should_retrieve":      true,
		"retrieve_from_vector_db": true,
		"retrieve_from_web":    true,
		"retrieve_from_api":    true,
		"check_relevance":      true,
		"route_query":          true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return nil
}

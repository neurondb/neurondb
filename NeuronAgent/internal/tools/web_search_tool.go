/*-------------------------------------------------------------------------
 *
 * web_search_tool.go
 *    Web search tool for retrieving information from the web
 *
 * Provides web search capabilities using various search APIs
 * (DuckDuckGo, Google, Bing, etc.) with result caching.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/web_search_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* WebSearchTool provides web search functionality */
type WebSearchTool struct {
	client      *http.Client
	cache       map[string]cacheEntry
	cacheMu     sync.RWMutex
	cacheExpiry time.Duration
}

type cacheEntry struct {
	results   []map[string]interface{}
	timestamp time.Time
}

/* NewWebSearchTool creates a new web search tool */
func NewWebSearchTool() *WebSearchTool {
	return &WebSearchTool{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		cache:       make(map[string]cacheEntry),
		cacheExpiry: 1 * time.Hour,
	}
}

/* Execute executes a web search operation */
func (t *WebSearchTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		action = "search" /* Default action */
	}

	switch action {
	case "search":
		return t.search(ctx, tool, args)
	case "search_with_filters":
		return t.searchWithFilters(ctx, tool, args)
	case "get_current_info":
		return t.getCurrentInfo(ctx, tool, args)
	default:
		return "", fmt.Errorf("unknown web search action: %s", action)
	}
}

/* search performs a web search */
func (t *WebSearchTool) search(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("search requires query parameter")
	}

	maxResults := 5
	if mr, ok := args["max_results"].(float64); ok {
		maxResults = int(mr)
	}

	/* Check cache */
	t.cacheMu.RLock()
	cached, ok := t.cache[query]
	t.cacheMu.RUnlock()
	if ok && time.Since(cached.timestamp) < t.cacheExpiry {
		return t.formatResults(cached.results, query, maxResults)
	}

	/* Perform search using DuckDuckGo Instant Answer API */
	results, err := t.searchDuckDuckGo(ctx, query, maxResults)
	if err != nil {
		/* Fallback to HTML scraping if API fails */
		results, err = t.searchHTML(ctx, query, maxResults)
		if err != nil {
			return "", fmt.Errorf("web search failed: %w", err)
		}
	}

	/* Cache results */
	t.cacheMu.Lock()
	t.cache[query] = cacheEntry{
		results:   results,
		timestamp: time.Now(),
	}
	t.cacheMu.Unlock()

	return t.formatResults(results, query, maxResults)
}

/* searchDuckDuckGo searches using DuckDuckGo Instant Answer API */
func (t *WebSearchTool) searchDuckDuckGo(ctx context.Context, query string, maxResults int) ([]map[string]interface{}, error) {
	if query == "" {
		return nil, fmt.Errorf("query cannot be empty")
	}

	apiURL := fmt.Sprintf("https://api.duckduckgo.com/?q=%s&format=json&no_html=1&skip_disambig=1", url.QueryEscape(query))
	if err := validation.ValidateURLForSSRF(apiURL, nil); err != nil {
		return nil, fmt.Errorf("web search URL validation failed: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	/* Add user agent to avoid blocking */
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; NeuronAgent/1.0; +https://neurondb.ai)")

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	/* Check status code */
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	/* Limit response size */
	maxBodySize := 1024 * 1024 /* 1MB */
	body, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxBodySize)))
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var ddgResponse map[string]interface{}
	if err := json.Unmarshal(body, &ddgResponse); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response: %w", err)
	}

	results := make([]map[string]interface{}, 0)

	/* Extract abstract/answer (highest priority) */
	if abstract, ok := ddgResponse["Abstract"].(string); ok && abstract != "" {
		heading := ""
		if h, ok := ddgResponse["Heading"].(string); ok {
			heading = h
		}
		abstractURL := ""
		if u, ok := ddgResponse["AbstractURL"].(string); ok {
			abstractURL = u
		}

		results = append(results, map[string]interface{}{
			"title":     heading,
			"content":   abstract,
			"url":       abstractURL,
			"source":    "duckduckgo",
			"relevance": 0.9, /* High relevance for instant answers */
		})
	}

	/* Extract related topics */
	if relatedTopics, ok := ddgResponse["RelatedTopics"].([]interface{}); ok {
		for i, topic := range relatedTopics {
			if len(results) >= maxResults {
				break
			}
			if topicMap, ok := topic.(map[string]interface{}); ok {
				text := ""
				url := ""
				title := ""

				if t, ok := topicMap["Text"].(string); ok {
					text = t
				}
				if u, ok := topicMap["FirstURL"].(string); ok {
					url = u
					title = u /* Use URL as title if no title available */
				}
				if t, ok := topicMap["Text"].(string); ok && len(t) > 0 {
					/* Extract first sentence as title if available */
					sentences := strings.Split(text, ".")
					if len(sentences) > 0 && len(sentences[0]) < 100 {
						title = sentences[0]
					}
				}

				if text != "" {
					results = append(results, map[string]interface{}{
						"title":     title,
						"content":   text,
						"url":       url,
						"source":    "duckduckgo",
						"relevance": 0.7 - float64(i)*0.05, /* Decreasing relevance */
					})
				}
			}
		}
	}

	/* Extract answer if available (for direct answers) */
	if answer, ok := ddgResponse["Answer"].(string); ok && answer != "" {
		/* Prepend answer as highest priority result */
		answerResult := map[string]interface{}{
			"title":     "Direct Answer",
			"content":   answer,
			"url":       "",
			"source":    "duckduckgo",
			"relevance": 1.0,
		}
		results = append([]map[string]interface{}{answerResult}, results...)
	}

	return results, nil
}

/* searchHTML performs HTML-based search (fallback) */
func (t *WebSearchTool) searchHTML(ctx context.Context, query string, maxResults int) ([]map[string]interface{}, error) {
	/* Use DuckDuckGo HTML search as fallback */
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; NeuronAgent/1.0; +https://neurondb.ai)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024)) /* Limit to 1MB */
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	/* Simple HTML parsing (in production, use proper HTML parser like goquery) */
	results := make([]map[string]interface{}, 0)
	bodyStr := string(body)

	/* Extract result links using regex-like pattern matching */
	/* Look for result links in DuckDuckGo HTML format */
	lines := strings.Split(bodyStr, "\n")
	inResult := false
	currentResult := make(map[string]interface{})

	for i, line := range lines {
		if len(results) >= maxResults {
			break
		}

		lineLower := strings.ToLower(line)

		/* Detect result start */
		if strings.Contains(lineLower, "result__a") || strings.Contains(lineLower, "result-link") {
			inResult = true
			currentResult = make(map[string]interface{})
			currentResult["source"] = "duckduckgo_html"

			/* Try to extract URL */
			if urlStart := strings.Index(line, "href=\""); urlStart != -1 {
				urlStart += 6
				if urlEnd := strings.Index(line[urlStart:], "\""); urlEnd != -1 {
					url := line[urlStart : urlStart+urlEnd]
					if strings.HasPrefix(url, "http") {
						currentResult["url"] = url
					}
				}
			}
		}

		/* Extract title */
		if inResult && strings.Contains(lineLower, "<a") && strings.Contains(lineLower, ">") {
			if titleStart := strings.Index(line, ">"); titleStart != -1 {
				if titleEnd := strings.Index(line[titleStart+1:], "<"); titleEnd != -1 {
					title := strings.TrimSpace(line[titleStart+1 : titleStart+1+titleEnd])
					if title != "" && len(title) < 200 {
						currentResult["title"] = title
					}
				}
			}
		}

		/* Extract snippet/description */
		if inResult && strings.Contains(lineLower, "result__snippet") {
			if snippetStart := strings.Index(line, ">"); snippetStart != -1 {
				if snippetEnd := strings.Index(line[snippetStart+1:], "<"); snippetEnd != -1 {
					snippet := strings.TrimSpace(line[snippetStart+1 : snippetStart+1+snippetEnd])
					if snippet != "" {
						currentResult["content"] = snippet
					}
				}
			}
		}

		/* End of result */
		if inResult && (strings.Contains(lineLower, "</div>") || strings.Contains(lineLower, "</li>")) {
			if title, ok := currentResult["title"].(string); ok && title != "" {
				if content, ok := currentResult["content"].(string); !ok || content == "" {
					currentResult["content"] = title /* Use title as content if no snippet */
				}
				results = append(results, currentResult)
			}
			inResult = false
			currentResult = nil
		}

		/* Safety limit */
		if i > maxResults*50 {
			break
		}
	}

	return results, nil
}

/* searchWithFilters performs search with additional filters */
func (t *WebSearchTool) searchWithFilters(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("search_with_filters requires query parameter")
	}

	/* Add filters to query */
	if filters, ok := args["filters"].(map[string]interface{}); ok {
		if site, ok := filters["site"].(string); ok {
			query = fmt.Sprintf("site:%s %s", site, query)
		}
		if dateRange, ok := filters["date_range"].(string); ok {
			query = fmt.Sprintf("%s %s", query, dateRange)
		}
	}

	/* Use regular search with modified query */
	return t.search(ctx, tool, map[string]interface{}{
		"query":       query,
		"max_results": args["max_results"],
	})
}

/* getCurrentInfo gets current information about a topic */
func (t *WebSearchTool) getCurrentInfo(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("get_current_info requires query parameter")
	}

	/* Add "current" or "latest" to query */
	query = fmt.Sprintf("latest %s", query)

	return t.search(ctx, tool, map[string]interface{}{
		"query":       query,
		"max_results": args["max_results"],
	})
}

/* formatResults formats search results as JSON */
func (t *WebSearchTool) formatResults(results []map[string]interface{}, query string, maxResults int) (string, error) {
	if len(results) > maxResults {
		results = results[:maxResults]
	}

	result := map[string]interface{}{
		"action":       "search",
		"query":        query,
		"results":      results,
		"result_count": len(results),
		"status":       "success",
	}

	resultJSON, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal results: %w", err)
	}

	return string(resultJSON), nil
}

/* Search implements WebSearchInterface for RetrievalTool */
func (t *WebSearchTool) Search(ctx context.Context, query string, maxResults int) ([]map[string]interface{}, error) {
	result, err := t.search(ctx, &db.Tool{Name: "web_search"}, map[string]interface{}{
		"query":       query,
		"max_results": float64(maxResults),
	})
	if err != nil {
		return nil, err
	}

	var resultMap map[string]interface{}
	if err := json.Unmarshal([]byte(result), &resultMap); err != nil {
		return nil, err
	}

	if results, ok := resultMap["results"].([]interface{}); ok {
		converted := make([]map[string]interface{}, len(results))
		for i, r := range results {
			if m, ok := r.(map[string]interface{}); ok {
				converted[i] = m
			}
		}
		return converted, nil
	}

	return nil, fmt.Errorf("invalid result format")
}

/* Validate validates tool arguments */
func (t *WebSearchTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}

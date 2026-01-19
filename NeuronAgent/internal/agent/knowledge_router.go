/*-------------------------------------------------------------------------
 *
 * knowledge_router.go
 *    Knowledge source router for intelligent query routing
 *
 * Routes queries to appropriate knowledge sources (vector DB, web search, APIs)
 * based on query characteristics and content type.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/knowledge_router.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* KnowledgeRouter routes queries to appropriate knowledge sources */
type KnowledgeRouter struct {
	llm   *LLMClient
	embed *neurondb.EmbeddingClient
}

/* NewKnowledgeRouter creates a new knowledge router */
func NewKnowledgeRouter(llm *LLMClient, embed *neurondb.EmbeddingClient) *KnowledgeRouter {
	return &KnowledgeRouter{
		llm:   llm,
		embed: embed,
	}
}

/* RouteQuery determines the best knowledge source(s) for a query */
func (r *KnowledgeRouter) RouteQuery(ctx context.Context, query string) ([]string, map[string]float64, error) {
	/* Classify query type using LLM */
	queryType, err := r.classifyQueryType(ctx, query)
	if err != nil {
		/* Fallback to heuristic-based classification */
		queryType = r.classifyQueryTypeHeuristic(query)
	}

	/* Determine sources based on query type */
	sources, scores := r.selectSources(queryType, query)

	return sources, scores, nil
}

/* QueryType represents the type of query */
type QueryType string

const (
	QueryTypeSemantic      QueryType = "semantic"      /* Past conversations, stored knowledge */
	QueryTypeCurrentEvents QueryType = "current_events" /* Real-time information */
	QueryTypeStructured    QueryType = "structured"    /* Structured data, APIs */
	QueryTypeFactual       QueryType = "factual"       /* General facts */
	QueryTypeHybrid        QueryType = "hybrid"        /* Multiple sources needed */
)

/* classifyQueryType uses LLM to classify query type */
func (r *KnowledgeRouter) classifyQueryType(ctx context.Context, query string) (QueryType, error) {
	if r.llm == nil {
		/* Fallback to heuristic if LLM not available */
		return r.classifyQueryTypeHeuristic(query), nil
	}

	/* Enhanced prompt with examples */
	prompt := fmt.Sprintf(`Classify the following query into one of these categories based on what type of information is needed:

Categories:
1. semantic - Past conversations, stored knowledge, personal preferences, user-specific information
   Examples: "What did I tell you about my preferences?", "Remember that I like coffee"
   
2. current_events - Real-time information, news, current status, live data
   Examples: "What's the weather today?", "Latest news about AI", "Current stock price"
   
3. structured - Structured data, APIs, databases, specific data lookups
   Examples: "Query the database for users", "Get data from API", "SQL query"
   
4. factual - General facts, definitions, explanations, knowledge base
   Examples: "What is machine learning?", "Explain quantum computing", "How does photosynthesis work?"
   
5. hybrid - Requires multiple sources or combination of above
   Examples: "Compare current AI trends with what I've discussed before"

Query: "%s"

Respond with ONLY the category name (one word: semantic, current_events, structured, factual, or hybrid).`, query)

	llmConfig := map[string]interface{}{
		"temperature": 0.1,
		"max_tokens":  20,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback to heuristic on error */
		metrics.WarnWithContext(ctx, "LLM classification failed, using heuristic", map[string]interface{}{
			"query": query,
			"error": err.Error(),
		})
		return r.classifyQueryTypeHeuristic(query), nil
	}

	category := strings.TrimSpace(strings.ToLower(response.Content))
	/* Extract category from response (handle cases where LLM adds extra text) */
	category = strings.Fields(category)[0] /* Take first word */
	
	switch category {
	case "semantic":
		return QueryTypeSemantic, nil
	case "current_events", "currentevents", "current":
		return QueryTypeCurrentEvents, nil
	case "structured", "structure":
		return QueryTypeStructured, nil
	case "factual", "fact":
		return QueryTypeFactual, nil
	case "hybrid":
		return QueryTypeHybrid, nil
	default:
		/* Default fallback to heuristic */
		return r.classifyQueryTypeHeuristic(query), nil
	}
}

/* classifyQueryTypeHeuristic uses heuristics to classify query type */
func (r *KnowledgeRouter) classifyQueryTypeHeuristic(query string) QueryType {
	queryLower := strings.ToLower(query)

	/* Score each category based on keyword matches */
	scores := make(map[QueryType]int)

	/* Current events keywords (higher weight) */
	currentEventKeywords := []string{"today", "now", "current", "latest", "recent", "news", "update", "status", "live", "happening", "right now"}
	for _, keyword := range currentEventKeywords {
		if strings.Contains(queryLower, keyword) {
			scores[QueryTypeCurrentEvents] += 2
		}
	}

	/* Structured data keywords (higher weight for "api" and "database") */
	structuredKeywords := []string{"api", "database", "table", "query", "sql", "json", "xml", "endpoint", "rest", "graphql", "fetch", "retrieve data"}
	for _, keyword := range structuredKeywords {
		weight := 2
		if keyword == "api" || keyword == "database" {
			weight = 3 /* Higher weight for clear structured data indicators */
		}
		if strings.Contains(queryLower, keyword) {
			scores[QueryTypeStructured] += weight
		}
	}

	/* Semantic/personal keywords (highest weight for personal context) */
	/* Use word boundaries to avoid false matches (e.g., "is" matching "i") */
	semanticKeywords := []string{"remember", "prefer", "like", "previous", "before", "past", "conversation", "told you", "mentioned", " we discussed", " my ", " i ", " i'm", " i've"}
	for _, keyword := range semanticKeywords {
		/* Check for word boundary matches to avoid false positives */
		if strings.Contains(queryLower, keyword) {
			/* Additional check: "i" should only match as a word, not in "is", "it", etc. */
			if keyword == " i " || keyword == " i'm" || keyword == " i've" {
				/* Only match if it's actually "i" as a word */
				if !strings.Contains(queryLower, " i ") && !strings.HasPrefix(queryLower, "i ") {
					continue
				}
			}
			scores[QueryTypeSemantic] += 3
		}
	}

	/* Factual/question keywords (higher weight for clear factual questions) */
	factualKeywords := []string{"what is", "what are", "how does", "explain", "define", "tell me about", "describe", "why", "when did"}
	for _, keyword := range factualKeywords {
		weight := 2
		if keyword == "what is" || keyword == "what are" || keyword == "explain" {
			weight = 4 /* Higher weight for clear factual questions */
		}
		if strings.Contains(queryLower, keyword) {
			scores[QueryTypeFactual] += weight
		}
	}

	/* Find category with highest score */
	maxScore := 0
	var bestCategory QueryType = QueryTypeFactual /* Default */
	for category, score := range scores {
		if score > maxScore {
			maxScore = score
			bestCategory = category
		}
	}

	/* If multiple categories have similar scores, prefer hybrid */
	if maxScore > 0 {
		highScoreCount := 0
		for _, score := range scores {
			if score >= maxScore-1 {
				highScoreCount++
			}
		}
		if highScoreCount > 1 {
			return QueryTypeHybrid
		}
	}

	return bestCategory
}

/* selectSources selects knowledge sources based on query type */
func (r *KnowledgeRouter) selectSources(queryType QueryType, query string) ([]string, map[string]float64) {
	sources := make([]string, 0)
	scores := make(map[string]float64)

	switch queryType {
	case QueryTypeSemantic:
		sources = append(sources, "vector_db")
		scores["vector_db"] = 0.9
		scores["web"] = 0.3
		scores["api"] = 0.1

	case QueryTypeCurrentEvents:
		sources = append(sources, "web")
		scores["web"] = 0.9
		scores["vector_db"] = 0.2
		scores["api"] = 0.4

	case QueryTypeStructured:
		sources = append(sources, "api")
		scores["api"] = 0.9
		scores["vector_db"] = 0.3
		scores["web"] = 0.2

	case QueryTypeFactual:
		sources = append(sources, "vector_db", "web")
		scores["vector_db"] = 0.7
		scores["web"] = 0.7
		scores["api"] = 0.3

	case QueryTypeHybrid:
		sources = append(sources, "vector_db", "web", "api")
		scores["vector_db"] = 0.6
		scores["web"] = 0.6
		scores["api"] = 0.5

	default:
		/* Default: try vector DB first, then web */
		sources = append(sources, "vector_db", "web")
		scores["vector_db"] = 0.6
		scores["web"] = 0.5
		scores["api"] = 0.3
	}

	return sources, scores
}

/* RouteQueryWithFallback routes query with fallback logic */
func (r *KnowledgeRouter) RouteQueryWithFallback(ctx context.Context, query string, primarySource string) ([]string, map[string]float64, error) {
	sources, scores, err := r.RouteQuery(ctx, query)
	if err != nil {
		return nil, nil, err
	}

	/* If primary source is specified and not in recommended sources, add it */
	if primarySource != "" {
		found := false
		for _, src := range sources {
			if src == primarySource {
				found = true
				break
			}
		}
		if !found {
			sources = append([]string{primarySource}, sources...)
			if scores[primarySource] == 0 {
				scores[primarySource] = 0.8
			}
		}
	}

	return sources, scores, nil
}

/* RecordRoutingDecision records routing decision for metrics */
func (r *KnowledgeRouter) RecordRoutingDecision(query string, sources []string, scores map[string]float64) {
	metrics.InfoWithContext(context.Background(), "Knowledge routing decision", map[string]interface{}{
		"query":        query,
		"sources":      sources,
		"source_scores": scores,
	})
}

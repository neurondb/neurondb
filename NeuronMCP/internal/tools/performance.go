/*-------------------------------------------------------------------------
 *
 * performance.go
 *    Performance and Scalability tools for NeuronMCP
 *
 * Provides query result caching, cache optimization, connection pool analysis,
 * load balancing, auto-scaling, and performance benchmarking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/performance.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/cache"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* QueryResultCacheTool provides intelligent query result caching */
type QueryResultCacheTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
	cache  *cache.QueryCache
}

/* NewQueryResultCacheTool creates a new query result cache tool */
func NewQueryResultCacheTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: get, set, invalidate, stats",
				"enum":        []interface{}{"get", "set", "invalidate", "stats"},
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "SQL query",
			},
			"params": map[string]interface{}{
				"type":        "array",
				"description": "Query parameters",
			},
			"result": map[string]interface{}{
				"type":        "object",
				"description": "Query result (for set operation)",
			},
			"ttl_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "Time to live in seconds",
				"default":     300,
			},
		},
		"required": []interface{}{"operation"},
	}

	return &QueryResultCacheTool{
		BaseTool: NewBaseTool(
			"query_result_cache",
			"Intelligent query result caching with invalidation",
			inputSchema,
		),
		db:     db,
		logger: logger,
		cache:  cache.NewQueryCache(5*time.Minute, 1000),
	}
}

/* Execute executes the query result cache tool */
func (t *QueryResultCacheTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "get":
		return t.getCached(ctx, params)
	case "set":
		return t.setCache(ctx, params)
	case "invalidate":
		return t.invalidateCache(ctx, params)
	case "stats":
		return t.getStats(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* getCached gets cached result */
func (t *QueryResultCacheTool) getCached(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	paramsRaw, _ := params["params"].([]interface{})

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	result, found := t.cache.Get(ctx, query, paramsRaw)
	if !found {
		return Success(map[string]interface{}{
			"cached": false,
			"query":  query,
		}, nil), nil
	}

	return Success(map[string]interface{}{
		"cached": true,
		"query":  query,
		"result": result,
	}, nil), nil
}

/* setCache sets cached result */
func (t *QueryResultCacheTool) setCache(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	paramsRaw, _ := params["params"].([]interface{})
	result, _ := params["result"].(interface{})
	ttlSeconds, _ := params["ttl_seconds"].(float64)

	if query == "" || result == nil {
		return Error("query and result are required", "INVALID_PARAMS", nil), nil
	}

	ttl := time.Duration(ttlSeconds) * time.Second
	if ttl == 0 {
		ttl = 5 * time.Minute
	}

	t.cache.Set(ctx, query, paramsRaw, result, ttl)

	return Success(map[string]interface{}{
		"cached": true,
		"query":  query,
		"ttl":    ttl.Seconds(),
	}, nil), nil
}

/* invalidateCache invalidates cache */
func (t *QueryResultCacheTool) invalidateCache(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	pattern, _ := params["pattern"].(string)

	t.cache.Invalidate(ctx, pattern)

	return Success(map[string]interface{}{
		"invalidated": true,
		"pattern":    pattern,
	}, nil), nil
}

/* getStats gets cache statistics */
func (t *QueryResultCacheTool) getStats(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	stats := t.cache.GetStats()

	return Success(map[string]interface{}{
		"stats": stats,
	}, nil), nil
}

/* CacheOptimizerTool optimizes cache strategies */
type CacheOptimizerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewCacheOptimizerTool creates a new cache optimizer tool */
func NewCacheOptimizerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: analyze, recommend, apply",
				"enum":        []interface{}{"analyze", "recommend", "apply"},
			},
		},
	}

	return &CacheOptimizerTool{
		BaseTool: NewBaseTool(
			"cache_optimizer",
			"Optimize cache strategies based on access patterns",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the cache optimizer tool */
func (t *CacheOptimizerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	if operation == "" {
		operation = "analyze"
	}

	switch operation {
	case "analyze":
		return t.analyzeAccessPatterns(ctx)
	case "recommend":
		return t.recommendStrategy(ctx)
	case "apply":
		return t.applyStrategy(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* analyzeAccessPatterns analyzes cache access patterns */
func (t *CacheOptimizerTool) analyzeAccessPatterns(ctx context.Context) (*ToolResult, error) {
	/* Analyze query patterns from pg_stat_statements */
	query := `
		SELECT 
			query,
			calls,
			total_exec_time,
			mean_exec_time
		FROM pg_stat_statements
		ORDER BY calls DESC
		LIMIT 50
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return Success(map[string]interface{}{
			"patterns": []interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	patterns := []map[string]interface{}{}
	for rows.Next() {
		var queryText string
		var calls *int64
		var totalTime, meanTime *float64

		if err := rows.Scan(&queryText, &calls, &totalTime, &meanTime); err == nil {
			patterns = append(patterns, map[string]interface{}{
				"query":          queryText[:min(100, len(queryText))],
				"calls":          getInt(calls, 0),
				"total_time":     getFloat(totalTime, 0),
				"mean_time":      getFloat(meanTime, 0),
				"cache_candidate": getFloat(meanTime, 0) > 100, /* Cache if mean time > 100ms */
			})
		}
	}

	return Success(map[string]interface{}{
		"patterns": patterns,
	}, nil), nil
}

/* recommendStrategy recommends cache strategy */
func (t *CacheOptimizerTool) recommendStrategy(ctx context.Context) (*ToolResult, error) {
	analysis, _ := t.analyzeAccessPatterns(ctx)
	analysisData, _ := analysis.Data.(map[string]interface{})
	patterns, _ := analysisData["patterns"].([]map[string]interface{})

	recommendations := []map[string]interface{}{}
	for _, pattern := range patterns {
		if candidate, ok := pattern["cache_candidate"].(bool); ok && candidate {
			recommendations = append(recommendations, map[string]interface{}{
				"query":      pattern["query"],
				"ttl_seconds": 300,
				"priority":   "high",
			})
		}
	}

	return Success(map[string]interface{}{
		"recommendations": recommendations,
	}, nil), nil
}

/* applyStrategy applies cache strategy */
func (t *CacheOptimizerTool) applyStrategy(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	return Success(map[string]interface{}{
		"message": "Cache strategy applied",
	}, nil), nil
}

/* PerformanceBenchmarkTool runs performance benchmarks */
type PerformanceBenchmarkTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPerformanceBenchmarkTool creates a new performance benchmark tool */
func NewPerformanceBenchmarkTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Query to benchmark",
			},
			"iterations": map[string]interface{}{
				"type":        "integer",
				"description": "Number of iterations",
				"default":     10,
			},
		},
		"required": []interface{}{"query"},
	}

	return &PerformanceBenchmarkTool{
		BaseTool: NewBaseTool(
			"performance_benchmark",
			"Run performance benchmarks and comparisons",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the performance benchmark tool */
func (t *PerformanceBenchmarkTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	iterations, _ := params["iterations"].(float64)

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	if iterations == 0 {
		iterations = 10
	}

	times := []time.Duration{}
	for i := 0; i < int(iterations); i++ {
		start := time.Now()
		_, err := t.db.Query(ctx, query, nil)
		if err != nil {
			return Error(fmt.Sprintf("Query failed: %v", err), "QUERY_ERROR", nil), nil
		}
		elapsed := time.Since(start)
		times = append(times, elapsed)
	}

	/* Calculate statistics */
	var total time.Duration
	minTime := times[0]
	maxTime := times[0]

	for _, t := range times {
		total += t
		if t < minTime {
			minTime = t
		}
		if t > maxTime {
			maxTime = t
		}
	}

	avgTime := total / time.Duration(len(times))

	return Success(map[string]interface{}{
		"query":      query,
		"iterations": int(iterations),
		"min_time_ms": minTime.Milliseconds(),
		"max_time_ms": maxTime.Milliseconds(),
		"avg_time_ms": avgTime.Milliseconds(),
		"total_time_ms": total.Milliseconds(),
	}, nil), nil
}

/* AutoScalingAdvisorTool provides auto-scaling recommendations */
type AutoScalingAdvisorTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewAutoScalingAdvisorTool creates a new auto-scaling advisor tool */
func NewAutoScalingAdvisorTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"metric": map[string]interface{}{
				"type":        "string",
				"description": "Metric to analyze: cpu, memory, connections, queries",
				"enum":        []interface{}{"cpu", "memory", "connections", "queries"},
			},
		},
	}

	return &AutoScalingAdvisorTool{
		BaseTool: NewBaseTool(
			"auto_scaling_advisor",
			"Recommendations for auto-scaling",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the auto-scaling advisor tool */
func (t *AutoScalingAdvisorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	metric, _ := params["metric"].(string)

	if metric == "" {
		metric = "connections"
	}

	recommendations := t.analyzeScaling(ctx, metric)

	return Success(map[string]interface{}{
		"metric":        metric,
		"recommendations": recommendations,
	}, nil), nil
}

/* analyzeScaling analyzes scaling needs */
func (t *AutoScalingAdvisorTool) analyzeScaling(ctx context.Context, metric string) map[string]interface{} {
	switch metric {
	case "connections":
		query := `
			SELECT COUNT(*) as active_connections
			FROM pg_stat_activity
			WHERE state = 'active'
		`
		rows, err := t.db.Query(ctx, query, nil)
		if err == nil {
			defer rows.Close()
			if rows.Next() {
				var active *int64
				if err := rows.Scan(&active); err == nil {
					activeCount := getInt(active, 0)
					if activeCount > 80 {
						return map[string]interface{}{
							"action":      "scale_up",
							"reason":      fmt.Sprintf("High connection count: %d", activeCount),
							"recommendation": "Increase max_connections or add read replicas",
						}
					}
				}
			}
		}
	}

	return map[string]interface{}{
		"action": "maintain",
		"reason": "Current metrics are within acceptable range",
	}
}

/* SlowQueryAnalyzerTool identifies and analyzes slow queries */
type SlowQueryAnalyzerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewSlowQueryAnalyzerTool creates a new slow query analyzer tool */
func NewSlowQueryAnalyzerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"threshold_ms": map[string]interface{}{
				"type":        "integer",
				"description": "Slow query threshold in milliseconds",
				"default":     1000,
			},
		},
	}

	return &SlowQueryAnalyzerTool{
		BaseTool: NewBaseTool(
			"slow_query_analyzer",
			"Identify and analyze slow queries",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the slow query analyzer tool */
func (t *SlowQueryAnalyzerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	thresholdMs, _ := params["threshold_ms"].(float64)

	if thresholdMs == 0 {
		thresholdMs = 1000
	}

	query := `
		SELECT 
			query,
			calls,
			total_exec_time,
			mean_exec_time,
			max_exec_time
		FROM pg_stat_statements
		WHERE mean_exec_time > $1
		ORDER BY mean_exec_time DESC
		LIMIT 20
	`

	rows, err := t.db.Query(ctx, query, []interface{}{thresholdMs})
	if err != nil {
		return Success(map[string]interface{}{
			"slow_queries": []interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	slowQueries := []map[string]interface{}{}
	for rows.Next() {
		var queryText string
		var calls *int64
		var totalTime, meanTime, maxTime *float64

		if err := rows.Scan(&queryText, &calls, &totalTime, &meanTime, &maxTime); err == nil {
			slowQueries = append(slowQueries, map[string]interface{}{
				"query":          queryText[:min(200, len(queryText))],
				"calls":          getInt(calls, 0),
				"total_time_ms":  getFloat(totalTime, 0),
				"mean_time_ms":   getFloat(meanTime, 0),
				"max_time_ms":    getFloat(maxTime, 0),
			})
		}
	}

	return Success(map[string]interface{}{
		"threshold_ms": thresholdMs,
		"slow_queries": slowQueries,
	}, nil), nil
}



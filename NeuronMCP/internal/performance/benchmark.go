/*-------------------------------------------------------------------------
 *
 * benchmark.go
 *    Performance benchmarking utilities for NeuronMCP
 *
 * Provides benchmarking infrastructure for measuring tool call latency,
 * throughput, memory usage, and concurrent request handling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/performance/benchmark.go
 *
 *-------------------------------------------------------------------------
 */

package performance

import (
	"context"
	"fmt"
	"sync"
	"time"
)

/* BenchmarkResult represents a single benchmark result */
type BenchmarkResult struct {
	ToolName      string            `json:"tool_name"`
	Operation     string            `json:"operation"`
	Duration      time.Duration     `json:"duration"`
	Success       bool              `json:"success"`
	Error         string            `json:"error,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

/* BenchmarkStats represents aggregated benchmark statistics */
type BenchmarkStats struct {
	ToolName      string            `json:"tool_name"`
	TotalRuns     int               `json:"total_runs"`
	SuccessCount  int               `json:"success_count"`
	FailureCount  int               `json:"failure_count"`
	MinDuration   time.Duration     `json:"min_duration"`
	MaxDuration   time.Duration     `json:"max_duration"`
	MeanDuration  time.Duration     `json:"mean_duration"`
	P50Duration   time.Duration     `json:"p50_duration"`
	P95Duration   time.Duration     `json:"p95_duration"`
	P99Duration   time.Duration     `json:"p99_duration"`
	Throughput    float64           `json:"throughput"` // requests per second
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

/* BenchmarkRunner runs performance benchmarks */
type BenchmarkRunner struct {
	results []BenchmarkResult
	mu      sync.RWMutex
}

/* NewBenchmarkRunner creates a new benchmark runner */
func NewBenchmarkRunner() *BenchmarkRunner {
	return &BenchmarkRunner{
		results: make([]BenchmarkResult, 0),
	}
}

/* RunBenchmark runs a single benchmark */
func (r *BenchmarkRunner) RunBenchmark(ctx context.Context, toolName, operation string, fn func() error) BenchmarkResult {
	start := time.Now()
	err := fn()
	duration := time.Since(start)

	result := BenchmarkResult{
		ToolName:  toolName,
		Operation: operation,
		Duration:  duration,
		Success:   err == nil,
		Metadata:  make(map[string]interface{}),
	}

	if err != nil {
		result.Error = err.Error()
	}

	r.mu.Lock()
	r.results = append(r.results, result)
	r.mu.Unlock()

	return result
}

/* RunConcurrentBenchmarks runs multiple benchmarks concurrently */
func (r *BenchmarkRunner) RunConcurrentBenchmarks(ctx context.Context, toolName, operation string, concurrency int, fn func() error) []BenchmarkResult {
	var wg sync.WaitGroup
	results := make([]BenchmarkResult, concurrency)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = r.RunBenchmark(ctx, toolName, operation, fn)
		}(i)
	}

	wg.Wait()
	return results
}

/* CalculateStats calculates statistics from benchmark results */
func (r *BenchmarkRunner) CalculateStats(toolName string) BenchmarkStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	/* Filter results for this tool */
	toolResults := make([]BenchmarkResult, 0)
	for _, result := range r.results {
		if result.ToolName == toolName {
			toolResults = append(toolResults, result)
		}
	}

	if len(toolResults) == 0 {
		return BenchmarkStats{
			ToolName: toolName,
		}
	}

	/* Calculate durations */
	durations := make([]time.Duration, 0, len(toolResults))
	successCount := 0
	var totalDuration time.Duration

	for _, result := range toolResults {
		if result.Success {
			durations = append(durations, result.Duration)
			totalDuration += result.Duration
			successCount++
		}
	}

	if len(durations) == 0 {
		return BenchmarkStats{
			ToolName:     toolName,
			TotalRuns:    len(toolResults),
			FailureCount: len(toolResults),
		}
	}

	/* Sort durations for percentile calculation */
	sortDurations(durations)

	stats := BenchmarkStats{
		ToolName:     toolName,
		TotalRuns:    len(toolResults),
		SuccessCount: successCount,
		FailureCount: len(toolResults) - successCount,
		MinDuration:  durations[0],
		MaxDuration:  durations[len(durations)-1],
		MeanDuration: totalDuration / time.Duration(len(durations)),
		P50Duration:  percentile(durations, 0.50),
		P95Duration:  percentile(durations, 0.95),
		P99Duration:  percentile(durations, 0.99),
		Metadata:     make(map[string]interface{}),
	}

	/* Calculate throughput */
	if stats.MeanDuration > 0 {
		stats.Throughput = float64(time.Second) / float64(stats.MeanDuration)
	}

	return stats
}

/* GetResults returns all benchmark results */
func (r *BenchmarkRunner) GetResults() []BenchmarkResult {
	r.mu.RLock()
	defer r.mu.RUnlock()

	results := make([]BenchmarkResult, len(r.results))
	copy(results, r.results)
	return results
}

/* ClearResults clears all benchmark results */
func (r *BenchmarkRunner) ClearResults() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.results = r.results[:0]
}

/* Helper functions */

func sortDurations(durations []time.Duration) {
	/* Simple bubble sort for small arrays */
	n := len(durations)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if durations[j] > durations[j+1] {
				durations[j], durations[j+1] = durations[j+1], durations[j]
			}
		}
	}
}

func percentile(durations []time.Duration, p float64) time.Duration {
	if len(durations) == 0 {
		return 0
	}
	idx := int(float64(len(durations)) * p)
	if idx >= len(durations) {
		idx = len(durations) - 1
	}
	return durations[idx]
}

/* BenchmarkReport generates a formatted benchmark report */
type BenchmarkReport struct {
	Timestamp   time.Time                  `json:"timestamp"`
	Stats       []BenchmarkStats           `json:"stats"`
	Summary     map[string]interface{}     `json:"summary"`
}

/* GenerateReport generates a benchmark report */
func (r *BenchmarkRunner) GenerateReport() BenchmarkReport {
	r.mu.RLock()
	defer r.mu.RUnlock()

	/* Get unique tool names */
	toolNames := make(map[string]bool)
	for _, result := range r.results {
		toolNames[result.ToolName] = true
	}

	/* Calculate stats for each tool */
	stats := make([]BenchmarkStats, 0, len(toolNames))
	for toolName := range toolNames {
		stats = append(stats, r.CalculateStats(toolName))
	}

	/* Generate summary */
	summary := map[string]interface{}{
		"total_tools":     len(toolNames),
		"total_runs":      len(r.results),
		"successful_runs": 0,
		"failed_runs":     0,
	}

	successCount := 0
	failCount := 0
	for _, result := range r.results {
		if result.Success {
			successCount++
		} else {
			failCount++
		}
	}
	summary["successful_runs"] = successCount
	summary["failed_runs"] = failCount

	return BenchmarkReport{
		Timestamp: time.Now(),
		Stats:     stats,
		Summary:   summary,
	}
}

/* FormatDuration formats duration for display */
func FormatDuration(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%.2fns", float64(d))
	} else if d < time.Millisecond {
		return fmt.Sprintf("%.2fµs", float64(d)/float64(time.Microsecond))
	} else if d < time.Second {
		return fmt.Sprintf("%.2fms", float64(d)/float64(time.Millisecond))
	}
	return fmt.Sprintf("%.2fs", float64(d)/float64(time.Second))
}





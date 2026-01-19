/*-------------------------------------------------------------------------
 *
 * framework.go
 *    Testing framework for NeuronMCP tools
 *
 * Provides utilities for testing tools in isolation with fixtures and mocks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/test/framework.go
 *
 *-------------------------------------------------------------------------
 */

package test

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

/* TestFramework provides testing utilities */
type TestFramework struct {
	db       *database.Database
	registry *tools.ToolRegistry
	logger   *logging.Logger
	fixtures *Fixtures
}

/* NewTestFramework creates a new test framework */
func NewTestFramework(db *database.Database, logger *logging.Logger) *TestFramework {
	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	return &TestFramework{
		db:       db,
		registry: registry,
		logger:   logger,
		fixtures: NewFixtures(db),
	}
}

/* TestTool tests a tool with given parameters */
func (tf *TestFramework) TestTool(ctx context.Context, toolName string, params map[string]interface{}) (*TestResult, error) {
	tool := tf.registry.GetTool(toolName)
	if tool == nil {
		return nil, fmt.Errorf("tool not found: %s", toolName)
	}

	startTime := time.Now()
	result, err := tool.Execute(ctx, params)
	duration := time.Since(startTime)

	return &TestResult{
		ToolName:    toolName,
		Params:      params,
		Result:      result,
		Error:       err,
		Duration:    duration,
		Success:     err == nil && (result == nil || result.Success),
	}, nil
}

/* TestResult represents a test result */
type TestResult struct {
	ToolName string
	Params   map[string]interface{}
	Result   *tools.ToolResult
	Error    error
	Duration time.Duration
	Success  bool
}

/* BenchmarkTool benchmarks a tool execution */
func (tf *TestFramework) BenchmarkTool(ctx context.Context, toolName string, params map[string]interface{}, iterations int) (*BenchmarkResult, error) {
	if iterations <= 0 {
		iterations = 1
	}

	durations := make([]time.Duration, 0, iterations)
	errors := 0

	for i := 0; i < iterations; i++ {
		startTime := time.Now()
		tool := tf.registry.GetTool(toolName)
		if tool == nil {
			return nil, fmt.Errorf("tool not found: %s", toolName)
		}

		_, err := tool.Execute(ctx, params)
		duration := time.Since(startTime)
		durations = append(durations, duration)

		if err != nil {
			errors++
		}
	}

	/* Calculate statistics */
	var totalDuration time.Duration
	minDuration := durations[0]
	maxDuration := durations[0]

	for _, d := range durations {
		totalDuration += d
		if d < minDuration {
			minDuration = d
		}
		if d > maxDuration {
			maxDuration = d
		}
	}

	avgDuration := totalDuration / time.Duration(iterations)

	return &BenchmarkResult{
		ToolName:     toolName,
		Iterations:   iterations,
		AvgDuration:  avgDuration,
		MinDuration:  minDuration,
		MaxDuration:  maxDuration,
		TotalDuration: totalDuration,
		Errors:       errors,
		SuccessRate:  float64(iterations-errors) / float64(iterations),
	}, nil
}

/* BenchmarkResult represents benchmark results */
type BenchmarkResult struct {
	ToolName      string
	Iterations    int
	AvgDuration   time.Duration
	MinDuration   time.Duration
	MaxDuration   time.Duration
	TotalDuration time.Duration
	Errors        int
	SuccessRate   float64
}

/* Fixtures provides test fixtures */
type Fixtures struct {
	db *database.Database
}

/* NewFixtures creates a new fixtures manager */
func NewFixtures(db *database.Database) *Fixtures {
	return &Fixtures{db: db}
}

/* SetupTestDatabase sets up a test database schema */
func (f *Fixtures) SetupTestDatabase(ctx context.Context) error {
	/* Create test schema */
	query := `
		CREATE SCHEMA IF NOT EXISTS test;
		CREATE EXTENSION IF NOT EXISTS neurondb;
	`
	_, err := f.db.Exec(ctx, query)
	return err
}

/* CleanupTestDatabase cleans up test database */
func (f *Fixtures) CleanupTestDatabase(ctx context.Context) error {
	query := `DROP SCHEMA IF EXISTS test CASCADE;`
	_, err := f.db.Exec(ctx, query)
	return err
}

/* CreateTestTable creates a test table */
func (f *Fixtures) CreateTestTable(ctx context.Context, tableName string, schema string) error {
	query := fmt.Sprintf("CREATE TABLE IF NOT EXISTS test.%s %s", tableName, schema)
	_, err := f.db.Exec(ctx, query)
	return err
}

/* InsertTestData inserts test data */
func (f *Fixtures) InsertTestData(ctx context.Context, tableName string, data []map[string]interface{}) error {
	if len(data) == 0 {
		return nil
	}

	/* Build INSERT query with proper escaping */
	for _, row := range data {
		if len(row) == 0 {
			continue
		}

		columns := make([]string, 0, len(row))
		values := make([]interface{}, 0, len(row))
		placeholders := make([]string, 0, len(row))

		idx := 1
		for col, val := range row {
			columns = append(columns, database.EscapeIdentifier(col))
			values = append(values, val)
			placeholders = append(placeholders, fmt.Sprintf("$%d", idx))
			idx++
		}

		query := fmt.Sprintf(
			"INSERT INTO test.%s (%s) VALUES (%s)",
			database.EscapeIdentifier(tableName),
			strings.Join(columns, ", "),
			strings.Join(placeholders, ", "),
		)

		_, err := f.db.Exec(ctx, query, values...)
		if err != nil {
			return fmt.Errorf("failed to insert test data: %w", err)
		}
	}

	return nil
}

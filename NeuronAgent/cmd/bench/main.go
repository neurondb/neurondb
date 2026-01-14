/*-------------------------------------------------------------------------
 *
 * main.go
 *    Bench CLI for NeuronAgent evaluation
 *
 * Runs 100 tasks from eval dataset and outputs single score plus diffs.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cmd/bench/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/config"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/eval"
	"github.com/neurondb/NeuronAgent/internal/tools"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <agent_id> <dataset_version> [limit]\n", os.Args[0])
		os.Exit(1)
	}

	agentIDStr := os.Args[1]
	datasetVersion := os.Args[2]
	limit := 100
	if len(os.Args) > 3 {
		fmt.Sscanf(os.Args[3], "%d", &limit)
	}

	agentID, err := uuid.Parse(agentIDStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid agent ID: %v\n", err)
		os.Exit(1)
	}

	/* Load configuration */
	cfg := config.DefaultConfig()
	if configPath := os.Getenv("CONFIG_PATH"); configPath != "" {
		var err error
		cfg, err = config.LoadConfig(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to load configuration from '%s': %v. Using default configuration.\n", configPath, err)
		}
	} else {
		config.LoadFromEnv(cfg)
	}

	/* Connect to database */
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		cfg.Database.Host, cfg.Database.Port, cfg.Database.User, cfg.Database.Password, cfg.Database.Database)

	database, err := db.NewDB(connStr, db.PoolConfig{
		MaxOpenConns:    cfg.Database.MaxOpenConns,
		MaxIdleConns:    cfg.Database.MaxIdleConns,
		ConnMaxLifetime: cfg.Database.ConnMaxLifetime,
		ConnMaxIdleTime: 10 * time.Minute,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to connect to database at %s:%d: %v\n", cfg.Database.Host, cfg.Database.Port, err)
		os.Exit(1)
	}
	defer database.Close()

	/* Initialize components */
	queries := db.NewQueries(database.DB)
	neurondbClient := neurondb.NewClient(database.DB)
	embedClient := neurondbClient.Embedding
	toolRegistry := tools.NewRegistryWithNeuronDB(queries, database, neurondbClient)
	runtime := agent.NewRuntime(database, queries, toolRegistry, embedClient)
	evaluator := eval.NewEvaluator(queries, runtime)

	/* Create eval run */
	evalRun := &db.EvalRun{
		DatasetVersion: datasetVersion,
		AgentID:        &agentID,
		TotalTasks:     limit,
	}
	if err := queries.CreateEvalRun(context.Background(), evalRun); err != nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to create evaluation run in database: %v\n", err)
		os.Exit(1)
	}

	/* Load eval tasks */
	tasks, err := queries.ListEvalTasks(context.Background(), nil, limit, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to load evaluation tasks from database: %v\n", err)
		os.Exit(1)
	}

	if len(tasks) == 0 {
		fmt.Fprintf(os.Stderr, "Error: No evaluation tasks found in database for dataset version '%s'\n", datasetVersion)
		os.Exit(1)
	}

	fmt.Printf("Starting evaluation: %d task(s) to process\n", len(tasks))

	/* Run evaluation */
	passed := 0
	failed := 0
	totalScore := 0.0
	var diffs []map[string]interface{}

	for _, task := range tasks {
		result, err := evaluator.EvaluateTask(context.Background(), &task, agentID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: Failed to evaluate task %s: %v\n", task.ID.String(), err)
			failed++
			continue
		}

		result.EvalRunID = evalRun.ID
		if err := queries.CreateEvalTaskResult(context.Background(), result); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to save evaluation result for task %s: %v\n", task.ID.String(), err)
		}

		if result.Passed {
			passed++
		} else {
			failed++
		}

		if result.Score != nil {
			totalScore += *result.Score
		}

		/* Collect diff if failed */
		if !result.Passed {
			diff := map[string]interface{}{
				"task_id":  task.ID.String(),
				"input":    task.Input,
				"expected": task.ExpectedOutput,
				"actual":   result.ActualOutput,
				"error":    result.ErrorMessage,
			}
			diffs = append(diffs, diff)
		}
	}

	/* Calculate final score */
	finalScore := totalScore / float64(len(tasks))
	evalRun.Score = &finalScore
	evalRun.PassedTasks = passed
	evalRun.FailedTasks = failed
	now := time.Now()
	evalRun.CompletedAt = &now

	if err := queries.UpdateEvalRun(context.Background(), evalRun); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Failed to update evaluation run with final results: %v\n", err)
	}

	/* Output results */
	output := map[string]interface{}{
		"score":           finalScore,
		"total_tasks":     len(tasks),
		"passed":          passed,
		"failed":          failed,
		"dataset_version": datasetVersion,
		"agent_id":        agentID.String(),
		"diffs":           diffs,
	}

	outputJSON, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to serialize evaluation results to JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(outputJSON))
}

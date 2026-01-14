/*-------------------------------------------------------------------------
 *
 * main.go
 *    Main entry point for NeuronAgent server
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cmd/agent-server/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/api"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/browser"
	"github.com/neurondb/NeuronAgent/internal/collaboration"
	"github.com/neurondb/NeuronAgent/internal/config"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/jobs"
	"github.com/neurondb/NeuronAgent/internal/eval"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/multimodal"
	"github.com/neurondb/NeuronAgent/internal/notifications"
	"github.com/neurondb/NeuronAgent/internal/replay"
	"github.com/neurondb/NeuronAgent/internal/session"
	"github.com/neurondb/NeuronAgent/internal/tools"
	"github.com/neurondb/NeuronAgent/internal/workflow"
	"github.com/neurondb/NeuronAgent/internal/worker"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

var (
	version   = "dev"
	buildDate = "unknown"
	gitCommit = "unknown"
)

func main() {
	var (
		showVersion      = flag.Bool("version", false, "Show version information")
		showVersionShort = flag.Bool("v", false, "Show version information (short)")
		configPath       = flag.String("c", "", "Path to configuration file")
		configPathLong   = flag.String("config", "", "Path to configuration file")
		showHelp         = flag.Bool("help", false, "Show help message")
		showHelpShort    = flag.Bool("h", false, "Show help message (short)")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "NeuronAgent Server - AI Agent server for NeuronDB\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s                    Start server with default configuration\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -c config.yaml     Start server with custom config file\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --version          Show version information\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --help             Show this help message\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nConfiguration:\n")
		fmt.Fprintf(os.Stderr, "  Configuration can be provided via:\n")
		fmt.Fprintf(os.Stderr, "  - Command line flag: -c or --config\n")
		fmt.Fprintf(os.Stderr, "  - Environment variable: CONFIG_PATH\n")
		fmt.Fprintf(os.Stderr, "  - Environment variables (see config package for details)\n")
	}
	flag.Parse()

	/* Handle version flag */
	if *showVersion || *showVersionShort {
		fmt.Printf("neuronagent version %s\n", version)
		fmt.Printf("Build date: %s\n", buildDate)
		fmt.Printf("Git commit: %s\n", gitCommit)
		os.Exit(0)
	}

	/* Handle help flag */
	if *showHelp || *showHelpShort {
		flag.Usage()
		os.Exit(0)
	}

	/* Load configuration */
	cfg := config.DefaultConfig()

	/* Determine config path - command line flag takes precedence over environment variable */
	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = *configPathLong
	}
	if cfgPath == "" {
		cfgPath = os.Getenv("CONFIG_PATH")
	}

	if cfgPath != "" {
		var err error
		cfg, err = config.LoadConfig(cfgPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to load configuration from file '%s': %v. Using default configuration.\n", cfgPath, err)
		}
	} else {
		/* Load from environment variables if no config file */
		config.LoadFromEnv(cfg)
	}

	/* Initialize logging */
	metrics.InitLogging(cfg.Logging.Level, cfg.Logging.Format)

	/* Connect to database */
	/* Add search_path to connection string to ensure it's set for all connections */
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable search_path=neurondb_agent,public",
		cfg.Database.Host, cfg.Database.Port, cfg.Database.User, cfg.Database.Password, cfg.Database.Database)

	connMaxIdleTime := 10 * time.Minute
	if cfg.Database.ConnMaxIdleTime > 0 {
		connMaxIdleTime = cfg.Database.ConnMaxIdleTime
	}

	database, err := db.NewDB(connStr, db.PoolConfig{
		MaxOpenConns:    cfg.Database.MaxOpenConns,
		MaxIdleConns:    cfg.Database.MaxIdleConns,
		ConnMaxLifetime: cfg.Database.ConnMaxLifetime,
		ConnMaxIdleTime: connMaxIdleTime,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: Failed to connect to database: %v\n", err)
		fmt.Fprintf(os.Stderr, "Connection string: host=%s port=%d user=%s dbname=%s\n",
			cfg.Database.Host, cfg.Database.Port, cfg.Database.User, cfg.Database.Database)
		os.Exit(1)
	}
	defer database.Close()

	/* Run migrations */
	migrationRunner, err := db.NewMigrationRunner(database.DB, "./sql")
	if err == nil {
		if err := migrationRunner.Run(context.Background()); err != nil {
			metrics.WarnWithContext(context.Background(), "Database migration failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}

	/* Initialize components */
	queries := db.NewQueries(database.DB)
	queries.SetConnInfoFunc(database.GetConnInfoString)

	/* Initialize NeuronDB client */
	neurondbClient := neurondb.NewClient(database.DB)
	embedClient := neurondbClient.Embedding

	/* Initialize advanced features */
	/* VFS uses database storage directly, so we pass nil for storage backend */
	vfs := agent.NewVirtualFileSystem(queries, nil, 100*1024*1024) /* 100MB max file size */
	pubsub := collaboration.NewPubSub()
	workspaceManager := collaboration.NewWorkspaceManager(queries, pubsub)

	/* Initialize notification services */
	emailService := notifications.NewEmailService("", 0, "", "", "") /* Configure via config */
	webhookService := notifications.NewWebhookService(30 * time.Second)

	/* Initialize basic runtime first to get hierarchical memory */
	toolRegistry := tools.NewRegistryWithNeuronDB(queries, database, neurondbClient)
	baseRuntime := agent.NewRuntime(database, queries, toolRegistry, embedClient)

	/* Get hierarchical memory from base runtime */
	hierMemory := baseRuntime.HierMemory()

	/* Re-initialize tool registry with all features */
	toolRegistry = tools.NewRegistryWithAllFeatures(queries, database, vfs, hierMemory, workspaceManager)

	/* Get browser driver for cleanup worker */
	var browserDriver *browser.Driver
	if browserTool := toolRegistry.GetBrowserTool(); browserTool != nil {
		browserDriver = browserTool.GetDriver()
	}

	/* Initialize runtime with all features */
	runtime := agent.NewRuntimeWithFeatures(database, queries, toolRegistry, embedClient, vfs, workspaceManager)

	/* Initialize async task executor and notifier */
	taskNotifier := agent.NewTaskNotifier(queries, emailService, webhookService)
	asyncExecutor := agent.NewAsyncTaskExecutor(queries, runtime, taskNotifier)
	runtime.SetAsyncExecutor(asyncExecutor)

	/* Initialize sub-agent manager */
	subAgentManager := agent.NewSubAgentManager(queries, runtime)
	runtime.SetSubAgentManager(subAgentManager)
	runtime.SetAlertManager(taskNotifier)

	/* Initialize enhanced multimodal processor with database connection */
	multimodalProcessor := multimodal.NewEnhancedMultimodalProcessorWithDB(database.DB)
	runtime.SetMultimodalProcessor(multimodalProcessor)

	/* Initialize session management */
	sessionCache := session.NewCache(5 * time.Minute)
	_ = session.NewManager(queries, sessionCache) /* Session manager for future use */
	sessionCleanup := session.NewCleanupService(queries, 1*time.Hour, 24*time.Hour)
	sessionCleanup.Start()
	defer sessionCleanup.Stop()

	/* Initialize browser session cleanup */
	browserCleanup := browser.NewCleanupWorker(database, browserDriver, 1*time.Hour, 24*time.Hour)
	browserCleanup.Start()
	defer browserCleanup.Stop()

	/* Initialize workflow engine */
	workflowEngine := workflow.NewEngine(queries)
	workflowEngine.SetRuntime(runtime)
	workflowEngine.SetToolRegistry(toolRegistry)
	workflowEngine.SetEmailService(emailService)
	workflowEngine.SetWebhookService(webhookService)
	workflowEngine.SetBaseURL(fmt.Sprintf("http://%s:%d", cfg.Server.Host, cfg.Server.Port))

	/* Initialize evaluation and replay services */
	evaluator := eval.NewEvaluator(queries, runtime)
	replayManager := replay.NewReplayManager(queries, runtime)

	/* Initialize API */
	handlers := api.NewHandlers(queries, runtime)
	collabHandlers := api.NewCollaborationHandlers(queries, workspaceManager)
	asyncTasksHandlers := api.NewAsyncTasksHandlers(queries, asyncExecutor)
	alertPrefsHandlers := api.NewAlertPreferencesHandlers(queries)
	workflowHandlers := api.NewWorkflowHandlers(queries, workflowEngine)
	eventStreamHandlers := api.NewEventStreamHandlers(queries, runtime)
	verificationHandlers := api.NewVerificationHandlers(queries, runtime)
	vfsHandlers := api.NewVFSHandlers(queries, runtime)
	evaluationHandlers := api.NewEvaluationHandlers(queries, evaluator)
	replayHandlers := api.NewReplayHandlers(queries, replayManager, runtime)
	specializationHandlers := api.NewSpecializationHandlers(queries)
	keyManager := auth.NewAPIKeyManager(queries)
	principalManager := auth.NewPrincipalManager(queries)
	rateLimiter := auth.NewRateLimiter()

	/* Setup router */
	router := mux.NewRouter()
	router.Use(api.RequestIDMiddleware)
	router.Use(api.SecurityHeadersMiddleware) /* Security headers must be set early */
	router.Use(api.CORSMiddleware)
	router.Use(api.LoggingMiddleware)
	router.Use(api.AuthMiddleware(keyManager, principalManager, rateLimiter))

	/* API routes */
	apiRouter := router.PathPrefix("/api/v1").Subrouter()
	apiRouter.HandleFunc("/agents", handlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/agents", handlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", handlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", handlers.UpdateAgent).Methods("PUT")
	apiRouter.HandleFunc("/agents/{id}", handlers.DeleteAgent).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{id}/clone", handlers.CloneAgent).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/plan", handlers.GeneratePlan).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/reflect", handlers.ReflectOnResponse).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/delegate", handlers.DelegateToAgent).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/metrics", handlers.GetAgentMetrics).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/costs", handlers.GetAgentCosts).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/versions", handlers.ListAgentVersions).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/versions", handlers.CreateAgentVersion).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/versions/{version}", handlers.GetAgentVersion).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/versions/{version}/activate", handlers.ActivateAgentVersion).Methods("PUT")
	apiRouter.HandleFunc("/agents/{id}/relationships", handlers.ListAgentRelationships).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/relationships", handlers.CreateAgentRelationship).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/relationships/{relationship_id}", handlers.DeleteAgentRelationship).Methods("DELETE")
	apiRouter.HandleFunc("/plans", handlers.ListPlans).Methods("GET")
	apiRouter.HandleFunc("/plans/{id}", handlers.GetPlan).Methods("GET")
	apiRouter.HandleFunc("/plans/{id}", handlers.UpdatePlanStatus).Methods("PUT")
	apiRouter.HandleFunc("/reflections", handlers.ListReflections).Methods("GET")
	apiRouter.HandleFunc("/reflections/{id}", handlers.GetReflection).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/memory", handlers.ListMemoryChunks).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/memory/search", handlers.SearchMemory).Methods("POST")
	apiRouter.HandleFunc("/memory/{chunk_id}", handlers.GetMemoryChunk).Methods("GET")
	apiRouter.HandleFunc("/memory/{chunk_id}", handlers.DeleteMemoryChunk).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{id}/budget", handlers.GetBudget).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/budget", handlers.SetBudget).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/budget", handlers.UpdateBudget).Methods("PUT")
	apiRouter.HandleFunc("/agents/batch", handlers.BatchCreateAgents).Methods("POST")
	apiRouter.HandleFunc("/agents/batch/delete", handlers.BatchDeleteAgents).Methods("POST")
	apiRouter.HandleFunc("/messages/batch/delete", handlers.BatchDeleteMessages).Methods("POST")
	apiRouter.HandleFunc("/tools/batch/delete", handlers.BatchDeleteTools).Methods("POST")
	apiRouter.HandleFunc("/webhooks", handlers.ListWebhooks).Methods("GET")
	apiRouter.HandleFunc("/webhooks", handlers.CreateWebhook).Methods("POST")
	apiRouter.HandleFunc("/webhooks/{id}", handlers.GetWebhook).Methods("GET")
	apiRouter.HandleFunc("/webhooks/{id}", handlers.UpdateWebhook).Methods("PUT")
	apiRouter.HandleFunc("/webhooks/{id}", handlers.DeleteWebhook).Methods("DELETE")
	apiRouter.HandleFunc("/webhooks/{id}/deliveries", handlers.ListWebhookDeliveries).Methods("GET")
	apiRouter.HandleFunc("/approval-requests", handlers.ListApprovalRequests).Methods("GET")
	apiRouter.HandleFunc("/approval-requests/{id}", handlers.GetApprovalRequest).Methods("GET")
	apiRouter.HandleFunc("/approval-requests/{id}/approve", handlers.ApproveRequest).Methods("POST")
	apiRouter.HandleFunc("/approval-requests/{id}/reject", handlers.RejectRequest).Methods("POST")
	apiRouter.HandleFunc("/feedback", handlers.SubmitFeedback).Methods("POST")
	apiRouter.HandleFunc("/feedback", handlers.ListFeedback).Methods("GET")
	apiRouter.HandleFunc("/feedback/stats", handlers.GetFeedbackStats).Methods("GET")
	apiRouter.HandleFunc("/sessions", handlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/sessions/{id}", handlers.GetSession).Methods("GET")
	apiRouter.HandleFunc("/sessions/{id}", handlers.UpdateSession).Methods("PUT")
	apiRouter.HandleFunc("/sessions/{id}", handlers.DeleteSession).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{agent_id}/sessions", handlers.ListSessions).Methods("GET")
	apiRouter.HandleFunc("/sessions/{session_id}/messages", handlers.SendMessage).Methods("POST")
	apiRouter.HandleFunc("/sessions/{session_id}/messages", handlers.GetMessages).Methods("GET")
	apiRouter.HandleFunc("/messages/{id}", handlers.GetMessage).Methods("GET")
	apiRouter.HandleFunc("/messages/{id}", handlers.UpdateMessage).Methods("PUT")
	apiRouter.HandleFunc("/messages/{id}", handlers.DeleteMessage).Methods("DELETE")
	apiRouter.HandleFunc("/tools", handlers.ListTools).Methods("GET")
	apiRouter.HandleFunc("/tools", handlers.CreateTool).Methods("POST")
	apiRouter.HandleFunc("/tools/{name}", handlers.GetTool).Methods("GET")
	apiRouter.HandleFunc("/tools/{name}", handlers.UpdateTool).Methods("PUT")
	apiRouter.HandleFunc("/tools/{name}", handlers.DeleteTool).Methods("DELETE")
	apiRouter.HandleFunc("/tools/{name}/analytics", handlers.GetToolAnalytics).Methods("GET")
	apiRouter.HandleFunc("/memory/{id}/summarize", handlers.SummarizeMemory).Methods("POST")
	apiRouter.HandleFunc("/analytics/overview", handlers.GetAnalyticsOverview).Methods("GET")
	apiRouter.HandleFunc("/ws", api.HandleWebSocket(runtime, keyManager)).Methods("GET")

	/* Collaboration workspace routes */
	apiRouter.HandleFunc("/workspaces", collabHandlers.CreateWorkspace).Methods("POST")
	apiRouter.HandleFunc("/workspaces/{id}", collabHandlers.GetWorkspace).Methods("GET")
	apiRouter.HandleFunc("/workspaces/{id}/participants", collabHandlers.AddParticipant).Methods("POST")

	/* Async task routes */
	apiRouter.HandleFunc("/async-tasks", asyncTasksHandlers.CreateAsyncTask).Methods("POST")
	apiRouter.HandleFunc("/async-tasks", asyncTasksHandlers.ListAsyncTasks).Methods("GET")
	apiRouter.HandleFunc("/async-tasks/{id}", asyncTasksHandlers.GetAsyncTaskStatus).Methods("GET")
	apiRouter.HandleFunc("/async-tasks/{id}/cancel", asyncTasksHandlers.CancelAsyncTask).Methods("POST")

	/* Alert preferences routes */
	apiRouter.HandleFunc("/alert-preferences", alertPrefsHandlers.SetAlertPreferences).Methods("POST")
	apiRouter.HandleFunc("/alert-preferences", alertPrefsHandlers.GetAlertPreferences).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/alert-preferences", alertPrefsHandlers.GetAlertPreferences).Methods("GET")

	/* Workflow routes */
	apiRouter.HandleFunc("/workflows", workflowHandlers.CreateWorkflow).Methods("POST")
	apiRouter.HandleFunc("/workflows", workflowHandlers.ListWorkflows).Methods("GET")
	apiRouter.HandleFunc("/workflows/{id}", workflowHandlers.GetWorkflow).Methods("GET")
	apiRouter.HandleFunc("/workflows/{id}", workflowHandlers.UpdateWorkflow).Methods("PUT")
	apiRouter.HandleFunc("/workflows/{id}", workflowHandlers.DeleteWorkflow).Methods("DELETE")
	apiRouter.HandleFunc("/workflows/{workflow_id}/steps", workflowHandlers.CreateWorkflowStep).Methods("POST")
	apiRouter.HandleFunc("/workflows/{workflow_id}/steps", workflowHandlers.ListWorkflowSteps).Methods("GET")
	apiRouter.HandleFunc("/workflows/{workflow_id}/execute", workflowHandlers.ExecuteWorkflow).Methods("POST")
	apiRouter.HandleFunc("/workflows/{workflow_id}/executions", workflowHandlers.ListWorkflowExecutions).Methods("GET")
	apiRouter.HandleFunc("/workflow-executions/{execution_id}", workflowHandlers.GetWorkflowExecution).Methods("GET")

	/* Event Stream routes */
	apiRouter.HandleFunc("/sessions/{session_id}/events", eventStreamHandlers.LogEvent).Methods("POST")
	apiRouter.HandleFunc("/sessions/{session_id}/events", eventStreamHandlers.GetEventHistory).Methods("GET")
	apiRouter.HandleFunc("/sessions/{session_id}/events/summarize", eventStreamHandlers.SummarizeEvents).Methods("POST")
	apiRouter.HandleFunc("/sessions/{session_id}/events/context", eventStreamHandlers.GetContextWindow).Methods("GET")
	apiRouter.HandleFunc("/sessions/{session_id}/events/count", eventStreamHandlers.GetEventCount).Methods("GET")

	/* Verification routes */
	apiRouter.HandleFunc("/agents/{agent_id}/verifications", verificationHandlers.QueueVerification).Methods("POST")
	apiRouter.HandleFunc("/verifications/{queue_id}/results", verificationHandlers.GetVerificationResults).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/verification-rules", verificationHandlers.ListVerificationRules).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/verification-rules", verificationHandlers.CreateVerificationRule).Methods("POST")
	apiRouter.HandleFunc("/verification-rules/{rule_id}", verificationHandlers.UpdateVerificationRule).Methods("PUT")
	apiRouter.HandleFunc("/verification-rules/{rule_id}", verificationHandlers.DeleteVerificationRule).Methods("DELETE")

	/* Virtual Filesystem routes */
	apiRouter.HandleFunc("/agents/{agent_id}/files", vfsHandlers.CreateFile).Methods("POST")
	apiRouter.HandleFunc("/agents/{agent_id}/files", vfsHandlers.ListFiles).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/files/{path:.*}", vfsHandlers.ReadFile).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/files/{path:.*}", vfsHandlers.WriteFile).Methods("PUT")
	apiRouter.HandleFunc("/agents/{agent_id}/files/{path:.*}", vfsHandlers.DeleteFile).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{agent_id}/files/{path:.*}/copy", vfsHandlers.CopyFile).Methods("POST")
	apiRouter.HandleFunc("/agents/{agent_id}/files/{path:.*}/move", vfsHandlers.MoveFile).Methods("POST")

	/* Evaluation framework routes */
	apiRouter.HandleFunc("/eval/tasks", evaluationHandlers.CreateEvalTask).Methods("POST")
	apiRouter.HandleFunc("/eval/tasks", evaluationHandlers.ListEvalTasks).Methods("GET")
	apiRouter.HandleFunc("/eval/tasks/{id}", evaluationHandlers.GetEvalTask).Methods("GET")
	apiRouter.HandleFunc("/eval/runs", evaluationHandlers.CreateEvalRun).Methods("POST")
	apiRouter.HandleFunc("/eval/runs", evaluationHandlers.ListEvalRuns).Methods("GET")
	apiRouter.HandleFunc("/eval/runs/{id}", evaluationHandlers.GetEvalRun).Methods("GET")
	apiRouter.HandleFunc("/eval/runs/{id}", evaluationHandlers.UpdateEvalRun).Methods("PUT")
	apiRouter.HandleFunc("/eval/runs/{run_id}/execute", evaluationHandlers.ExecuteEvalRun).Methods("POST")
	apiRouter.HandleFunc("/eval/runs/{run_id}/results", evaluationHandlers.GetEvalRunResults).Methods("GET")
	apiRouter.HandleFunc("/eval/runs/{run_id}/results/{result_id}/retrieval", evaluationHandlers.CreateEvalRetrievalResult).Methods("POST")

	/* Execution snapshots and replay routes */
	apiRouter.HandleFunc("/sessions/{session_id}/snapshots", replayHandlers.CreateSnapshot).Methods("POST")
	apiRouter.HandleFunc("/sessions/{session_id}/snapshots", replayHandlers.ListSnapshotsBySession).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/snapshots", replayHandlers.ListSnapshotsByAgent).Methods("GET")
	apiRouter.HandleFunc("/snapshots/{id}", replayHandlers.GetSnapshot).Methods("GET")
	apiRouter.HandleFunc("/snapshots/{id}/replay", replayHandlers.ReplaySnapshot).Methods("POST")
	apiRouter.HandleFunc("/snapshots/{id}", replayHandlers.DeleteSnapshot).Methods("DELETE")

	/* Workflow schedule routes */
	apiRouter.HandleFunc("/workflows/{workflow_id}/schedule", workflowHandlers.CreateWorkflowSchedule).Methods("POST")
	apiRouter.HandleFunc("/workflows/{workflow_id}/schedule", workflowHandlers.GetWorkflowSchedule).Methods("GET")
	apiRouter.HandleFunc("/workflows/{workflow_id}/schedule", workflowHandlers.UpdateWorkflowSchedule).Methods("PUT")
	apiRouter.HandleFunc("/workflows/{workflow_id}/schedule", workflowHandlers.DeleteWorkflowSchedule).Methods("DELETE")
	apiRouter.HandleFunc("/workflow-schedules", workflowHandlers.ListWorkflowSchedules).Methods("GET")

	/* Agent specialization routes */
	apiRouter.HandleFunc("/agents/{agent_id}/specialization", specializationHandlers.CreateSpecialization).Methods("POST")
	apiRouter.HandleFunc("/agents/{agent_id}/specialization", specializationHandlers.GetSpecialization).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/specialization", specializationHandlers.UpdateSpecialization).Methods("PUT")
	apiRouter.HandleFunc("/agents/{agent_id}/specialization", specializationHandlers.DeleteSpecialization).Methods("DELETE")
	apiRouter.HandleFunc("/specializations", specializationHandlers.ListSpecializations).Methods("GET")

	/* Health check */
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if err := database.HealthCheck(r.Context()); err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}).Methods("GET")

	/* Metrics endpoint (no auth required) */
	router.Handle("/metrics", metrics.Handler()).Methods("GET")

	/* Start background workers */
	queue := jobs.NewQueue(queries)
	processor := jobs.NewProcessor(database)
	jobWorker := jobs.NewWorker(queue, processor, 5)
	jobWorker.Start()
	defer jobWorker.Stop()

	/* Start job scheduler */
	scheduler := jobs.NewScheduler(queue)
	scheduler.Start()
	defer scheduler.Stop()

	/* Start memory promoter worker */
	if runtime.HierMemory() != nil {
		memoryPromoter := worker.NewMemoryPromoter(runtime.HierMemory(), queries, 5*time.Minute)
		go func() {
			ctx := context.Background()
			if err := memoryPromoter.Start(ctx); err != nil {
				metrics.ErrorWithContext(ctx, "Memory promoter worker failed", err, nil)
			}
		}()
	}

	/* Start verifier worker */
	if runtime.Verifier() != nil {
		verifierWorker := worker.NewVerifierWorker(queries, runtime, 10*time.Second, 3)
		go func() {
			ctx := context.Background()
			if err := verifierWorker.Start(ctx); err != nil {
				metrics.ErrorWithContext(ctx, "Verifier worker failed", err, nil)
			}
		}()
	}

	/* Start async task worker */
	if asyncExecutor != nil {
		asyncTaskWorker := worker.NewAsyncTaskWorker(queries, asyncExecutor, 5*time.Second, 5)
		go func() {
			ctx := context.Background()
			if err := asyncTaskWorker.Start(ctx); err != nil {
				metrics.ErrorWithContext(ctx, "Async task worker failed", err, nil)
			}
		}()
	}

	/* Start connection pool metrics collector */
	metricsCtx, metricsCancel := context.WithCancel(context.Background())
	defer metricsCancel()
	go func() {
		ticker := time.NewTicker(10 * time.Second) /* Collect metrics every 10 seconds */
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				openConns, idleConns, inUse := database.GetPoolStats()
				metrics.RecordDBPoolStats(cfg.Database.Database, openConns, idleConns, inUse)
			case <-metricsCtx.Done():
				return
			}
		}
	}()

	/* Start server */
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	/* Graceful shutdown */
	go func() {
		metrics.InfoWithContext(context.Background(), "NeuronAgent server starting", map[string]interface{}{
			"address": addr,
			"host":    cfg.Server.Host,
			"port":    cfg.Server.Port,
		})
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			metrics.ErrorWithContext(context.Background(), "Server failed to start", err, map[string]interface{}{
				"address": addr,
			})
			os.Exit(1)
		}
	}()

	/* Wait for interrupt signal */
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	metrics.InfoWithContext(context.Background(), "Shutdown signal received, gracefully shutting down server", nil)

	/* Stop metrics collector */
	metricsCancel()

	/* Cleanup resources */
	if toolRegistry != nil {
		toolRegistry.Cleanup()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		metrics.ErrorWithContext(context.Background(), "Server shutdown timeout exceeded, forcing shutdown", err, nil)
	}

	metrics.InfoWithContext(context.Background(), "Server shutdown complete", nil)
}

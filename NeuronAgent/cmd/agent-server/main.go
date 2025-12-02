package main

import (
	"context"
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
	"github.com/neurondb/NeuronAgent/internal/config"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/jobs"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/session"
	"github.com/neurondb/NeuronAgent/internal/tools"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

func main() {
	// Load configuration
	cfg := config.DefaultConfig()
	if configPath := os.Getenv("CONFIG_PATH"); configPath != "" {
		var err error
		cfg, err = config.LoadConfig(configPath)
		if err != nil {
			fmt.Printf("Failed to load config: %v, using defaults\n", err)
		}
	}

	// Initialize logging
	metrics.InitLogging(cfg.Logging.Level, cfg.Logging.Format)

	// Connect to database
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
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
		panic(fmt.Sprintf("Failed to connect to database: %v", err))
	}
	defer database.Close()

	// Run migrations
	migrationRunner, err := db.NewMigrationRunner(database.DB, "./migrations")
	if err == nil {
		if err := migrationRunner.Run(context.Background()); err != nil {
			fmt.Printf("Warning: Migration failed: %v\n", err)
		}
	}

	// Initialize components
	queries := db.NewQueries(database.DB)
	queries.SetConnInfoFunc(database.GetConnInfoString)
	embedClient := neurondb.NewEmbeddingClient(database.DB)
	toolRegistry := tools.NewRegistry(queries, database)
	runtime := agent.NewRuntime(database, queries, toolRegistry, embedClient)

	// Initialize session management
	sessionCache := session.NewCache(5 * time.Minute)
	_ = session.NewManager(queries, sessionCache) // Session manager for future use
	sessionCleanup := session.NewCleanupService(queries, 1*time.Hour, 24*time.Hour)
	sessionCleanup.Start()
	defer sessionCleanup.Stop()

	// Initialize API
	handlers := api.NewHandlers(queries, runtime)
	keyManager := auth.NewAPIKeyManager(queries)
	rateLimiter := auth.NewRateLimiter()

	// Setup router
	router := mux.NewRouter()
	router.Use(api.RequestIDMiddleware)
	router.Use(api.CORSMiddleware)
	router.Use(api.LoggingMiddleware)
	router.Use(api.AuthMiddleware(keyManager, rateLimiter))

	// API routes
	apiRouter := router.PathPrefix("/api/v1").Subrouter()
	apiRouter.HandleFunc("/agents", handlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/agents", handlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", handlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", handlers.UpdateAgent).Methods("PUT")
	apiRouter.HandleFunc("/agents/{id}", handlers.DeleteAgent).Methods("DELETE")
	apiRouter.HandleFunc("/sessions", handlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/sessions/{id}", handlers.GetSession).Methods("GET")
	apiRouter.HandleFunc("/agents/{agent_id}/sessions", handlers.ListSessions).Methods("GET")
	apiRouter.HandleFunc("/sessions/{session_id}/messages", handlers.SendMessage).Methods("POST")
	apiRouter.HandleFunc("/sessions/{session_id}/messages", handlers.GetMessages).Methods("GET")
	apiRouter.HandleFunc("/ws", api.HandleWebSocket(runtime)).Methods("GET")

	// Health check
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if err := database.HealthCheck(r.Context()); err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}).Methods("GET")

	// Metrics endpoint (no auth required)
	router.Handle("/metrics", metrics.Handler()).Methods("GET")

	// Start background workers
	queue := jobs.NewQueue(queries)
	processor := jobs.NewProcessor(database)
	worker := jobs.NewWorker(queue, processor, 5)
	worker.Start()
	defer worker.Stop()

	// Start job scheduler
	scheduler := jobs.NewScheduler(queue)
	scheduler.Start()
	defer scheduler.Stop()

	// Start server
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// Graceful shutdown
	go func() {
		fmt.Printf("Server starting on %s\n", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			panic(fmt.Sprintf("Server failed: %v", err))
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	fmt.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		fmt.Printf("Server forced to shutdown: %v\n", err)
	}

	fmt.Println("Server exited")
}


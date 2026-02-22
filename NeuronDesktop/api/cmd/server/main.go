package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/auth/oidc"
	"github.com/neurondb/NeuronDesktop/api/internal/config"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/handlers"
	"github.com/neurondb/NeuronDesktop/api/internal/initialization"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/metrics"
	"github.com/neurondb/NeuronDesktop/api/internal/middleware"
	"github.com/neurondb/NeuronDesktop/api/internal/session"
)

var (
	version   = "dev"
	buildDate = "unknown"
	gitCommit = "unknown"
)

func main() {
	var (
		showVersion = flag.Bool("version", false, "Show version information")
		showVersionShort = flag.Bool("v", false, "Show version information (short)")
		configPath  = flag.String("c", "", "Path to configuration file (currently configuration is loaded from environment variables)")
		configPathLong = flag.String("config", "", "Path to configuration file (currently configuration is loaded from environment variables)")
		showHelp    = flag.Bool("help", false, "Show help message")
		showHelpShort = flag.Bool("h", false, "Show help message (short)")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "NeuronDesktop API Server - Desktop application API server for NeuronDB\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s                    Start server with default configuration\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --version          Show version information\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --help             Show this help message\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nConfiguration:\n")
		fmt.Fprintf(os.Stderr, "  Configuration is currently loaded from environment variables.\n")
		fmt.Fprintf(os.Stderr, "  See the config package for available environment variables.\n")
		fmt.Fprintf(os.Stderr, "  Note: Config file support is planned for a future release.\n")
	}
	flag.Parse()

	/* Handle version flag */
	if *showVersion || *showVersionShort {
		fmt.Printf("neurondesktop version %s\n", version)
		fmt.Printf("Build date: %s\n", buildDate)
		fmt.Printf("Git commit: %s\n", gitCommit)
		os.Exit(0)
	}

	/* Handle help flag */
	if *showHelp || *showHelpShort {
		flag.Usage()
		os.Exit(0)
	}

	/* Load config from file if provided */
	configPathValue := ""
	if *configPath != "" {
		configPathValue = *configPath
	} else if *configPathLong != "" {
		configPathValue = *configPathLong
	}

	/* Load config (from file if provided, otherwise from env) */
	cfg := config.LoadFromFile(configPathValue)

	logger := logging.NewLogger(cfg.Logging.Level, cfg.Logging.Format, cfg.Logging.Output)
	if configPathValue != "" {
		logger.Info("Loaded configuration from file", map[string]interface{}{"path": configPathValue})
	}
	logger.Info("Starting NeuronDesktop API server", nil)

	database, err := sql.Open("pgx", cfg.Database.DSN())
	if err != nil {
		logger.Error("Failed to open database", err, nil)
		os.Exit(1)
	}
	defer database.Close()

	database.SetMaxOpenConns(cfg.Database.MaxOpenConns)
	database.SetMaxIdleConns(cfg.Database.MaxIdleConns)
	database.SetConnMaxLifetime(cfg.Database.ConnMaxLifetime)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := database.PingContext(ctx); err != nil {
		logger.Error("Failed to ping database", err, nil)
		os.Exit(1)
	}

	logger.Info("Connected to database", map[string]interface{}{
		"host": cfg.Database.Host,
		"port": cfg.Database.Port,
		"name": cfg.Database.Name,
	})

	queries := db.NewQueries(database)
	keyManager := auth.NewAPIKeyManager(queries) /* Keep for backwards compatibility if needed */

	if cfg.Auth.Mode == "jwt" || cfg.Auth.Mode == "hybrid" {
		if cfg.Auth.JWTSecret == "" {
			logger.Error("JWT_SECRET is required when using JWT authentication", fmt.Errorf("JWT_SECRET environment variable not set"), nil)
			os.Exit(1)
		}
		if len(cfg.Auth.JWTSecret) < 32 {
			logger.Error("JWT_SECRET must be at least 32 bytes for security", fmt.Errorf("JWT_SECRET length is %d", len(cfg.Auth.JWTSecret)), nil)
			os.Exit(1)
		}
	}

	sessionMgr := session.NewManager(
		database,
		cfg.Session.AccessTTL,
		cfg.Session.RefreshTTL,
		cfg.Session.CookieDomain,
		cfg.Session.CookieSecure,
		cfg.Session.CookieSameSite,
	)

	var oidcProvider *oidc.Provider
	var oidcHandlers *handlers.OIDCHandlers
	if cfg.Auth.Mode == "oidc" || cfg.Auth.Mode == "hybrid" {
		if cfg.Auth.OIDC.IssuerURL != "" {
			var err error
			oidcProvider, err = oidc.NewProvider(
				ctx,
				cfg.Auth.OIDC.IssuerURL,
				cfg.Auth.OIDC.ClientID,
				cfg.Auth.OIDC.ClientSecret,
				cfg.Auth.OIDC.RedirectURL,
				cfg.Auth.OIDC.Scopes,
			)
			if err != nil {
				logger.Error("Failed to initialize OIDC provider", err, nil)
			} else {
				oidcHandlers = handlers.NewOIDCHandlers(oidcProvider, sessionMgr, queries, cfg.Auth.OIDC.IssuerURL)
				logger.Info("OIDC provider initialized", map[string]interface{}{
					"issuer": cfg.Auth.OIDC.IssuerURL,
				})
			}
		}
	}

	initCtx, initCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer initCancel()

	bootstrap := initialization.NewBootstrap(queries, logger)
	if err := bootstrap.Initialize(initCtx); err != nil {
		logger.Error("Failed to bootstrap application", err, nil)
	}

	authHandlers := handlers.NewAuthHandlers(queries)
	mcpManager := handlers.NewMCPManager(queries)
	mcpHandlers := handlers.NewMCPHandlers(mcpManager)
	mcpHandlers.SetCORSConfig(cfg.CORS.AllowedOrigins)
	mcpHandlers.SetLogger(logger)
	neurondbHandlers := handlers.NewNeuronDBHandlers(queries, cfg.Security.EnableSQLConsole)
	agentHandlers := handlers.NewAgentHandlers(queries)
	agentHandlers.SetCORSConfig(cfg.CORS.AllowedOrigins)
	agentHandlers.SetLogger(logger)
	templateHandlers := handlers.NewTemplateHandlers(queries, logger)
	profileHandlers := handlers.NewProfileHandlers(queries)
	profileHandlers.SetHandlers(neurondbHandlers, agentHandlers, mcpManager)
	metricsHandlers := handlers.NewMetricsHandlers()
	factoryHandlers := handlers.NewFactoryHandlers(queries)
	systemMetricsHandlers := handlers.NewSystemMetricsHandlers(logger)
	analyticsHandlers := handlers.NewAnalyticsHandlers(queries, database)
	requestLogHandlers := handlers.NewRequestLogHandlers(queries)
	auditLogHandlers := handlers.NewAuditLogHandlers(queries)

	rateLimiter := middleware.NewRateLimiter(10000, 1*time.Minute)

	router := mux.NewRouter()
	router.StrictSlash(true)

	router.Use(middleware.RequestIDMiddleware())
	router.Use(middleware.RecoveryMiddleware(logger))
	router.Use(middleware.SecurityHeadersMiddleware)
	router.Use(middleware.PrometheusMetricsMiddleware)
	router.Use(middleware.LoggingMiddleware(logger, queries))

	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "ok",
			"service":   "neurondesk-api",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods("GET")

	// Prometheus metrics endpoint (no auth required)
	router.Handle("/metrics", metrics.Handler()).Methods("GET")

	systemMetricsRouter := router.PathPrefix("/api/v1/system-metrics").Subrouter()
	systemMetricsRouter.HandleFunc("", systemMetricsHandlers.GetSystemMetrics).Methods("GET")
	systemMetricsRouter.HandleFunc("/ws", systemMetricsHandlers.SystemMetricsWebSocket).Methods("GET")

	databaseTestHandlers := handlers.NewDatabaseTestHandlers()

	authRouter := router.PathPrefix("/api/v1/auth").Subrouter()
	if cfg.Auth.EnableLocalAuth {
		authRouter.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			authHandlers.Register(w, r)
		}).Methods("POST", "OPTIONS")
		authRouter.HandleFunc("/login", func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			authHandlers.Login(w, r)
		}).Methods("POST", "OPTIONS")
	}

	if oidcHandlers != nil {
		authRouter.HandleFunc("/oidc/start", oidcHandlers.StartOIDCFlow).Methods("GET")
		authRouter.HandleFunc("/oidc/callback", oidcHandlers.OIDCCallback).Methods("GET")
		authRouter.HandleFunc("/refresh", oidcHandlers.RefreshToken).Methods("POST")
	}

	authRouter.HandleFunc("/logout", func(w http.ResponseWriter, r *http.Request) {
		if oidcHandlers != nil {
			oidcHandlers.Logout(w, r)
		} else {
			handlers.WriteSuccess(w, map[string]interface{}{"message": "logged out"}, http.StatusOK)
		}
	}).Methods("POST")

	router.HandleFunc("/api/v1/database/test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		databaseTestHandlers.TestConnection(w, r)
	}).Methods("POST", "OPTIONS")

	// Health endpoint under /api/v1 (for frontend compatibility, no auth required)
	// Register on main router with exact path match to avoid subrouter conflicts
	router.HandleFunc("/api/v1/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "ok",
			"service":   "neurondesk-api",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods("GET").Name("api_health")

	apiRouter := router.PathPrefix("/api/v1").Subrouter()

	if cfg.Auth.Mode == "oidc" && oidcHandlers != nil {
		apiRouter.Use(sessionMgr.SessionMiddleware())
	} else if cfg.Auth.Mode == "hybrid" {
		apiRouter.Use(func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if sessionMgr.GetAccessTokenFromRequest(r) != "" {
					sessionMgr.SessionMiddleware()(next).ServeHTTP(w, r)
					return
				}
				auth.JWTMiddleware()(next).ServeHTTP(w, r)
			})
		})
	} else {
		apiRouter.Use(auth.JWTMiddleware())
	}

	apiRouter.Use(middleware.RateLimitMiddleware(rateLimiter))
	apiRouter.Use(middleware.CSRFMiddleware())

	apiRouter.HandleFunc("/auth/me", authHandlers.GetCurrentUser).Methods("GET")

	apiKeyHandlers := handlers.NewAPIKeyHandlers(keyManager, queries)
	apiRouter.HandleFunc("/api-keys", apiKeyHandlers.GenerateAPIKey).Methods("POST")
	apiRouter.HandleFunc("/api-keys", apiKeyHandlers.ListAPIKeys).Methods("GET")
	apiRouter.HandleFunc("/api-keys/{id}", apiKeyHandlers.DeleteAPIKey).Methods("DELETE")

	apiRouter.HandleFunc("/profiles", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method == "GET" {
			profileHandlers.ListProfiles(w, r)
		} else if r.Method == "POST" {
			profileHandlers.CreateProfile(w, r)
		}
	}).Methods("GET", "POST", "OPTIONS")
	apiRouter.HandleFunc("/profiles/{id}", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method == "GET" {
			profileHandlers.GetProfile(w, r)
		} else if r.Method == "PUT" {
			profileHandlers.UpdateProfile(w, r)
		} else if r.Method == "DELETE" {
			profileHandlers.DeleteProfile(w, r)
		}
	}).Methods("GET", "PUT", "DELETE", "OPTIONS")
	apiRouter.HandleFunc("/profiles/{id}/clone", profileHandlers.CloneProfile).Methods("POST")
	apiRouter.HandleFunc("/profiles/validate", profileHandlers.ValidateProfile).Methods("POST")
	apiRouter.HandleFunc("/profiles/{id}/health", profileHandlers.HealthCheckProfile).Methods("GET")

	dashboardHandlers := handlers.NewDashboardHandlers(database)
	apiRouter.HandleFunc("/profiles/{profile_id}/dashboard", dashboardHandlers.GetDashboard).Methods("GET")

	unifiedQueryHandlers := handlers.NewUnifiedQueryHandlers(queries)
	apiRouter.HandleFunc("/profiles/{profile_id}/unified-query", unifiedQueryHandlers.ExecuteUnifiedQuery).Methods("POST")

	apiRouter.HandleFunc("/mcp/connections", mcpHandlers.ListConnections).Methods("GET")
	apiRouter.HandleFunc("/mcp/test", mcpHandlers.TestMCPConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools", mcpHandlers.ListTools).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools/call", mcpHandlers.CallTool).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/ws", mcpHandlers.MCPWebSocket).Methods("GET")

	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads", mcpHandlers.ListThreads).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads", mcpHandlers.CreateThread).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}", mcpHandlers.GetThread).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}", mcpHandlers.UpdateThread).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}", mcpHandlers.DeleteThread).Methods("DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}/messages", mcpHandlers.AddMessage).Methods("POST")

	/* MCP Resources */
	mcpResourcesHandlers := handlers.NewMCPResourcesHandlers(queries, mcpManager)
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/resources", mcpResourcesHandlers.ListResources).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/resources/{uri:.*}", mcpResourcesHandlers.GetResource).Methods("GET")

	/* MCP Datasets */
	mcpDatasetsHandlers := handlers.NewMCPDatasetsHandlers(queries, mcpManager)
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/datasets/load", mcpDatasetsHandlers.LoadDataset).Methods("POST")

	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/collections", neurondbHandlers.ListCollections).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/search", neurondbHandlers.Search).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/sql", neurondbHandlers.ExecuteSQL).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/sql/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		neurondbHandlers.ExecuteSQLFull(w, r)
	}).Methods("POST", "OPTIONS")

	datasetHandlers := handlers.NewDatasetHandlers(queries, nil)
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/ingest", datasetHandlers.IngestDataset).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/ingest/{job_id}", datasetHandlers.GetIngestStatus).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/ingest", datasetHandlers.ListIngestJobs).Methods("GET")

	/* RAG routes */
	ragHandlers := handlers.NewRAGHandlers(queries, neurondbHandlers)
	apiRouter.HandleFunc("/profiles/{profileId}/rag/query", ragHandlers.RAGQuery).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profileId}/rag/ingest", ragHandlers.RAGIngest).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profileId}/rag/evaluate", ragHandlers.RAGEvaluate).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profileId}/rag/pipelines", ragHandlers.ListRAGPipelines).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profileId}/rag/pipelines", ragHandlers.CreateRAGPipeline).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profileId}/rag/pipelines/{id}", ragHandlers.GetRAGPipeline).Methods("GET")

	apiRouter.HandleFunc("/agent/test", agentHandlers.TestAgentConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}", agentHandlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}", agentHandlers.UpdateAgent).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}", agentHandlers.DeleteAgent).Methods("DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/export", agentHandlers.ExportAgent).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/import", agentHandlers.ImportAgent).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/models", agentHandlers.ListModels).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions", agentHandlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/sessions", agentHandlers.ListSessions).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}", agentHandlers.GetSession).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/messages", agentHandlers.SendMessage).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/messages", agentHandlers.GetMessages).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/ws", agentHandlers.AgentWebSocket).Methods("GET")

	/* Agent retrieval and memory handlers */
	agentRetrievalHandlers := handlers.NewAgentRetrievalHandlers(queries, agentHandlers)
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/retrieval-stats", agentRetrievalHandlers.GetRetrievalStats).Methods("GET")

	agentMemoryHandlers := handlers.NewAgentMemoryHandlers(queries, agentHandlers)
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/memory/{memory_id}/feedback", agentMemoryHandlers.SubmitMemoryFeedback).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/memory/consolidate", agentMemoryHandlers.ConsolidateMemory).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/memory/quality", agentMemoryHandlers.GetMemoryQuality).Methods("GET")

	/* Generic proxy for NeuronAgent endpoints */
	agentProxyHandlers := handlers.NewAgentProxyHandlers(queries, agentHandlers)
	/* Workflows */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/workflows", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/workflows/{id}", agentProxyHandlers.ProxyRequest).Methods("GET", "PUT", "DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/workflows/{workflow_id}/steps", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/workflows/{workflow_id}/execute", agentProxyHandlers.ProxyRequest).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/workflows/{workflow_id}/executions", agentProxyHandlers.ProxyRequest).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/workflow-executions/{execution_id}", agentProxyHandlers.ProxyRequest).Methods("GET")
	/* Budgets */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/budgets", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/budgets/{id}", agentProxyHandlers.ProxyRequest).Methods("GET", "PUT", "DELETE")
	/* Evaluation */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/eval/tasks", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/eval/runs", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/eval/runs/{run_id}/execute", agentProxyHandlers.ProxyRequest).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/eval/runs/{run_id}/results", agentProxyHandlers.ProxyRequest).Methods("GET")
	/* Snapshots */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/snapshots", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/snapshots", agentProxyHandlers.ProxyRequest).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/snapshots/{id}/replay", agentProxyHandlers.ProxyRequest).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/snapshots/{id}", agentProxyHandlers.ProxyRequest).Methods("DELETE")
	/* Virtual Filesystem */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/filesystem", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/filesystem/{path:.*}", agentProxyHandlers.ProxyRequest).Methods("GET", "PUT", "DELETE")
	/* Event Streams */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/events", agentProxyHandlers.ProxyRequest).Methods("GET", "POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/events/summarize", agentProxyHandlers.ProxyRequest).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/events/context", agentProxyHandlers.ProxyRequest).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/events/count", agentProxyHandlers.ProxyRequest).Methods("GET")
	/* Observability */
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/observability/executions/{id}/decision-tree", agentProxyHandlers.ProxyRequest).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/observability/executions/{id}/tool-chain", agentProxyHandlers.ProxyRequest).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/observability/executions/{id}/performance", agentProxyHandlers.ProxyRequest).Methods("GET")

	apiRouter.HandleFunc("/templates", templateHandlers.ListTemplates).Methods("GET")
	apiRouter.HandleFunc("/templates/{id}", templateHandlers.GetTemplate).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/templates/{id}/deploy", templateHandlers.DeployTemplate).Methods("POST")

	modelConfigHandlers := handlers.NewModelConfigHandlers(queries)
	apiRouter.HandleFunc("/profiles/{profile_id}/models", modelConfigHandlers.ListModelConfigs).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/models", modelConfigHandlers.CreateModelConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/{id}", modelConfigHandlers.UpdateModelConfig).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/{id}", modelConfigHandlers.DeleteModelConfig).Methods("DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/default", modelConfigHandlers.GetDefaultModelConfig).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/{id}/set-default", modelConfigHandlers.SetDefaultModelConfig).Methods("POST")

	// JSON stats endpoint (moved from /metrics to avoid confusion with Prometheus)
	apiRouter.HandleFunc("/stats", metricsHandlers.GetMetrics).Methods("GET")
	apiRouter.HandleFunc("/stats/reset", metricsHandlers.ResetMetrics).Methods("POST")

	apiRouter.HandleFunc("/profiles/{profile_id}/analytics", analyticsHandlers.GetAnalytics).Methods("GET")
	apiRouter.HandleFunc("/analytics", analyticsHandlers.GetAnalytics).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/analytics/export", analyticsHandlers.ExportAnalytics).Methods("GET")
	apiRouter.HandleFunc("/analytics/export", analyticsHandlers.ExportAnalytics).Methods("GET")

	/* Request logs routes */
	apiRouter.HandleFunc("/profiles/{profile_id}/logs", requestLogHandlers.ListLogs).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/logs/{id}", requestLogHandlers.GetLog).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/logs/{id}", requestLogHandlers.DeleteLog).Methods("DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/logs/export", requestLogHandlers.ExportLogs).Methods("GET")

	/* Audit log routes (admin only) */
	apiRouter.HandleFunc("/audit-logs", auditLogHandlers.ListAuditLogs).Methods("GET")
	apiRouter.HandleFunc("/audit-logs/{id}", auditLogHandlers.GetAuditLog).Methods("GET")

	apiRouter.HandleFunc("/factory/status", factoryHandlers.GetFactoryStatus).Methods("GET")
	apiRouter.HandleFunc("/factory/setup-state", factoryHandlers.GetSetupState).Methods("GET")
	apiRouter.HandleFunc("/factory/setup-state", factoryHandlers.SetSetupState).Methods("POST")

	/* CORS handler wrapper
	 * Important: we wrap the router at the HTTP handler level (instead of router.Use),
	 * so CORS headers and OPTIONS preflight responses work even when gorilla/mux would
	 * otherwise return 404 for method-mismatches (e.g. OPTIONS on a GET-only route). */
	corsHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		/* Check if this is a WebSocket upgrade request
		 * WebSocket upgrades need direct access to the underlying connection (Hijacker interface)
		 * so we bypass the CORS wrapper for WebSocket requests */
		if r.Header.Get("Upgrade") == "websocket" {
			router.ServeHTTP(w, r)
			return
		}

		origin := r.Header.Get("Origin")
		allowed := false
		allowAll := false

		for _, allowedOrigin := range cfg.CORS.AllowedOrigins {
			if allowedOrigin == "*" {
				allowAll = true
				allowed = true
				break
			} else if allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed {
			if allowAll && origin != "" {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			} else if allowAll {
				w.Header().Set("Access-Control-Allow-Origin", "*")
			} else if origin != "" {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}
		}

		/* Do not set Allow-Credentials when Origin is * (forbidden by spec) */
		if allowed && !allowAll && origin != "" {
			w.Header().Set("Access-Control-Allow-Credentials", "true")
		}

		w.Header().Set("Access-Control-Allow-Methods", joinStrings(cfg.CORS.AllowedMethods, ", "))
		w.Header().Set("Access-Control-Allow-Headers", joinStrings(cfg.CORS.AllowedHeaders, ", "))

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		router.ServeHTTP(w, r)
	})

	addr := cfg.Server.Host + ":" + cfg.Server.Port
	srv := &http.Server{
		Addr:         addr,
		Handler:      corsHandler,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	go func() {
		logger.Info("Server starting", map[string]interface{}{
			"address": addr,
		})
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("Server failed", err, nil)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server", nil)

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	/* Close all client connections gracefully */
	logger.Info("Closing client connections", nil)
	neurondbHandlers.CloseAll()
	agentHandlers.CloseAll()
	mcpManager.CloseAll()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("Server shutdown failed", err, nil)
	}

	logger.Info("Server stopped", nil)
}

func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}

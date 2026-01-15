/*
 * Server implements the main MCP server for NeuronMCP
 *
 * Provides MCP protocol communication, request routing, middleware execution,
 * and tool/resource management for PostgreSQL and vector operations.
 */

package server

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/neurondb/NeuronMCP/internal/batch"
	"github.com/neurondb/NeuronMCP/internal/cache"
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/health"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/metrics"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/progress"
	"github.com/neurondb/NeuronMCP/internal/prompts"
	"github.com/neurondb/NeuronMCP/internal/resources"
	"github.com/neurondb/NeuronMCP/internal/sampling"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/internal/transport"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/*
 * Server is the main MCP server
 */
type Server struct {
	mcpServer           *mcp.Server
	db                  *database.Database
	config              *config.ConfigManager
	logger              *logging.Logger
	middleware          *middleware.Manager
	toolRegistry        *tools.ToolRegistry
	resources           *resources.Manager
	prompts             *prompts.Manager
	sampling            *sampling.Manager
	health              *health.Checker
	progress            *progress.Tracker
	batch               *batch.Processor
	capabilitiesManager *CapabilitiesManager
	idempotencyCache    *cache.IdempotencyCache
	metricsCollector    *metrics.Collector
	prometheusExporter  *metrics.PrometheusExporter
	httpServer          *HTTPServer // HTTP server for /metrics and /health
	httpTransport       *transport.HTTPTransport // HTTP transport for MCP protocol
}

/*
 * NewServer creates a new server
 */
func NewServer() (*Server, error) {
	return NewServerWithConfig("")
}

/*
 * NewServerWithConfig creates a new server with a specific config path
 */
func NewServerWithConfig(configPath string) (*Server, error) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())

	db := database.NewDatabase()
	dbCfg := cfgMgr.GetDatabaseConfig()
	logger.Info("Database configuration", map[string]interface{}{
		"host":         dbCfg.GetHost(),
		"port":         dbCfg.GetPort(),
		"database":     dbCfg.GetDatabase(),
		"user":         dbCfg.GetUser(),
		"has_password": dbCfg.Password != nil && *dbCfg.Password != "",
	})

	if err := db.Connect(dbCfg); err != nil {
		logger.Warn("Failed to connect to database at startup", map[string]interface{}{
			"error":    err.Error(),
			"host":     dbCfg.GetHost(),
			"port":     dbCfg.GetPort(),
			"database": dbCfg.GetDatabase(),
			"user":     dbCfg.GetUser(),
			"note":     "Server will start but tools may fail. Database connection will be retried on first use.",
		})
	} else {
		logger.Info("Connected to database", map[string]interface{}{
			"host":     dbCfg.GetHost(),
			"database": dbCfg.GetDatabase(),
			"user":     dbCfg.GetUser(),
		})
	}

	serverSettings := cfgMgr.GetServerSettings()

	maxRequestSize := int64(0)
	if serverSettings.MaxRequestSize != nil && *serverSettings.MaxRequestSize > 0 {
		maxRequestSize = int64(*serverSettings.MaxRequestSize)
	}

	mcpServer := mcp.NewServerWithMaxRequestSize(serverSettings.GetName(), serverSettings.GetVersion(), maxRequestSize)

	mwManager := middleware.NewManager(logger)
	setupBuiltInMiddleware(mwManager, cfgMgr, logger)

	toolRegistry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(toolRegistry, db, logger)

	capabilitiesManager := NewCapabilitiesManager(serverSettings.GetName(), serverSettings.GetVersion(), toolRegistry)

	resourcesManager := resources.NewManager(db)
	resources.RegisterAllResources(resourcesManager, db)
	promptsManager := prompts.NewManager(db, logger)
	samplingManager := sampling.NewManager(db, logger)
	samplingManager.SetToolRegistry(toolRegistry) /* Enable tool calling in sampling */
	healthChecker := health.NewChecker(db, logger)
	healthChecker.SetToolRegistry(toolRegistry)
	healthChecker.SetResources(resourcesManager)
	progressTracker := progress.NewTracker()
	batchProcessor := batch.NewProcessor(db, toolRegistry, logger)
	batchProcessor.SetProgressTracker(progressTracker)

	/* Create idempotency cache with 1 hour TTL */
	idempotencyCache := cache.NewIdempotencyCache(time.Hour)

	/* Create metrics collector */
	metricsCollector := metrics.NewCollectorWithDB(db)
	prometheusExporter := metrics.NewPrometheusExporter(metricsCollector)

	var httpServer *HTTPServer
	enableHTTPMetrics := false

	if envEnable := os.Getenv("NEURONMCP_ENABLE_HTTP_METRICS"); envEnable != "" {
		enableHTTPMetrics, _ = strconv.ParseBool(envEnable)
	}

	if enableHTTPMetrics {
		httpAddr := ":8082"
		if envAddr := os.Getenv("NEURONMCP_HTTP_ADDR"); envAddr != "" {
			httpAddr = envAddr
		}
		httpServer = NewHTTPServerWithLogger(httpAddr, prometheusExporter.Handler(), logger)
		logger.Info("HTTP metrics server enabled", map[string]interface{}{
			"address": httpAddr,
		})
	}

	s := &Server{
		mcpServer:           mcpServer,
		db:                  db,
		config:              cfgMgr,
		logger:              logger,
		middleware:          mwManager,
		toolRegistry:        toolRegistry,
		resources:           resourcesManager,
		prompts:             promptsManager,
		sampling:            samplingManager,
		health:              healthChecker,
		progress:            progressTracker,
		batch:               batchProcessor,
		capabilitiesManager: capabilitiesManager,
		idempotencyCache:    idempotencyCache,
		metricsCollector:    metricsCollector,
		prometheusExporter:  prometheusExporter,
		httpServer:          httpServer,
	}

	/* Create HTTP transport if enabled (after server is created so we can pass it) */
	httpTransportCfg := serverSettings.HTTPTransport
	if httpTransportCfg != nil && httpTransportCfg.GetEnabled() {
		httpAddr := httpTransportCfg.GetAddress()
		/* Create adapter for HTTP transport interface */
		httpHandler := &httpRequestHandlerAdapter{server: s}
		s.httpTransport = transport.NewHTTPTransport(httpAddr, mcpServer, mwManager, httpHandler, prometheusExporter.Handler())
		logger.Info("HTTP transport enabled", map[string]interface{}{
			"address": httpAddr,
		})
	}

	s.setupHandlers()
	s.setupExperimentalHandlers()

	return s, nil
}

func (s *Server) setupHandlers() {
	s.setupToolHandlers()
	s.setupResourceHandlers()
	s.setupPromptHandlers()
	s.setupSamplingHandlers()
	s.setupHealthHandlers()
	s.setupProgressHandlers()
	s.setupBatchHandlers()

	/* Set capabilities using capabilities manager */
	if s.capabilitiesManager != nil {
		caps := s.capabilitiesManager.GetServerCapabilities()
		s.mcpServer.SetCapabilities(caps)
	} else {
		/* Fallback to empty capabilities */
		s.mcpServer.SetCapabilities(mcp.ServerCapabilities{
			Tools: mcp.ToolsCapability{
				ListChanged: false,
			},
			Resources: mcp.ResourcesCapability{
				Subscribe:   false,
				ListChanged: false,
			},
			Prompts:  make(map[string]interface{}),
			Sampling: make(map[string]interface{}),
		})
	}
}

/* Start starts the server */
func (s *Server) Start(ctx context.Context) error {
	s.logger.Info("Starting Neurondb MCP server", nil)

	/* Start HTTP metrics server in background (only if enabled) */
	if s.httpServer != nil {
		s.httpServer.Start()
		s.logger.Info("HTTP metrics server started", map[string]interface{}{
			"endpoint": "/metrics",
		})
	}

	/* Start HTTP transport in background (only if enabled) */
	if s.httpTransport != nil {
		go func() {
			if err := s.httpTransport.Start(); err != nil {
				s.logger.Error("HTTP transport failed", err, map[string]interface{}{
					"error": err.Error(),
				})
			}
		}()
		s.logger.Info("HTTP transport started", map[string]interface{}{
			"endpoint": "/mcp",
		})
	}

	/* Run the MCP server - this will block until context is cancelled or EOF */
	err := s.mcpServer.Run(ctx)
	if err != nil && err != context.Canceled {
		s.logger.Warn("MCP server stopped", map[string]interface{}{
			"error": err.Error(),
		})
	}
	return err
}

/* Stop stops the server gracefully */
func (s *Server) Stop() error {
	if s == nil {
		return fmt.Errorf("server instance is nil")
	}
	
	shutdownTimeout := 30 * time.Second
	if s.logger != nil {
		s.logger.Info("Stopping Neurondb MCP server", map[string]interface{}{
			"shutdown_timeout_seconds": shutdownTimeout.Seconds(),
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	/* Collect shutdown errors */
	var shutdownErrors []error

	/* Step 1: Shutdown HTTP transport first (external connections) */
	if s.httpTransport != nil {
		httpTransportCtx, httpTransportCancel := context.WithTimeout(ctx, 5*time.Second)
		if err := s.httpTransport.Shutdown(httpTransportCtx); err != nil {
			shutdownErrors = append(shutdownErrors, fmt.Errorf("HTTP transport shutdown error: %w", err))
			if s.logger != nil {
				s.logger.Warn("HTTP transport shutdown error", map[string]interface{}{
					"error": err.Error(),
				})
			}
		}
		httpTransportCancel()
	}

	/* Step 2: Shutdown HTTP metrics server (external connections) */
	if s.httpServer != nil {
		httpCtx, httpCancel := context.WithTimeout(ctx, 5*time.Second)
		if err := s.httpServer.Shutdown(httpCtx); err != nil {
			shutdownErrors = append(shutdownErrors, fmt.Errorf("HTTP metrics server shutdown error: %w", err))
			if s.logger != nil {
				s.logger.Warn("HTTP metrics server shutdown error", map[string]interface{}{
					"error": err.Error(),
				})
			}
		}
		httpCancel()
	}

	/* Step 3: Close idempotency cache (in-memory resources) */
	if s.idempotencyCache != nil {
		if err := func() error {
			defer func() {
				if r := recover(); r != nil {
					if s.logger != nil {
						s.logger.Warn("Panic during idempotency cache close", map[string]interface{}{
							"panic": r,
						})
					}
				}
			}()
			s.idempotencyCache.Close()
			return nil
		}(); err != nil {
			shutdownErrors = append(shutdownErrors, fmt.Errorf("idempotency cache close error: %w", err))
		}
		s.idempotencyCache = nil
	}

	/* Step 4: Close database connections (external resources) */
	if s.db != nil {
		if err := func() error {
			defer func() {
				if r := recover(); r != nil {
					if s.logger != nil {
						s.logger.Warn("Panic during database close", map[string]interface{}{
							"panic": r,
						})
					}
				}
			}()
			s.db.Close()
			return nil
		}(); err != nil {
			shutdownErrors = append(shutdownErrors, fmt.Errorf("database close error: %w", err))
		}
	}

	/* Step 5: Clean up other resources */
	if s.progress != nil {
		/* Progress tracker cleanup if needed */
		s.progress = nil
	}

	if s.batch != nil {
		/* Batch processor cleanup if needed */
		s.batch = nil
	}

	/* Step 6: Clean up metrics and exporters */
	if s.metricsCollector != nil {
		/* Metrics collector cleanup if needed */
		s.metricsCollector = nil
	}

	if s.prometheusExporter != nil {
		/* Prometheus exporter cleanup if needed */
		s.prometheusExporter = nil
	}

	/* Note: HTTP server and SSE connections are managed by transport manager
	 * If HTTP transport is used, it should be shut down via transport manager:
	 * transportManager.Shutdown(ctx)
	 * Currently, the server uses stdio transport by default, so HTTP/SSE cleanup
	 * is not needed unless HTTP transport is explicitly started.
	 */

	/* Context cancellation will handle goroutine cleanup */
	/* All goroutines should respect context cancellation */

	if len(shutdownErrors) > 0 {
		if s.logger != nil {
			s.logger.Warn("Server shutdown completed with errors", map[string]interface{}{
				"error_count": len(shutdownErrors),
				"errors":      shutdownErrors,
			})
		}
		return fmt.Errorf("server shutdown completed with %d error(s): %v", len(shutdownErrors), shutdownErrors)
	}

	if s.logger != nil {
		s.logger.Info("NeuronMCP server stopped successfully", nil)
	}
	return nil
}

/* GetToolRegistry returns the tool registry (for testing) */
func (s *Server) GetToolRegistry() *tools.ToolRegistry {
	return s.toolRegistry
}

/* GetDatabase returns the database connection (for testing) */
func (s *Server) GetDatabase() *database.Database {
	return s.db
}

/* GetMetricsCollector returns the metrics collector */
func (s *Server) GetMetricsCollector() *metrics.Collector {
	return s.metricsCollector
}

/* GetPrometheusExporter returns the Prometheus exporter */
func (s *Server) GetPrometheusExporter() *metrics.PrometheusExporter {
	return s.prometheusExporter
}

/* GetConfig returns the config manager */
func (s *Server) GetConfig() *config.ConfigManager {
	return s.config
}

/* httpRequestHandlerAdapter adapts Server to HTTPRequestHandler interface */
type httpRequestHandlerAdapter struct {
	server *Server
}

/* HandleHTTPRequest implements HTTPRequestHandler */
func (a *httpRequestHandlerAdapter) HandleHTTPRequest(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	return a.server.HandleHTTPRequest(ctx, mcpReq)
}

/* GetConfig implements HTTPRequestHandler */
func (a *httpRequestHandlerAdapter) GetConfig() transport.HTTPConfigProvider {
	return &configAdapter{a.server.config}
}

/* configAdapter adapts ConfigManager to HTTPConfigProvider interface */
type configAdapter struct {
	*config.ConfigManager
}

/* GetServerSettings implements HTTPConfigProvider */
func (a *configAdapter) GetServerSettings() transport.HTTPServerSettingsProvider {
	return &serverSettingsAdapter{a.ConfigManager.GetServerSettings()}
}

/* serverSettingsAdapter adapts ServerSettings to HTTPServerSettingsProvider interface */
type serverSettingsAdapter struct {
	*config.ServerSettings
}

/* GetMaxRequestSize implements HTTPServerSettingsProvider */
func (a *serverSettingsAdapter) GetMaxRequestSize() *int {
	return a.ServerSettings.MaxRequestSize
}

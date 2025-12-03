package server

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/resources"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

// Server is the main MCP server
type Server struct {
	mcpServer    *mcp.Server
	db           *database.Database
	config       *config.ConfigManager
	logger       *logging.Logger
	middleware   *middleware.Manager
	toolRegistry *tools.ToolRegistry
	resources    *resources.Manager
}

// NewServer creates a new server
func NewServer() (*Server, error) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load("")
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())

	db := database.NewDatabase()
	// Log database config for debugging
	dbCfg := cfgMgr.GetDatabaseConfig()
	logger.Info("Database configuration", map[string]interface{}{
		"host":     dbCfg.GetHost(),
		"port":     dbCfg.GetPort(),
		"database": dbCfg.GetDatabase(),
		"user":     dbCfg.GetUser(),
		"has_password": dbCfg.Password != nil && *dbCfg.Password != "",
	})
	
	// Try to connect, but don't fail server startup if it fails
	// The server can start and tools will fail gracefully with proper error messages
	if err := db.Connect(dbCfg); err != nil {
		logger.Warn("Failed to connect to database at startup", map[string]interface{}{
			"error": err.Error(),
			"host":     dbCfg.GetHost(),
			"port":     dbCfg.GetPort(),
			"database": dbCfg.GetDatabase(),
			"user":     dbCfg.GetUser(),
			"note":  "Server will start but tools may fail. Database connection will be retried on first use.",
		})
		// Continue anyway - tools will handle connection errors gracefully
	} else {
		logger.Info("Connected to database", map[string]interface{}{
			"host":     dbCfg.GetHost(),
			"database": dbCfg.GetDatabase(),
			"user":     dbCfg.GetUser(),
		})
	}

	serverSettings := cfgMgr.GetServerSettings()
	mcpServer := mcp.NewServer(serverSettings.GetName(), serverSettings.GetVersion())

	mwManager := middleware.NewManager(logger)
	setupBuiltInMiddleware(mwManager, cfgMgr, logger)

	toolRegistry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(toolRegistry, db, logger)

	resourcesManager := resources.NewManager(db)

	s := &Server{
		mcpServer:    mcpServer,
		db:           db,
		config:       cfgMgr,
		logger:       logger,
		middleware:   mwManager,
		toolRegistry: toolRegistry,
		resources:    resourcesManager,
	}

	s.setupHandlers()

	return s, nil
}

func (s *Server) setupHandlers() {
	s.setupToolHandlers()
	s.setupResourceHandlers()
	
	// Set capabilities
	s.mcpServer.SetCapabilities(mcp.ServerCapabilities{
		Tools:     make(map[string]interface{}),
		Resources: make(map[string]interface{}),
	})
}

// Start starts the server
func (s *Server) Start(ctx context.Context) error {
	s.logger.Info("Starting Neurondb MCP server", nil)
	// Run the MCP server - this will block until context is cancelled or EOF
	err := s.mcpServer.Run(ctx)
	if err != nil && err != context.Canceled {
		s.logger.Warn("MCP server stopped", map[string]interface{}{
			"error": err.Error(),
		})
	}
	return err
}

// Stop stops the server
func (s *Server) Stop() error {
	s.logger.Info("Stopping Neurondb MCP server", nil)
	s.db.Close()
	return nil
}


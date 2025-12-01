package server

import (
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/middleware/builtin"
)

// setupBuiltInMiddleware registers all built-in middleware
func setupBuiltInMiddleware(mgr *middleware.Manager, cfgMgr *config.ConfigManager, logger *logging.Logger) {
	loggingCfg := cfgMgr.GetLoggingConfig()
	serverCfg := cfgMgr.GetServerSettings()

	// Validation middleware (order: 1)
	mgr.Register(builtin.NewValidationMiddleware())

	// Logging middleware (order: 2)
	mgr.Register(builtin.NewLoggingMiddleware(
		logger,
		loggingCfg.EnableRequestLogging != nil && *loggingCfg.EnableRequestLogging,
		loggingCfg.EnableResponseLogging != nil && *loggingCfg.EnableResponseLogging,
	))

	// Timeout middleware (order: 3) - only if timeout is configured
	if serverCfg.Timeout != nil {
		mgr.Register(builtin.NewTimeoutMiddleware(serverCfg.GetTimeout(), logger))
	}

	// Error handling middleware (order: 100) - always last
	mgr.Register(builtin.NewErrorHandlingMiddleware(
		logger,
		loggingCfg.EnableErrorStack != nil && *loggingCfg.EnableErrorStack,
	))
}


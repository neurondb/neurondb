/*-------------------------------------------------------------------------
 *
 * middleware_setup.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/middleware_setup.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"os"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/audit"
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/middleware/builtin"
	"github.com/neurondb/NeuronMCP/internal/observability"
	"github.com/neurondb/NeuronMCP/internal/reliability"
	"github.com/neurondb/NeuronMCP/internal/safety"
)

/* setupBuiltInMiddleware registers all built-in middleware */
func setupBuiltInMiddleware(mgr *middleware.Manager, cfgMgr *config.ConfigManager, logger *logging.Logger) {
	loggingCfg := cfgMgr.GetLoggingConfig()
	serverCfg := cfgMgr.GetServerSettings()

	/* Correlation ID middleware (order: -1, runs first) */
	mgr.Register(builtin.NewCorrelationMiddleware(logger))

  /* Audit middleware (order: 0, runs early to capture all requests) */
	auditLogger := audit.NewLogger(logger)
	mgr.Register(builtin.NewAuditMiddleware(auditLogger))

  /* Authentication middleware (order: 0) - enabled by default for security */
	authDisabled := strings.ToLower(strings.TrimSpace(os.Getenv("NEURONMCP_AUTH_DISABLED"))) == "true" || os.Getenv("NEURONMCP_AUTH_DISABLED") == "1"
	authConfig := &builtin.AuthConfig{
		Enabled: !authDisabled,
	}
	if authDisabled {
		logger.Warn("Authentication is DISABLED (NEURONMCP_AUTH_DISABLED=true) - insecure for production", nil)
	}
	mgr.Register(builtin.NewAuthMiddleware(authConfig, logger))

  /* Scoped authentication middleware (order: 1, runs after auth) */
	/* Use default scope checker - can be replaced with custom implementation */
	scopeChecker := builtin.NewDefaultScopeChecker()
	mgr.Register(builtin.NewScopedAuthMiddleware(scopeChecker))

  /* Rate limiting middleware (order: 10) - enabled by default for security */
	/* Can be disabled via NEURONMCP_RATE_LIMIT_DISABLED=true environment variable */
	rateLimitDisabled := strings.ToLower(strings.TrimSpace(os.Getenv("NEURONMCP_RATE_LIMIT_DISABLED"))) == "true" || os.Getenv("NEURONMCP_RATE_LIMIT_DISABLED") == "1"
	rateLimitConfig := &builtin.RateLimitConfig{
		Enabled:        !rateLimitDisabled, /* Enabled by default */
		RequestsPerMin: 60,
		BurstSize:      10,
		PerUser:        false,
		PerTool:        false,
	}
	if rateLimitDisabled {
		logger.Warn("Rate limiting is DISABLED (NEURONMCP_RATE_LIMIT_DISABLED=true) - insecure for production", nil)
	}
	mgr.Register(builtin.NewRateLimitMiddleware(rateLimitConfig, logger))

  /* Validation middleware (order: 1) */
	mgr.Register(builtin.NewValidationMiddleware())

  /* Safety middleware (order: 5) - enforce safety modes */
	safetyCfg := cfgMgr.GetSafetyConfig()
	safetyMode := safety.SafetyMode(safetyCfg.DefaultMode)
	if safetyMode == "" {
		safetyMode = safety.SafetyModeReadOnly
	}
	var allowlist *safety.StatementAllowlist
	if safetyMode == safety.SafetyModeAllowlist {
		allowlist = safety.NewStatementAllowlist(safetyCfg.StatementAllowlist)
	}
	safetyManager := safety.NewSafetyManager(safetyMode, allowlist, logger)
	mgr.Register(builtin.NewSafetyMiddleware(safetyManager, logger))

  /* Idempotency middleware (order: 18, after validation, before logging) */
	mgr.Register(builtin.NewIdempotencyMiddleware(logger, true))

  /* Tracing middleware (order: 2) - before logging */
	observabilityCfg := cfgMgr.GetObservabilityConfig()
	enableTracing := observabilityCfg.EnableTracing
	var tracer *observability.Tracer
	var dbTiming *observability.DBTimingTracker
	if enableTracing {
		tracer = observability.NewTracer()
		dbTiming = observability.NewDBTimingTracker(1 * time.Second)
	}
	mgr.Register(builtin.NewTracingMiddleware(tracer, dbTiming, logger, enableTracing))

  /* Logging middleware (order: 2) */
	mgr.Register(builtin.NewLoggingMiddleware(
		logger,
		loggingCfg.EnableRequestLogging != nil && *loggingCfg.EnableRequestLogging,
		loggingCfg.EnableResponseLogging != nil && *loggingCfg.EnableResponseLogging,
	))

  /* Timeout middleware (order: 3) - with per-tool timeout support */
	reliabilityCfg := cfgMgr.GetReliabilityConfig()
	var timeoutManager *reliability.TimeoutManager
	if serverCfg.Timeout != nil || reliabilityCfg.DefaultTimeout > 0 {
		defaultTimeout := serverCfg.GetTimeout()
		if reliabilityCfg.DefaultTimeout > 0 {
			defaultTimeout = time.Duration(reliabilityCfg.DefaultTimeout) * time.Second
		}
		timeoutManager = reliability.NewTimeoutManager(defaultTimeout, reliabilityCfg.ToolTimeouts)
		mgr.Register(builtin.NewTimeoutMiddlewareWithManager(timeoutManager, logger))
	} else {
		/* Fallback to default timeout if nothing configured */
		mgr.Register(builtin.NewTimeoutMiddleware(60*time.Second, logger))
	}

  /* Retry middleware (order: 4) - with exponential backoff and circuit breaker */
	retryConfig := &builtin.RetryConfig{
		Enabled:         false, /* Disabled by default */
		MaxRetries:      3,
		InitialBackoff:  100 * time.Millisecond,
		MaxBackoff:      5 * time.Second,
		BackoffMultiplier: 2.0,
		CircuitBreaker: &builtin.CircuitBreakerConfig{
			Enabled:          false, /* Disabled by default */
			FailureThreshold: 5,
			SuccessThreshold: 2,
			Timeout:          60 * time.Second,
		},
	}
	mgr.Register(builtin.NewRetryMiddleware(retryConfig, logger))

  /* Circuit breaker middleware (order: 6) - for fault tolerance */
	circuitBreakerConfig := middleware.CircuitBreakerConfig{
		FailureThreshold:     5,
		SuccessThreshold:     2,
		Timeout:              60 * time.Second,
		EnablePerToolBreaker: false, /* Disabled by default */
	}
	mgr.Register(builtin.NewCircuitBreakerAdapter(logger, circuitBreakerConfig))

  /* Resource quota middleware (order: 7) - for resource limits */
	resourceQuotaConfig := middleware.ResourceQuotaConfig{
		MaxMemoryMB:      1024,  /* 1GB default */
		MaxVectorDim:     10000,
		MaxBatchSize:     10000,
		MaxConcurrent:    100,
		EnableThrottling: false, /* Disabled by default */
	}
	mgr.Register(builtin.NewResourceQuotaAdapter(logger, resourceQuotaConfig))

  /* Error handling middleware (order: 100) - always last */
	mgr.Register(builtin.NewErrorHandlingMiddleware(
		logger,
		loggingCfg.EnableErrorStack != nil && *loggingCfg.EnableErrorStack,
	))
}


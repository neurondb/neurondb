package initialization

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
	"golang.org/x/crypto/bcrypt"
)

/* Bootstrap handles all application initialization tasks */
type Bootstrap struct {
	queries   *db.Queries
	logger    *logging.Logger
	validator *Validator
}

/* NewBootstrap creates a new bootstrap instance */
func NewBootstrap(queries *db.Queries, logger *logging.Logger) *Bootstrap {
	return &Bootstrap{
		queries:   queries,
		logger:    logger,
		validator: NewValidator(logger),
	}
}

/* Initialize performs all initialization tasks in the correct order */
func (b *Bootstrap) Initialize(ctx context.Context) error {
	metrics := NewBootstrapMetrics()
	defer metrics.Finish()
	defer metrics.LogMetrics(b.logger)

	b.logger.Info("Starting application bootstrap sequence", nil)

	stepStart := time.Now()
	adminUser, err := b.ensureAdminUserWithRetry(ctx)
	if err != nil {
		metrics.TrackStep("admin_user", time.Since(stepStart), false)
		return fmt.Errorf("failed to ensure admin user: %w", err)
	}
	metrics.TrackStep("admin_user", time.Since(stepStart), true)

	validationStart := time.Now()
	if validation := b.validator.ValidateAdminUser(ctx, b.queries); !validation.Valid {
		b.logger.Info("Admin user validation failed", map[string]interface{}{
			"errors": validation.Errors,
		})
		for _, errMsg := range validation.Errors {
			b.logger.Error("Validation error", fmt.Errorf("%s", errMsg), nil)
		}
		metrics.TrackStep("validation", time.Since(validationStart), false)
	} else {
		metrics.TrackStep("validation", time.Since(validationStart), true)
	}

	profileStart := time.Now()
	defaultProfile, err := b.ensureDefaultProfileWithRetry(ctx, adminUser)
	if err != nil {
		metrics.TrackStep("profile", time.Since(profileStart), false)
		return fmt.Errorf("failed to ensure default profile: %w", err)
	}
	metrics.TrackStep("profile", time.Since(profileStart), true)

	if defaultProfile != nil {
		schemaStart := time.Now()
		if err := b.initializeProfileSchemaWithRetry(ctx, defaultProfile); err != nil {
			b.logger.Info("Warning: Failed to initialize profile schema", map[string]interface{}{
				"error": err.Error(),
			})
			metrics.TrackStep("schema", time.Since(schemaStart), false)
		} else {
			metrics.TrackStep("schema", time.Since(schemaStart), true)
		}
	}

	if defaultProfile != nil {
		if validation := b.validator.ValidateProfile(ctx, b.queries); !validation.Valid {
			b.logger.Info("Profile validation failed", map[string]interface{}{
				"errors": validation.Errors,
			})
		}
	}

	if defaultProfile != nil {
		b.verifyConnections(ctx, defaultProfile)
	}

	/* Create demo agent if profile has agent endpoint configured */
	if defaultProfile != nil {
		demoAgentStart := time.Now()
		if err := b.ensureDemoAgentWithRetry(ctx, defaultProfile); err != nil {
			b.logger.Info("Warning: Failed to create demo agent", map[string]interface{}{
				"error": err.Error(),
			})
			metrics.TrackStep("demo_agent", time.Since(demoAgentStart), false)
		} else {
			metrics.TrackStep("demo_agent", time.Since(demoAgentStart), true)
		}
	}

	healthStart := time.Now()
	healthChecker := NewHealthChecker(b.queries, b.logger)
	healthStatus := healthChecker.CheckAll(ctx)
	metrics.TrackStep("health_check", time.Since(healthStart), healthStatus.Overall)
	if !healthStatus.Overall {
		b.logger.Info("Health check completed with issues", map[string]interface{}{
			"status": healthStatus.Status,
			"checks": healthStatus.Checks,
		})
	} else {
		b.logger.Info("Health check passed", map[string]interface{}{
			"status": healthStatus.Status,
		})
	}

	b.logger.Info("Application bootstrap completed successfully", map[string]interface{}{
		"total_duration": metrics.Duration.String(),
	})
	return nil
}

/* ensureAdminUserWithRetry ensures admin user exists with retry logic */
func (b *Bootstrap) ensureAdminUserWithRetry(ctx context.Context) (*db.User, error) {
	var adminUser *db.User
	var err error

	retryFunc := func(ctx context.Context) error {
		adminUser, err = b.ensureAdminUser(ctx)
		return err
	}

	if err := RetryWithBackoff(ctx, b.logger, "ensure admin user", retryFunc); err != nil {
		return nil, err
	}

	return adminUser, nil
}

/* ensureDefaultProfileWithRetry ensures default profile exists with retry logic */
func (b *Bootstrap) ensureDefaultProfileWithRetry(ctx context.Context, adminUser *db.User) (*db.Profile, error) {
	var defaultProfile *db.Profile
	var err error

	retryFunc := func(ctx context.Context) error {
		defaultProfile, err = b.ensureDefaultProfile(ctx, adminUser)
		return err
	}

	if err := RetryWithBackoff(ctx, b.logger, "ensure default profile", retryFunc); err != nil {
		return nil, err
	}

	return defaultProfile, nil
}

/* initializeProfileSchemaWithRetry initializes schema with retry logic */
func (b *Bootstrap) initializeProfileSchemaWithRetry(ctx context.Context, profile *db.Profile) error {
	retryFunc := func(ctx context.Context) error {
		return b.initializeProfileSchema(ctx, profile)
	}

	return RetryWithBackoff(ctx, b.logger, "initialize profile schema", retryFunc)
}

/* ensureAdminUser ensures the default admin user exists */
func (b *Bootstrap) ensureAdminUser(ctx context.Context) (*db.User, error) {
	b.logger.Info("Checking for default admin user", nil)

	adminUser, err := b.queries.GetUserByUsername(ctx, "admin")
	if err != nil {
		/* Admin user doesn't exist, create it */
		b.logger.Info("Creating default admin user", nil)

		/* Get admin password from environment or generate a random one */
		adminPassword := os.Getenv("ADMIN_PASSWORD")
		if adminPassword == "" {
			/* Generate a random password and log it (one-time setup) */
			adminPassword = fmt.Sprintf("admin-%d", time.Now().Unix())
			b.logger.Info("⚠️  ADMIN_PASSWORD not set - using temporary password", map[string]interface{}{
				"password": adminPassword,
				"warning":  "Please set ADMIN_PASSWORD environment variable and change this password immediately",
			})
		}

		passwordHash, hashErr := bcrypt.GenerateFromPassword([]byte(adminPassword), bcrypt.DefaultCost)
		if hashErr != nil {
			b.logger.Error("Failed to hash admin password", hashErr, nil)
			return nil, fmt.Errorf("failed to hash admin password: %w", hashErr)
		}

		adminUser = &db.User{
			Username:     "admin",
			IsAdmin:      true,
			PasswordHash: string(passwordHash),
		}

		if createErr := b.queries.CreateUser(ctx, adminUser); createErr != nil {
			b.logger.Error("Failed to create admin user", createErr, nil)
			return nil, fmt.Errorf("failed to create admin user: %w", createErr)
		}

		b.logger.Info("Default admin user created successfully", map[string]interface{}{
			"username": "admin",
			"user_id":  adminUser.ID,
		})
		return adminUser, nil
	}

	b.logger.Info("Admin user already exists", map[string]interface{}{
		"username": "admin",
		"user_id":  adminUser.ID,
	})
	return adminUser, nil
}

/* ensureDefaultProfile ensures the default profile exists */
func (b *Bootstrap) ensureDefaultProfile(ctx context.Context, adminUser *db.User) (*db.Profile, error) {
	b.logger.Info("Checking for default profile", nil)

	defaultProfile, err := b.queries.GetDefaultProfile(ctx)
	if err != nil || defaultProfile == nil {
		b.logger.Info("Creating default profile", nil)

		/* Determine user ID for profile */
		userID := "default"
		if adminUser != nil {
			userID = adminUser.ID
		}

		/* Get default configuration */
		neurondbDSN := utils.GetDefaultNeuronDBDSN()
		mcpConfig := utils.GetDefaultMCPConfig()
		agentEndpoint := os.Getenv("NEURONAGENT_ENDPOINT")
		agentAPIKey := os.Getenv("NEURONAGENT_API_KEY")

		/* Hash password for admin profile (use same as admin user password) */
		adminPassword := os.Getenv("ADMIN_PASSWORD")
		if adminPassword == "" {
			adminPassword = fmt.Sprintf("admin-%d", time.Now().Unix())
		}
		profilePasswordHash, hashErr := bcrypt.GenerateFromPassword([]byte(adminPassword), bcrypt.DefaultCost)
		if hashErr != nil {
			b.logger.Error("Failed to hash admin profile password", hashErr, nil)
			return nil, fmt.Errorf("failed to hash admin profile password: %w", hashErr)
		}

		defaultProfile = &db.Profile{
			UserID:          userID,
			Name:            "admin",
			ProfileUsername: "admin",
			ProfilePassword: string(profilePasswordHash),
			NeuronDBDSN:     neurondbDSN,
			MCPConfig:       mcpConfig,
			AgentEndpoint:   agentEndpoint,
			AgentAPIKey:     agentAPIKey,
			IsDefault:       true,
		}

		if err := b.queries.CreateProfile(ctx, defaultProfile); err != nil {
			b.logger.Error("Failed to create default profile", err, nil)
			return nil, fmt.Errorf("failed to create default profile: %w", err)
		}

		if err := b.queries.SetDefaultProfile(ctx, defaultProfile.ID); err != nil {
			b.logger.Error("Failed to set default profile", err, nil)
			return nil, fmt.Errorf("failed to set default profile: %w", err)
		}

		/* Create default model configurations */
		if err := utils.CreateDefaultModelsForProfile(ctx, b.queries, defaultProfile.ID); err != nil {
			b.logger.Error("Failed to create default models", err, map[string]interface{}{
				"profile_id": defaultProfile.ID,
			})
		} else {
			b.logger.Info("Default models created successfully", map[string]interface{}{
				"profile_id": defaultProfile.ID,
			})
		}

		b.logger.Info("Default profile created successfully", map[string]interface{}{
			"profile_id": defaultProfile.ID,
			"name":       defaultProfile.Name,
			"user_id":    defaultProfile.UserID,
		})

		/* Reload to get full profile data */
		defaultProfile, _ = b.queries.GetDefaultProfile(ctx)
		return defaultProfile, nil
	}

	b.logger.Info("Default profile already exists", map[string]interface{}{
		"profile_id": defaultProfile.ID,
		"name":       defaultProfile.Name,
	})
	return defaultProfile, nil
}

/* initializeProfileSchema initializes the database schema for a profile */
func (b *Bootstrap) initializeProfileSchema(ctx context.Context, profile *db.Profile) error {
	b.logger.Info("Initializing database schema for profile", map[string]interface{}{
		"profile_id": profile.ID,
		"dsn":        profile.NeuronDBDSN,
	})

	if err := utils.InitSchema(ctx, profile.NeuronDBDSN); err != nil {
		return fmt.Errorf("failed to initialize schema: %w", err)
	}

	b.logger.Info("Schema initialized successfully for profile database", map[string]interface{}{
		"profile_id": profile.ID,
	})
	return nil
}

/* verifyConnections verifies connections for a profile */
func (b *Bootstrap) verifyConnections(ctx context.Context, profile *db.Profile) {
	b.logger.Info("Verifying connections for default profile", map[string]interface{}{
		"profile_id": profile.ID,
	})

	/* Verify PostgreSQL (NeuronDB) connection */
	b.verifyPostgreSQLConnection(ctx, profile)

	/* Verify MCP server connection */
	if profile.MCPConfig != nil {
		b.verifyMCPConnection(ctx, profile)
	}
}

/* verifyPostgreSQLConnection verifies the PostgreSQL connection */
func (b *Bootstrap) verifyPostgreSQLConnection(ctx context.Context, profile *db.Profile) {
	b.logger.Info("Verifying PostgreSQL (NeuronDB) connection", map[string]interface{}{
		"dsn": profile.NeuronDBDSN,
	})

	conn, err := sql.Open("pgx", profile.NeuronDBDSN)
	if err != nil {
		b.logger.Info("⚠ Failed to open PostgreSQL (NeuronDB) connection", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}
	defer conn.Close()

	testCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := conn.PingContext(testCtx); err != nil {
		b.logger.Info("⚠ PostgreSQL (NeuronDB) connection failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	b.logger.Info("✓ PostgreSQL (NeuronDB) connection verified", map[string]interface{}{
		"dsn": profile.NeuronDBDSN,
	})

	/* Test NeuronDB extension (optional - may not be installed) */
	var version string
	if err := conn.QueryRowContext(testCtx, "SELECT neurondb.version()").Scan(&version); err == nil {
		b.logger.Info("✓ NeuronDB extension verified", map[string]interface{}{
			"version": version,
		})
	} else {
		b.logger.Info("⚠ NeuronDB extension not found (database may not have extension installed)", nil)
	}
}

/* verifyMCPConnection verifies the MCP server connection */
func (b *Bootstrap) verifyMCPConnection(ctx context.Context, profile *db.Profile) {
	if profile.MCPConfig == nil {
		b.logger.Info("MCP connection verification skipped - no MCP config", map[string]interface{}{
			"profile_id": profile.ID,
		})
		return
	}

	b.logger.Info("Verifying MCP connection", map[string]interface{}{
		"profile_id": profile.ID,
		"command":    profile.MCPConfig["command"],
	})

	/* Create MCP config from profile */
	defaultCmd := utils.FindNeuronMCPBinary()
	if defaultCmd == "" {
		defaultCmd = "neurondb-mcp"
	}
	mcpConfig := mcp.MCPConfig{
		Command: defaultCmd,
		Args:    []string{},
		Env:     make(map[string]string),
	}

	/* Set default database environment variables from profile's NeuronDB DSN */
	if profile.NeuronDBDSN != "" {
		mcpConfig.Env["NEURONDB_CONNECTION_STRING"] = profile.NeuronDBDSN
	}

	/* Override with profile MCP config */
	if cmd, ok := profile.MCPConfig["command"].(string); ok && cmd != "" {
		mcpConfig.Command = cmd
	}
	if args, ok := profile.MCPConfig["args"].([]interface{}); ok {
		for _, arg := range args {
			if s, ok := arg.(string); ok {
				mcpConfig.Args = append(mcpConfig.Args, s)
			}
		}
	}
	if env, ok := profile.MCPConfig["env"].(map[string]interface{}); ok {
		for k, v := range env {
			if s, ok := v.(string); ok {
				mcpConfig.Env[k] = s
			}
		}
	}

	/* Retry logic with exponential backoff */
	maxRetries := 3
	baseDelay := 2 * time.Second
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			delay := baseDelay * time.Duration(1<<uint(attempt-1)) // Exponential backoff
			b.logger.Info("Retrying MCP connection verification", map[string]interface{}{
				"attempt": attempt + 1,
				"delay":    delay.String(),
			})
			select {
			case <-ctx.Done():
				b.logger.Info("MCP connection verification cancelled", nil)
				return
			case <-time.After(delay):
			}
		}

		/* Create a temporary client for verification */
		testClient, err := mcp.NewClient(mcpConfig)
		if err != nil {
			lastErr = err
			b.logger.Info("Failed to create MCP client for verification", map[string]interface{}{
				"attempt": attempt + 1,
				"error":   err.Error(),
			})
			continue
		}

		/* Verify client is alive */
		if !testClient.IsAlive() {
			testClient.Close()
			lastErr = fmt.Errorf("MCP client is not alive")
			b.logger.Info("MCP client not alive", map[string]interface{}{
				"attempt": attempt + 1,
			})
			continue
		}

		/* Try to list tools as a health check */
		_, err = testClient.ListTools()

		if err != nil {
			testClient.Close()
			lastErr = err
			b.logger.Info("Failed to list MCP tools during verification", map[string]interface{}{
				"attempt": attempt + 1,
				"error":   err.Error(),
			})
			continue
		}

		/* Success - close test client and log success */
		testClient.Close()
		b.logger.Info("✓ MCP connection verified successfully", map[string]interface{}{
			"profile_id": profile.ID,
			"command":    mcpConfig.Command,
			"attempts":   attempt + 1,
		})
		return
	}

	/* All retries failed */
	b.logger.Info("⚠ MCP connection verification failed after retries", map[string]interface{}{
		"profile_id": profile.ID,
		"command":    mcpConfig.Command,
		"error":      lastErr.Error(),
		"attempts":   maxRetries,
	})
}

/* ensureDemoAgentWithRetry ensures demo agent exists with retry logic */
func (b *Bootstrap) ensureDemoAgentWithRetry(ctx context.Context, profile *db.Profile) error {
	retryFunc := func(ctx context.Context) error {
		return b.ensureDemoAgent(ctx, profile)
	}

	return RetryWithBackoff(ctx, b.logger, "ensure demo agent", retryFunc)
}

/* ensureDemoAgent ensures a demo agent exists for the profile */
func (b *Bootstrap) ensureDemoAgent(ctx context.Context, profile *db.Profile) error {
	/* Check if profile has agent endpoint configured */
	if profile.AgentEndpoint == "" {
		b.logger.Info("Skipping demo agent creation - no agent endpoint configured", map[string]interface{}{
			"profile_id": profile.ID,
		})
		return nil
	}

	b.logger.Info("Checking for demo agent", map[string]interface{}{
		"profile_id":     profile.ID,
		"agent_endpoint": profile.AgentEndpoint,
	})

	/* Create agent client */
	agentClient := agent.NewClient(profile.AgentEndpoint, profile.AgentAPIKey)

	/* Check if NeuronAgent is accessible */
	healthCtx, healthCancel := context.WithTimeout(ctx, 5*time.Second)
	defer healthCancel()

	/* Try to list agents as a health check */
	existingAgents, err := agentClient.ListAgents(healthCtx)
	if err != nil {
		b.logger.Info("⚠ NeuronAgent not accessible, skipping demo agent creation", map[string]interface{}{
			"profile_id":     profile.ID,
			"agent_endpoint": profile.AgentEndpoint,
			"error":          err.Error(),
		})
		return nil /* Don't fail bootstrap if agent server is not available */
	}

	/* Check if demo agent already exists */
	demoAgentName := "demo-agent"
	for _, existingAgent := range existingAgents {
		if existingAgent.Name == demoAgentName {
			b.logger.Info("Demo agent already exists", map[string]interface{}{
				"profile_id": profile.ID,
				"agent_name": demoAgentName,
				"agent_id":   existingAgent.ID,
			})
			return nil
		}
	}

	/* Create demo agent */
	b.logger.Info("Creating demo agent", map[string]interface{}{
		"profile_id":     profile.ID,
		"agent_endpoint": profile.AgentEndpoint,
		"agent_name":     demoAgentName,
	})

	createReq := agent.CreateAgentRequest{
		Name:         demoAgentName,
		Description:  "Demo agent for NeuronDesktop - a helpful assistant that can answer questions and help with tasks using NeuronAgent",
		SystemPrompt: "You are a helpful, harmless, and honest assistant. Answer questions accurately and helpfully. You can use tools to help users with their tasks.",
		ModelName:    "gpt-4",
		EnabledTools: []string{"sql", "http"},
		Config: map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  1000,
		},
	}

	createCtx, createCancel := context.WithTimeout(ctx, 10*time.Second)
	defer createCancel()

	createdAgent, err := agentClient.CreateAgent(createCtx, createReq)
	if err != nil {
		return fmt.Errorf("failed to create demo agent: %w", err)
	}

	b.logger.Info("Demo agent created successfully", map[string]interface{}{
		"profile_id": profile.ID,
		"agent_name": demoAgentName,
		"agent_id":   createdAgent.ID,
	})

	return nil
}

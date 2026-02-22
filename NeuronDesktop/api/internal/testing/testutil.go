package testing

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"regexp"
	"testing"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"golang.org/x/crypto/bcrypt"
)

/* safeIdentifierRegex matches valid SQL identifiers (alphanumeric, underscore, must start with letter or underscore) */
var safeIdentifierRegex = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

/* TestDB holds test database connection */
type TestDB struct {
	DB      *sql.DB
	Queries *db.Queries
}

/* SetupTestDB creates a test database connection */
func SetupTestDB(t *testing.T) *TestDB {
	t.Helper()

	/* Use test database from environment or default */
	testDBName := os.Getenv("TEST_DB_NAME")
	if testDBName == "" {
		testDBName = "neurondesk_test"
	}

	dsn := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		getEnv("TEST_DB_HOST", "localhost"),
		getEnv("TEST_DB_PORT", "5432"),
		getEnv("TEST_DB_USER", "neurondesk"),
		getEnv("TEST_DB_PASSWORD", "neurondesk"),
		testDBName,
	)

	/* Connect to postgres database first to create test database */
	postgresDSN := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=postgres sslmode=disable",
		getEnv("TEST_DB_HOST", "localhost"),
		getEnv("TEST_DB_PORT", "5432"),
		getEnv("TEST_DB_USER", "neurondesk"),
		getEnv("TEST_DB_PASSWORD", "neurondesk"),
	)

	postgresDB, err := sql.Open("pgx", postgresDSN)
	if err != nil {
		t.Fatalf("Failed to connect to postgres: %v", err)
	}
	defer postgresDB.Close()

	/* Skip test when PostgreSQL is not running (e.g. CI without postgres) */
	pingCtx, pingCancel := context.WithTimeout(context.Background(), 2*time.Second)
	if err := postgresDB.PingContext(pingCtx); err != nil {
		pingCancel()
		t.Skipf("PostgreSQL not available (skipping integration test): %v", err)
	}
	pingCancel()

	/* Validate test database name to prevent SQL injection */
	if !safeIdentifierRegex.MatchString(testDBName) {
		t.Fatalf("Invalid test database name (must match ^[a-zA-Z_][a-zA-Z0-9_]*$): %s", testDBName)
	}

	/* Create test database if it doesn't exist - use parameterized query for existence check */
	checkCtx, checkCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer checkCancel()
	var exists int
	err = postgresDB.QueryRowContext(checkCtx, "SELECT 1 FROM pg_database WHERE datname = $1", testDBName).Scan(&exists)
	if err != nil && err != sql.ErrNoRows {
		t.Fatalf("Failed to check test database existence: %v", err)
	}
	if err == sql.ErrNoRows || exists != 1 {
		/* Database doesn't exist, create it (testDBName already validated as safe identifier) */
		_, err = postgresDB.Exec(fmt.Sprintf("CREATE DATABASE %s", testDBName))
		if err != nil {
			t.Fatalf("Failed to create test database: %v", err)
		}
	}

	/* Connect to test database */
	testDB, err := sql.Open("pgx", dsn)
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := testDB.PingContext(ctx); err != nil {
		testDB.Close()
		t.Fatalf("Failed to ping test database: %v", err)
	}

	/* Run migrations */
	if err := runMigrations(testDB); err != nil {
		testDB.Close()
		t.Fatalf("Failed to run migrations: %v", err)
	}

	queries := db.NewQueries(testDB)

	return &TestDB{
		DB:      testDB,
		Queries: queries,
	}
}

/* CleanupTestDB cleans up test database */
func (tdb *TestDB) CleanupTestDB(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	/* Truncate all tables - only allow tables in the whitelist to prevent SQL injection */
	allowedTables := map[string]bool{
		"mcp_chat_messages": true,
		"mcp_chat_threads":  true,
		"model_configs":     true,
		"api_keys":          true,
		"request_logs":      true,
		"profiles":          true,
		"users":             true,
		"app_settings":      true,
	}
	tables := []string{
		"mcp_chat_messages",
		"mcp_chat_threads",
		"model_configs",
		"api_keys",
		"request_logs",
		"profiles",
		"users",
		"app_settings",
	}

	for _, table := range tables {
		if !allowedTables[table] || !safeIdentifierRegex.MatchString(table) {
			continue
		}
		_, err := tdb.DB.ExecContext(ctx, fmt.Sprintf("TRUNCATE TABLE %s CASCADE", table))
		if err != nil {
			/* Table might not exist, ignore */
			t.Logf("Warning: Failed to truncate %s: %v", table, err)
		}
	}

	tdb.DB.Close()
}

/* CreateTestUser creates a test user */
func CreateTestUser(ctx context.Context, queries *db.Queries, username, password string) (*db.User, error) {
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	user := &db.User{
		Username:     username,
		PasswordHash: string(passwordHash),
		IsAdmin:      false,
	}

	if err := queries.CreateUser(ctx, user); err != nil {
		return nil, err
	}

	return user, nil
}

/* CreateTestAdmin creates a test admin user */
func CreateTestAdmin(ctx context.Context, queries *db.Queries, username, password string) (*db.User, error) {
	/* Create user with admin flag set */
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	user := &db.User{
		Username:     username,
		PasswordHash: string(passwordHash),
		IsAdmin:      true,
	}

	if err := queries.CreateUser(ctx, user); err != nil {
		return nil, err
	}

	return user, nil
}

/* CreateTestProfile creates a test profile */
func CreateTestProfile(ctx context.Context, queries *db.Queries, userID string) (*db.Profile, error) {
	profile := &db.Profile{
		UserID:      userID,
		Name:        "Test Profile",
		NeuronDBDSN: "host=localhost port=5432 user=neurondb dbname=neurondb",
		MCPConfig: map[string]interface{}{
			"command": "echo",
			"args":    []string{"test"},
		},
		IsDefault: false,
	}

	if err := queries.CreateProfile(ctx, profile); err != nil {
		return nil, err
	}

	return profile, nil
}

/* CreateTestProfileWithPassword creates a test profile with credentials */
func CreateTestProfileWithPassword(ctx context.Context, queries *db.Queries, userID, username, password string) (*db.Profile, error) {
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	profile := &db.Profile{
		UserID:          userID,
		Name:            "Test Profile",
		ProfileUsername: username,
		ProfilePassword: string(passwordHash),
		NeuronDBDSN:     "host=localhost port=5432 user=neurondb dbname=neurondb",
		MCPConfig: map[string]interface{}{
			"command": "echo",
			"args":    []string{"test"},
		},
		IsDefault: false,
	}

	if err := queries.CreateProfile(ctx, profile); err != nil {
		return nil, err
	}

	return profile, nil
}

/* runMigrations runs database migrations */
func runMigrations(db *sql.DB) error {
	migrations := []string{
		`CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`,
		`CREATE TABLE IF NOT EXISTS users (
			id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
			username TEXT NOT NULL UNIQUE,
			password_hash TEXT NOT NULL,
			is_admin BOOLEAN NOT NULL DEFAULT false,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
		`CREATE TABLE IF NOT EXISTS profiles (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			name TEXT NOT NULL,
			user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			profile_username TEXT,
			profile_password_hash TEXT,
			mcp_config JSONB,
			neurondb_dsn TEXT NOT NULL,
			agent_endpoint TEXT,
			agent_api_key TEXT,
			default_collection TEXT,
			is_default BOOLEAN NOT NULL DEFAULT false,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
		`CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);`,
		`CREATE TABLE IF NOT EXISTS api_keys (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			key_hash TEXT NOT NULL,
			key_prefix TEXT NOT NULL UNIQUE,
			user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
			rate_limit INTEGER NOT NULL DEFAULT 100,
			last_used_at TIMESTAMPTZ,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
		`CREATE TABLE IF NOT EXISTS request_logs (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
			endpoint TEXT NOT NULL,
			method TEXT NOT NULL,
			request_body JSONB,
			response_body JSONB,
			status_code INTEGER NOT NULL,
			duration_ms INTEGER NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
		`CREATE TABLE IF NOT EXISTS model_configs (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
			model_provider TEXT NOT NULL,
			model_name TEXT NOT NULL,
			api_key TEXT,
			base_url TEXT,
			is_default BOOLEAN NOT NULL DEFAULT false,
			is_free BOOLEAN NOT NULL DEFAULT false,
			metadata JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			UNIQUE(profile_id, model_provider, model_name)
		);`,
		`CREATE TABLE IF NOT EXISTS app_settings (
			key TEXT PRIMARY KEY,
			value JSONB NOT NULL,
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
		`CREATE TABLE IF NOT EXISTS mcp_chat_threads (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
			title TEXT NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
		`CREATE TABLE IF NOT EXISTS mcp_chat_messages (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			thread_id UUID NOT NULL REFERENCES mcp_chat_threads(id) ON DELETE CASCADE,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			tool_name TEXT,
			data JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);`,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for _, migration := range migrations {
		if _, err := db.ExecContext(ctx, migration); err != nil {
			return fmt.Errorf("migration failed: %w", err)
		}
	}

	return nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

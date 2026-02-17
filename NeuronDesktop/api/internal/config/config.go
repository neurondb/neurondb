package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

/* Config holds application configuration */
type Config struct {
	Database DatabaseConfig
	Server   ServerConfig
	Logging  LoggingConfig
	CORS     CORSConfig
	Auth     AuthConfig
	Session  SessionConfig
	Security SecurityConfig
	Redis    RedisConfig
}

/* DatabaseConfig holds database configuration */
type DatabaseConfig struct {
	Host            string
	Port            string
	User            string
	Password        string
	Name            string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
}

/* ServerConfig holds server configuration */
type ServerConfig struct {
	Host         string
	Port         string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

/* LoggingConfig holds logging configuration */
type LoggingConfig struct {
	Level  string
	Format string
	Output string
}

/* CORSConfig holds CORS configuration */
type CORSConfig struct {
	AllowedOrigins []string
	AllowedMethods []string
	AllowedHeaders []string
}

/* AuthConfig holds authentication configuration */
type AuthConfig struct {
	Mode            string // "oidc", "jwt", "hybrid"
	OIDC            OIDCConfig
	JWTSecret       string
	EnableLocalAuth bool
}

/* OIDCConfig holds OIDC configuration */
type OIDCConfig struct {
	IssuerURL    string
	ClientID     string
	ClientSecret string
	RedirectURL  string
	Scopes       []string
}

/* SessionConfig holds session configuration */
type SessionConfig struct {
	CookieDomain   string
	CookieSecure   bool
	CookieSameSite string // "Lax", "Strict", "None"
	AccessTTL      time.Duration
	RefreshTTL     time.Duration
}

/* SecurityConfig holds security-related configuration */
type SecurityConfig struct {
	EnableSQLConsole      bool   // Enable arbitrary SQL execution endpoint (default: false)
	LogRetentionDays      int    // Number of days to retain request logs (default: 30)
	MaxRequestSize        int64  // Maximum request body size in bytes (default: 10MB)
	EnablePIISanitization bool   // Enable PII sanitization in logs (default: true)
}

/* RedisConfig holds Redis configuration for CSRF token storage */
type RedisConfig struct {
	Enabled  bool   // Enable Redis for CSRF token storage (default: false)
	Host     string // Redis host (default: localhost)
	Port     string // Redis port (default: 6379)
	Password string // Redis password (default: empty)
	DB       int    // Redis database number (default: 0)
}

/* Load loads configuration from environment variables and optionally from a config file */
func Load() *Config {
	return LoadFromFile("")
}

/* LoadFromFile loads configuration from a file (if provided) and environment variables (which override file values) */
func LoadFromFile(configPath string) *Config {
	var fileCfg *Config

	/* Load from file if provided */
	if configPath != "" {
		if fc, err := loadFromFile(configPath); err == nil {
			fileCfg = fc
		}
	}

	/* Load from environment variables */
	envCfg := &Config{
		Database: DatabaseConfig{
			Host:            getEnv("DB_HOST", "localhost"),
			Port:            getEnv("DB_PORT", "5432"),
			User:            getEnv("DB_USER", "neurondesk"),
			Password:        getEnv("DB_PASSWORD", "neurondesk"),
			Name:            getEnv("DB_NAME", "neurondesk"),
			MaxOpenConns:    getEnvInt("DB_MAX_OPEN_CONNS", 25),
			MaxIdleConns:    getEnvInt("DB_MAX_IDLE_CONNS", 5),
			ConnMaxLifetime: getEnvDuration("DB_CONN_MAX_LIFETIME", 5*time.Minute),
		},
		Server: ServerConfig{
			Host:         getEnv("SERVER_HOST", "0.0.0.0"),
			Port:         getEnv("SERVER_PORT", "8081"),
			ReadTimeout:  getEnvDuration("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout: getEnvDuration("SERVER_WRITE_TIMEOUT", 30*time.Second),
		},
		Logging: LoggingConfig{
			Level:  getEnv("LOG_LEVEL", "info"),
			Format: getEnv("LOG_FORMAT", "json"),
			Output: getEnv("LOG_OUTPUT", "stdout"),
		},
		CORS: CORSConfig{
			AllowedOrigins: getEnvSlice("CORS_ALLOWED_ORIGINS", []string{"*"}),
			AllowedMethods: getEnvSlice("CORS_ALLOWED_METHODS", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
			AllowedHeaders: getEnvSlice("CORS_ALLOWED_HEADERS", []string{"Content-Type", "Authorization"}),
		},
		Auth: AuthConfig{
			Mode:            getEnv("AUTH_MODE", "oidc"),
			JWTSecret:       getEnv("JWT_SECRET", ""),
			EnableLocalAuth: getEnv("ENABLE_LOCAL_AUTH", "true") == "true",
			OIDC: OIDCConfig{
				IssuerURL:    getEnv("OIDC_ISSUER_URL", ""),
				ClientID:     getEnv("OIDC_CLIENT_ID", ""),
				ClientSecret: getEnv("OIDC_CLIENT_SECRET", ""),
				RedirectURL:  getEnv("OIDC_REDIRECT_URL", ""),
				Scopes:       getEnvSlice("OIDC_SCOPES", []string{"openid", "profile", "email"}),
			},
		},
		Session: SessionConfig{
			CookieDomain:   getEnv("SESSION_COOKIE_DOMAIN", ""),
			CookieSecure:   getEnv("SESSION_SECURE", "true") == "true",
			CookieSameSite: getEnv("SESSION_SAMESITE", "Lax"),
			AccessTTL:      getEnvDuration("SESSION_ACCESS_TTL", 15*time.Minute),
			RefreshTTL:     getEnvDuration("SESSION_REFRESH_TTL", 30*24*time.Hour),
		},
		Security: SecurityConfig{
			EnableSQLConsole:      getEnv("ENABLE_SQL_CONSOLE", "false") == "true",
			LogRetentionDays:      getEnvInt("LOG_RETENTION_DAYS", 30),
			MaxRequestSize:        int64(getEnvInt("MAX_REQUEST_SIZE_MB", 10)) * 1024 * 1024,
			EnablePIISanitization: getEnv("ENABLE_PII_SANITIZATION", "true") == "true",
		},
		Redis: RedisConfig{
			Enabled:  getEnv("REDIS_ENABLED", "false") == "true",
			Host:     getEnv("REDIS_HOST", "localhost"),
			Port:     getEnv("REDIS_PORT", "6379"),
			Password: getEnv("REDIS_PASSWORD", ""),
			DB:       getEnvInt("REDIS_DB", 0),
		},
	}

	/* Merge file config with env config (env takes precedence) */
	if fileCfg != nil {
		return mergeConfig(fileCfg, envCfg)
	}

	return envCfg
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getEnvSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		parts := []string{}
		for _, part := range splitString(value, ",") {
			parts = append(parts, trimSpace(part))
		}
		if len(parts) > 0 {
			return parts
		}
	}
	return defaultValue
}

func splitString(s, sep string) []string {
	parts := []string{}
	current := ""
	for _, char := range s {
		if string(char) == sep {
			if current != "" {
				parts = append(parts, current)
				current = ""
			}
		} else {
			current += string(char)
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	return parts
}

func trimSpace(s string) string {
	start := 0
	end := len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t') {
		end--
	}
	return s[start:end]
}

/* loadFromFile loads configuration from a YAML or JSON file */
func loadFromFile(configPath string) (*Config, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	cfg := &Config{}

	/* Try YAML first, then JSON */
	ext := strings.ToLower(filepath.Ext(configPath))
	if ext == ".yaml" || ext == ".yml" {
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("failed to parse YAML config: %w", err)
		}
	} else if ext == ".json" {
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("failed to parse JSON config: %w", err)
		}
	} else {
		/* Try both formats */
		if err := yaml.Unmarshal(data, cfg); err != nil {
			if err2 := json.Unmarshal(data, cfg); err2 != nil {
				return nil, fmt.Errorf("failed to parse config (tried YAML and JSON): %w, %w", err, err2)
			}
		}
	}

	return cfg, nil
}

/* mergeConfig merges file config with environment variables (env takes precedence) */
func mergeConfig(fileCfg, envCfg *Config) *Config {
	merged := *fileCfg

	/* Database */
	if envCfg.Database.Host != "" {
		merged.Database.Host = envCfg.Database.Host
	}
	if envCfg.Database.Port != "" {
		merged.Database.Port = envCfg.Database.Port
	}
	if envCfg.Database.User != "" {
		merged.Database.User = envCfg.Database.User
	}
	if envCfg.Database.Password != "" {
		merged.Database.Password = envCfg.Database.Password
	}
	if envCfg.Database.Name != "" {
		merged.Database.Name = envCfg.Database.Name
	}
	if envCfg.Database.MaxOpenConns > 0 {
		merged.Database.MaxOpenConns = envCfg.Database.MaxOpenConns
	}
	if envCfg.Database.MaxIdleConns > 0 {
		merged.Database.MaxIdleConns = envCfg.Database.MaxIdleConns
	}
	if envCfg.Database.ConnMaxLifetime > 0 {
		merged.Database.ConnMaxLifetime = envCfg.Database.ConnMaxLifetime
	}

	/* Server */
	if envCfg.Server.Host != "" {
		merged.Server.Host = envCfg.Server.Host
	}
	if envCfg.Server.Port != "" {
		merged.Server.Port = envCfg.Server.Port
	}
	if envCfg.Server.ReadTimeout > 0 {
		merged.Server.ReadTimeout = envCfg.Server.ReadTimeout
	}
	if envCfg.Server.WriteTimeout > 0 {
		merged.Server.WriteTimeout = envCfg.Server.WriteTimeout
	}

	/* Logging */
	if envCfg.Logging.Level != "" {
		merged.Logging.Level = envCfg.Logging.Level
	}
	if envCfg.Logging.Format != "" {
		merged.Logging.Format = envCfg.Logging.Format
	}
	if envCfg.Logging.Output != "" {
		merged.Logging.Output = envCfg.Logging.Output
	}

	/* CORS */
	if len(envCfg.CORS.AllowedOrigins) > 0 {
		merged.CORS.AllowedOrigins = envCfg.CORS.AllowedOrigins
	}
	if len(envCfg.CORS.AllowedMethods) > 0 {
		merged.CORS.AllowedMethods = envCfg.CORS.AllowedMethods
	}
	if len(envCfg.CORS.AllowedHeaders) > 0 {
		merged.CORS.AllowedHeaders = envCfg.CORS.AllowedHeaders
	}

	/* Auth */
	if envCfg.Auth.Mode != "" {
		merged.Auth.Mode = envCfg.Auth.Mode
	}
	if envCfg.Auth.JWTSecret != "" {
		merged.Auth.JWTSecret = envCfg.Auth.JWTSecret
	}
	merged.Auth.EnableLocalAuth = envCfg.Auth.EnableLocalAuth
	if envCfg.Auth.OIDC.IssuerURL != "" {
		merged.Auth.OIDC.IssuerURL = envCfg.Auth.OIDC.IssuerURL
	}
	if envCfg.Auth.OIDC.ClientID != "" {
		merged.Auth.OIDC.ClientID = envCfg.Auth.OIDC.ClientID
	}
	if envCfg.Auth.OIDC.ClientSecret != "" {
		merged.Auth.OIDC.ClientSecret = envCfg.Auth.OIDC.ClientSecret
	}
	if envCfg.Auth.OIDC.RedirectURL != "" {
		merged.Auth.OIDC.RedirectURL = envCfg.Auth.OIDC.RedirectURL
	}
	if len(envCfg.Auth.OIDC.Scopes) > 0 {
		merged.Auth.OIDC.Scopes = envCfg.Auth.OIDC.Scopes
	}

	/* Session */
	if envCfg.Session.CookieDomain != "" {
		merged.Session.CookieDomain = envCfg.Session.CookieDomain
	}
	merged.Session.CookieSecure = envCfg.Session.CookieSecure
	if envCfg.Session.CookieSameSite != "" {
		merged.Session.CookieSameSite = envCfg.Session.CookieSameSite
	}
	if envCfg.Session.AccessTTL > 0 {
		merged.Session.AccessTTL = envCfg.Session.AccessTTL
	}
	if envCfg.Session.RefreshTTL > 0 {
		merged.Session.RefreshTTL = envCfg.Session.RefreshTTL
	}

	/* Security */
	merged.Security.EnableSQLConsole = envCfg.Security.EnableSQLConsole

	return &merged
}

/* DSN returns the database connection string (do not log this; use SanitizedDSN for logs) */
func (c *DatabaseConfig) DSN() string {
	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		c.Host, c.Port, c.User, c.Password, c.Name)
}

/* SanitizedDSN returns a DSN with password redacted for safe logging */
func (c *DatabaseConfig) SanitizedDSN() string {
	return fmt.Sprintf("host=%s port=%s user=%s password=*** dbname=%s sslmode=disable",
		c.Host, c.Port, c.User, c.Name)
}

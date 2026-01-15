/*-------------------------------------------------------------------------
 *
 * env.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/config/env.go
 *
 *-------------------------------------------------------------------------
 */

package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

/* LoadFromEnv loads configuration from environment variables */
func LoadFromEnv(cfg *Config) error {
	/* Server config */
	if host := os.Getenv("SERVER_HOST"); host != "" {
		cfg.Server.Host = host
	}
	if port := os.Getenv("SERVER_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			cfg.Server.Port = p
		}
	}
	if timeout := os.Getenv("SERVER_READ_TIMEOUT"); timeout != "" {
		if d, err := time.ParseDuration(timeout); err == nil {
			cfg.Server.ReadTimeout = d
		}
	}
	if timeout := os.Getenv("SERVER_WRITE_TIMEOUT"); timeout != "" {
		if d, err := time.ParseDuration(timeout); err == nil {
			cfg.Server.WriteTimeout = d
		}
	}

	/* Database config */
	if host := os.Getenv("DB_HOST"); host != "" {
		cfg.Database.Host = host
	}
	if port := os.Getenv("DB_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			cfg.Database.Port = p
		}
	}
	if db := os.Getenv("DB_NAME"); db != "" {
		cfg.Database.Database = db
	}
	if user := os.Getenv("DB_USER"); user != "" {
		cfg.Database.User = user
	}
	if pass := os.Getenv("DB_PASSWORD"); pass != "" {
		cfg.Database.Password = pass
	}
	if maxOpen := os.Getenv("DB_MAX_OPEN_CONNS"); maxOpen != "" {
		if n, err := strconv.Atoi(maxOpen); err == nil {
			cfg.Database.MaxOpenConns = n
		}
	}
	if maxIdle := os.Getenv("DB_MAX_IDLE_CONNS"); maxIdle != "" {
		if n, err := strconv.Atoi(maxIdle); err == nil {
			cfg.Database.MaxIdleConns = n
		}
	}
	if lifetime := os.Getenv("DB_CONN_MAX_LIFETIME"); lifetime != "" {
		if d, err := time.ParseDuration(lifetime); err == nil {
			cfg.Database.ConnMaxLifetime = d
		}
	}

	/* Auth config */
	if header := os.Getenv("AUTH_API_KEY_HEADER"); header != "" {
		cfg.Auth.APIKeyHeader = header
	}

	/* Logging config */
	if level := os.Getenv("LOG_LEVEL"); level != "" {
		cfg.Logging.Level = level
	}
	if format := os.Getenv("LOG_FORMAT"); format != "" {
		cfg.Logging.Format = format
	}

	/* Distributed config */
	if enabled := os.Getenv("DISTRIBUTED_ENABLED"); enabled == "true" {
		cfg.Distributed.Enabled = true
	}
	if addr := os.Getenv("NODE_ADDRESS"); addr != "" {
		cfg.Distributed.NodeAddress = addr
	}
	if port := os.Getenv("NODE_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			cfg.Distributed.NodePort = p
		}
	}
	if timeout := os.Getenv("RPC_TIMEOUT"); timeout != "" {
		if d, err := time.ParseDuration(timeout); err == nil {
			cfg.Distributed.RPCTimeout = d
		}
	}

	/* Cache config */
	if enabled := os.Getenv("DISTRIBUTED_CACHE_ENABLED"); enabled == "true" {
		cfg.Cache.Enabled = true
	}
	if ttl := os.Getenv("CACHE_TTL"); ttl != "" {
		if d, err := time.ParseDuration(ttl); err == nil {
			cfg.Cache.TTL = d
		}
	}
	if syncInterval := os.Getenv("CACHE_SYNC_INTERVAL"); syncInterval != "" {
		if d, err := time.ParseDuration(syncInterval); err == nil {
			cfg.Cache.SyncInterval = d
		}
	}

	/* Multimodal config */
	if provider := os.Getenv("OCR_PROVIDER"); provider != "" {
		cfg.Multimodal.OCRProvider = provider
	}

	return nil
}

/* GetEnvOrDefault gets environment variable or returns default */
func GetEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

/* GetEnvIntOrDefault gets environment variable as int or returns default */
func GetEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if n, err := strconv.Atoi(value); err == nil {
			return n
		}
	}
	return defaultValue
}

/* GetEnvDurationOrDefault gets environment variable as duration or returns default */
func GetEnvDurationOrDefault(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if d, err := time.ParseDuration(value); err == nil {
			return d
		}
	}
	return defaultValue
}

/* ValidateEnv validates required environment variables */
func ValidateEnv() error {
	required := []string{"DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"}
	for _, key := range required {
		if os.Getenv(key) == "" {
			return fmt.Errorf("required environment variable %s is not set", key)
		}
	}
	return nil
}

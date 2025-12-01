package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/config"
)

func main() {
	var (
		orgID     = flag.String("org", "", "Organization ID")
		userID    = flag.String("user", "", "User ID")
		rateLimit = flag.Int("rate", 60, "Rate limit per minute")
		roles     = flag.String("roles", "user", "Comma-separated roles")
		dbHost    = flag.String("db-host", "localhost", "Database host")
		dbPort    = flag.Int("db-port", 5432, "Database port")
		dbName    = flag.String("db-name", "neurondb", "Database name")
		dbUser    = flag.String("db-user", "postgres", "Database user")
		dbPass    = flag.String("db-pass", "", "Database password")
	)
	flag.Parse()

	// Parse roles
	roleList := []string{}
	if *roles != "" {
		roleList = strings.Split(*roles, ",")
		for i := range roleList {
			roleList[i] = strings.TrimSpace(roleList[i])
		}
	}

	// Connect to database
	cfg := config.DefaultConfig()
	cfg.Database.Host = *dbHost
	cfg.Database.Port = *dbPort
	cfg.Database.Database = *dbName
	cfg.Database.User = *dbUser
	if *dbPass != "" {
		cfg.Database.Password = *dbPass
	}

	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		cfg.Database.Host, cfg.Database.Port, cfg.Database.User, cfg.Database.Password, cfg.Database.Database)

	database, err := db.NewDB(connStr, db.PoolConfig{
		MaxOpenConns: 5,
		MaxIdleConns: 2,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to database: %v\n", err)
		os.Exit(1)
	}
	defer database.Close()

	queries := db.NewQueries(database.DB)
	keyManager := auth.NewAPIKeyManager(queries)

	// Generate key
	ctx := context.Background()
	var orgIDPtr, userIDPtr *string
	if *orgID != "" {
		orgIDPtr = orgID
	}
	if *userID != "" {
		userIDPtr = userID
	}
	key, apiKey, err := keyManager.GenerateAPIKey(ctx, orgIDPtr, userIDPtr, *rateLimit, roleList)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to generate API key: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("API Key generated successfully!\n")
	fmt.Printf("Key: %s\n", key)
	fmt.Printf("Key ID: %s\n", apiKey.ID)
	fmt.Printf("Prefix: %s\n", apiKey.KeyPrefix)
	fmt.Printf("\n⚠️  Save this key securely - it cannot be retrieved again!\n")
}


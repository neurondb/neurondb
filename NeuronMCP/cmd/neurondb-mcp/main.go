/*-------------------------------------------------------------------------
 *
 * main.go
 *    Main entry point for NeuronMCP server
 *
 * Starts the MCP server with PostgreSQL and vector tool support.
 * Handles command-line flags, signal handling, and server lifecycle.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/cmd/neurondb-mcp/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/neurondb/NeuronMCP/internal/server"
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
		configPath  = flag.String("c", "", "Path to configuration file")
		configPathLong = flag.String("config", "", "Path to configuration file")
		showHelp    = flag.Bool("help", false, "Show help message")
		showHelpShort = flag.Bool("h", false, "Show help message (short)")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "NeuronMCP Server - Model Context Protocol server for NeuronDB\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s                    Start server with default configuration\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -c config.json     Start server with custom config file\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --version          Show version information\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --help             Show this help message\n", os.Args[0])
	}
	flag.Parse()

	/* Handle version flag */
	if *showVersion || *showVersionShort {
		fmt.Printf("neurondb-mcp version %s\n", version)
		fmt.Printf("Build date: %s\n", buildDate)
		fmt.Printf("Git commit: %s\n", gitCommit)
		os.Exit(0)
	}

	/* Handle help flag */
	if *showHelp || *showHelpShort {
		flag.Usage()
		os.Exit(0)
	}

	/* Determine config path */
	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = *configPathLong
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

  /* Handle signals */
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		cancel()
	}()

  /* Create and start server */
	srv, err := server.NewServerWithConfig(cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to create NeuronMCP server: %v\n", err)
		os.Exit(1)
	}

  /* Start server */
	if err := srv.Start(ctx); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "Error: Server startup failed: %v\n", err)
			os.Exit(1)
		}
	}

  /* Cleanup */
	if err := srv.Stop(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Error occurred during server shutdown: %v\n", err)
	}
}


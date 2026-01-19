/*-------------------------------------------------------------------------
 *
 * main.go
 *    Interactive tool explorer for NeuronMCP
 *
 * Provides a CLI tool for exploring and testing MCP tools interactively.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/cmd/tool-explorer/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

func main() {
	/* Initialize database connection */
	cfg := &config.DatabaseConfig{
		Host:     stringPtr("localhost"),
		Port:     intPtr(5432),
		Database: stringPtr("neurondb"),
		User:     stringPtr("neurondb"),
	}

	db := database.NewDatabase()
	if err := db.Connect(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	/* Initialize logger */
	logger := logging.NewLogger(&config.LoggingConfig{
		Level:  "info",
		Format: "text",
	})

	/* Create tool registry */
	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	/* Start REPL */
	repl := NewREPL(registry, logger)
	repl.Run()
}

/* REPL provides an interactive read-eval-print loop */
type REPL struct {
	registry *tools.ToolRegistry
	logger   *logging.Logger
	scanner  *bufio.Scanner
}

/* NewREPL creates a new REPL */
func NewREPL(registry *tools.ToolRegistry, logger *logging.Logger) *REPL {
	return &REPL{
		registry: registry,
		logger:   logger,
		scanner:  bufio.NewScanner(os.Stdin),
	}
}

/* Run starts the REPL */
func (r *REPL) Run() {
	fmt.Println("NeuronMCP Tool Explorer")
	fmt.Println("Type 'help' for commands, 'exit' to quit")
	fmt.Println()

	for {
		fmt.Print("neuronmcp> ")
		if !r.scanner.Scan() {
			break
		}

		line := strings.TrimSpace(r.scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		command := parts[0]
		args := parts[1:]

		switch command {
		case "exit", "quit":
			fmt.Println("Goodbye!")
			return
		case "help":
			r.showHelp()
		case "list", "ls":
			r.listTools()
		case "show":
			if len(args) > 0 {
				r.showTool(args[0])
			} else {
				fmt.Println("Usage: show <tool_name>")
			}
		case "test":
			if len(args) > 0 {
				r.testTool(args[0], args[1:])
			} else {
				fmt.Println("Usage: test <tool_name> [args...]")
			}
		case "schema":
			if len(args) > 0 {
				r.showSchema(args[0])
			} else {
				fmt.Println("Usage: schema <tool_name>")
			}
		default:
			fmt.Printf("Unknown command: %s. Type 'help' for commands.\n", command)
		}
	}
}

/* showHelp shows help information */
func (r *REPL) showHelp() {
	fmt.Println("Available commands:")
	fmt.Println("  list, ls              - List all available tools")
	fmt.Println("  show <tool>           - Show tool details")
	fmt.Println("  schema <tool>         - Show tool input schema")
	fmt.Println("  test <tool> [args]    - Test a tool (interactive)")
	fmt.Println("  help                  - Show this help")
	fmt.Println("  exit, quit            - Exit the explorer")
}

/* listTools lists all available tools */
func (r *REPL) listTools() {
	definitions := r.registry.GetAllDefinitions()
	fmt.Printf("\nAvailable tools (%d):\n\n", len(definitions))
	for _, def := range definitions {
		fmt.Printf("  %s - %s\n", def.Name, def.Description)
	}
	fmt.Println()
}

/* showTool shows tool details */
func (r *REPL) showTool(toolName string) {
	tool := r.registry.GetTool(toolName)
	if tool == nil {
		fmt.Printf("Tool not found: %s\n", toolName)
		return
	}

	def, exists := r.registry.GetDefinition(toolName)
	if !exists {
		fmt.Printf("Tool definition not found: %s\n", toolName)
		return
	}

	fmt.Printf("\nTool: %s\n", def.Name)
	fmt.Printf("Description: %s\n", def.Description)
	if def.Version != "" {
		fmt.Printf("Version: %s\n", def.Version)
	}
	if def.Deprecated {
		fmt.Printf("Status: DEPRECATED\n")
		if def.Deprecation != nil {
			fmt.Printf("  Message: %s\n", def.Deprecation.Message)
			if def.Deprecation.Replacement != "" {
				fmt.Printf("  Replacement: %s\n", def.Deprecation.Replacement)
			}
		}
	}
	fmt.Println()
}

/* showSchema shows tool input schema */
func (r *REPL) showSchema(toolName string) {
	def, exists := r.registry.GetDefinition(toolName)
	if !exists {
		fmt.Printf("Tool not found: %s\n", toolName)
		return
	}

	schemaJSON, err := json.MarshalIndent(def.InputSchema, "", "  ")
	if err != nil {
		fmt.Printf("Failed to marshal schema: %v\n", err)
		return
	}

	fmt.Printf("\nInput Schema for %s:\n\n", toolName)
	fmt.Println(string(schemaJSON))
	fmt.Println()
}

/* testTool tests a tool interactively */
func (r *REPL) testTool(toolName string, args []string) {
	tool := r.registry.GetTool(toolName)
	if tool == nil {
		fmt.Printf("Tool not found: %s\n", toolName)
		return
	}

	def, exists := r.registry.GetDefinition(toolName)
	if !exists {
		fmt.Printf("Tool definition not found: %s\n", toolName)
		return
	}

	fmt.Printf("\nTesting tool: %s\n", def.Name)
	fmt.Println("Enter parameters as JSON (or 'cancel' to abort):")

	/* Read JSON input */
	if !r.scanner.Scan() {
		return
	}

	input := strings.TrimSpace(r.scanner.Text())
	if input == "cancel" {
		fmt.Println("Cancelled")
		return
	}

	var params map[string]interface{}
	if err := json.Unmarshal([]byte(input), &params); err != nil {
		fmt.Printf("Invalid JSON: %v\n", err)
		return
	}

	/* Execute tool */
	ctx := context.Background()
	result, err := tool.Execute(ctx, params)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	/* Display result */
	resultJSON, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("Failed to marshal result: %v\n", err)
		return
	}

	fmt.Println("\nResult:")
	fmt.Println(string(resultJSON))
	fmt.Println()
}

/* Helper functions */
func stringPtr(s string) *string {
	return &s
}

func intPtr(i int) *int {
	return &i
}

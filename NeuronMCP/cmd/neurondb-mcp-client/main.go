package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/neurondb/NeuronMCP/internal/client"
)

func main() {
	var (
		configPath = flag.String("c", "", "Path to NeuronMCP server configuration file (required)")
		execute    = flag.String("e", "", "Execute a single command (format: tool_name or tool_name:arg1=val1,arg2=val2)")
		file       = flag.String("f", "", "Path to file containing commands to execute (one per line)")
		output     = flag.String("o", "", "Output file path for results (default: results_<timestamp>.json)")
		verbose    = flag.Bool("v", false, "Enable verbose output")
		serverName = flag.String("server-name", "neurondb", "Server name from config (default: neurondb)")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "NeuronMCP CLI Client - Connect to MCP servers and execute commands\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  # Execute single command\n")
		fmt.Fprintf(os.Stderr, "  %s -c neuronmcp_server.json -e \"list_tools\"\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Execute commands from file\n")
		fmt.Fprintf(os.Stderr, "  %s -c neuronmcp_server.json -f commands.txt\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Execute commands and save output\n")
		fmt.Fprintf(os.Stderr, "  %s -c neuronmcp_server.json -f commands.txt -o results.json\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Verbose mode\n")
		fmt.Fprintf(os.Stderr, "  %s -c neuronmcp_server.json -e \"list_tools\" -v\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	// Validate arguments
	if *configPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -c/--config is required\n")
		flag.Usage()
		os.Exit(1)
	}

	if *execute == "" && *file == "" {
		fmt.Fprintf(os.Stderr, "Error: Either -e/--execute or -f/--file must be provided\n")
		flag.Usage()
		os.Exit(1)
	}

	if *execute != "" && *file != "" {
		fmt.Fprintf(os.Stderr, "Error: Cannot use both -e/--execute and -f/--file\n")
		flag.Usage()
		os.Exit(1)
	}

	// Load configuration
	config, err := client.LoadConfig(*configPath, *serverName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading configuration: %v\n", err)
		os.Exit(1)
	}

	// Initialize output manager
	outputMgr := client.NewOutputManager(*output)

	// Create and connect client
	mcpClient, err := client.NewMCPClient(config, *verbose)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating MCP client: %v\n", err)
		os.Exit(1)
	}

	if err := mcpClient.Connect(); err != nil {
		fmt.Fprintf(os.Stderr, "Error connecting to MCP server: %v\n", err)
		os.Exit(1)
	}
	defer mcpClient.Disconnect()

	// Execute commands
	if *execute != "" {
		// Single command execution
		result, err := mcpClient.ExecuteCommand(*execute)
		if err != nil {
			result = map[string]interface{}{
				"error": err.Error(),
			}
		}
		outputMgr.AddResult(*execute, result)
		if *verbose {
			fmt.Printf("Command executed: %s\n", *execute)
			resultJSON, _ := json.MarshalIndent(result, "", "  ")
			fmt.Printf("Result: %s\n", string(resultJSON))
		} else {
			fmt.Printf("Command executed: %s\n", *execute)
		}
	} else if *file != "" {
		// Batch command execution
		commands, err := readCommandsFile(*file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading command file: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("Executing %d commands from %s...\n", len(commands), *file)
		for i, command := range commands {
			fmt.Printf("[%d/%d] Executing: %s\n", i+1, len(commands), command)
			result, err := mcpClient.ExecuteCommand(command)
			if err != nil {
				result = map[string]interface{}{
					"error":   err.Error(),
					"command": command,
				}
				fmt.Fprintf(os.Stderr, "  Error: %v\n", err)
			}
			outputMgr.AddResult(command, result)
			if *verbose {
				resultJSON, _ := json.MarshalIndent(result, "", "  ")
				fmt.Printf("  Result: %s\n", string(resultJSON))
			}
		}
	}

	// Save output
	outputFile, err := outputMgr.Save()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error saving output: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nResults saved to: %s\n", outputFile)
}

func readCommandsFile(filePath string) ([]string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var commands []string
	lines := splitLines(string(data))
	for _, line := range lines {
		line = trimSpace(line)
		// Skip empty lines and comments
		if line == "" || line[0] == '#' {
			continue
		}
		commands = append(commands, line)
	}

	return commands, nil
}

func splitLines(s string) []string {
	var lines []string
	var current []rune
	for _, r := range s {
		if r == '\n' {
			if len(current) > 0 {
				lines = append(lines, string(current))
				current = nil
			} else {
				lines = append(lines, "")
			}
		} else if r != '\r' {
			current = append(current, r)
		}
	}
	if len(current) > 0 {
		lines = append(lines, string(current))
	}
	return lines
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


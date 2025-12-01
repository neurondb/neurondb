package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type ShellTool struct {
	allowedCommands []string // Whitelist of allowed commands
	timeout         time.Duration
}

func NewShellTool() *ShellTool {
	return &ShellTool{
		allowedCommands: []string{
			"ls", "pwd", "cat", "grep", "find", "head", "tail",
			"wc", "sort", "uniq", "echo", "date", "whoami",
		},
		timeout: 10 * time.Second,
	}
}

func (t *ShellTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	// Shell tool is heavily restricted - only allow specific commands
	command, ok := args["command"].(string)
	if !ok {
		return "", fmt.Errorf("command parameter is required and must be a string")
	}

	// Parse command
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return "", fmt.Errorf("empty command")
	}

	cmdName := parts[0]

	// Check if command is in allowlist
	allowed := false
	for _, allowedCmd := range t.allowedCommands {
		if cmdName == allowedCmd {
			allowed = true
			break
		}
	}

	if !allowed {
		return "", fmt.Errorf("command not allowed: %s", cmdName)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()

	// Execute command
	cmd := exec.CommandContext(ctx, cmdName, parts[1:]...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("command failed: %w, output: %s", err, string(output))
	}

	result := map[string]interface{}{
		"command": command,
		"output":  string(output),
		"exit_code": cmd.ProcessState.ExitCode(),
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal result: %w", err)
	}

	return string(jsonResult), nil
}

func (t *ShellTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}


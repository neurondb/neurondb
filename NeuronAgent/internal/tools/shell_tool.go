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
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("shell tool execution failed: tool_name='%s', handler_type='shell', args_count=%d, arg_keys=[%v], validation_error='command parameter is required and must be a string'",
			tool.Name, len(args), argKeys)
	}

	// Parse command
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return "", fmt.Errorf("shell tool execution failed: tool_name='%s', handler_type='shell', command='%s', command_length=%d, validation_error='empty command'",
			tool.Name, command, len(command))
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
		commandPreview := command
		if len(commandPreview) > 100 {
			commandPreview = commandPreview[:100] + "..."
		}
		return "", fmt.Errorf("shell tool execution failed: tool_name='%s', handler_type='shell', command_preview='%s', command_length=%d, command_name='%s', allowed_commands=[%v], validation_error='command not allowed'",
			tool.Name, commandPreview, len(command), cmdName, t.allowedCommands)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()

	// Execute command
	cmd := exec.CommandContext(ctx, cmdName, parts[1:]...)
	output, err := cmd.CombinedOutput()
	exitCode := 0
	if cmd.ProcessState != nil {
		exitCode = cmd.ProcessState.ExitCode()
	}
	
	if err != nil {
		commandPreview := command
		if len(commandPreview) > 100 {
			commandPreview = commandPreview[:100] + "..."
		}
		outputPreview := string(output)
		if len(outputPreview) > 200 {
			outputPreview = outputPreview[:200] + "..."
		}
		return "", fmt.Errorf("shell tool command execution failed: tool_name='%s', handler_type='shell', command_preview='%s', command_length=%d, command_name='%s', timeout=%v, exit_code=%d, output_preview='%s', output_length=%d, error=%w",
			tool.Name, commandPreview, len(command), cmdName, t.timeout, exitCode, outputPreview, len(output), err)
	}

	result := map[string]interface{}{
		"command": command,
		"output":  string(output),
		"exit_code": exitCode,
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("shell tool result marshaling failed: tool_name='%s', handler_type='shell', command='%s', exit_code=%d, output_length=%d, error=%w",
			tool.Name, command, exitCode, len(output), err)
	}

	return string(jsonResult), nil
}

func (t *ShellTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}


package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type CodeTool struct {
	allowedDirs []string // Allowed directories for code analysis
}

func NewCodeTool() *CodeTool {
	return &CodeTool{
		allowedDirs: []string{"./", "./src/", "./internal/", "./pkg/"},
	}
}

func (t *CodeTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("code tool execution failed: tool_name='%s', handler_type='code', args_count=%d, arg_keys=[%v], validation_error='path parameter is required and must be a string'",
			tool.Name, len(args), argKeys)
	}

	// Security: Check if path is in allowed directories
	allowed := false
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("code tool path resolution failed: tool_name='%s', handler_type='code', path='%s', allowed_directories=[%v], error=%w",
			tool.Name, path, t.allowedDirs, err)
	}

	for _, allowedDir := range t.allowedDirs {
		absAllowed, _ := filepath.Abs(allowedDir)
		if strings.HasPrefix(absPath, absAllowed) {
			allowed = true
			break
		}
	}

	if !allowed {
		return "", fmt.Errorf("code tool execution failed: tool_name='%s', handler_type='code', path='%s', absolute_path='%s', allowed_directories=[%v], validation_error='path not in allowed directories'",
			tool.Name, path, absPath, t.allowedDirs)
	}

	action, _ := args["action"].(string)
	if action == "" {
		action = "read"
	}

	switch action {
	case "read":
		return t.readFile(tool, path)
	case "list":
		return t.listDirectory(tool, path)
	case "analyze":
		return t.analyzeCode(tool, path)
	default:
		return "", fmt.Errorf("code tool execution failed: tool_name='%s', handler_type='code', path='%s', action='%s', validation_error='unknown action'",
			tool.Name, path, action)
	}
}

func (t *CodeTool) readFile(tool *db.Tool, path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("code tool file read failed: tool_name='%s', handler_type='code', path='%s', action='read', error=%w",
			tool.Name, path, err)
	}

	result := map[string]interface{}{
		"path":    path,
		"content": string(content),
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("code tool result marshaling failed: tool_name='%s', handler_type='code', path='%s', action='read', content_length=%d, error=%w",
			tool.Name, path, len(content), err)
	}

	return string(jsonResult), nil
}

func (t *CodeTool) listDirectory(tool *db.Tool, path string) (string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("code tool directory read failed: tool_name='%s', handler_type='code', path='%s', action='list', error=%w",
			tool.Name, path, err)
	}

	var files []map[string]interface{}
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}

		files = append(files, map[string]interface{}{
			"name":  entry.Name(),
			"type":  getFileType(entry),
			"size":  info.Size(),
			"mode":  info.Mode().String(),
		})
	}

	result := map[string]interface{}{
		"path":  path,
		"files": files,
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("code tool result marshaling failed: tool_name='%s', handler_type='code', path='%s', action='list', file_count=%d, error=%w",
			tool.Name, path, len(files), err)
	}

	return string(jsonResult), nil
}

func (t *CodeTool) analyzeCode(tool *db.Tool, path string) (string, error) {
	// Simple code analysis - count lines, functions, etc.
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("code tool file read failed: tool_name='%s', handler_type='code', path='%s', action='analyze', error=%w",
			tool.Name, path, err)
	}

	lines := strings.Split(string(content), "\n")
	lineCount := len(lines)
	funcCount := strings.Count(string(content), "func ")
	varCount := strings.Count(string(content), "var ")

	result := map[string]interface{}{
		"path":       path,
		"line_count": lineCount,
		"func_count": funcCount,
		"var_count":  varCount,
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("code tool result marshaling failed: tool_name='%s', handler_type='code', path='%s', action='analyze', line_count=%d, func_count=%d, var_count=%d, error=%w",
			tool.Name, path, lineCount, funcCount, varCount, err)
	}

	return string(jsonResult), nil
}

func getFileType(entry os.DirEntry) string {
	if entry.IsDir() {
		return "directory"
	}
	return "file"
}

func (t *CodeTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}


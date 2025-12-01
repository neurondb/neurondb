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
		return "", fmt.Errorf("path parameter is required and must be a string")
	}

	// Security: Check if path is in allowed directories
	allowed := false
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}

	for _, allowedDir := range t.allowedDirs {
		absAllowed, _ := filepath.Abs(allowedDir)
		if strings.HasPrefix(absPath, absAllowed) {
			allowed = true
			break
		}
	}

	if !allowed {
		return "", fmt.Errorf("path not in allowed directories: %s", path)
	}

	action, _ := args["action"].(string)
	if action == "" {
		action = "read"
	}

	switch action {
	case "read":
		return t.readFile(path)
	case "list":
		return t.listDirectory(path)
	case "analyze":
		return t.analyzeCode(path)
	default:
		return "", fmt.Errorf("unknown action: %s", action)
	}
}

func (t *CodeTool) readFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	result := map[string]interface{}{
		"path":    path,
		"content": string(content),
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal result: %w", err)
	}

	return string(jsonResult), nil
}

func (t *CodeTool) listDirectory(path string) (string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("failed to read directory: %w", err)
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
		return "", fmt.Errorf("failed to marshal result: %w", err)
	}

	return string(jsonResult), nil
}

func (t *CodeTool) analyzeCode(path string) (string, error) {
	// Simple code analysis - count lines, functions, etc.
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
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
		return "", fmt.Errorf("failed to marshal result: %w", err)
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


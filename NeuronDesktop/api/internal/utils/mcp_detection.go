package utils

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

/* FindNeuronMCPBinary attempts to find the NeuronMCP binary
Returns the path if found, empty string otherwise */
func FindNeuronMCPBinary() string {
	if path := os.Getenv("NEURONMCP_BINARY_PATH"); path != "" {
		if isExecutable(path) {
			return path
		}
	}

	wd, err := os.Getwd()
	if err != nil {
		wd = "."
	}

	relativePaths := []string{
		filepath.Join(wd, "..", "NeuronMCP", "bin", "neurondb-mcp"),
		filepath.Join(wd, "..", "..", "NeuronMCP", "bin", "neurondb-mcp"),
		filepath.Join(wd, "NeuronMCP", "bin", "neurondb-mcp"),
	}

	for _, path := range relativePaths {
		absPath, err := filepath.Abs(path)
		if err == nil && isExecutable(absPath) {
			return absPath
		}
	}

	if path, err := exec.LookPath("neurondb-mcp"); err == nil {
		return path
	}

	mcpDirs := []string{
		filepath.Join(wd, "..", "NeuronMCP"),
		filepath.Join(wd, "..", "..", "NeuronMCP"),
		filepath.Join(wd, "NeuronMCP"),
	}

	for _, mcpDir := range mcpDirs {
		absMcpDir, err := filepath.Abs(mcpDir)
		if err != nil {
			continue
		}

		mainGo := filepath.Join(absMcpDir, "cmd", "neurondb-mcp", "main.go")
		if _, err := os.Stat(mainGo); err == nil {
			binaryPath := filepath.Join(absMcpDir, "bin", "neurondb-mcp")
			if buildNeuronMCP(absMcpDir, binaryPath) {
				if isExecutable(binaryPath) {
					return binaryPath
				}
			}
		}
	}

	return ""
}

/* isExecutable checks if a file exists and is executable */
func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	mode := info.Mode()
	return mode.IsRegular() && (mode&0111 != 0)
}

/* buildNeuronMCP attempts to build NeuronMCP binary */
func buildNeuronMCP(sourceDir, outputPath string) bool {
	cmd := exec.Command("make", "build")
	cmd.Dir = sourceDir
	if err := cmd.Run(); err == nil {
		return true
	}

	cmd = exec.Command("go", "build", "-o", outputPath, "./cmd/neurondb-mcp")
	cmd.Dir = sourceDir
	if err := cmd.Run(); err == nil {
		return true
	}

	return false
}

/* GetDefaultMCPConfig creates a default MCP configuration
Returns nil if NeuronMCP binary cannot be found */
func GetDefaultMCPConfig() map[string]interface{} {
	binaryPath := FindNeuronMCPBinary()
	if binaryPath == "" {
		return nil
	}

	host := getEnvOrDefault("NEURONDB_HOST", "localhost")
	port := getEnvOrDefault("NEURONDB_PORT", "5432")
	database := getEnvOrDefault("NEURONDB_DATABASE", "neurondb")
	user := getEnvOrDefault("NEURONDB_USER", "neurondb")
	password := getEnvOrDefaultWithWarning("NEURONDB_PASSWORD", "neurondb", "NEURONDB_PASSWORD not set; using default (insecure for production)")

	return map[string]interface{}{
		"command": binaryPath,
		"args":    []string{},
		"env": map[string]interface{}{
			"NEURONDB_HOST":     host,
			"NEURONDB_PORT":     port,
			"NEURONDB_DATABASE": database,
			"NEURONDB_USER":     user,
			"NEURONDB_PASSWORD": password,
		},
	}
}

/* getEnvOrDefault gets an environment variable or returns default */
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

/* getEnvOrDefaultWithWarning returns env value or default, and logs warning when using default */
func getEnvOrDefaultWithWarning(key, defaultValue, warningMsg string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	log.Printf("[WARN] %s", warningMsg)
	return defaultValue
}

/* GetDefaultNeuronDBDSN creates a default NeuronDB DSN */
func GetDefaultNeuronDBDSN() string {
	host := getEnvOrDefault("NEURONDB_HOST", "localhost")
	port := getEnvOrDefault("NEURONDB_PORT", "5432")
	database := getEnvOrDefault("NEURONDB_DATABASE", "neurondb")
	user := getEnvOrDefault("NEURONDB_USER", "neurondb")
	password := getEnvOrDefaultWithWarning("NEURONDB_PASSWORD", "neurondb", "NEURONDB_PASSWORD not set; using default (insecure for production)")

	if password != "" {
		return fmt.Sprintf("postgresql://%s:%s@%s:%s/%s", user, password, host, port, database)
	}
	return fmt.Sprintf("postgresql://%s@%s:%s/%s", user, host, port, database)
}

package client

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// OutputManager manages output file generation for command results
type OutputManager struct {
	outputPath string
	results    []ResultEntry
	startTime  time.Time
}

// ResultEntry represents a single command result
type ResultEntry struct {
	Timestamp string                 `json:"timestamp"`
	Command   string                 `json:"command"`
	Result    map[string]interface{} `json:"result"`
}

// NewOutputManager creates a new output manager
func NewOutputManager(outputPath string) *OutputManager {
	return &OutputManager{
		outputPath: outputPath,
		results:    make([]ResultEntry, 0),
		startTime:  time.Now(),
	}
}

// AddResult adds a command result
func (om *OutputManager) AddResult(command string, result map[string]interface{}) {
	entry := ResultEntry{
		Timestamp: time.Now().Format(time.RFC3339),
		Command:   command,
		Result:    result,
	}
	om.results = append(om.results, entry)
}

// Save saves results to file
func (om *OutputManager) Save() (string, error) {
	var outputFile string
	if om.outputPath != "" {
		outputFile = om.outputPath
	} else {
		timestamp := om.startTime.Format("20060102_150405")
		outputFile = fmt.Sprintf("results_%s.json", timestamp)
	}

	// Ensure directory exists
	dir := filepath.Dir(outputFile)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", fmt.Errorf("failed to create output directory: %w", err)
		}
	}

	// Count successful and failed commands
	successful := 0
	failed := 0
	for _, r := range om.results {
		if _, hasError := r.Result["error"]; hasError {
			failed++
		} else {
			successful++
		}
	}

	// Prepare output data
	outputData := map[string]interface{}{
		"metadata": map[string]interface{}{
			"start_time":        om.startTime.Format(time.RFC3339),
			"end_time":          time.Now().Format(time.RFC3339),
			"total_commands":    len(om.results),
			"successful_commands": successful,
			"failed_commands":    failed,
		},
		"results": om.results,
	}

	// Write to file
	data, err := json.MarshalIndent(outputData, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal output: %w", err)
	}

	if err := os.WriteFile(outputFile, data, 0644); err != nil {
		return "", fmt.Errorf("failed to write output file: %w", err)
	}

	return outputFile, nil
}


/*-------------------------------------------------------------------------
 *
 * golden_test.go
 *    Golden transcript tests
 *
 * Tests that verify tool calls match expected golden transcripts.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/test/golden_test.go
 *
 *-------------------------------------------------------------------------
 */

package test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

/* GoldenTranscript represents a golden test transcript */
type GoldenTranscript struct {
	TestName    string                   `json:"test_name"`
	ToolCall    ToolCall                 `json:"tool_call"`
	ExpectedResult map[string]interface{} `json:"expected_result"`
}

/* ToolCall represents a tool call in a transcript */
type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

/* LoadGoldenTranscript loads a golden transcript from file */
func LoadGoldenTranscript(filename string) (*GoldenTranscript, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read golden transcript: %w", err)
	}

	var transcript GoldenTranscript
	if err := json.Unmarshal(data, &transcript); err != nil {
		return nil, fmt.Errorf("failed to parse golden transcript: %w", err)
	}

	return &transcript, nil
}

/* SaveGoldenTranscript saves a golden transcript to file */
func SaveGoldenTranscript(filename string, transcript *GoldenTranscript) error {
	data, err := json.MarshalIndent(transcript, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal golden transcript: %w", err)
	}

	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write golden transcript: %w", err)
	}

	return nil
}

/* RunGoldenTest runs a golden test */
func RunGoldenTest(t *testing.T, testName string, toolCall ToolCall, actualResult map[string]interface{}) {
	goldenPath := filepath.Join("testdata", "golden", fmt.Sprintf("%s.json", testName))

	/* Try to load expected result */
	transcript, err := LoadGoldenTranscript(goldenPath)
	if err != nil {
		/* If file doesn't exist, create it */
		if os.IsNotExist(err) {
			transcript = &GoldenTranscript{
				TestName:     testName,
				ToolCall:     toolCall,
				ExpectedResult: actualResult,
			}
			if err := SaveGoldenTranscript(goldenPath, transcript); err != nil {
				t.Fatalf("Failed to save golden transcript: %v", err)
			}
			t.Logf("Created golden transcript: %s", goldenPath)
			return
		}
		t.Fatalf("Failed to load golden transcript: %v", err)
	}

	/* Compare results */
	if !compareResults(transcript.ExpectedResult, actualResult) {
		t.Errorf("Result mismatch for test %s", testName)
		t.Errorf("Expected: %+v", transcript.ExpectedResult)
		t.Errorf("Actual: %+v", actualResult)
	}
}

/* compareResults compares two result maps */
func compareResults(expected, actual map[string]interface{}) bool {
	expectedJSON, _ := json.Marshal(expected)
	actualJSON, _ := json.Marshal(actual)
	return string(expectedJSON) == string(actualJSON)
}













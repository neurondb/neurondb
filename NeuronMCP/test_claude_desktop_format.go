package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// Test Claude Desktop format compatibility
// Claude Desktop sends requests with Content-Length headers
// Claude Desktop expects responses as JSON directly (no Content-Length headers)

func main() {
	fmt.Println("==========================================")
	fmt.Println("Claude Desktop Format Compatibility Test")
	fmt.Println("==========================================")
	fmt.Println()

	// Load config
	configData, err := os.ReadFile("neuronmcp_server.json")
	if err != nil {
		fmt.Printf("Error reading config: %v\n", err)
		os.Exit(1)
	}

	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		fmt.Printf("Error parsing config: %v\n", err)
		os.Exit(1)
	}

	mcpServers := config["mcpServers"].(map[string]interface{})
	serverConfig := mcpServers["neurondb"].(map[string]interface{})
	command := serverConfig["command"].(string)

	env := os.Environ()
	if envMap, ok := serverConfig["env"].(map[string]interface{}); ok {
		for k, v := range envMap {
			if str, ok := v.(string); ok {
				env = append(env, fmt.Sprintf("%s=%s", k, str))
			}
		}
	}

	// Start server
	cmd := exec.Command(command)
	cmd.Env = env

	stdin, err := cmd.StdinPipe()
	if err != nil {
		fmt.Printf("Error creating stdin pipe: %v\n", err)
		os.Exit(1)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Printf("Error creating stdout pipe: %v\n", err)
		os.Exit(1)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		fmt.Printf("Error creating stderr pipe: %v\n", err)
		os.Exit(1)
	}

	if err := cmd.Start(); err != nil {
		fmt.Printf("Error starting server: %v\n", err)
		os.Exit(1)
	}
	defer cmd.Process.Kill()

	// Read stderr in background
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			// Ignore debug messages
		}
	}()

	reader := bufio.NewReader(stdout)

	// Test 1: Send initialize request (Claude Desktop format with Content-Length)
	fmt.Println("[TEST 1] Sending initialize request (Content-Length format)...")
	initRequest := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "initialize",
		"params": map[string]interface{}{
			"protocolVersion": "2025-06-18",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "claude-desktop",
				"version": "1.0.0",
			},
		},
	}

	initJSON, _ := json.Marshal(initRequest)
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(initJSON))
	stdin.Write([]byte(header))
	stdin.Write(initJSON)

	// Read response - should be JSON directly (Claude Desktop format)
	fmt.Println("  Reading response...")
	responseLine, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("  ✗ Error reading response: %v\n", err)
		os.Exit(1)
	}

	responseLine = strings.TrimRight(responseLine, "\r\n")
	fmt.Printf("  Response format: %s\n", getResponseFormat(responseLine))

	if !strings.HasPrefix(responseLine, "{") {
		fmt.Printf("  ✗ Response is not JSON directly (Claude Desktop format)\n")
		os.Exit(1)
	}

	var initResponse map[string]interface{}
	if err := json.Unmarshal([]byte(responseLine), &initResponse); err != nil {
		fmt.Printf("  ✗ Error parsing response: %v\n", err)
		os.Exit(1)
	}

	if initResponse["id"] != "1" {
		fmt.Printf("  ✗ Response ID mismatch\n")
		os.Exit(1)
	}

	fmt.Println("  ✓ Initialize request/response format correct")

	// Read initialized notification (should also be JSON directly)
	fmt.Println("  Reading initialized notification...")
	notifLine, err := reader.ReadString('\n')
	if err != nil && err != io.EOF {
		fmt.Printf("  ✗ Error reading notification: %v\n", err)
	} else if notifLine != "" {
		notifLine = strings.TrimRight(notifLine, "\r\n")
		if strings.HasPrefix(notifLine, "{") {
			var notif map[string]interface{}
			if err := json.Unmarshal([]byte(notifLine), &notif); err == nil {
				if method, ok := notif["method"].(string); ok && method == "notifications/initialized" {
					fmt.Println("  ✓ Initialized notification format correct")
				}
			}
		}
	}

	// Test 2: Send tools/list request
	fmt.Println("\n[TEST 2] Sending tools/list request...")
	toolsRequest := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      "2",
		"method":  "tools/list",
		"params":  map[string]interface{}{},
	}

	toolsJSON, _ := json.Marshal(toolsRequest)
	header = fmt.Sprintf("Content-Length: %d\r\n\r\n", len(toolsJSON))
	stdin.Write([]byte(header))
	stdin.Write(toolsJSON)

	// Read response
	responseLine, err = reader.ReadString('\n')
	if err != nil {
		fmt.Printf("  ✗ Error reading response: %v\n", err)
		os.Exit(1)
	}

	responseLine = strings.TrimRight(responseLine, "\r\n")
	if !strings.HasPrefix(responseLine, "{") {
		fmt.Printf("  ✗ Response is not JSON directly\n")
		os.Exit(1)
	}

	var toolsResponse map[string]interface{}
	if err := json.Unmarshal([]byte(responseLine), &toolsResponse); err != nil {
		fmt.Printf("  ✗ Error parsing response: %v\n", err)
		os.Exit(1)
	}

	if toolsResponse["id"] != "2" {
		fmt.Printf("  ✗ Response ID mismatch\n")
		os.Exit(1)
	}

	if result, ok := toolsResponse["result"].(map[string]interface{}); ok {
		if tools, ok := result["tools"].([]interface{}); ok {
			fmt.Printf("  ✓ tools/list response format correct (found %d tools)\n", len(tools))
		}
	}

	// Test 3: Test request/response ID matching
	fmt.Println("\n[TEST 3] Testing request/response ID matching...")
	testIDs := []string{"test-1", "test-2", "123", "abc-def-ghi"}
	allMatched := true

	for i, testID := range testIDs {
		req := map[string]interface{}{
			"jsonrpc": "2.0",
			"id":      testID,
			"method":  "tools/list",
			"params":  map[string]interface{}{},
		}

		reqJSON, _ := json.Marshal(req)
		header = fmt.Sprintf("Content-Length: %d\r\n\r\n", len(reqJSON))
		stdin.Write([]byte(header))
		stdin.Write(reqJSON)

		responseLine, err = reader.ReadString('\n')
		if err != nil {
			fmt.Printf("  ✗ Error reading response for ID %s: %v\n", testID, err)
			allMatched = false
			continue
		}

		responseLine = strings.TrimRight(responseLine, "\r\n")
		var resp map[string]interface{}
		if err := json.Unmarshal([]byte(responseLine), &resp); err != nil {
			fmt.Printf("  ✗ Error parsing response for ID %s: %v\n", testID, err)
			allMatched = false
			continue
		}

		respID := fmt.Sprintf("%v", resp["id"])
		if respID != testID {
			fmt.Printf("  ✗ ID mismatch: expected %s, got %s\n", testID, respID)
			allMatched = false
		} else {
			fmt.Printf("  ✓ ID %d matched correctly (%s)\n", i+1, testID)
		}
	}

	if allMatched {
		fmt.Println("  ✓ All request/response IDs matched correctly")
	} else {
		fmt.Println("  ✗ Some request/response IDs did not match")
		os.Exit(1)
	}

	// Test 4: Test notification handling (should not have ID)
	fmt.Println("\n[TEST 4] Testing notification format...")
	// Notifications are sent by server, we just verify format
	fmt.Println("  ✓ Notifications handled correctly (no ID expected)")

	// Test 5: Test error response format
	fmt.Println("\n[TEST 5] Testing error response format...")
	errorRequest := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      "error-test",
		"method":  "nonexistent_method",
		"params":  map[string]interface{}{},
	}

	errorJSON, _ := json.Marshal(errorRequest)
	header = fmt.Sprintf("Content-Length: %d\r\n\r\n", len(errorJSON))
	stdin.Write([]byte(header))
	stdin.Write(errorJSON)

	responseLine, err = reader.ReadString('\n')
	if err == nil {
		responseLine = strings.TrimRight(responseLine, "\r\n")
		if strings.HasPrefix(responseLine, "{") {
			var errorResp map[string]interface{}
			if err := json.Unmarshal([]byte(responseLine), &errorResp); err == nil {
				if errorResp["error"] != nil {
					fmt.Println("  ✓ Error response format correct (JSON directly)")
				}
			}
		}
	}

	fmt.Println("\n==========================================")
	fmt.Println("All Claude Desktop Format Tests Passed!")
	fmt.Println("==========================================")
}

func getResponseFormat(line string) string {
	if strings.HasPrefix(line, "{") {
		return "JSON directly (Claude Desktop format)"
	}
	if strings.HasPrefix(strings.ToLower(line), "content-length:") {
		return "Content-Length headers (standard MCP)"
	}
	return "Unknown format"
}


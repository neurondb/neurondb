package client

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ParseCommand parses a command string into tool name and arguments
// Format: tool_name or tool_name:arg1=val1,arg2=val2
func ParseCommand(commandStr string) (string, map[string]interface{}, error) {
	commandStr = strings.TrimSpace(commandStr)
	if commandStr == "" {
		return "", nil, fmt.Errorf("empty command")
	}

	// Check if command has arguments
	parts := strings.SplitN(commandStr, ":", 2)
	toolName := strings.TrimSpace(parts[0])

	var arguments map[string]interface{}
	if len(parts) > 1 {
		var err error
		arguments, err = parseArguments(parts[1])
		if err != nil {
			return "", nil, fmt.Errorf("failed to parse arguments: %w", err)
		}
	} else {
		arguments = make(map[string]interface{})
	}

	return toolName, arguments, nil
}

// parseArguments parses argument string into map
// Format: arg1=val1,arg2=val2,arg3=[1,2,3]
func parseArguments(argsStr string) (map[string]interface{}, error) {
	args := make(map[string]interface{})
	if strings.TrimSpace(argsStr) == "" {
		return args, nil
	}

	// Simple parser for key=value pairs
	// Handles: strings, numbers, booleans, arrays, objects
	var currentKey string
	var currentValue strings.Builder
	inString := false
	inArray := false
	inObject := false
	stringChar := byte(0)
	bracketDepth := 0
	braceDepth := 0

	addArg := func() {
		if currentKey != "" {
			valueStr := currentValue.String()
			args[currentKey] = parseValue(valueStr)
			currentKey = ""
			currentValue.Reset()
		}
	}

	for i := 0; i < len(argsStr); i++ {
		char := argsStr[i]

		if (char == '"' || char == '\'') && !inArray && !inObject {
			if !inString {
				inString = true
				stringChar = char
			} else if char == stringChar {
				inString = false
				stringChar = 0
			} else {
				currentValue.WriteByte(char)
			}
		} else if inString {
			currentValue.WriteByte(char)
		} else if char == '[' {
			inArray = true
			bracketDepth++
			currentValue.WriteByte(char)
		} else if char == ']' {
			bracketDepth--
			currentValue.WriteByte(char)
			if bracketDepth == 0 {
				inArray = false
			}
		} else if char == '{' {
			inObject = true
			braceDepth++
			currentValue.WriteByte(char)
		} else if char == '}' {
			braceDepth--
			currentValue.WriteByte(char)
			if braceDepth == 0 {
				inObject = false
			}
		} else if char == '=' && !inArray && !inObject {
			if currentKey == "" {
				currentKey = strings.TrimSpace(currentValue.String())
				currentValue.Reset()
			}
		} else if char == ',' && !inArray && !inObject {
			addArg()
		} else {
			currentValue.WriteByte(char)
		}
	}

	// Add last argument
	addArg()

	return args, nil
}

// parseValue parses a value string into Go value
func parseValue(valueStr string) interface{} {
	valueStr = strings.TrimSpace(valueStr)

	// None/null
	if valueStr == "null" || valueStr == "none" || valueStr == "None" {
		return nil
	}

	// Boolean
	if valueStr == "true" || valueStr == "True" {
		return true
	}
	if valueStr == "false" || valueStr == "False" {
		return false
	}

	// Try JSON parsing (for arrays, objects, numbers)
	var jsonValue interface{}
	if err := json.Unmarshal([]byte(valueStr), &jsonValue); err == nil {
		return jsonValue
	}

	// Try number parsing
	if strings.Contains(valueStr, ".") {
		if f, err := parseFloat(valueStr); err == nil {
			return f
		}
	} else {
		if i, err := parseInt(valueStr); err == nil {
			return i
		}
	}

	// Remove quotes if present
	if len(valueStr) >= 2 {
		if (valueStr[0] == '"' && valueStr[len(valueStr)-1] == '"') ||
			(valueStr[0] == '\'' && valueStr[len(valueStr)-1] == '\'') {
			return valueStr[1 : len(valueStr)-1]
		}
	}

	// Return as string
	return valueStr
}

func parseInt(s string) (int, error) {
	var result int
	_, err := fmt.Sscanf(s, "%d", &result)
	return result, err
}

func parseFloat(s string) (float64, error) {
	var result float64
	_, err := fmt.Sscanf(s, "%f", &result)
	return result, err
}


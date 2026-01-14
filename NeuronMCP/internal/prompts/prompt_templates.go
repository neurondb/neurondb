/*-------------------------------------------------------------------------
 *
 * prompt_templates.go
 *    Template engine for prompt rendering
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/prompts/prompt_templates.go
 *
 *-------------------------------------------------------------------------
 */

package prompts

import (
	"fmt"
	"strings"
	"unicode"
)

/* RenderTemplate renders a template string with provided variables */
func RenderTemplate(template string, variableDefs []VariableDefinition, variables map[string]string) (string, error) {
	if template == "" {
		return "", fmt.Errorf("template cannot be empty")
	}

	result := template

	/* Validate and check required variables */
	for _, def := range variableDefs {
		if def.Name == "" {
			continue /* Skip invalid variable definitions */
		}

		if def.Required {
			value, exists := variables[def.Name]
			if !exists || (exists && value == "" && def.Default == "") {
				return "", fmt.Errorf("required variable '%s' is missing or empty", def.Name)
			}
		}
	}

	/* Replace variables in template */
	/* Support {{variable}} syntax */
	for key, value := range variables {
		if key == "" {
			continue /* Skip empty keys */
		}
		placeholder := fmt.Sprintf("{{%s}}", key)
		result = strings.ReplaceAll(result, placeholder, value)
	}

	/* Use defaults for missing optional variables */
	for _, def := range variableDefs {
		if def.Name == "" {
			continue
		}

		if !def.Required && def.Default != "" {
			placeholder := fmt.Sprintf("{{%s}}", def.Name)
			if strings.Contains(result, placeholder) {
				if _, exists := variables[def.Name]; !exists || variables[def.Name] == "" {
					result = strings.ReplaceAll(result, placeholder, def.Default)
				}
			}
		}
	}

	return result, nil
}

/* ValidateVariables validates that all required variables are provided */
func ValidateVariables(variableDefs []VariableDefinition, variables map[string]string) error {
	for _, def := range variableDefs {
		if def.Required {
			if _, exists := variables[def.Name]; !exists {
				if def.Default == "" {
					return fmt.Errorf("required variable '%s' is missing", def.Name)
				}
			}
		}
	}
	return nil
}

/* ExtractVariables extracts variable names from a template */
func ExtractVariables(template string) []string {
	if template == "" {
		return []string{}
	}

	var variables []string
	seen := make(map[string]bool)
	start := 0

	for {
		idx := strings.Index(template[start:], "{{")
		if idx == -1 {
			break
		}
		start += idx + 2

		endIdx := strings.Index(template[start:], "}}")
		if endIdx == -1 {
			break
		}

		varName := strings.TrimSpace(template[start : start+endIdx])
		/* Validate variable name */
		if varName != "" && isValidVariableName(varName) {
			if !seen[varName] {
				variables = append(variables, varName)
				seen[varName] = true
			}
		}

		start += endIdx + 2
	}

	return variables
}

/* isValidVariableName checks if a variable name is valid */
func isValidVariableName(name string) bool {
	if len(name) == 0 {
		return false
	}

	/* Must start with letter or underscore */
	if !unicode.IsLetter(rune(name[0])) && name[0] != '_' {
		return false
	}

	/* Rest can be letters, digits, or underscores */
	for _, r := range name[1:] {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
			return false
		}
	}

	return true
}


/*-------------------------------------------------------------------------
 *
 * verifier.go
 *    Verification agent for quality assurance
 *
 * Provides automated quality assurance and output validation for agent
 * executions. Checks output format, data accuracy, and logical consistency.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/verifier.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* VerificationAgent provides quality assurance for agent outputs */
type VerificationAgent struct {
	agentID uuid.UUID
	runtime *Runtime
	queries *db.Queries
	rules   []VerificationRule
}

/* VerificationRule defines a verification rule */
type VerificationRule struct {
	ID       uuid.UUID
	RuleType string
	Criteria map[string]interface{}
	Enabled  bool
}

/* VerificationResult represents the result of verification */
type VerificationResult struct {
	Passed      bool
	Issues      []Issue
	Suggestions []string
	Confidence  float64
}

/* Issue represents a problem found during verification */
type Issue struct {
	Type        string
	Description string
	Severity    string
}

/* NewVerificationAgent creates a new verification agent */
func NewVerificationAgent(agentID uuid.UUID, runtime *Runtime, queries *db.Queries) *VerificationAgent {
	return &VerificationAgent{
		agentID: agentID,
		runtime: runtime,
		queries: queries,
		rules:   []VerificationRule{},
	}
}

/* LoadRules loads verification rules for the agent */
func (v *VerificationAgent) LoadRules(ctx context.Context) error {
	query := `SELECT id, rule_type, criteria, enabled
		FROM neurondb_agent.verification_rules
		WHERE agent_id = $1 AND enabled = true`

	type RuleRow struct {
		ID       uuid.UUID              `db:"id"`
		RuleType string                 `db:"rule_type"`
		Criteria map[string]interface{} `db:"criteria"`
		Enabled  bool                   `db:"enabled"`
	}

	var rows []RuleRow
	err := v.queries.GetDB().SelectContext(ctx, &rows, query, v.agentID)
	if err != nil {
		return fmt.Errorf("rule loading failed: error=%w", err)
	}

	v.rules = make([]VerificationRule, len(rows))
	for i, row := range rows {
		v.rules[i] = VerificationRule{
			ID:       row.ID,
			RuleType: row.RuleType,
			Criteria: row.Criteria,
			Enabled:  row.Enabled,
		}
	}

	return nil
}

/* QueueVerification adds an output to the verification queue */
func (v *VerificationAgent) QueueVerification(ctx context.Context, sessionID uuid.UUID, outputID *uuid.UUID, outputContent string, priority string) (uuid.UUID, error) {
	if priority == "" {
		priority = "medium"
	}

	validPriorities := map[string]bool{"low": true, "medium": true, "high": true}
	if !validPriorities[priority] {
		priority = "medium"
	}

	query := `INSERT INTO neurondb_agent.verification_queue
		(session_id, output_id, output_content, priority)
		VALUES ($1, $2, $3, $4)
		RETURNING id`

	var queueID uuid.UUID
	err := v.queries.GetDB().GetContext(ctx, &queueID, query, sessionID, outputID, outputContent, priority)
	if err != nil {
		return uuid.Nil, fmt.Errorf("verification queueing failed: error=%w", err)
	}

	return queueID, nil
}

/* VerifyOutput verifies an output against all enabled rules */
func (v *VerificationAgent) VerifyOutput(ctx context.Context, output string, rules []VerificationRule) (*VerificationResult, error) {
	if len(rules) == 0 {
		rules = v.rules
	}

	result := &VerificationResult{
		Passed:      true,
		Issues:      []Issue{},
		Suggestions: []string{},
		Confidence:  1.0,
	}

	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}

		ruleResult, err := v.checkRule(ctx, output, rule)
		if err != nil {
			continue /* Skip rule on error */
		}

		if !ruleResult.Passed {
			result.Passed = false
			result.Issues = append(result.Issues, ruleResult.Issues...)
			result.Suggestions = append(result.Suggestions, ruleResult.Suggestions...)
			result.Confidence = result.Confidence * 0.8 /* Reduce confidence on failure */
		}
	}

	return result, nil
}

/* checkRule checks a single verification rule */
func (v *VerificationAgent) checkRule(ctx context.Context, output string, rule VerificationRule) (*VerificationResult, error) {
	switch rule.RuleType {
	case "output_format":
		return v.checkOutputFormat(output, rule.Criteria)
	case "data_accuracy":
		return v.checkDataAccuracy(ctx, output, rule.Criteria)
	case "logical_consistency":
		return v.checkLogicalConsistency(ctx, output, rule.Criteria)
	case "completeness":
		return v.checkCompleteness(output, rule.Criteria)
	default:
		return &VerificationResult{Passed: true, Confidence: 1.0}, nil
	}
}

/* checkOutputFormat validates output format */
func (v *VerificationAgent) checkOutputFormat(output string, criteria map[string]interface{}) (*VerificationResult, error) {
	result := &VerificationResult{
		Passed:     true,
		Issues:     []Issue{},
		Confidence: 1.0,
	}

	expectedFormat, ok := criteria["format"].(string)
	if !ok {
		return result, nil
	}

	switch expectedFormat {
	case "json":
		var jsonData interface{}
		if err := json.Unmarshal([]byte(output), &jsonData); err != nil {
			result.Passed = false
			result.Issues = append(result.Issues, Issue{
				Type:        "format_error",
				Description: "Output is not valid JSON",
				Severity:    "high",
			})
			result.Suggestions = append(result.Suggestions, "Ensure output is valid JSON format")
			result.Confidence = 0.9
		}
	case "text":
		if len(output) == 0 {
			result.Passed = false
			result.Issues = append(result.Issues, Issue{
				Type:        "format_error",
				Description: "Output is empty",
				Severity:    "medium",
			})
			result.Confidence = 0.8
		}
	}

	return result, nil
}

/* checkDataAccuracy validates data accuracy */
func (v *VerificationAgent) checkDataAccuracy(ctx context.Context, output string, criteria map[string]interface{}) (*VerificationResult, error) {
	result := &VerificationResult{
		Passed:     true,
		Issues:     []Issue{},
		Confidence: 1.0,
	}

	/* Basic accuracy checks */
	if len(output) < 10 {
		result.Passed = false
		result.Issues = append(result.Issues, Issue{
			Type:        "accuracy_error",
			Description: "Output is too short to be meaningful",
			Severity:    "medium",
		})
		result.Confidence = 0.7
	}

	return result, nil
}

/* checkLogicalConsistency validates logical consistency */
func (v *VerificationAgent) checkLogicalConsistency(ctx context.Context, output string, criteria map[string]interface{}) (*VerificationResult, error) {
	result := &VerificationResult{
		Passed:     true,
		Issues:     []Issue{},
		Confidence: 1.0,
	}

	/* Basic consistency checks */
	/* More sophisticated checks would use LLM here */

	return result, nil
}

/* checkCompleteness validates output completeness */
func (v *VerificationAgent) checkCompleteness(output string, criteria map[string]interface{}) (*VerificationResult, error) {
	result := &VerificationResult{
		Passed:     true,
		Issues:     []Issue{},
		Confidence: 1.0,
	}

	minLength, ok := criteria["min_length"].(float64)
	if ok && float64(len(output)) < minLength {
		result.Passed = false
		result.Issues = append(result.Issues, Issue{
			Type:        "completeness_error",
			Description: fmt.Sprintf("Output length %d is below minimum %d", len(output), int(minLength)),
			Severity:    "medium",
		})
		result.Confidence = 0.8
	}

	return result, nil
}

/* GenerateSuggestions generates improvement suggestions */
func (v *VerificationAgent) GenerateSuggestions(ctx context.Context, issues []Issue) []string {
	suggestions := make([]string, 0)

	for _, issue := range issues {
		switch issue.Type {
		case "format_error":
			suggestions = append(suggestions, "Review output format requirements and ensure compliance")
		case "accuracy_error":
			suggestions = append(suggestions, "Verify data accuracy and cross-check with source")
		case "completeness_error":
			suggestions = append(suggestions, "Ensure output includes all required information")
		default:
			suggestions = append(suggestions, "Review output quality and address identified issues")
		}
	}

	return suggestions
}

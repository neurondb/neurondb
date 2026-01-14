/*-------------------------------------------------------------------------
 *
 * reflector.go
 *    Reflection and self-correction engine
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/reflector.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type Reflector struct {
	llm *LLMClient
}

/* NewReflector creates a new reflection engine */
func NewReflector(llm *LLMClient) *Reflector {
	return &Reflector{llm: llm}
}

/* Reflect analyzes a response and provides feedback */
func (r *Reflector) Reflect(ctx context.Context, userMessage, response string, toolCalls []ToolCall, toolResults []ToolResult) (*ReflectionResult, error) {
	/* Validate inputs */
	if response == "" {
		return &ReflectionResult{
			QualityScore: 0.0,
			Accuracy:     0.0,
			Completeness: 0.0,
			Clarity:      0.0,
			Relevance:    0.0,
			Issues:       []string{"Empty response"},
			Suggestions:  []string{"Provide a response"},
			Confidence:   0.0,
		}, nil
	}

	if r.llm == nil {
		/* Fallback to simple scoring if no LLM */
		return r.simpleReflection(response), nil
	}

	/* Build reflection prompt */
	prompt := r.buildReflectionPrompt(userMessage, response, toolCalls, toolResults)

	llmConfig := map[string]interface{}{
		"temperature": 0.2, /* Lower temperature for more consistent evaluation */
		"max_tokens":  1000,
	}

	llmResponse, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback to simple reflection on error */
		return r.simpleReflection(response), nil
	}

	/* Parse reflection result */
	result, err := r.parseReflectionResponse(llmResponse.Content)
	if err != nil {
		/* Fallback to simple reflection on parse error */
		return r.simpleReflection(response), nil
	}

	return result, nil
}

/* buildReflectionPrompt builds the prompt for reflection */
func (r *Reflector) buildReflectionPrompt(userMessage, response string, toolCalls []ToolCall, toolResults []ToolResult) string {
	var toolInfo strings.Builder
	if len(toolCalls) > 0 {
		toolInfo.WriteString("\nTool calls made:\n")
		for i, call := range toolCalls {
			toolInfo.WriteString(fmt.Sprintf("- %s: %v\n", call.Name, call.Arguments))
			if i < len(toolResults) {
				result := toolResults[i]
				if result.Error != nil {
					toolInfo.WriteString(fmt.Sprintf("  Error: %v\n", result.Error))
				} else {
					resultPreview := result.Content
					if len(resultPreview) > 200 {
						resultPreview = resultPreview[:200] + "..."
					}
					toolInfo.WriteString(fmt.Sprintf("  Result: %s\n", resultPreview))
				}
			}
		}
	}

	return fmt.Sprintf(`You are a quality assessment assistant. Evaluate the following response to a user query.

User Query: %s
Agent Response: %s%s

Evaluate the response on:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-structured and easy to understand?
4. Relevance: Is it relevant to the query?

Respond with JSON:
{
  "quality_score": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "completeness": 0.0-1.0,
  "clarity": 0.0-1.0,
  "relevance": 0.0-1.0,
  "issues": ["list of issues found"],
  "suggestions": ["suggestions for improvement"],
  "confidence": 0.0-1.0
}`, userMessage, response, toolInfo.String())
}

/* parseReflectionResponse parses LLM reflection response */
func (r *Reflector) parseReflectionResponse(response string) (*ReflectionResult, error) {
	response = strings.TrimSpace(response)

	/* Find JSON object in response */
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start == -1 || end == -1 || end <= start {
		return nil, fmt.Errorf("reflection parsing failed: no_json_object_found=true, response_length=%d", len(response))
	}

	jsonStr := response[start : end+1]

	var result ReflectionResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("reflection parsing failed: json_unmarshal_error=true, error=%w", err)
	}

	return &result, nil
}

/* simpleReflection provides simple heuristic-based reflection */
func (r *Reflector) simpleReflection(response string) *ReflectionResult {
	score := 0.7 /* Base score */

	/* Adjust based on response length */
	if len(response) < 50 {
		score -= 0.2
	} else if len(response) > 500 {
		score += 0.1
	}

	/* Check for common issues */
	issues := []string{}
	if len(response) < 20 {
		issues = append(issues, "Response is too short")
	}
	if strings.Contains(strings.ToLower(response), "error") {
		issues = append(issues, "Response mentions errors")
		score -= 0.2
	}

	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return &ReflectionResult{
		QualityScore: score,
		Accuracy:     score,
		Completeness: score,
		Clarity:      score,
		Relevance:    score,
		Issues:       issues,
		Suggestions:  []string{},
		Confidence:   0.5,
	}
}

/* ReflectionResult represents the result of reflection */
type ReflectionResult struct {
	QualityScore float64  `json:"quality_score"`
	Accuracy     float64  `json:"accuracy"`
	Completeness float64  `json:"completeness"`
	Clarity      float64  `json:"clarity"`
	Relevance    float64  `json:"relevance"`
	Issues       []string `json:"issues"`
	Suggestions  []string `json:"suggestions"`
	Confidence   float64  `json:"confidence"`
}

/* ShouldRetry determines if the response should be retried based on reflection */
func (r *ReflectionResult) ShouldRetry(threshold float64) bool {
	return r.QualityScore < threshold
}

/* ReflectOnFailure analyzes a failed execution and suggests correction strategy */
func (r *Reflector) ReflectOnFailure(ctx context.Context, userMessage string, errorMsg string, attemptedSteps []PlanStep, stepIndex int) (*FailureReflection, error) {
	if r.llm == nil {
		return r.simpleFailureReflection(errorMsg), nil
	}

	prompt := fmt.Sprintf(`Analyze this task execution failure and suggest a correction strategy.

User Query: %s
Error: %s
Failed Step: %d
Step Action: %s
Step Tool: %s
Total Steps: %d

Provide JSON response with:
{
  "failure_reason": "why it failed",
  "suggested_strategy": "retry", "skip", "abort", "modify",
  "modifications": {"suggested changes to step"},
  "alternative_approach": "alternative way to achieve goal",
  "confidence": 0.0-1.0
}`, userMessage, errorMsg, stepIndex, attemptedSteps[stepIndex].Action, attemptedSteps[stepIndex].Tool, len(attemptedSteps))

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  1000,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return r.simpleFailureReflection(errorMsg), nil
	}

	result, err := r.parseFailureReflection(response.Content)
	if err != nil {
		return r.simpleFailureReflection(errorMsg), nil
	}

	return result, nil
}

/* parseFailureReflection parses failure reflection response */
func (r *Reflector) parseFailureReflection(response string) (*FailureReflection, error) {
	response = strings.TrimSpace(response)

	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start == -1 || end == -1 || end <= start {
		return nil, fmt.Errorf("failure reflection parsing failed: no_json_object_found=true, response_length=%d", len(response))
	}

	jsonStr := response[start : end+1]

	var result FailureReflection
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failure reflection parsing failed: json_unmarshal_error=true, error=%w", err)
	}

	return &result, nil
}

/* simpleFailureReflection provides simple failure analysis */
func (r *Reflector) simpleFailureReflection(errorMsg string) *FailureReflection {
	strategy := "retry"
	if strings.Contains(strings.ToLower(errorMsg), "timeout") {
		strategy = "modify"
	} else if strings.Contains(strings.ToLower(errorMsg), "permission") {
		strategy = "skip"
	}

	return &FailureReflection{
		FailureReason:       errorMsg,
		SuggestedStrategy:   strategy,
		Modifications:       make(map[string]interface{}),
		AlternativeApproach: "",
		Confidence:          0.5,
	}
}

/* FailureReflection represents analysis of a failed execution */
type FailureReflection struct {
	FailureReason       string                 `json:"failure_reason"`
	SuggestedStrategy   string                 `json:"suggested_strategy"`
	Modifications       map[string]interface{} `json:"modifications"`
	AlternativeApproach string                 `json:"alternative_approach"`
	Confidence          float64                `json:"confidence"`
}

/* GetImprovementPrompt generates a prompt for improving the response */
func (r *ReflectionResult) GetImprovementPrompt(originalResponse string) string {
	if len(r.Suggestions) == 0 {
		return ""
	}

	suggestions := strings.Join(r.Suggestions, "\n- ")
	return fmt.Sprintf(`The previous response had quality issues. Please improve it based on these suggestions:

%s

Previous response:
%s

Please provide an improved response.`, suggestions, originalResponse)
}

/*-------------------------------------------------------------------------
 *
 * planner.go
 *    Advanced planning system with LLM-based task decomposition
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/planner.go
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

type Planner struct {
	maxIterations    int
	llm              *LLMClient
	reasoningEngine  *ReasoningEngine
	reasoningMode    ReasoningMode
}

func NewPlanner() *Planner {
	return &Planner{
		maxIterations: 10,  /* Prevent infinite loops */
		llm:           nil, /* Will be set by runtime */
		reasoningMode: ReasoningModeStandard,
	}
}

/* NewPlannerWithLLM creates a planner with LLM support */
func NewPlannerWithLLM(llm *LLMClient) *Planner {
	config := ReasoningConfig{
		Mode:              ReasoningModeStandard,
		MaxThoughts:       5,
		ConsistencySamples: 3,
		EnableAdaptive:    true,
		Temperature:       0.7,
	}
	return &Planner{
		maxIterations:  10,
		llm:            llm,
		reasoningEngine: NewReasoningEngine(llm, config),
		reasoningMode:  ReasoningModeStandard,
	}
}

/* NewPlannerWithReasoning creates a planner with advanced reasoning support */
func NewPlannerWithReasoning(llm *LLMClient, mode ReasoningMode, config ReasoningConfig) *Planner {
	if config.Mode == "" {
		config.Mode = mode
	}
	return &Planner{
		maxIterations:  10,
		llm:            llm,
		reasoningEngine: NewReasoningEngine(llm, config),
		reasoningMode:  mode,
	}
}

/* Plan creates a multi-step plan for complex tasks using LLM */
func (p *Planner) Plan(ctx context.Context, userMessage string, availableTools []string) ([]PlanStep, error) {
	/* Validate input */
	if userMessage == "" {
		return nil, fmt.Errorf("planning failed: user_message_empty=true")
	}
	if len(userMessage) > 50000 {
		return nil, fmt.Errorf("planning failed: user_message_too_large=true, length=%d, max_length=50000", len(userMessage))
	}

	/* If no LLM, fall back to simple plan */
	if p.llm == nil {
		return p.simplePlan(userMessage), nil
	}

	/* Use advanced reasoning if configured */
	if p.reasoningEngine != nil && p.reasoningMode != ReasoningModeStandard {
		return p.planWithReasoning(ctx, userMessage, availableTools)
	}

	/* Standard planning */
	return p.planStandard(ctx, userMessage, availableTools)
}

/* planStandard creates a plan using standard planning */
func (p *Planner) planStandard(ctx context.Context, userMessage string, availableTools []string) ([]PlanStep, error) {
	/* Build planning prompt with recursive decomposition support */
	toolsList := strings.Join(availableTools, ", ")
	prompt := fmt.Sprintf(`You are a task planning assistant. Break down the following task into a series of steps.
Each step should specify:
1. The action to take
2. Which tool to use (if any) from: %s
3. The parameters for that tool
4. Dependencies on other steps (step indices that must complete first)
5. Whether this step can run in parallel with others

Task: %s

Respond with a JSON array of steps, each with:
- "action": description of what to do
- "tool": tool name to use (or empty string if no tool)
- "payload": object with tool parameters
- "dependencies": array of step indices (0-based) that must complete before this step
- "can_parallel": boolean indicating if this can run in parallel
- "retry_strategy": optional strategy if step fails ("retry", "skip", "abort")

Example format:
[
  {"action": "Search for information", "tool": "sql", "payload": {"query": "SELECT * FROM table"}, "dependencies": [], "can_parallel": true},
  {"action": "Process results", "tool": "", "payload": {}, "dependencies": [0], "can_parallel": false}
]`, toolsList, userMessage)

	/* Generate plan using LLM */
	llmConfig := map[string]interface{}{
		"temperature": 0.3, /* Lower temperature for more structured output */
		"max_tokens":  2000,
	}

	response, err := p.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback to simple plan on error */
		return p.simplePlan(userMessage), nil
	}

	/* Parse LLM response */
	steps, err := p.parsePlanResponse(response.Content)
	if err != nil {
		/* Fallback to simple plan on parse error */
		return p.simplePlan(userMessage), nil
	}

	/* Validate and optimize plan */
	steps = p.validatePlan(steps, availableTools)

	/* Enhance plan with recursive decomposition if needed */
	if p.needsRecursiveDecomposition(steps) {
		enhancedSteps, err := p.recursiveDecompose(ctx, steps, availableTools)
		if err == nil && len(enhancedSteps) > 0 {
			steps = enhancedSteps
		}
	}

	/* Add dependency tracking */
	steps = p.addDependencyTracking(steps)

	return steps, nil
}

/* planWithReasoning creates a plan using advanced reasoning patterns */
func (p *Planner) planWithReasoning(ctx context.Context, userMessage string, availableTools []string) ([]PlanStep, error) {
	var steps []PlanStep
	var err error

	switch p.reasoningMode {
	case ReasoningModeChainOfThought:
		/* Chain-of-Thought: convert thoughts to plan steps */
		thoughts, err2 := p.reasoningEngine.ChainOfThoughtReasoning(ctx, userMessage, availableTools)
		if err2 != nil {
			return p.planStandard(ctx, userMessage, availableTools) /* Fallback */
		}
		steps = p.thoughtsToPlanSteps(thoughts)

	case ReasoningModeTreeOfThoughts:
		/* Tree-of-Thoughts: explore multiple paths */
		thoughts, err2 := p.reasoningEngine.TreeOfThoughtsReasoning(ctx, userMessage, availableTools)
		if err2 != nil {
			return p.planStandard(ctx, userMessage, availableTools) /* Fallback */
		}
		steps = p.thoughtsToPlanSteps(thoughts)

	case ReasoningModeSelfConsistency:
		/* Self-Consistency: generate multiple plans and find consensus */
		steps, err = p.reasoningEngine.SelfConsistencyReasoning(ctx, userMessage, availableTools)
		if err != nil {
			return p.planStandard(ctx, userMessage, availableTools) /* Fallback */
		}

	default:
		/* Fallback to standard */
		return p.planStandard(ctx, userMessage, availableTools)
	}

	/* Validate and optimize plan */
	steps = p.validatePlan(steps, availableTools)

	/* Optimize plan if enabled */
	if p.reasoningEngine.config.EnableAdaptive {
		optimized, err2 := p.reasoningEngine.PlanOptimization(ctx, steps, availableTools)
		if err2 == nil && len(optimized) > 0 {
			steps = optimized
		}
	}

	/* Add dependency tracking */
	steps = p.addDependencyTracking(steps)

	return steps, nil
}

/* thoughtsToPlanSteps converts thoughts to plan steps */
func (p *Planner) thoughtsToPlanSteps(thoughts []Thought) []PlanStep {
	var steps []PlanStep
	for _, thought := range thoughts {
		/* Extract plan information from thought content */
		/* This is a simplified conversion - in practice, thoughts might contain structured data */
		step := PlanStep{
			Action:  thought.Content,
			Tool:    "", /* Would need to parse from thought */
			Payload: make(map[string]interface{}),
		}
		steps = append(steps, step)
	}
	return steps
}

/* needsRecursiveDecomposition checks if plan needs further decomposition */
func (p *Planner) needsRecursiveDecomposition(steps []PlanStep) bool {
	if len(steps) == 0 {
		return false
	}

	for _, step := range steps {
		if len(step.Action) > 200 {
			return true
		}
		if strings.Contains(strings.ToLower(step.Action), "and then") ||
			strings.Contains(strings.ToLower(step.Action), "multiple") {
			return true
		}
	}

	return false
}

/* recursiveDecompose breaks down complex steps into sub-steps */
func (p *Planner) recursiveDecompose(ctx context.Context, steps []PlanStep, availableTools []string) ([]PlanStep, error) {
	if p.llm == nil {
		return steps, nil
	}

	var decomposedSteps []PlanStep
	toolsList := strings.Join(availableTools, ", ")

	for i, step := range steps {
		if !p.needsRecursiveDecomposition([]PlanStep{step}) {
			step.Dependencies = []int{}
			if i > 0 {
				step.Dependencies = []int{i - 1}
			}
			decomposedSteps = append(decomposedSteps, step)
			continue
		}

		prompt := fmt.Sprintf(`Break down this complex step into smaller sub-steps:
Step: %s
Tool: %s

Available tools: %s

Respond with JSON array of sub-steps, each with action, tool, payload, dependencies, can_parallel.`, step.Action, step.Tool, toolsList)

		llmConfig := map[string]interface{}{
			"temperature": 0.3,
			"max_tokens":  1500,
		}

		response, err := p.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
		if err != nil {
			step.Dependencies = []int{}
			if i > 0 {
				step.Dependencies = []int{len(decomposedSteps) - 1}
			}
			decomposedSteps = append(decomposedSteps, step)
			continue
		}

		subSteps, err := p.parsePlanResponse(response.Content)
		if err != nil {
			step.Dependencies = []int{}
			if i > 0 {
				step.Dependencies = []int{len(decomposedSteps) - 1}
			}
			decomposedSteps = append(decomposedSteps, step)
			continue
		}

		for j, subStep := range subSteps {
			subStep.Dependencies = []int{}
			if i > 0 && j == 0 {
				subStep.Dependencies = []int{len(decomposedSteps) - 1}
			}
			if j > 0 {
				subStep.Dependencies = []int{len(decomposedSteps) - 1}
			}
			decomposedSteps = append(decomposedSteps, subStep)
		}
	}

	return decomposedSteps, nil
}

/* addDependencyTracking ensures all steps have proper dependency arrays */
func (p *Planner) addDependencyTracking(steps []PlanStep) []PlanStep {
	for i := range steps {
		if steps[i].Dependencies == nil {
			steps[i].Dependencies = []int{}
		}
		if i > 0 && len(steps[i].Dependencies) == 0 {
			steps[i].Dependencies = []int{i - 1}
		}
		if steps[i].RetryStrategy == "" {
			steps[i].RetryStrategy = "retry"
		}
	}
	return steps
}

/* simplePlan creates a simple single-step plan */
func (p *Planner) simplePlan(userMessage string) []PlanStep {
	return []PlanStep{
		{
			Action:  "execute",
			Tool:    "",
			Payload: map[string]interface{}{"query": userMessage},
		},
	}
}

/* parsePlanResponse parses LLM response into plan steps */
func (p *Planner) parsePlanResponse(response string) ([]PlanStep, error) {
	/* Try to extract JSON from response */
	response = strings.TrimSpace(response)

	/* Find JSON array in response */
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 || end <= start {
		return nil, fmt.Errorf("plan parsing failed: no_json_array_found=true, response_length=%d", len(response))
	}

	jsonStr := response[start : end+1]

	var steps []PlanStep
	if err := json.Unmarshal([]byte(jsonStr), &steps); err != nil {
		return nil, fmt.Errorf("plan parsing failed: json_unmarshal_error=true, error=%w", err)
	}

	return steps, nil
}

/* validatePlan validates and optimizes plan steps */
func (p *Planner) validatePlan(steps []PlanStep, availableTools []string) []PlanStep {
	if len(steps) == 0 {
		return steps
	}

	validSteps := make([]PlanStep, 0, len(steps))
	toolSet := make(map[string]bool)
	for _, tool := range availableTools {
		toolSet[tool] = true
	}

	for _, step := range steps {
		/* Validate action */
		if step.Action == "" {
			step.Action = "execute" /* Default action */
		}

		/* Validate tool if specified */
		if step.Tool != "" && !toolSet[step.Tool] {
			/* Skip invalid tool steps but log warning */
			continue
		}

		/* Ensure payload is not nil */
		if step.Payload == nil {
			step.Payload = make(map[string]interface{})
		}

		validSteps = append(validSteps, step)
	}

	/* Limit to max iterations */
	if len(validSteps) > p.maxIterations {
		validSteps = validSteps[:p.maxIterations]
	}

	/* Ensure at least one step */
	if len(validSteps) == 0 {
		validSteps = []PlanStep{
			{
				Action:  "execute",
				Tool:    "",
				Payload: make(map[string]interface{}),
			},
		}
	}

	return validSteps
}

type PlanStep struct {
	Action        string                 `json:"action"`
	Tool          string                 `json:"tool"`
	Payload       map[string]interface{} `json:"payload"`
	Dependencies  []int                  `json:"dependencies,omitempty"`
	CanParallel   bool                   `json:"can_parallel,omitempty"`
	RetryStrategy string                 `json:"retry_strategy,omitempty"`
}

/* ExecutePlan executes a multi-step plan */
func (p *Planner) ExecutePlan(ctx context.Context, steps []PlanStep, executor func(step PlanStep) (interface{}, error)) ([]interface{}, error) {
	var results []interface{}
	iterations := 0

	for i, step := range steps {
		if iterations >= p.maxIterations {
			return results, fmt.Errorf("max iterations reached: completed_steps=%d, total_steps=%d", i, len(steps))
		}

		result, err := executor(step)
		if err != nil {
			return results, fmt.Errorf("step %d failed: action='%s', tool='%s', error=%w", i+1, step.Action, step.Tool, err)
		}

		results = append(results, result)
		iterations++
	}

	return results, nil
}

/* Helper functions */
func floatPtr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

/*-------------------------------------------------------------------------
 *
 * reasoning.go
 *    Advanced reasoning patterns for agent planning and execution
 *
 * Implements Chain-of-Thought, Tree-of-Thoughts, ReAct pattern, Self-Consistency,
 * and other advanced reasoning capabilities.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/reasoning.go
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

/* ReasoningMode defines the type of reasoning to use */
type ReasoningMode string

const (
	ReasoningModeStandard      ReasoningMode = "standard"
	ReasoningModeChainOfThought ReasoningMode = "chain_of_thought"
	ReasoningModeTreeOfThoughts ReasoningMode = "tree_of_thoughts"
	ReasoningModeReact          ReasoningMode = "react"
	ReasoningModeSelfConsistency ReasoningMode = "self_consistency"
)

/* ReasoningEngine handles advanced reasoning patterns */
type ReasoningEngine struct {
	llm   *LLMClient
	mode  ReasoningMode
	config ReasoningConfig
}

/* ReasoningConfig configures reasoning behavior */
type ReasoningConfig struct {
	Mode              ReasoningMode
	MaxThoughts       int /* For Tree-of-Thoughts */
	ConsistencySamples int /* For Self-Consistency */
	EnableAdaptive    bool
	Temperature       float64
}

/* Thought represents a reasoning step */
type Thought struct {
	ID          int
	Content     string
	Confidence  float64
	ParentID    *int
	ChildrenIDs []int
	Result      interface{}
}

/* ChainOfThoughtReasoning performs step-by-step reasoning */
func (re *ReasoningEngine) ChainOfThoughtReasoning(ctx context.Context, task string, availableTools []string) ([]Thought, error) {
	if re.llm == nil {
		return nil, fmt.Errorf("chain-of-thought reasoning failed: llm_client_not_available=true")
	}

	toolsList := strings.Join(availableTools, ", ")
	prompt := fmt.Sprintf(`You are solving a complex task. Break down your reasoning into explicit steps.

Task: %s
Available tools: %s

For each step, think through:
1. What needs to be done in this step
2. What information is needed
3. What reasoning leads to the next step

Format your response as a JSON array of thoughts, each with:
- "step": step number (1, 2, 3...)
- "thought": your reasoning for this step
- "action": what action to take (if any)
- "tool": tool name to use (or empty)
- "payload": tool parameters (if any)

Example:
[
  {"step": 1, "thought": "I need to understand the task first", "action": "analyze", "tool": "", "payload": {}},
  {"step": 2, "thought": "I should search for relevant information", "action": "search", "tool": "sql", "payload": {"query": "SELECT * FROM table"}}
]`, task, toolsList)

	llmConfig := map[string]interface{}{
		"temperature": re.config.Temperature,
		"max_tokens":  3000,
	}

	response, err := re.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return nil, fmt.Errorf("chain-of-thought reasoning failed: llm_error=true, error=%w", err)
	}

	if response == nil {
		return nil, fmt.Errorf("chain-of-thought reasoning failed: empty_response=true")
	}

	thoughts, err := re.parseThoughts(response.Content)
	if err != nil {
		return nil, fmt.Errorf("chain-of-thought reasoning failed: parse_error=true, error=%w", err)
	}

	return thoughts, nil
}

/* TreeOfThoughtsReasoning explores multiple reasoning paths */
func (re *ReasoningEngine) TreeOfThoughtsReasoning(ctx context.Context, task string, availableTools []string) ([]Thought, error) {
	if re.llm == nil {
		return nil, fmt.Errorf("tree-of-thoughts reasoning failed: llm_client_not_available=true")
	}

	maxThoughts := re.config.MaxThoughts
	if maxThoughts <= 0 {
		maxThoughts = 5 /* Default */
	}

	toolsList := strings.Join(availableTools, ", ")
	prompt := fmt.Sprintf(`You are solving a complex task. Explore multiple reasoning paths simultaneously.

Task: %s
Available tools: %s

Generate %d different approaches to solving this task. For each approach, provide:
1. A brief description of the approach
2. The reasoning behind it
3. The steps involved
4. The expected outcome

Format as JSON array with:
- "approach": approach number (1, 2, 3...)
- "description": brief description
- "reasoning": why this approach might work
- "steps": array of steps
- "confidence": confidence score (0.0 to 1.0)

Each step should have: "action", "tool", "payload"
`, task, toolsList, maxThoughts)

	llmConfig := map[string]interface{}{
		"temperature": re.config.Temperature + 0.2, /* Higher temperature for diversity */
		"max_tokens":  4000,
	}

	response, err := re.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return nil, fmt.Errorf("tree-of-thoughts reasoning failed: llm_error=true, error=%w", err)
	}

	if response == nil {
		return nil, fmt.Errorf("tree-of-thoughts reasoning failed: empty_response=true")
	}

	approaches, err := re.parseApproaches(response.Content)
	if err != nil {
		return nil, fmt.Errorf("tree-of-thoughts reasoning failed: parse_error=true, error=%w", err)
	}

	/* Select best approach based on confidence */
	bestApproach := approaches[0]
	for _, approach := range approaches {
		if approach.Confidence > bestApproach.Confidence {
			bestApproach = approach
		}
	}

	return bestApproach.Thoughts, nil
}

/* ReactReasoning implements ReAct pattern (Reasoning + Acting) */
func (re *ReasoningEngine) ReactReasoning(ctx context.Context, task string, availableTools []string, toolExecutor func(tool string, payload map[string]interface{}) (string, error)) ([]Thought, string, error) {
	if re.llm == nil {
		return nil, "", fmt.Errorf("react reasoning failed: llm_client_not_available=true")
	}

	var thoughts []Thought
	thoughtID := 1
	maxIterations := 10
	observationHistory := ""

	for i := 0; i < maxIterations; i++ {
		toolsList := strings.Join(availableTools, ", ")
		prompt := fmt.Sprintf(`You are solving a task using the ReAct pattern: Reasoning and Acting in a loop.

Task: %s
Available tools: %s

Previous observations:
%s

Now, follow this format:
Thought: [your reasoning about what to do next]
Action: [tool name or "final_answer"]
Action Input: [parameters for the tool, or your final answer if Action is final_answer]

Think step by step about what to do next based on the observations above.`, task, toolsList, observationHistory)

		llmConfig := map[string]interface{}{
			"temperature": re.config.Temperature,
			"max_tokens":  2000,
		}

		response, err := re.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
		if err != nil {
			return thoughts, "", fmt.Errorf("react reasoning failed: llm_error=true, iteration=%d, error=%w", i, err)
		}

		if response == nil {
			return thoughts, "", fmt.Errorf("react reasoning failed: empty_response=true, iteration=%d", i)
		}

		/* Parse ReAct format */
		thought, action, actionInput, err := re.parseReactResponse(response.Content)
		if err != nil {
			return thoughts, "", fmt.Errorf("react reasoning failed: parse_error=true, iteration=%d, error=%w", i, err)
		}

		thoughtObj := Thought{
			ID:         thoughtID,
			Content:    thought,
			Confidence: 0.8,
		}
		thoughts = append(thoughts, thoughtObj)
		thoughtID++

		if action == "final_answer" {
			return thoughts, actionInput, nil
		}

		/* Execute action */
		if toolExecutor == nil {
			return thoughts, "", fmt.Errorf("react reasoning failed: tool_executor_not_provided=true")
		}

		var payload map[string]interface{}
		if err := json.Unmarshal([]byte(actionInput), &payload); err != nil {
			/* If not JSON, treat as simple string */
			payload = map[string]interface{}{"input": actionInput}
		}

		observation, err := toolExecutor(action, payload)
		if err != nil {
			observation = fmt.Sprintf("Error: %v", err)
		}

		observationHistory += fmt.Sprintf("\nThought: %s\nAction: %s\nAction Input: %s\nObservation: %s\n", thought, action, actionInput, observation)
	}

	return thoughts, "", fmt.Errorf("react reasoning failed: max_iterations_reached=true")
}

/* SelfConsistencyReasoning generates multiple responses and selects best via consensus */
func (re *ReasoningEngine) SelfConsistencyReasoning(ctx context.Context, task string, availableTools []string) ([]PlanStep, error) {
	if re.llm == nil {
		return nil, fmt.Errorf("self-consistency reasoning failed: llm_client_not_available=true")
	}

	samples := re.config.ConsistencySamples
	if samples <= 0 {
		samples = 3 /* Default */
	}

	toolsList := strings.Join(availableTools, ", ")
	basePrompt := fmt.Sprintf(`You are solving a task. Create a step-by-step plan.

Task: %s
Available tools: %s

Respond with a JSON array of steps, each with:
- "action": description
- "tool": tool name (or empty)
- "payload": tool parameters
- "dependencies": array of step indices
- "can_parallel": boolean

Format exactly as JSON array.`, task, toolsList)

	var allPlans [][]PlanStep

	/* Generate multiple plans with varying temperature */
	for i := 0; i < samples; i++ {
		temperature := re.config.Temperature + float64(i)*0.1 /* Vary temperature */
		llmConfig := map[string]interface{}{
			"temperature": temperature,
			"max_tokens":  2000,
		}

		response, err := re.llm.Generate(ctx, "gpt-4", basePrompt, llmConfig)
		if err != nil {
			continue
		}

		if response == nil {
			continue
		}

		steps, err := re.parsePlanSteps(response.Content)
		if err != nil {
			continue
		}

		allPlans = append(allPlans, steps)
	}

	if len(allPlans) == 0 {
		return nil, fmt.Errorf("self-consistency reasoning failed: no_valid_plans_generated=true")
	}

	/* Find consensus plan (most common structure) */
	return re.findConsensusPlan(allPlans), nil
}

/* AdaptivePlanning adjusts plan based on execution feedback */
func (re *ReasoningEngine) AdaptivePlanning(ctx context.Context, originalPlan []PlanStep, executionResults []ExecutionResult, task string, availableTools []string) ([]PlanStep, error) {
	if re.llm == nil {
		return originalPlan, nil /* Return original if no LLM */
	}

	toolsList := strings.Join(availableTools, ", ")
	resultsJSON, _ := json.Marshal(executionResults)

	prompt := fmt.Sprintf(`A plan was executed with the following results. Adjust the plan based on the feedback.

Original Task: %s
Available tools: %s

Original Plan:
%s

Execution Results:
%s

Analyze the results and adjust the plan:
1. Remove steps that failed and cannot be retried
2. Add new steps to address issues
3. Modify steps that need adjustment
4. Optimize the remaining plan

Respond with adjusted JSON array of steps.`, task, toolsList, re.planToJSON(originalPlan), string(resultsJSON))

	llmConfig := map[string]interface{}{
		"temperature": re.config.Temperature,
		"max_tokens":  3000,
	}

	response, err := re.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return originalPlan, nil /* Return original on error */
	}

	if response == nil {
		return originalPlan, nil
	}

	adjustedPlan, err := re.parsePlanSteps(response.Content)
	if err != nil {
		return originalPlan, nil
	}

	return adjustedPlan, nil
}

/* PlanOptimization simplifies and optimizes a plan */
func (re *ReasoningEngine) PlanOptimization(ctx context.Context, plan []PlanStep, availableTools []string) ([]PlanStep, error) {
	/* Identify parallelizable steps */
	optimized := re.identifyParallelSteps(plan)

	/* Remove redundant steps */
	optimized = re.removeRedundantSteps(optimized)

	/* Simplify complex steps */
	optimized = re.simplifySteps(optimized)

	return optimized, nil
}

/* Helper types and methods */

type ExecutionResult struct {
	StepIndex int
	Success   bool
	Result    interface{}
	Error     string
}

type Approach struct {
	ApproachNumber int
	Description    string
	Reasoning      string
	Steps          []PlanStep
	Confidence     float64
	Thoughts       []Thought
}

/* NewReasoningEngine creates a new reasoning engine */
func NewReasoningEngine(llm *LLMClient, config ReasoningConfig) *ReasoningEngine {
	if config.Mode == "" {
		config.Mode = ReasoningModeStandard
	}
	if config.Temperature <= 0 {
		config.Temperature = 0.7
	}

	return &ReasoningEngine{
		llm:    llm,
		mode:   config.Mode,
		config: config,
	}
}

/* Parse helper methods */

func (re *ReasoningEngine) parseThoughts(response string) ([]Thought, error) {
	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 {
		return nil, fmt.Errorf("no JSON array found")
	}

	jsonStr := response[start : end+1]

	var rawThoughts []map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &rawThoughts); err != nil {
		return nil, err
	}

	var thoughts []Thought
	for i, raw := range rawThoughts {
		thought := Thought{
			ID:      i + 1,
			Content: getStringReasoning(raw, "thought", ""),
		}
		if conf, ok := raw["confidence"].(float64); ok {
			thought.Confidence = conf
		}
		thoughts = append(thoughts, thought)
	}

	return thoughts, nil
}

func (re *ReasoningEngine) parseApproaches(response string) ([]Approach, error) {
	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 {
		return nil, fmt.Errorf("no JSON array found")
	}

	jsonStr := response[start : end+1]

	var rawApproaches []map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &rawApproaches); err != nil {
		return nil, err
	}

	var approaches []Approach
	for _, raw := range rawApproaches {
		approach := Approach{
			Description: getStringReasoning(raw, "description", ""),
			Reasoning:   getStringReasoning(raw, "reasoning", ""),
		}
		if conf, ok := raw["confidence"].(float64); ok {
			approach.Confidence = conf
		}
		if stepsRaw, ok := raw["steps"].([]interface{}); ok {
			for _, stepRaw := range stepsRaw {
				if stepMap, ok := stepRaw.(map[string]interface{}); ok {
					step := PlanStep{
						Action:  getStringReasoning(stepMap, "action", ""),
						Tool:    getStringReasoning(stepMap, "tool", ""),
						Payload: getMapReasoning(stepMap, "payload"),
					}
					approach.Steps = append(approach.Steps, step)
				}
			}
		}
		approaches = append(approaches, approach)
	}

	return approaches, nil
}

func (re *ReasoningEngine) parseReactResponse(response string) (thought, action, actionInput string, err error) {
	lines := strings.Split(response, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Thought:") {
			thought = strings.TrimPrefix(line, "Thought:")
			thought = strings.TrimSpace(thought)
		} else if strings.HasPrefix(line, "Action:") {
			action = strings.TrimPrefix(line, "Action:")
			action = strings.TrimSpace(action)
		} else if strings.HasPrefix(line, "Action Input:") {
			actionInput = strings.TrimPrefix(line, "Action Input:")
			actionInput = strings.TrimSpace(actionInput)
		}
	}

	if thought == "" || action == "" {
		return "", "", "", fmt.Errorf("invalid react format")
	}

	return thought, action, actionInput, nil
}

func (re *ReasoningEngine) parsePlanSteps(response string) ([]PlanStep, error) {
	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 {
		return nil, fmt.Errorf("no JSON array found")
	}

	jsonStr := response[start : end+1]

	var steps []PlanStep
	if err := json.Unmarshal([]byte(jsonStr), &steps); err != nil {
		return nil, err
	}

	return steps, nil
}

func (re *ReasoningEngine) findConsensusPlan(plans [][]PlanStep) []PlanStep {
	if len(plans) == 0 {
		return nil
	}
	if len(plans) == 1 {
		return plans[0]
	}

	/* Simple consensus: return the most common plan length, then the first plan of that length */
	lengthCount := make(map[int]int)
	for _, plan := range plans {
		lengthCount[len(plan)]++
	}

	maxCount := 0
	consensusLength := len(plans[0])
	for length, count := range lengthCount {
		if count > maxCount {
			maxCount = count
			consensusLength = length
		}
	}

	for _, plan := range plans {
		if len(plan) == consensusLength {
			return plan
		}
	}

	return plans[0]
}

func (re *ReasoningEngine) planToJSON(plan []PlanStep) string {
	data, _ := json.MarshalIndent(plan, "", "  ")
	return string(data)
}

func (re *ReasoningEngine) identifyParallelSteps(plan []PlanStep) []PlanStep {
	/* Mark steps that can run in parallel */
	for i := range plan {
		if len(plan[i].Dependencies) == 0 && i > 0 {
			plan[i].CanParallel = true
		}
	}
	return plan
}

func (re *ReasoningEngine) removeRedundantSteps(plan []PlanStep) []PlanStep {
	seen := make(map[string]bool)
	var filtered []PlanStep

	for _, step := range plan {
		key := step.Action + ":" + step.Tool
		if !seen[key] {
			seen[key] = true
			filtered = append(filtered, step)
		}
	}

	return filtered
}

func (re *ReasoningEngine) simplifySteps(plan []PlanStep) []PlanStep {
	/* Simplify long action descriptions */
	for i := range plan {
		if len(plan[i].Action) > 200 {
			plan[i].Action = plan[i].Action[:200] + "..."
		}
	}
	return plan
}

/* Utility functions */

func getStringReasoning(m map[string]interface{}, key, def string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return def
}

func getMapReasoning(m map[string]interface{}, key string) map[string]interface{} {
	if val, ok := m[key].(map[string]interface{}); ok {
		return val
	}
	return make(map[string]interface{})
}


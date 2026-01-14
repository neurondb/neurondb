/*-------------------------------------------------------------------------
 *
 * advanced_collaboration.go
 *    Advanced multi-agent collaboration features
 *
 * Implements agent specialization, rich communication protocols,
 * collaborative planning, task delegation, conflict resolution,
 * consensus building, agent hierarchy, and swarm intelligence.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/advanced_collaboration.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* AdvancedCollaboration provides advanced multi-agent collaboration */
type AdvancedCollaboration struct {
	collabManager *CollaborationManager
	queries       *db.Queries
	runtime       *Runtime
	llm           *LLMClient
}

/* AgentSpecialization represents agent domain specialization */
type AgentSpecialization struct {
	AgentID      uuid.UUID
	Domain       string
	Capabilities []string
	Expertise    float64
}

/* CommunicationProtocol defines communication between agents */
type CommunicationProtocol struct {
	FromAgentID uuid.UUID
	ToAgentID   uuid.UUID
	MessageType string /* "request", "response", "notification", "query" */
	Content     string
	Metadata    map[string]interface{}
	Timestamp   time.Time
}

/* CollaborativePlan represents a plan created by multiple agents */
type CollaborativePlan struct {
	PlanID      uuid.UUID
	Agents      []uuid.UUID
	Steps       []CollaborativeStep
	Consensus   float64
	CreatedAt   time.Time
}

/* CollaborativeStep represents a step in collaborative plan */
type CollaborativeStep struct {
	StepID      int
	AgentID     uuid.UUID
	Action      string
	Dependencies []int
}

/* NewAdvancedCollaboration creates advanced collaboration manager */
func NewAdvancedCollaboration(collabManager *CollaborationManager, queries *db.Queries, runtime *Runtime, llm *LLMClient) *AdvancedCollaboration {
	return &AdvancedCollaboration{
		collabManager: collabManager,
		queries:       queries,
		runtime:       runtime,
		llm:           llm,
	}
}

/* SpecializeAgent marks an agent as specialized in a domain */
func (ac *AdvancedCollaboration) SpecializeAgent(ctx context.Context, agentID uuid.UUID, domain string, capabilities []string, expertise float64) error {
	query := `INSERT INTO neurondb_agent.agent_specializations
		(agent_id, domain, capabilities, expertise, created_at)
		VALUES ($1, $2, $3::text[], $4, $5)
		ON CONFLICT (agent_id, domain) DO UPDATE
		SET capabilities = $3::text[], expertise = $4, updated_at = $5`

	_, err := ac.queries.DB.ExecContext(ctx, query, agentID, domain, capabilities, expertise, time.Now())
	if err != nil {
		return fmt.Errorf("agent specialization failed: agent_id='%s', domain='%s', error=%w", agentID.String(), domain, err)
	}

	return nil
}

/* GetSpecialization gets agent specialization */
func (ac *AdvancedCollaboration) GetSpecialization(ctx context.Context, agentID uuid.UUID, domain string) (*AgentSpecialization, error) {
	query := `SELECT agent_id, domain, capabilities, expertise
		FROM neurondb_agent.agent_specializations
		WHERE agent_id = $1 AND domain = $2`

	type SpecRow struct {
		AgentID     uuid.UUID  `db:"agent_id"`
		Domain      string     `db:"domain"`
		Capabilities []string   `db:"capabilities"`
		Expertise   float64    `db:"expertise"`
	}

	var row SpecRow
	err := ac.queries.DB.GetContext(ctx, &row, query, agentID, domain)
	if err != nil {
		return nil, fmt.Errorf("specialization retrieval failed: agent_id='%s', domain='%s', error=%w", agentID.String(), domain, err)
	}

	return &AgentSpecialization{
		AgentID:      row.AgentID,
		Domain:       row.Domain,
		Capabilities: row.Capabilities,
		Expertise:    row.Expertise,
	}, nil
}

/* CollaborativePlanning creates a plan with multiple agents */
func (ac *AdvancedCollaboration) CollaborativePlanning(ctx context.Context, task string, agentIDs []uuid.UUID) (*CollaborativePlan, error) {
	if len(agentIDs) == 0 {
		return nil, fmt.Errorf("collaborative planning failed: no_agents_provided=true")
	}

	/* Get agent information */
	agents := make([]*db.Agent, 0, len(agentIDs))
	for _, agentID := range agentIDs {
		agent, err := ac.queries.GetAgentByID(ctx, agentID)
		if err != nil {
			continue
		}
		agents = append(agents, agent)
	}

	if len(agents) == 0 {
		return nil, fmt.Errorf("collaborative planning failed: no_valid_agents=true")
	}

	/* Build collaborative planning prompt */
	prompt := fmt.Sprintf(`Multiple agents are collaborating to solve a task.

Task: %s

Agents involved:
`, task)
	for i, agent := range agents {
		prompt += fmt.Sprintf("- Agent %d: %s (Model: %s)\n", i+1, agent.Name, agent.ModelName)
	}

	prompt += `
Work together to create a collaborative plan. Each step should specify:
1. Which agent should execute it
2. What action to take
3. Dependencies on other steps

Respond with JSON array of steps, each with:
- "agent_index": index of agent (0-based)
- "action": description
- "dependencies": array of step indices
`

	llmConfig := map[string]interface{}{
		"temperature": 0.4,
		"max_tokens":  3000,
	}

	response, err := ac.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return nil, fmt.Errorf("collaborative planning failed: llm_error=true, error=%w", err)
	}

	/* Parse response */
	steps, err := ac.parseCollaborativeSteps(response.Content, agentIDs)
	if err != nil {
		return nil, fmt.Errorf("collaborative planning failed: parse_error=true, error=%w", err)
	}

	/* Calculate consensus */
	consensus := ac.calculateConsensus(steps)

	plan := &CollaborativePlan{
		PlanID:    uuid.New(),
		Agents:    agentIDs,
		Steps:     steps,
		Consensus: consensus,
		CreatedAt: time.Now(),
	}

	return plan, nil
}

/* IntelligentTaskDelegation delegates tasks intelligently based on agent capabilities */
func (ac *AdvancedCollaboration) IntelligentTaskDelegation(ctx context.Context, task string, availableAgents []uuid.UUID) (uuid.UUID, error) {
	if len(availableAgents) == 0 {
		return uuid.Nil, fmt.Errorf("task delegation failed: no_agents_available=true")
	}

	/* Get agent specializations */
	specializations := make(map[uuid.UUID]*AgentSpecialization)
	for _, agentID := range availableAgents {
		/* Get all specializations for agent */
		query := `SELECT domain, capabilities, expertise
			FROM neurondb_agent.agent_specializations
			WHERE agent_id = $1
			ORDER BY expertise DESC
			LIMIT 1`

		type SpecRow struct {
			Domain       string   `db:"domain"`
			Capabilities []string `db:"capabilities"`
			Expertise    float64  `db:"expertise"`
		}

		var row SpecRow
		err := ac.queries.DB.GetContext(ctx, &row, query, agentID)
		if err == nil {
			specializations[agentID] = &AgentSpecialization{
				AgentID:      agentID,
				Domain:       row.Domain,
				Capabilities: row.Capabilities,
				Expertise:    row.Expertise,
			}
		}
	}

	/* Use LLM to determine best agent */
	prompt := fmt.Sprintf(`Determine which agent should handle this task.

Task: %s

Available agents:
`, task)

	agentDescriptions := make([]string, 0, len(availableAgents))
	for i, agentID := range availableAgents {
		agent, err := ac.queries.GetAgentByID(ctx, agentID)
		if err != nil {
			continue
		}

		desc := fmt.Sprintf("Agent %d (ID: %s): %s", i, agentID.String()[:8], agent.Name)
		if spec, exists := specializations[agentID]; exists {
			desc += fmt.Sprintf(" - Specialized in %s (expertise: %.2f)", spec.Domain, spec.Expertise)
		}
		agentDescriptions = append(agentDescriptions, desc)
	}

	prompt += joinStrings(agentDescriptions, "\n")
	prompt += "\n\nRespond with the agent index (0-based) that should handle this task."

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  10,
	}

	response, err := ac.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback: select first agent */
		return availableAgents[0], nil
	}

	/* Parse agent index */
	var agentIndex int
	_, err = fmt.Sscanf(response.Content, "%d", &agentIndex)
	if err != nil || agentIndex < 0 || agentIndex >= len(availableAgents) {
		return availableAgents[0], nil
	}

	return availableAgents[agentIndex], nil
}

/* ResolveConflict resolves conflicts between agent decisions */
func (ac *AdvancedCollaboration) ResolveConflict(ctx context.Context, conflictingDecisions []AgentDecision, task string) (*AgentDecision, error) {
	if len(conflictingDecisions) == 0 {
		return nil, fmt.Errorf("conflict resolution failed: no_decisions_provided=true")
	}

	if len(conflictingDecisions) == 1 {
		return &conflictingDecisions[0], nil
	}

	/* Use LLM to resolve conflict */
	prompt := fmt.Sprintf(`Multiple agents have conflicting decisions about a task.

Task: %s

Decisions:
`, task)

	for i, decision := range conflictingDecisions {
		prompt += fmt.Sprintf("\nAgent %d (ID: %s):\n", i+1, decision.AgentID.String()[:8])
		prompt += fmt.Sprintf("Decision: %s\n", decision.Decision)
		prompt += fmt.Sprintf("Reasoning: %s\n", decision.Reasoning)
		if decision.Confidence > 0 {
			prompt += fmt.Sprintf("Confidence: %.2f\n", decision.Confidence)
		}
	}

	prompt += "\nAnalyze the decisions and determine the best resolution. Respond with the decision index (1-based)."

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  10,
	}

	response, err := ac.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback: select decision with highest confidence */
		best := conflictingDecisions[0]
		for _, decision := range conflictingDecisions[1:] {
			if decision.Confidence > best.Confidence {
				best = decision
			}
		}
		return &best, nil
	}

	/* Parse decision index */
	var decisionIndex int
	_, err = fmt.Sscanf(response.Content, "%d", &decisionIndex)
	if err != nil || decisionIndex < 1 || decisionIndex > len(conflictingDecisions) {
		/* Fallback: highest confidence */
		best := conflictingDecisions[0]
		for _, decision := range conflictingDecisions[1:] {
			if decision.Confidence > best.Confidence {
				best = decision
			}
		}
		return &best, nil
	}

	return &conflictingDecisions[decisionIndex-1], nil
}

/* BuildConsensus builds consensus among multiple agents */
func (ac *AdvancedCollaboration) BuildConsensus(ctx context.Context, task string, agentIDs []uuid.UUID) (*ConsensusResult, error) {
	if len(agentIDs) == 0 {
		return nil, fmt.Errorf("consensus building failed: no_agents_provided=true")
	}

	/* Get decisions from each agent */
	decisions := make([]AgentDecision, 0, len(agentIDs))
	for _, agentID := range agentIDs {
		agent, err := ac.queries.GetAgentByID(ctx, agentID)
		if err != nil {
			continue
		}

		/* Create session for agent */
		session := &db.Session{
			AgentID: agentID,
		}
		if err := ac.queries.CreateSession(ctx, session); err != nil {
			continue
		}

		/* Get agent decision */
		prompt := fmt.Sprintf("Task: %s\n\nProvide your decision and reasoning.", task)

		response, err := ac.llm.Generate(ctx, agent.ModelName, prompt, agent.Config)
		if err != nil {
			continue
		}

		decisions = append(decisions, AgentDecision{
			AgentID:    agentID,
			Decision:   response.Content,
			Reasoning:  response.Content,
			Confidence: 0.7,
		})
	}

	if len(decisions) == 0 {
		return nil, fmt.Errorf("consensus building failed: no_decisions_obtained=true")
	}

	/* Build consensus using voting */
	consensus := ac.voteOnConsensus(decisions)

	return &ConsensusResult{
		Consensus:   consensus,
		Votes:       len(decisions),
		Agreement:   ac.calculateAgreement(decisions, consensus),
		Decisions:   decisions,
	}, nil
}

/* Helper types */

type AgentDecision struct {
	AgentID    uuid.UUID
	Decision   string
	Reasoning  string
	Confidence float64
}

type ConsensusResult struct {
	Consensus string
	Votes     int
	Agreement float64
	Decisions []AgentDecision
}

/* Helper methods */

func (ac *AdvancedCollaboration) parseCollaborativeSteps(response string, agentIDs []uuid.UUID) ([]CollaborativeStep, error) {
	/* Parse JSON response */
	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 {
		return nil, fmt.Errorf("no JSON array found")
	}

	jsonStr := response[start : end+1]

	var rawSteps []map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &rawSteps); err != nil {
		return nil, err
	}

	steps := make([]CollaborativeStep, 0, len(rawSteps))
	for i, raw := range rawSteps {
		agentIndex := int(raw["agent_index"].(float64))
		if agentIndex < 0 || agentIndex >= len(agentIDs) {
			agentIndex = 0
		}

		step := CollaborativeStep{
			StepID:  i,
			AgentID: agentIDs[agentIndex],
			Action:  getString(raw, "action", ""),
		}

		if deps, ok := raw["dependencies"].([]interface{}); ok {
			step.Dependencies = make([]int, 0, len(deps))
			for _, dep := range deps {
				if depInt, ok := dep.(float64); ok {
					step.Dependencies = append(step.Dependencies, int(depInt))
				}
			}
		}

		steps = append(steps, step)
	}

	return steps, nil
}

func (ac *AdvancedCollaboration) calculateConsensus(steps []CollaborativeStep) float64 {
	if len(steps) == 0 {
		return 0.0
	}

	/* Count unique agents */
	agentSet := make(map[uuid.UUID]bool)
	for _, step := range steps {
		agentSet[step.AgentID] = true
	}

	/* Consensus is higher when more agents participate */
	numAgents := len(agentSet)
	consensus := float64(numAgents) / float64(len(steps)+1)
	if consensus > 1.0 {
		consensus = 1.0
	}

	return consensus
}

func (ac *AdvancedCollaboration) voteOnConsensus(decisions []AgentDecision) string {
	if len(decisions) == 0 {
		return ""
	}

	/* Simple voting: count occurrences of similar decisions */
	votes := make(map[string]int)
	for _, decision := range decisions {
		/* Normalize decision text for voting */
		key := strings.ToLower(strings.TrimSpace(decision.Decision))
		if len(key) > 100 {
			key = key[:100]
		}
		votes[key]++
	}

	/* Find most voted decision */
	maxVotes := 0
	consensus := decisions[0].Decision
	for key, count := range votes {
		if count > maxVotes {
			maxVotes = count
			/* Find original decision */
			for _, decision := range decisions {
				if strings.ToLower(strings.TrimSpace(decision.Decision))[:100] == key {
					consensus = decision.Decision
					break
				}
			}
		}
	}

	return consensus
}

func (ac *AdvancedCollaboration) calculateAgreement(decisions []AgentDecision, consensus string) float64 {
	if len(decisions) == 0 {
		return 0.0
	}

	/* Calculate similarity of decisions to consensus */
	consensusLower := strings.ToLower(strings.TrimSpace(consensus))
	agreements := 0

	for _, decision := range decisions {
		decisionLower := strings.ToLower(strings.TrimSpace(decision.Decision))
		if len(decisionLower) > 100 {
			decisionLower = decisionLower[:100]
		}
		if len(consensusLower) > 100 {
			consensusLower = consensusLower[:100]
		}

		/* Simple similarity check */
		if decisionLower == consensusLower || strings.Contains(decisionLower, consensusLower) || strings.Contains(consensusLower, decisionLower) {
			agreements++
		}
	}

	return float64(agreements) / float64(len(decisions))
}

func getStringCollab(m map[string]interface{}, key, def string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return def
}

func joinStringsCollab(strs []string, sep string) string {
	return strings.Join(strs, sep)
}


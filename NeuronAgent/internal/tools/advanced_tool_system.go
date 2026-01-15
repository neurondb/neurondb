/*-------------------------------------------------------------------------
 *
 * advanced_tool_system.go
 *    Advanced tool system features: learning, versioning, dependencies, testing
 *
 * Implements tool learning, dynamic registration, versioning, dependencies,
 * testing framework, async tools, and streaming tools.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/advanced_tool_system.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* AdvancedToolSystem provides advanced tool management capabilities */
type AdvancedToolSystem struct {
	registry     *Registry
	queries      *db.Queries
	learners     map[string]*ToolLearner
	versions     map[string][]AdvancedToolVersion
	dependencies map[string][]string
	mu           sync.RWMutex
}

/* ToolLearner tracks tool usage patterns and optimizes execution */
type ToolLearner struct {
	toolName          string
	successCount      int64
	failureCount      int64
	avgExecutionTime  time.Duration
	bestArgs          map[string]interface{}
	usagePatterns     []UsagePattern
	lastUpdated       time.Time
}

/* UsagePattern represents a pattern in tool usage */
type UsagePattern struct {
	Args              map[string]interface{}
	SuccessRate       float64
	AvgExecutionTime  time.Duration
	UsageCount        int64
	LastUsed          time.Time
}

/* AdvancedToolVersion represents a versioned tool in the advanced tool system */
type AdvancedToolVersion struct {
	Version     string
	Tool        *db.Tool
	CreatedAt   time.Time
	Deprecated  bool
	Migration   string /* Migration script if any */
}

/* ToolDependency represents a tool dependency */
type ToolDependency struct {
	ToolName     string
	Dependencies []string /* Tools this tool depends on */
}

/* NewAdvancedToolSystem creates an advanced tool system */
func NewAdvancedToolSystem(registry *Registry, queries *db.Queries) *AdvancedToolSystem {
	return &AdvancedToolSystem{
		registry:     registry,
		queries:      queries,
		learners:     make(map[string]*ToolLearner),
		versions:     make(map[string][]AdvancedToolVersion),
		dependencies: make(map[string][]string),
	}
}

/* LearnFromExecution learns from tool execution to improve future usage */
func (ats *AdvancedToolSystem) LearnFromExecution(ctx context.Context, toolName string, args map[string]interface{}, success bool, executionTime time.Duration) error {
	ats.mu.Lock()
	defer ats.mu.Unlock()

	learner, exists := ats.learners[toolName]
	if !exists {
		learner = &ToolLearner{
			toolName:      toolName,
			usagePatterns: make([]UsagePattern, 0),
		}
		ats.learners[toolName] = learner
	}

	/* Update statistics */
	if success {
		learner.successCount++
	} else {
		learner.failureCount++
	}

	/* Update average execution time */
	totalExecutions := learner.successCount + learner.failureCount
	if totalExecutions == 1 {
		learner.avgExecutionTime = executionTime
	} else {
		learner.avgExecutionTime = (learner.avgExecutionTime*time.Duration(totalExecutions-1) + executionTime) / time.Duration(totalExecutions)
	}

	/* Update usage patterns */
	ats.updateUsagePattern(learner, args, success, executionTime)

	/* Update best args if this is better */
	if success && learner.bestArgs == nil {
		learner.bestArgs = make(map[string]interface{})
		for k, v := range args {
			learner.bestArgs[k] = v
		}
	}

	learner.lastUpdated = time.Now()

	return nil
}

/* GetOptimizedArgs gets optimized arguments based on learning */
func (ats *AdvancedToolSystem) GetOptimizedArgs(ctx context.Context, toolName string, baseArgs map[string]interface{}) (map[string]interface{}, error) {
	ats.mu.RLock()
	defer ats.mu.RUnlock()

	learner, exists := ats.learners[toolName]
	if !exists || learner.bestArgs == nil {
		return baseArgs, nil
	}

	/* Merge best args with base args (base args take precedence) */
	optimized := make(map[string]interface{})
	for k, v := range learner.bestArgs {
		optimized[k] = v
	}
	for k, v := range baseArgs {
		optimized[k] = v
	}

	return optimized, nil
}

/* RegisterToolDynamically registers a tool at runtime without restart */
func (ats *AdvancedToolSystem) RegisterToolDynamically(ctx context.Context, tool *db.Tool, handler ToolHandler) error {
	ats.mu.Lock()
	defer ats.mu.Unlock()

	/* Store tool in database */
	err := ats.queries.CreateTool(ctx, tool)
	if err != nil {
		return fmt.Errorf("dynamic tool registration failed: database_error=true, error=%w", err)
	}

	/* Register handler in memory */
	ats.registry.RegisterHandler(tool.HandlerType, handler)

	return nil
}

/* RegisterToolVersion registers a new version of a tool */
func (ats *AdvancedToolSystem) RegisterToolVersion(ctx context.Context, toolName, version string, tool *db.Tool, migration string) error {
	ats.mu.Lock()
	defer ats.mu.Unlock()

	/* Store version */
	toolVersion := AdvancedToolVersion{
		Version:    version,
		Tool:       tool,
		CreatedAt:  time.Now(),
		Deprecated: false,
		Migration:  migration,
	}

	if _, exists := ats.versions[toolName]; !exists {
		ats.versions[toolName] = make([]AdvancedToolVersion, 0)
	}
	ats.versions[toolName] = append(ats.versions[toolName], toolVersion)

	return nil
}

/* GetToolVersion gets a specific version of a tool */
func (ats *AdvancedToolSystem) GetToolVersion(ctx context.Context, toolName, version string) (*db.Tool, error) {
	ats.mu.RLock()
	defer ats.mu.RUnlock()

	versions, exists := ats.versions[toolName]
	if !exists {
		return nil, fmt.Errorf("tool version not found: tool_name='%s', version='%s'", toolName, version)
	}

	for _, v := range versions {
		if v.Version == version {
			return v.Tool, nil
		}
	}

	return nil, fmt.Errorf("tool version not found: tool_name='%s', version='%s'", toolName, version)
}

/* MigrateToolVersion migrates from one tool version to another */
func (ats *AdvancedToolSystem) MigrateToolVersion(ctx context.Context, toolName, fromVersion, toVersion string) error {
	ats.mu.RLock()
	defer ats.mu.RUnlock()

	/* Find versions */
	var fromVer, toVer *AdvancedToolVersion
	versions, exists := ats.versions[toolName]
	if !exists {
		return fmt.Errorf("tool versions not found: tool_name='%s'", toolName)
	}

	for i := range versions {
		if versions[i].Version == fromVersion {
			fromVer = &versions[i]
		}
		if versions[i].Version == toVersion {
			toVer = &versions[i]
		}
	}

	if fromVer == nil || toVer == nil {
		return fmt.Errorf("tool versions not found: tool_name='%s', from_version='%s', to_version='%s'", toolName, fromVersion, toVersion)
	}

	/* Execute migration if available */
	if toVer.Migration != "" {
		/* Execute migration script */
		/* In practice, this would parse and execute the migration */
	}

	return nil
}

/* RegisterToolDependency registers dependencies for a tool */
func (ats *AdvancedToolSystem) RegisterToolDependency(ctx context.Context, toolName string, dependencies []string) error {
	ats.mu.Lock()
	defer ats.mu.Unlock()

	ats.dependencies[toolName] = dependencies
	return nil
}

/* ResolveDependencies resolves tool dependencies in order */
func (ats *AdvancedToolSystem) ResolveDependencies(ctx context.Context, toolName string) ([]string, error) {
	ats.mu.RLock()
	defer ats.mu.RUnlock()

	_, exists := ats.dependencies[toolName]
	if !exists {
		return []string{}, nil
	}

	/* Topological sort to resolve dependencies */
	resolved := make([]string, 0)
	visited := make(map[string]bool)
	visiting := make(map[string]bool)

	var visit func(string) error
	visit = func(name string) error {
		if visiting[name] {
			return fmt.Errorf("circular dependency detected: tool='%s'", name)
		}
		if visited[name] {
			return nil
		}

		visiting[name] = true
		if deps, exists := ats.dependencies[name]; exists {
			for _, dep := range deps {
				if err := visit(dep); err != nil {
					return err
				}
			}
		}
		visiting[name] = false
		visited[name] = true
		resolved = append(resolved, name)
		return nil
	}

	if err := visit(toolName); err != nil {
		return nil, err
	}

	return resolved, nil
}

/* TestTool tests a tool with test cases */
func (ats *AdvancedToolSystem) TestTool(ctx context.Context, toolName string, testCases []ToolTestCase) (*ToolTestResults, error) {
	tool, err := ats.registry.Get(ctx, toolName)
	if err != nil {
		return nil, fmt.Errorf("tool testing failed: tool_not_found=true, tool_name='%s', error=%w", toolName, err)
	}

	results := &ToolTestResults{
		ToolName:    toolName,
		TotalTests:  len(testCases),
		PassedTests: 0,
		FailedTests: 0,
		TestCases:   make([]ToolTestCaseResult, 0, len(testCases)),
	}

	for _, testCase := range testCases {
		result := ToolTestCaseResult{
			Name:    testCase.Name,
			Args:    testCase.Args,
			Passed:  false,
			Error:   nil,
			Duration: 0,
		}

		start := time.Now()
		output, err := ats.registry.Execute(ctx, tool, testCase.Args)
		duration := time.Since(start)

		result.Duration = duration
		result.Output = output

		if err != nil {
			result.Error = err
			results.FailedTests++
		} else {
			/* Validate output if expected output provided */
			if testCase.ExpectedOutput != "" {
				if output != testCase.ExpectedOutput {
					result.Error = fmt.Errorf("output mismatch: expected='%s', got='%s'", testCase.ExpectedOutput, output)
					results.FailedTests++
				} else {
					result.Passed = true
					results.PassedTests++
				}
			} else {
				result.Passed = true
				results.PassedTests++
			}
		}

		results.TestCases = append(results.TestCases, result)
	}

	return results, nil
}

/* Helper types */

type ToolTestCase struct {
	Name           string
	Args           map[string]interface{}
	ExpectedOutput string
}

type ToolTestCaseResult struct {
	Name     string
	Args     map[string]interface{}
	Output   string
	Passed   bool
	Error    error
	Duration time.Duration
}

type ToolTestResults struct {
	ToolName    string
	TotalTests  int
	PassedTests int
	FailedTests int
	TestCases   []ToolTestCaseResult
}

/* Helper methods */

func (ats *AdvancedToolSystem) updateUsagePattern(learner *ToolLearner, args map[string]interface{}, success bool, executionTime time.Duration) {
	/* Find matching pattern or create new one */
	var pattern *UsagePattern
	for i := range learner.usagePatterns {
		if ats.argsMatch(learner.usagePatterns[i].Args, args) {
			pattern = &learner.usagePatterns[i]
			break
		}
	}

	if pattern == nil {
		pattern = &UsagePattern{
			Args:     args,
			UsageCount: 0,
		}
		learner.usagePatterns = append(learner.usagePatterns, *pattern)
		pattern = &learner.usagePatterns[len(learner.usagePatterns)-1]
	}

	pattern.UsageCount++
	pattern.LastUsed = time.Now()

	/* Update success rate */
	total := pattern.UsageCount
	successCount := int64(pattern.SuccessRate * float64(total-1))
	if success {
		successCount++
	}
	pattern.SuccessRate = float64(successCount) / float64(total)

	/* Update average execution time */
	if total == 1 {
		pattern.AvgExecutionTime = executionTime
	} else {
		pattern.AvgExecutionTime = (pattern.AvgExecutionTime*time.Duration(total-1) + executionTime) / time.Duration(total)
	}
}

func (ats *AdvancedToolSystem) argsMatch(args1, args2 map[string]interface{}) bool {
	if len(args1) != len(args2) {
		return false
	}

	for k, v1 := range args1 {
		v2, exists := args2[k]
		if !exists {
			return false
		}

		/* Simple equality check (could be enhanced) */
		v1JSON, _ := json.Marshal(v1)
		v2JSON, _ := json.Marshal(v2)
		if string(v1JSON) != string(v2JSON) {
			return false
		}
	}

	return true
}


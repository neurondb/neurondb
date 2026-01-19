/*-------------------------------------------------------------------------
 *
 * memory_maintenance.go
 *    Background jobs for memory maintenance
 *
 * Provides periodic jobs for memory corruption detection, forgetting,
 * conflict resolution, quality scoring, and cross-session deduplication.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/jobs/memory_maintenance.go
 *
 *-------------------------------------------------------------------------
 */

package jobs

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MemoryMaintenanceJob handles all memory maintenance tasks */
type MemoryMaintenanceJob struct {
	db                    *db.DB
	queries               *db.Queries
	corruptionDetector    *agent.MemoryCorruptionDetector
	forgettingManager     *agent.MemoryForgettingManager
	conflictResolver      *agent.MemoryConflictResolver
	qualityScorer         *agent.MemoryQualityScorer
	crossSessionManager   *agent.CrossSessionMemoryManager
	corruptionInterval    time.Duration
	forgettingInterval    time.Duration
	conflictInterval      time.Duration
	qualityInterval       time.Duration
	deduplicationInterval time.Duration
	stopChan              chan struct{}
}

/* NewMemoryMaintenanceJob creates a new memory maintenance job */
func NewMemoryMaintenanceJob(
	db *db.DB,
	queries *db.Queries,
	llm *agent.LLMClient,
	embed *neurondb.EmbeddingClient,
) *MemoryMaintenanceJob {
	return &MemoryMaintenanceJob{
		db:                    db,
		queries:               queries,
		corruptionDetector:    agent.NewMemoryCorruptionDetector(db, queries, embed),
		forgettingManager:     agent.NewMemoryForgettingManager(db, queries, embed),
		conflictResolver:      agent.NewMemoryConflictResolver(db, queries, llm, embed),
		qualityScorer:         agent.NewMemoryQualityScorer(db, queries),
		crossSessionManager:   agent.NewCrossSessionMemoryManager(db, queries, embed),
		corruptionInterval:    24 * time.Hour,  /* Daily */
		forgettingInterval:    1 * time.Hour,   /* Hourly */
		conflictInterval:      1 * time.Hour,   /* Hourly */
		qualityInterval:       24 * time.Hour,  /* Daily */
		deduplicationInterval: 24 * time.Hour,  /* Daily */
		stopChan:              make(chan struct{}),
	}
}

/* Start starts all maintenance jobs */
func (j *MemoryMaintenanceJob) Start(ctx context.Context) {
	/* Start corruption detection job (daily) */
	go j.runCorruptionDetection(ctx)

	/* Start forgetting job (hourly) */
	go j.runForgetting(ctx)

	/* Start conflict resolution job (hourly) */
	go j.runConflictResolution(ctx)

	/* Start quality scoring job (daily) */
	go j.runQualityScoring(ctx)

	/* Start deduplication job (daily) */
	go j.runDeduplication(ctx)
}

/* Stop stops all maintenance jobs */
func (j *MemoryMaintenanceJob) Stop() {
	close(j.stopChan)
}

/* runCorruptionDetection runs corruption detection daily */
func (j *MemoryMaintenanceJob) runCorruptionDetection(ctx context.Context) {
	ticker := time.NewTicker(j.corruptionInterval)
	defer ticker.Stop()

	/* Run immediately on start */
	j.executeCorruptionDetection(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-j.stopChan:
			return
		case <-ticker.C:
			j.executeCorruptionDetection(ctx)
		}
	}
}

/* executeCorruptionDetection executes corruption detection for all agents */
func (j *MemoryMaintenanceJob) executeCorruptionDetection(ctx context.Context) {
	startTime := time.Now()
	agents, err := j.getAllAgents(ctx)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to get agents for corruption detection", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	totalIssues := 0
	totalRepaired := 0
	agentsProcessed := 0
	agentsFailed := 0

	for _, agentID := range agents {
		/* Create context with timeout for each agent */
		agentCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
		
		issues, repaired, err := j.corruptionDetector.ValidateAndRepair(agentCtx, agentID)
		cancel()
		
		if err != nil {
			agentsFailed++
			metrics.WarnWithContext(ctx, "Corruption detection failed", map[string]interface{}{
				"agent_id": agentID.String(),
				"error":    err.Error(),
			})
			continue
		}

		agentsProcessed++
		totalIssues += issues
		totalRepaired += repaired

		if issues > 0 {
			metrics.InfoWithContext(ctx, "Corruption detection completed", map[string]interface{}{
				"agent_id": agentID.String(),
				"issues":   issues,
				"repaired": repaired,
			})
		}
	}

	duration := time.Since(startTime)
	metrics.InfoWithContext(ctx, "Corruption detection job completed", map[string]interface{}{
		"agents_processed": agentsProcessed,
		"agents_failed":    agentsFailed,
		"total_issues":     totalIssues,
		"total_repaired":   totalRepaired,
		"duration_seconds": duration.Seconds(),
	})
}

/* runForgetting runs forgetting job hourly */
func (j *MemoryMaintenanceJob) runForgetting(ctx context.Context) {
	ticker := time.NewTicker(j.forgettingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-j.stopChan:
			return
		case <-ticker.C:
			j.executeForgetting(ctx)
		}
	}
}

/* executeForgetting executes forgetting for all agents */
func (j *MemoryMaintenanceJob) executeForgetting(ctx context.Context) {
	startTime := time.Now()
	agents, err := j.getAllAgents(ctx)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to get agents for forgetting", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	totalForgotten := 0
	agentsProcessed := 0
	agentsFailed := 0
	tierStats := make(map[string]int)

	for _, agentID := range agents {
		/* Create context with timeout for each agent */
		agentCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
		
		config := agent.DefaultForgettingConfig()
		agentForgotten := 0
		agentFailed := false

		for _, tier := range []string{"stm", "mtm", "lpm"} {
			forgotten, err := j.forgettingManager.ForgetMemories(agentCtx, agentID, tier, config)
			if err != nil {
				agentFailed = true
				metrics.WarnWithContext(ctx, "Forgetting failed", map[string]interface{}{
					"agent_id": agentID.String(),
					"tier":     tier,
					"error":    err.Error(),
				})
				continue
			}
			tierStats[tier] += forgotten
			agentForgotten += forgotten
		}
		
		cancel()

		if agentFailed {
			agentsFailed++
		} else {
			agentsProcessed++
		}

		totalForgotten += agentForgotten

		if agentForgotten > 0 {
			metrics.InfoWithContext(ctx, "Forgetting completed", map[string]interface{}{
				"agent_id":        agentID.String(),
				"forgotten_count": agentForgotten,
			})
		}
	}

	duration := time.Since(startTime)
	if totalForgotten > 0 || agentsFailed > 0 {
		metrics.InfoWithContext(ctx, "Forgetting job completed", map[string]interface{}{
			"agents_processed": agentsProcessed,
			"agents_failed":    agentsFailed,
			"total_forgotten":  totalForgotten,
			"tier_stats":       tierStats,
			"duration_seconds": duration.Seconds(),
		})
	}
}

/* runConflictResolution runs conflict resolution hourly */
func (j *MemoryMaintenanceJob) runConflictResolution(ctx context.Context) {
	ticker := time.NewTicker(j.conflictInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-j.stopChan:
			return
		case <-ticker.C:
			j.executeConflictResolution(ctx)
		}
	}
}

/* executeConflictResolution executes conflict detection and resolution */
func (j *MemoryMaintenanceJob) executeConflictResolution(ctx context.Context) {
	startTime := time.Now()
	agents, err := j.getAllAgents(ctx)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to get agents for conflict resolution", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	totalConflicts := 0
	totalResolved := 0
	agentsProcessed := 0
	agentsFailed := 0

	for _, agentID := range agents {
		/* Create context with timeout for each agent */
		agentCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
		
		/* Detect conflicts */
		conflicts, err := j.conflictResolver.DetectConflicts(agentCtx, agentID)
		if err != nil {
			agentsFailed++
			cancel()
			metrics.WarnWithContext(ctx, "Conflict detection failed", map[string]interface{}{
				"agent_id": agentID.String(),
				"error":    err.Error(),
			})
			continue
		}

		totalConflicts += len(conflicts)

		/* Resolve conflicts with intelligent strategy selection */
		resolved := 0
		resolutionResults := make(map[string]int) /* strategy -> count */
		
		for _, conflict := range conflicts {
			if !conflict.Resolved {
				/* Select strategy based on conflict type */
				strategy := agent.ResolutionStrategyTimestamp
				switch conflict.ConflictType {
				case "semantic":
					/* For semantic conflicts, prefer timestamp (newer is better) */
					strategy = agent.ResolutionStrategyTimestamp
				case "factual":
					/* For factual conflicts, prefer confidence or LLM-based */
					strategy = agent.ResolutionStrategyConfidence
				}

				if err := j.conflictResolver.ResolveConflict(agentCtx, conflict, strategy); err == nil {
					resolved++
					resolutionResults[string(strategy)]++
				} else {
					/* Try fallback strategy */
					fallbackStrategy := agent.ResolutionStrategyTimestamp
					if err := j.conflictResolver.ResolveConflict(agentCtx, conflict, fallbackStrategy); err == nil {
						resolved++
						resolutionResults["fallback_"+string(fallbackStrategy)]++
					}
				}
			}
		}
		
		cancel()

		agentsProcessed++
		totalResolved += resolved

		if len(conflicts) > 0 {
			metrics.InfoWithContext(ctx, "Conflict resolution completed", map[string]interface{}{
				"agent_id":          agentID.String(),
				"conflicts_found":   len(conflicts),
				"resolved":          resolved,
				"resolution_strategies": resolutionResults,
			})
		}
	}

	duration := time.Since(startTime)
	if totalConflicts > 0 || agentsFailed > 0 {
		metrics.InfoWithContext(ctx, "Conflict resolution job completed", map[string]interface{}{
			"agents_processed": agentsProcessed,
			"agents_failed":    agentsFailed,
			"total_conflicts":  totalConflicts,
			"total_resolved":   totalResolved,
			"duration_seconds": duration.Seconds(),
		})
	}
}

/* runQualityScoring runs quality scoring daily */
func (j *MemoryMaintenanceJob) runQualityScoring(ctx context.Context) {
	ticker := time.NewTicker(j.qualityInterval)
	defer ticker.Stop()

	/* Run immediately on start */
	j.executeQualityScoring(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-j.stopChan:
			return
		case <-ticker.C:
			j.executeQualityScoring(ctx)
		}
	}
}

/* executeQualityScoring updates quality scores for all memories */
func (j *MemoryMaintenanceJob) executeQualityScoring(ctx context.Context) {
	agents, err := j.getAllAgents(ctx)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to get agents for quality scoring", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	for _, agentID := range agents {
		updated, err := j.qualityScorer.UpdateAllMemoryQuality(ctx, agentID)
		if err != nil {
			metrics.WarnWithContext(ctx, "Quality scoring failed", map[string]interface{}{
				"agent_id": agentID.String(),
				"error":    err.Error(),
			})
			continue
		}

		metrics.InfoWithContext(ctx, "Quality scoring completed", map[string]interface{}{
			"agent_id": agentID.String(),
			"updated":  updated,
		})
	}
}

/* runDeduplication runs cross-session deduplication daily */
func (j *MemoryMaintenanceJob) runDeduplication(ctx context.Context) {
	ticker := time.NewTicker(j.deduplicationInterval)
	defer ticker.Stop()

	/* Run immediately on start */
	j.executeDeduplication(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-j.stopChan:
			return
		case <-ticker.C:
			j.executeDeduplication(ctx)
		}
	}
}

/* executeDeduplication executes deduplication for all agents */
func (j *MemoryMaintenanceJob) executeDeduplication(ctx context.Context) {
	agents, err := j.getAllAgents(ctx)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to get agents for deduplication", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	for _, agentID := range agents {
		totalMerged := 0

		for _, tier := range []string{"stm", "mtm", "lpm"} {
			merged, err := j.crossSessionManager.DeduplicateMemories(ctx, agentID, tier, 0.9)
			if err != nil {
				metrics.WarnWithContext(ctx, "Deduplication failed", map[string]interface{}{
					"agent_id": agentID.String(),
					"tier":     tier,
					"error":    err.Error(),
				})
				continue
			}
			totalMerged += merged
		}

		if totalMerged > 0 {
			metrics.InfoWithContext(ctx, "Deduplication completed", map[string]interface{}{
				"agent_id":    agentID.String(),
				"merged":      totalMerged,
			})
		}
	}
}

/* getAllAgents retrieves all agent IDs */
func (j *MemoryMaintenanceJob) getAllAgents(ctx context.Context) ([]uuid.UUID, error) {
	query := `SELECT id FROM neurondb_agent.agents WHERE id IS NOT NULL`

	type AgentRow struct {
		ID uuid.UUID `db:"id"`
	}

	var agents []AgentRow
	err := j.db.DB.SelectContext(ctx, &agents, query)
	if err != nil {
		return nil, err
	}

	agentIDs := make([]uuid.UUID, len(agents))
	for i, agent := range agents {
		agentIDs[i] = agent.ID
	}

	return agentIDs, nil
}

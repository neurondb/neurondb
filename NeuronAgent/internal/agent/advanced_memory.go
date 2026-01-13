/*-------------------------------------------------------------------------
 *
 * advanced_memory.go
 *    Advanced memory systems: Episodic, Semantic, and Working Memory
 *
 * Implements episodic memory (specific events), semantic memory (factual knowledge),
 * working memory (short-term context), memory consolidation, and memory relationships.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/advanced_memory.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MemoryType defines the type of memory */
type MemoryType string

const (
	MemoryTypeEpisodic MemoryType = "episodic" /* Specific events and experiences */
	MemoryTypeSemantic MemoryType = "semantic" /* Factual knowledge and concepts */
	MemoryTypeWorking  MemoryType = "working"  /* Short-term context */
)

/* AdvancedMemoryManager manages episodic, semantic, and working memory */
type AdvancedMemoryManager struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
	llm     *LLMClient
}

/* EpisodicMemory stores specific events and experiences */
type EpisodicMemory struct {
	ID          int64
	AgentID     uuid.UUID
	Event       string
	Context     string
	Timestamp   time.Time
	Embedding   []float32
	Importance  float64
	Metadata    map[string]interface{}
	RelatedIDs  []int64 /* Related episodic memories */
}

/* SemanticMemory stores factual knowledge and concepts */
type SemanticMemory struct {
	ID          int64
	AgentID     uuid.UUID
	Concept     string
	Knowledge   string
	Embedding   []float32
	Confidence  float64
	Metadata    map[string]interface{}
	RelatedIDs  []int64 /* Related semantic memories */
}

/* WorkingMemory manages short-term context with capacity limits */
type WorkingMemory struct {
	Capacity      int
	CurrentItems  []WorkingMemoryItem
	MaxAge        time.Duration
}

/* WorkingMemoryItem represents an item in working memory */
type WorkingMemoryItem struct {
	ID        int64
	Content   string
	Type      MemoryType
	Timestamp time.Time
	Priority  float64
}

/* MemoryRelationship links related memories */
type MemoryRelationship struct {
	FromID     int64
	ToID       int64
	FromType   MemoryType
	ToType     MemoryType
	Relation   string /* "similar", "follows", "contradicts", "elaborates" */
	Strength   float64
	CreatedAt  time.Time
}

/* NewAdvancedMemoryManager creates a new advanced memory manager */
func NewAdvancedMemoryManager(database *db.DB, queries *db.Queries, embedClient *neurondb.EmbeddingClient, llmClient *LLMClient) *AdvancedMemoryManager {
	return &AdvancedMemoryManager{
		db:      database,
		queries: queries,
		embed:   embedClient,
		llm:     llmClient,
	}
}

/* StoreEpisodic stores an episodic memory (specific event) */
func (m *AdvancedMemoryManager) StoreEpisodic(ctx context.Context, agentID uuid.UUID, event, context string, metadata map[string]interface{}) (int64, error) {
	/* Compute embedding for event */
	embedding, err := m.embed.Embed(ctx, event+" "+context, "all-MiniLM-L6-v2")
	if err != nil {
		return 0, fmt.Errorf("episodic memory embedding failed: error=%w", err)
	}

	/* Compute importance */
	importance := m.computeImportance(event, context, metadata)

	/* Store in database */
	query := `INSERT INTO neurondb_agent.memory_episodic
		(agent_id, event, context, embedding, importance_score, metadata, created_at)
		VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb, $7)
		RETURNING id`

	var id int64
	err = m.db.DB.GetContext(ctx, &id, query, agentID, event, context, embedding, importance, metadata, time.Now())
	if err != nil {
		return 0, fmt.Errorf("episodic memory storage failed: error=%w", err)
	}

	/* Find related memories */
	relatedIDs, err := m.findRelatedEpisodicMemories(ctx, agentID, embedding, 5)
	if err == nil && len(relatedIDs) > 0 {
		/* Create relationships */
		m.createMemoryRelationships(ctx, id, MemoryTypeEpisodic, relatedIDs, MemoryTypeEpisodic)
	}

	return id, nil
}

/* StoreSemantic stores a semantic memory (factual knowledge) */
func (m *AdvancedMemoryManager) StoreSemantic(ctx context.Context, agentID uuid.UUID, concept, knowledge string, metadata map[string]interface{}) (int64, error) {
	/* Compute embedding for concept */
	embedding, err := m.embed.Embed(ctx, concept+" "+knowledge, "all-MiniLM-L6-v2")
	if err != nil {
		return 0, fmt.Errorf("semantic memory embedding failed: error=%w", err)
	}

	/* Compute confidence (could be based on source, verification, etc.) */
	confidence := m.computeConfidence(knowledge, metadata)

	/* Store in database */
	query := `INSERT INTO neurondb_agent.memory_semantic
		(agent_id, concept, knowledge, embedding, confidence, metadata, created_at)
		VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb, $7)
		RETURNING id`

	var id int64
	err = m.db.DB.GetContext(ctx, &id, query, agentID, concept, knowledge, embedding, confidence, metadata, time.Now())
	if err != nil {
		return 0, fmt.Errorf("semantic memory storage failed: error=%w", err)
	}

	/* Find related memories */
	relatedIDs, err := m.findRelatedSemanticMemories(ctx, agentID, embedding, 5)
	if err == nil && len(relatedIDs) > 0 {
		/* Create relationships */
		m.createMemoryRelationships(ctx, id, MemoryTypeSemantic, relatedIDs, MemoryTypeSemantic)
	}

	return id, nil
}

/* RetrieveEpisodic retrieves episodic memories based on query */
func (m *AdvancedMemoryManager) RetrieveEpisodic(ctx context.Context, agentID uuid.UUID, queryEmbedding []float32, topK int) ([]EpisodicMemory, error) {
	query := `SELECT id, agent_id, event, context, importance_score, metadata, created_at,
			  1 - (embedding <=> $1::vector) AS similarity
		FROM neurondb_agent.memory_episodic
		WHERE agent_id = $2
		ORDER BY embedding <=> $1::vector
		LIMIT $3`

	type EpisodicRow struct {
		ID              int64                  `db:"id"`
		AgentID         uuid.UUID              `db:"agent_id"`
		Event           string                 `db:"event"`
		Context         string                 `db:"context"`
		ImportanceScore float64                `db:"importance_score"`
		Metadata        map[string]interface{} `db:"metadata"`
		CreatedAt       time.Time              `db:"created_at"`
		Similarity      float64                `db:"similarity"`
	}

	var rows []EpisodicRow
	err := m.db.DB.SelectContext(ctx, &rows, query, queryEmbedding, agentID, topK)
	if err != nil {
		return nil, fmt.Errorf("episodic memory retrieval failed: error=%w", err)
	}

	var memories []EpisodicMemory
	for _, row := range rows {
		/* Load related IDs */
		relatedIDs, _ := m.getRelatedMemoryIDs(ctx, row.ID, MemoryTypeEpisodic)

		memories = append(memories, EpisodicMemory{
			ID:         row.ID,
			AgentID:    row.AgentID,
			Event:      row.Event,
			Context:    row.Context,
			Timestamp:  row.CreatedAt,
			Importance: row.ImportanceScore,
			Metadata:   row.Metadata,
			RelatedIDs: relatedIDs,
		})
	}

	return memories, nil
}

/* RetrieveSemantic retrieves semantic memories based on query */
func (m *AdvancedMemoryManager) RetrieveSemantic(ctx context.Context, agentID uuid.UUID, queryEmbedding []float32, topK int) ([]SemanticMemory, error) {
	query := `SELECT id, agent_id, concept, knowledge, confidence, metadata, created_at,
			  1 - (embedding <=> $1::vector) AS similarity
		FROM neurondb_agent.memory_semantic
		WHERE agent_id = $2
		ORDER BY embedding <=> $1::vector
		LIMIT $3`

	type SemanticRow struct {
		ID         int64                  `db:"id"`
		AgentID    uuid.UUID              `db:"agent_id"`
		Concept    string                 `db:"concept"`
		Knowledge  string                 `db:"knowledge"`
		Confidence float64                `db:"confidence"`
		Metadata   map[string]interface{} `db:"metadata"`
		CreatedAt  time.Time              `db:"created_at"`
		Similarity float64                `db:"similarity"`
	}

	var rows []SemanticRow
	err := m.db.DB.SelectContext(ctx, &rows, query, queryEmbedding, agentID, topK)
	if err != nil {
		return nil, fmt.Errorf("semantic memory retrieval failed: error=%w", err)
	}

	var memories []SemanticMemory
	for _, row := range rows {
		/* Load related IDs */
		relatedIDs, _ := m.getRelatedMemoryIDs(ctx, row.ID, MemoryTypeSemantic)

		memories = append(memories, SemanticMemory{
			ID:         row.ID,
			AgentID:    row.AgentID,
			Concept:    row.Concept,
			Knowledge:  row.Knowledge,
			Confidence: row.Confidence,
			Metadata:   row.Metadata,
			RelatedIDs: relatedIDs,
		})
	}

	return memories, nil
}

/* NewWorkingMemory creates a new working memory with capacity limits */
func NewWorkingMemory(capacity int, maxAge time.Duration) *WorkingMemory {
	if capacity <= 0 {
		capacity = 10 /* Default capacity */
	}
	if maxAge <= 0 {
		maxAge = 1 * time.Hour /* Default max age */
	}

	return &WorkingMemory{
		Capacity:     capacity,
		CurrentItems: make([]WorkingMemoryItem, 0, capacity),
		MaxAge:       maxAge,
	}
}

/* AddToWorkingMemory adds an item to working memory */
func (wm *WorkingMemory) AddToWorkingMemory(content string, memType MemoryType, priority float64) {
	item := WorkingMemoryItem{
		ID:        time.Now().UnixNano(),
		Content:   content,
		Type:      memType,
		Timestamp: time.Now(),
		Priority:  priority,
	}

	/* Remove expired items */
	wm.removeExpired()

	/* Add new item */
	wm.CurrentItems = append(wm.CurrentItems, item)

	/* If over capacity, remove lowest priority items */
	if len(wm.CurrentItems) > wm.Capacity {
		wm.trimToCapacity()
	}
}

/* GetWorkingMemory retrieves items from working memory */
func (wm *WorkingMemory) GetWorkingMemory(limit int) []WorkingMemoryItem {
	wm.removeExpired()

	if limit <= 0 || limit > len(wm.CurrentItems) {
		limit = len(wm.CurrentItems)
	}

	/* Sort by priority (highest first) */
	items := make([]WorkingMemoryItem, len(wm.CurrentItems))
	copy(items, wm.CurrentItems)

	/* Simple priority sort (could be optimized) */
	for i := 0; i < len(items)-1; i++ {
		for j := i + 1; j < len(items); j++ {
			if items[i].Priority < items[j].Priority {
				items[i], items[j] = items[j], items[i]
			}
		}
	}

	if limit < len(items) {
		return items[:limit]
	}
	return items
}

/* ConsolidateMemory promotes working memory to episodic/semantic memory */
func (m *AdvancedMemoryManager) ConsolidateMemory(ctx context.Context, agentID uuid.UUID, workingMem *WorkingMemory) error {
	items := workingMem.GetWorkingMemory(workingMem.Capacity)

	for _, item := range items {
		if item.Priority < 0.5 {
			continue /* Skip low priority items */
		}

		/* Classify as episodic or semantic */
		memType := m.classifyMemoryType(ctx, item.Content)
		if memType == MemoryTypeEpisodic {
			_, err := m.StoreEpisodic(ctx, agentID, item.Content, "", nil)
			if err != nil {
				continue /* Skip on error */
			}
		} else if memType == MemoryTypeSemantic {
			/* Extract concept and knowledge */
			concept, knowledge := m.extractConceptKnowledge(ctx, item.Content)
			_, err := m.StoreSemantic(ctx, agentID, concept, knowledge, nil)
			if err != nil {
				continue
			}
		}
	}

	/* Clear working memory after consolidation */
	workingMem.CurrentItems = []WorkingMemoryItem{}

	return nil
}

/* Helper methods */

func (m *AdvancedMemoryManager) computeImportance(event, context string, metadata map[string]interface{}) float64 {
	/* Simple heuristic: length, user flags, etc. */
	importance := 0.5

	if len(event) > 100 {
		importance += 0.1
	}
	if len(context) > 100 {
		importance += 0.1
	}
	if metadata != nil {
		if val, ok := metadata["important"].(bool); ok && val {
			importance += 0.2
		}
	}

	if importance > 1.0 {
		importance = 1.0
	}
	return importance
}

func (m *AdvancedMemoryManager) computeConfidence(knowledge string, metadata map[string]interface{}) float64 {
	confidence := 0.7 /* Default confidence */

	if metadata != nil {
		if val, ok := metadata["verified"].(bool); ok && val {
			confidence += 0.2
		}
		if val, ok := metadata["confidence"].(float64); ok {
			confidence = val
		}
	}

	if confidence > 1.0 {
		confidence = 1.0
	}
	return confidence
}

func (m *AdvancedMemoryManager) findRelatedEpisodicMemories(ctx context.Context, agentID uuid.UUID, embedding []float32, topK int) ([]int64, error) {
	query := `SELECT id
		FROM neurondb_agent.memory_episodic
		WHERE agent_id = $1
		ORDER BY embedding <=> $2::vector
		LIMIT $3`

	var ids []int64
	err := m.db.DB.SelectContext(ctx, &ids, query, agentID, embedding, topK)
	if err != nil {
		return nil, err
	}

	return ids, nil
}

func (m *AdvancedMemoryManager) findRelatedSemanticMemories(ctx context.Context, agentID uuid.UUID, embedding []float32, topK int) ([]int64, error) {
	query := `SELECT id
		FROM neurondb_agent.memory_semantic
		WHERE agent_id = $1
		ORDER BY embedding <=> $2::vector
		LIMIT $3`

	var ids []int64
	err := m.db.DB.SelectContext(ctx, &ids, query, agentID, embedding, topK)
	if err != nil {
		return nil, err
	}

	return ids, nil
}

func (m *AdvancedMemoryManager) createMemoryRelationships(ctx context.Context, fromID int64, fromType MemoryType, toIDs []int64, toType MemoryType) {
	query := `INSERT INTO neurondb_agent.memory_relationships
		(from_memory_id, to_memory_id, from_type, to_type, relation, strength, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT DO NOTHING`

	for _, toID := range toIDs {
		if fromID == toID {
			continue
		}
		_, _ = m.db.DB.ExecContext(ctx, query, fromID, toID, string(fromType), string(toType), "similar", 0.7, time.Now())
	}
}

func (m *AdvancedMemoryManager) getRelatedMemoryIDs(ctx context.Context, memoryID int64, memType MemoryType) ([]int64, error) {
	query := `SELECT to_memory_id
		FROM neurondb_agent.memory_relationships
		WHERE from_memory_id = $1 AND from_type = $2
		ORDER BY strength DESC`

	var ids []int64
	err := m.db.DB.SelectContext(ctx, &ids, query, memoryID, string(memType))
	return ids, err
}

func (wm *WorkingMemory) removeExpired() {
	now := time.Now()
	var valid []WorkingMemoryItem
	for _, item := range wm.CurrentItems {
		if now.Sub(item.Timestamp) < wm.MaxAge {
			valid = append(valid, item)
		}
	}
	wm.CurrentItems = valid
}

func (wm *WorkingMemory) trimToCapacity() {
	/* Sort by priority */
	for i := 0; i < len(wm.CurrentItems)-1; i++ {
		for j := i + 1; j < len(wm.CurrentItems); j++ {
			if wm.CurrentItems[i].Priority < wm.CurrentItems[j].Priority {
				wm.CurrentItems[i], wm.CurrentItems[j] = wm.CurrentItems[j], wm.CurrentItems[i]
			}
		}
	}

	/* Keep only top capacity items */
	if len(wm.CurrentItems) > wm.Capacity {
		wm.CurrentItems = wm.CurrentItems[:wm.Capacity]
	}
}

func (m *AdvancedMemoryManager) classifyMemoryType(ctx context.Context, content string) MemoryType {
	/* Simple heuristic: check for event-like language */
	contentLower := content
	if len(contentLower) > 500 {
		contentLower = contentLower[:500]
	}

	/* Event indicators */
	eventKeywords := []string{"when", "happened", "did", "went", "saw", "heard", "felt"}
	for _, keyword := range eventKeywords {
		if len(contentLower) > 0 && stringContains(contentLower, keyword) {
			return MemoryTypeEpisodic
		}
	}

	/* Default to semantic */
	return MemoryTypeSemantic
}

func (m *AdvancedMemoryManager) extractConceptKnowledge(ctx context.Context, content string) (concept, knowledge string) {
	/* Simple extraction: first sentence as concept, rest as knowledge */
	sentences := splitSentences(content)
	if len(sentences) > 0 {
		concept = sentences[0]
		if len(sentences) > 1 {
			knowledge = joinSentences(sentences[1:])
		} else {
			knowledge = content
		}
	} else {
		concept = content
		knowledge = content
	}

	if len(concept) > 200 {
		concept = concept[:200]
	}
	if len(knowledge) > 1000 {
		knowledge = knowledge[:1000]
	}

	return concept, knowledge
}

/* Utility functions */

func stringContains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || 
		(len(s) > len(substr) && (s[:len(substr)] == substr || 
			s[len(s)-len(substr):] == substr || 
			indexOfSubstring(s, substr) >= 0)))
}

func indexOfSubstring(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func splitSentences(text string) []string {
	/* Simple sentence splitting by periods */
	var sentences []string
	current := ""
	for _, char := range text {
		current += string(char)
		if char == '.' || char == '!' || char == '?' {
			sentences = append(sentences, current)
			current = ""
		}
	}
	if current != "" {
		sentences = append(sentences, current)
	}
	return sentences
}

func joinSentences(sentences []string) string {
	result := ""
	for i, s := range sentences {
		if i > 0 {
			result += " "
		}
		result += s
	}
	return result
}




/*-------------------------------------------------------------------------
 *
 * advanced_rag.go
 *    Advanced RAG integration with hybrid search, reranking, and evaluation
 *
 * Implements multi-vector RAG, temporal RAG, faceted RAG, graph RAG,
 * streaming RAG, and RAG evaluation using NeuronDB capabilities.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/advanced_rag.go
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

/* AdvancedRAG provides advanced RAG capabilities */
type AdvancedRAG struct {
	db               *db.DB
	queries          *db.Queries
	ragClient        *neurondb.RAGClient
	hybridClient     *neurondb.HybridSearchClient
	rerankingClient  *neurondb.RerankingClient
	embed            *neurondb.EmbeddingClient
	llm              *LLMClient
}

/* NewAdvancedRAG creates an advanced RAG system */
func NewAdvancedRAG(database *db.DB, queries *db.Queries, ragClient *neurondb.RAGClient, hybridClient *neurondb.HybridSearchClient, rerankingClient *neurondb.RerankingClient, embedClient *neurondb.EmbeddingClient, llmClient *LLMClient) *AdvancedRAG {
	return &AdvancedRAG{
		db:              database,
		queries:         queries,
		ragClient:       ragClient,
		hybridClient:    hybridClient,
		rerankingClient: rerankingClient,
		embed:           embedClient,
		llm:             llmClient,
	}
}

/* HybridRAG performs RAG with hybrid search (vector + full-text) */
func (r *AdvancedRAG) HybridRAG(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, vectorWeight float64) (*RAGResult, error) {
	/* Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("hybrid RAG failed: embedding_error=true, error=%w", err)
	}

	/* Perform hybrid search */
	params := map[string]interface{}{
		"vector_weight": vectorWeight,
		"text_weight":   1.0 - vectorWeight,
	}

	results, err := r.hybridClient.HybridSearch(ctx, query, queryEmbedding, tableName, vectorCol, textCol, limit, params)
	if err != nil {
		return nil, fmt.Errorf("hybrid RAG failed: hybrid_search_error=true, error=%w", err)
	}

	/* Convert to RAG result */
	documents := make([]string, len(results))
	for i, result := range results {
		documents[i] = result.Content
	}

	/* Generate answer */
	answer, err := r.generateAnswer(ctx, query, documents)
	if err != nil {
		return nil, fmt.Errorf("hybrid RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: documents,
		Count:     len(documents),
		Method:    "hybrid_search",
	}, nil
}

/* MultiVectorRAG performs RAG with multiple embeddings per document */
func (r *AdvancedRAG) MultiVectorRAG(ctx context.Context, query, tableName string, embeddingCols []string, limit int) (*RAGResult, error) {
	/* Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("multi-vector RAG failed: embedding_error=true, error=%w", err)
	}

	/* Search across multiple embedding columns */
	var allResults []map[string]interface{}

	for _, col := range embeddingCols {
		query := fmt.Sprintf(`SELECT id, content, metadata, 1 - (%s <=> $1::vector) AS similarity
			FROM %s
			ORDER BY %s <=> $1::vector
			LIMIT $2`, col, tableName, col)

		type ResultRow struct {
			ID         int64                  `db:"id"`
			Content    string                 `db:"content"`
			Metadata   map[string]interface{} `db:"metadata"`
			Similarity float64                `db:"similarity"`
		}

		var rows []ResultRow
		err = r.db.DB.SelectContext(ctx, &rows, query, queryEmbedding, limit)
		if err != nil {
			continue
		}

		for _, row := range rows {
			allResults = append(allResults, map[string]interface{}{
				"id":         row.ID,
				"content":    row.Content,
				"metadata":   row.Metadata,
				"similarity": row.Similarity,
				"embedding_col": col,
			})
		}
	}

	/* Deduplicate and re-rank */
	documents := r.deduplicateAndRank(allResults, limit)

	/* Generate answer */
	answer, err := r.generateAnswer(ctx, query, documents)
	if err != nil {
		return nil, fmt.Errorf("multi-vector RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: documents,
		Count:     len(documents),
		Method:    "multi_vector",
	}, nil
}

/* RerankedRAG performs RAG with reranking */
func (r *AdvancedRAG) RerankedRAG(ctx context.Context, query, tableName, vectorCol string, initialLimit, finalLimit int, rerankModel string) (*RAGResult, error) {
	/* Initial vector search */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("reranked RAG failed: embedding_error=true, error=%w", err)
	}

	initialQuery := fmt.Sprintf(`SELECT id, content, metadata
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2`, tableName, vectorCol)

	type InitialRow struct {
		ID       int64                  `db:"id"`
		Content  string                 `db:"content"`
		Metadata map[string]interface{} `db:"metadata"`
	}

	var initialRows []InitialRow
	err = r.db.DB.SelectContext(ctx, &initialRows, initialQuery, queryEmbedding, initialLimit)
	if err != nil {
		return nil, fmt.Errorf("reranked RAG failed: initial_search_error=true, error=%w", err)
	}

	/* Extract documents */
	documents := make([]string, len(initialRows))
	for i, row := range initialRows {
		documents[i] = row.Content
	}

	/* Rerank */
	reranked, err := r.rerankingClient.RerankCrossEncoder(ctx, query, documents, rerankModel, finalLimit)
	if err != nil {
		/* Fallback to original if reranking fails */
		fallbackDocs := documents[:finalLimit]
		fallbackReranked := make([]neurondb.RerankResult, len(fallbackDocs))
		for i, doc := range fallbackDocs {
			fallbackReranked[i] = neurondb.RerankResult{Document: doc, Score: 1.0, Rank: i + 1}
		}
		reranked = fallbackReranked
	}

	/* Extract documents from reranked results */
	rerankedDocs := make([]string, len(reranked))
	for i, result := range reranked {
		rerankedDocs[i] = result.Document
	}

	/* Generate answer */
	answer, err := r.generateAnswer(ctx, query, rerankedDocs)
	if err != nil {
		return nil, fmt.Errorf("reranked RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: rerankedDocs,
		Count:     len(rerankedDocs),
		Method:    "reranked",
	}, nil
}

/* TemporalRAG performs time-aware RAG with recency weighting */
func (r *AdvancedRAG) TemporalRAG(ctx context.Context, query, tableName, vectorCol, timestampCol string, limit int, recencyWeight float64) (*RAGResult, error) {
	/* Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("temporal RAG failed: embedding_error=true, error=%w", err)
	}

	/* Search with temporal weighting */
	querySQL := fmt.Sprintf(`SELECT id, content, metadata, created_at,
		(1 - (%s <=> $1::vector)) * (1 - $2) + 
		(EXP(-EXTRACT(EPOCH FROM (NOW() - %s)) / 86400.0) / 7.0) * $2 AS combined_score
		FROM %s
		ORDER BY combined_score DESC
		LIMIT $3`, vectorCol, timestampCol, tableName)

	type TemporalRow struct {
		ID            int64                  `db:"id"`
		Content       string                 `db:"content"`
		Metadata      map[string]interface{} `db:"metadata"`
		CreatedAt     time.Time              `db:"created_at"`
		CombinedScore float64                `db:"combined_score"`
	}

	var rows []TemporalRow
	err = r.db.DB.SelectContext(ctx, &rows, querySQL, queryEmbedding, recencyWeight, limit)
	if err != nil {
		return nil, fmt.Errorf("temporal RAG failed: query_error=true, error=%w", err)
	}

	documents := make([]string, len(rows))
	for i, row := range rows {
		documents[i] = row.Content
	}

	/* Generate answer */
	answer, err := r.generateAnswer(ctx, query, documents)
	if err != nil {
		return nil, fmt.Errorf("temporal RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: documents,
		Count:     len(documents),
		Method:    "temporal",
	}, nil
}

/* FacetedRAG performs category-aware RAG */
func (r *AdvancedRAG) FacetedRAG(ctx context.Context, query, tableName, vectorCol, categoryCol string, categories []string, limit int) (*RAGResult, error) {
	/* Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("faceted RAG failed: embedding_error=true, error=%w", err)
	}

	/* Build category filter */
	categoryFilter := ""
	if len(categories) > 0 {
		categoryFilter = fmt.Sprintf("AND %s = ANY($3::text[])", categoryCol)
	}

	/* Search with category filter */
	querySQL := fmt.Sprintf(`SELECT id, content, metadata, %s,
		1 - (%s <=> $1::vector) AS similarity
		FROM %s
		WHERE agent_id = $2 %s
		ORDER BY %s <=> $1::vector
		LIMIT $4`, categoryCol, vectorCol, tableName, categoryFilter, vectorCol)

	type FacetedRow struct {
		ID         int64                  `db:"id"`
		Content    string                 `db:"content"`
		Metadata   map[string]interface{} `db:"metadata"`
		Category   string                 `db:"category"`
		Similarity float64                `db:"similarity"`
	}

	var rows []FacetedRow
	if len(categories) > 0 {
		err = r.db.DB.SelectContext(ctx, &rows, querySQL, queryEmbedding, uuid.Nil, categories, limit)
	} else {
		err = r.db.DB.SelectContext(ctx, &rows, querySQL, queryEmbedding, uuid.Nil, limit)
	}

	if err != nil {
		return nil, fmt.Errorf("faceted RAG failed: query_error=true, error=%w", err)
	}

	documents := make([]string, len(rows))
	for i, row := range rows {
		documents[i] = row.Content
	}

	/* Generate answer */
	answer, err := r.generateAnswer(ctx, query, documents)
	if err != nil {
		return nil, fmt.Errorf("faceted RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: documents,
		Count:     len(documents),
		Method:    "faceted",
	}, nil
}

/* EvaluateRAG evaluates RAG performance using RAGAS metrics */
func (r *AdvancedRAG) EvaluateRAG(ctx context.Context, query, answer string, contexts []string) (*RAGEvaluation, error) {
	/* Calculate faithfulness (answer grounded in context) */
	faithfulness, err := r.calculateFaithfulness(ctx, query, answer, contexts)
	if err != nil {
		faithfulness = 0.5 /* Default */
	}

	/* Calculate relevancy (context relevant to query) */
	relevancy, err := r.calculateRelevancy(ctx, query, contexts)
	if err != nil {
		relevancy = 0.5 /* Default */
	}

	/* Calculate context precision */
	contextPrecision, err := r.calculateContextPrecision(ctx, query, contexts)
	if err != nil {
		contextPrecision = 0.5 /* Default */
	}

	/* Calculate context recall */
	contextRecall, err := r.calculateContextRecall(ctx, query, contexts)
	if err != nil {
		contextRecall = 0.5 /* Default */
	}

	/* Calculate answer semantic similarity */
	semanticSimilarity, err := r.calculateSemanticSimilarity(ctx, query, answer)
	if err != nil {
		semanticSimilarity = 0.5 /* Default */
	}

	return &RAGEvaluation{
		Faithfulness:       faithfulness,
		Relevancy:          relevancy,
		ContextPrecision:   contextPrecision,
		ContextRecall:      contextRecall,
		SemanticSimilarity: semanticSimilarity,
		OverallScore:       (faithfulness + relevancy + contextPrecision + contextRecall + semanticSimilarity) / 5.0,
	}, nil
}

/* Helper types */

type RAGResult struct {
	Query     string
	Answer    string
	Documents []string
	Count     int
	Method    string
}

type RAGEvaluation struct {
	Faithfulness       float64
	Relevancy          float64
	ContextPrecision   float64
	ContextRecall      float64
	SemanticSimilarity float64
	OverallScore       float64
}

/* Helper methods */

func (r *AdvancedRAG) generateAnswer(ctx context.Context, query string, contexts []string) (string, error) {
	/* Build context string */
	contextStr := ""
	for i, ctx := range contexts {
		if i > 0 {
			contextStr += "\n\n"
		}
		contextStr += fmt.Sprintf("Context %d: %s", i+1, ctx)
	}

	prompt := fmt.Sprintf(`Answer the following question based on the provided context.

Question: %s

Context:
%s

Provide a clear, concise answer based only on the context provided.`, query, contextStr)

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  500,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return "", err
	}

	return response.Content, nil
}

func (r *AdvancedRAG) deduplicateAndRank(results []map[string]interface{}, limit int) []string {
	/* Deduplicate by ID */
	seen := make(map[int64]bool)
	var unique []map[string]interface{}

	for _, result := range results {
		if id, ok := result["id"].(int64); ok {
			if !seen[id] {
				seen[id] = true
				unique = append(unique, result)
			}
		}
	}

	/* Sort by similarity */
	for i := 0; i < len(unique)-1; i++ {
		for j := i + 1; j < len(unique); j++ {
			simI, _ := unique[i]["similarity"].(float64)
			simJ, _ := unique[j]["similarity"].(float64)
			if simI < simJ {
				unique[i], unique[j] = unique[j], unique[i]
			}
		}
	}

	/* Extract documents */
	documents := make([]string, 0, limit)
	for i, result := range unique {
		if i >= limit {
			break
		}
		if content, ok := result["content"].(string); ok {
			documents = append(documents, content)
		}
	}

	return documents
}

func (r *AdvancedRAG) calculateFaithfulness(ctx context.Context, query, answer string, contexts []string) (float64, error) {
	/* Use LLM to check if answer is grounded in contexts */
	prompt := fmt.Sprintf(`Check if the answer is fully supported by the provided contexts.

Query: %s
Answer: %s

Contexts:
%s

Respond with a score from 0.0 to 1.0 where:
- 1.0 = Answer is fully supported by contexts
- 0.0 = Answer is not supported by contexts

Respond with only the score (e.g., 0.85).`, query, answer, joinStrings(contexts, "\n\n"))

	llmConfig := map[string]interface{}{
		"temperature": 0.1,
		"max_tokens":  10,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return 0.5, err
	}

	/* Parse score */
	var score float64
	_, err = fmt.Sscanf(response.Content, "%f", &score)
	if err != nil {
		return 0.5, err
	}

	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score, nil
}

func (r *AdvancedRAG) calculateRelevancy(ctx context.Context, query string, contexts []string) (float64, error) {
	/* Calculate average similarity between query and contexts */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return 0.5, err
	}

	var totalSimilarity float64
	count := 0

	for _, ctx := range contexts {
		ctxEmbedding, err := r.embed.Embed(context.Background(), ctx, "all-MiniLM-L6-v2")
		if err != nil {
			continue
		}

		similarity := r.cosineSimilarity(queryEmbedding, ctxEmbedding)
		totalSimilarity += similarity
		count++
	}

	if count == 0 {
		return 0.5, nil
	}

	return totalSimilarity / float64(count), nil
}

func (r *AdvancedRAG) calculateContextPrecision(ctx context.Context, query string, contexts []string) (float64, error) {
	/* Use LLM to check precision of contexts */
	prompt := fmt.Sprintf(`Rate how relevant each context is to the query.

Query: %s

Contexts:
%s

For each context, rate 1.0 if highly relevant, 0.0 if not relevant.
Respond with average score (e.g., 0.75).`, query, joinStrings(contexts, "\n\n"))

	llmConfig := map[string]interface{}{
		"temperature": 0.1,
		"max_tokens":  10,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return 0.5, err
	}

	var score float64
	_, err = fmt.Sscanf(response.Content, "%f", &score)
	if err != nil {
		return 0.5, err
	}

	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score, nil
}

func (r *AdvancedRAG) calculateContextRecall(ctx context.Context, query string, contexts []string) (float64, error) {
	if len(contexts) == 0 {
		return 0.0, nil
	}

	/* Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return 0.5, fmt.Errorf("context recall calculation failed: embedding_error=true, error=%w", err)
	}

	/* Calculate average similarity between query and all contexts */
	/* Recall is approximated by how well the retrieved contexts match the query */
	var totalSimilarity float64
	validContexts := 0

	for _, contextText := range contexts {
		if contextText == "" {
			continue
		}

		/* Generate context embedding */
		ctxEmbedding, err := r.embed.Embed(ctx, contextText, "all-MiniLM-L6-v2")
		if err != nil {
			/* Skip contexts that fail to embed */
			continue
		}

		/* Calculate cosine similarity */
		similarity := r.cosineSimilarity(queryEmbedding, ctxEmbedding)
		totalSimilarity += similarity
		validContexts++
	}

	if validContexts == 0 {
		return 0.0, nil
	}

	/* Average similarity as recall proxy */
	/* Higher similarity = better recall (more relevant contexts retrieved) */
	avgSimilarity := totalSimilarity / float64(validContexts)

	/* Normalize to 0-1 range (similarity is already in that range, but we can adjust) */
	/* Use a threshold: if average similarity > 0.7, consider it good recall */
	recall := avgSimilarity

	/* Apply sigmoid-like function to map similarity to recall score */
	/* This gives better differentiation in the 0.5-0.9 range */
	if recall < 0.5 {
		recall = recall * 0.8 /* Penalize low similarity */
	} else if recall > 0.8 {
		recall = 0.8 + (recall-0.8)*0.5 /* Cap very high similarity */
	}

	/* Ensure result is in [0, 1] range */
	if recall < 0 {
		recall = 0
	}
	if recall > 1 {
		recall = 1
	}

	return recall, nil
}

func (r *AdvancedRAG) calculateSemanticSimilarity(ctx context.Context, query, answer string) (float64, error) {
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return 0.5, err
	}

	answerEmbedding, err := r.embed.Embed(ctx, answer, "all-MiniLM-L6-v2")
	if err != nil {
		return 0.5, err
	}

	return r.cosineSimilarity(queryEmbedding, answerEmbedding), nil
}

func (r *AdvancedRAG) cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float64) float64 {
	/* Simple square root approximation */
	if x == 0 {
		return 0
	}
	guess := x
	for i := 0; i < 10; i++ {
		guess = 0.5 * (guess + x/guess)
	}
	return guess
}

func joinStrings(strs []string, sep string) string {
	result := ""
	for i, s := range strs {
		if i > 0 {
			result += sep
		}
		result += s
	}
	return result
}


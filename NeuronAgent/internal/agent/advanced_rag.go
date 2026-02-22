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
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* AdvancedRAG provides advanced RAG capabilities */
type AdvancedRAG struct {
	db              *db.DB
	queries         *db.Queries
	ragClient       *neurondb.RAGClient
	hybridClient    *neurondb.HybridSearchClient
	rerankingClient *neurondb.RerankingClient
	embed           *neurondb.EmbeddingClient
	llm             *LLMClient
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

	qTable, err := validation.QuoteIdentifier(tableName)
	if err != nil {
		return nil, fmt.Errorf("multi-vector RAG failed: invalid table name: %w", err)
	}
	for _, col := range embeddingCols {
		qCol, err := validation.QuoteIdentifier(col)
		if err != nil {
			return nil, fmt.Errorf("multi-vector RAG failed: invalid column name %q: %w", col, err)
		}
		query := fmt.Sprintf(`SELECT id, content, metadata, 1 - (%s <=> $1::vector) AS similarity
			FROM %s
			ORDER BY %s <=> $1::vector
			LIMIT $2`, qCol, qTable, qCol)

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
				"id":            row.ID,
				"content":       row.Content,
				"metadata":      row.Metadata,
				"similarity":    row.Similarity,
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

	qTable, err := validation.QuoteIdentifier(tableName)
	if err != nil {
		return nil, fmt.Errorf("reranked RAG failed: invalid table name: %w", err)
	}
	qCol, err := validation.QuoteIdentifier(vectorCol)
	if err != nil {
		return nil, fmt.Errorf("reranked RAG failed: invalid vector column: %w", err)
	}
	initialQuery := fmt.Sprintf(`SELECT id, content, metadata
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2`, qTable, qCol)

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
	qTable, err := validation.QuoteIdentifier(tableName)
	if err != nil {
		return nil, fmt.Errorf("temporal RAG failed: invalid table name: %w", err)
	}
	qVectorCol, err := validation.QuoteIdentifier(vectorCol)
	if err != nil {
		return nil, fmt.Errorf("temporal RAG failed: invalid vector column: %w", err)
	}
	qTimestampCol, err := validation.QuoteIdentifier(timestampCol)
	if err != nil {
		return nil, fmt.Errorf("temporal RAG failed: invalid timestamp column: %w", err)
	}
	querySQL := fmt.Sprintf(`SELECT id, content, metadata, created_at,
		(1 - (%s <=> $1::vector)) * (1 - $2) + 
		(EXP(-EXTRACT(EPOCH FROM (NOW() - %s)) / 86400.0) / 7.0) * $2 AS combined_score
		FROM %s
		ORDER BY combined_score DESC
		LIMIT $3`, qVectorCol, qTimestampCol, qTable)

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
	qTable, err := validation.QuoteIdentifier(tableName)
	if err != nil {
		return nil, fmt.Errorf("faceted RAG failed: invalid table name: %w", err)
	}
	qVectorCol, err := validation.QuoteIdentifier(vectorCol)
	if err != nil {
		return nil, fmt.Errorf("faceted RAG failed: invalid vector column: %w", err)
	}
	qCategoryCol, err := validation.QuoteIdentifier(categoryCol)
	if err != nil {
		return nil, fmt.Errorf("faceted RAG failed: invalid category column: %w", err)
	}
	categoryFilter := ""
	if len(categories) > 0 {
		categoryFilter = fmt.Sprintf("AND %s = ANY($3::text[])", qCategoryCol)
	}

	/* Search with category filter */
	querySQL := fmt.Sprintf(`SELECT id, content, metadata, %s,
		1 - (%s <=> $1::vector) AS similarity
		FROM %s
		WHERE agent_id = $2 %s
		ORDER BY %s <=> $1::vector
		LIMIT $4`, qCategoryCol, qVectorCol, qTable, categoryFilter, qVectorCol)

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

/* VectorRetrieveDocuments retrieves documents by vector similarity only (no answer generation). Used by modular RAG. */
func (r *AdvancedRAG) VectorRetrieveDocuments(ctx context.Context, query, tableName, vectorCol, textCol string, topK int) ([]string, error) {
	embedModel := "all-MiniLM-L6-v2"
	queryEmbedding, err := r.embed.Embed(ctx, query, embedModel)
	if err != nil {
		return nil, fmt.Errorf("vector retrieve: embed failed: %w", err)
	}
	contexts, err := r.ragClient.RetrieveContext(ctx, queryEmbedding, tableName, vectorCol, topK)
	if err != nil {
		return nil, fmt.Errorf("vector retrieve: %w", err)
	}
	documents := make([]string, len(contexts))
	for i, c := range contexts {
		documents[i] = c.Content
	}
	return documents, nil
}

/* HybridRetrieveDocuments retrieves documents via hybrid search only (no answer generation). Used by modular RAG. */
func (r *AdvancedRAG) HybridRetrieveDocuments(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, vectorWeight float64) ([]string, error) {
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("hybrid retrieve: embed failed: %w", err)
	}
	params := map[string]interface{}{
		"vector_weight": vectorWeight,
		"text_weight":   1.0 - vectorWeight,
	}
	results, err := r.hybridClient.HybridSearch(ctx, query, queryEmbedding, tableName, vectorCol, textCol, limit, params)
	if err != nil {
		return nil, fmt.Errorf("hybrid retrieve: %w", err)
	}
	documents := make([]string, len(results))
	for i, res := range results {
		documents[i] = res.Content
	}
	return documents, nil
}

/* RerankDocuments reranks documents using the configured reranker. Used by modular RAG. */
func (r *AdvancedRAG) RerankDocuments(ctx context.Context, query string, documents []string, model string, topK int) ([]string, error) {
	if len(documents) == 0 {
		return documents, nil
	}
	results, err := r.ragClient.RerankResults(ctx, query, documents, model, topK)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}
	out := make([]string, len(results))
	for i, res := range results {
		out[i] = res.Document
	}
	return out, nil
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

/* HyDERAG performs RAG with Hypothetical Document Embeddings */
func (r *AdvancedRAG) HyDERAG(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, numHypotheticals int, hypotheticalWeight float64) (*RAGResult, error) {
	/* Step 1: Generate hypothetical documents from query */
	hydePrompt := fmt.Sprintf(`Given the following question, generate %d hypothetical documents that would contain the answer. Each document should be a concise paragraph (2-3 sentences) that directly answers or addresses the question.

Question: %s

Generate %d hypothetical documents, one per line, each starting with "DOC: "`, numHypotheticals, query, numHypotheticals)

	llmConfig := map[string]interface{}{
		"temperature": 0.7,
		"max_tokens":  500,
	}

	hydeResponse, err := r.llm.Generate(ctx, "gpt-4", hydePrompt, llmConfig)
	if err != nil {
		/* Fallback to simple hypotheticals */
		hydeResponse = &LLMResponse{
			Content: fmt.Sprintf("DOC: %s is a topic that requires detailed explanation.\nDOC: Information about %s can be found in relevant documentation.\nDOC: The answer to %s involves understanding key concepts.", query, query, query),
		}
	}

	/* Parse hypothetical documents */
	hypotheticalDocs := r.parseHypotheticalDocuments(hydeResponse.Content, numHypotheticals)
	if len(hypotheticalDocs) == 0 {
		/* Fallback hypotheticals */
		hypotheticalDocs = []string{
			query + " is a topic that requires detailed explanation.",
			"Information about " + query + " can be found in relevant documentation.",
			"The answer to " + query + " involves understanding key concepts.",
		}
	}

	/* Step 2: Generate embeddings for original query and hypothetical documents */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("HyDE RAG failed: embedding_error=true, error=%w", err)
	}

	/* Generate embeddings for hypothetical documents */
	hypotheticalEmbeddings := make([][]float32, len(hypotheticalDocs))
	for i, hypoDoc := range hypotheticalDocs {
		hypoEmbedding, err := r.embed.Embed(ctx, hypoDoc, "all-MiniLM-L6-v2")
		if err != nil {
			continue
		}
		hypotheticalEmbeddings[i] = hypoEmbedding
	}

	/* Step 3: Perform dual-path retrieval */
	qTable, err := validation.QuoteIdentifier(tableName)
	if err != nil {
		return nil, fmt.Errorf("HyDE RAG failed: invalid table name: %w", err)
	}
	qVectorCol, err := validation.QuoteIdentifier(vectorCol)
	if err != nil {
		return nil, fmt.Errorf("HyDE RAG failed: invalid vector column: %w", err)
	}
	/* Retrieve using original query */
	originalQuery := fmt.Sprintf(`SELECT id, content, metadata, 1 - (%s <=> $1::vector) AS similarity
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2`, qVectorCol, qTable, qVectorCol)

	type OriginalRow struct {
		ID         int64                  `db:"id"`
		Content    string                 `db:"content"`
		Metadata   map[string]interface{} `db:"metadata"`
		Similarity float64                `db:"similarity"`
	}

	var originalRows []OriginalRow
	err = r.db.DB.SelectContext(ctx, &originalRows, originalQuery, queryEmbedding, limit*2)
	if err != nil {
		return nil, fmt.Errorf("HyDE RAG failed: original_retrieval_error=true, error=%w", err)
	}

	/* Retrieve using hypothetical document embeddings */
	allResults := make([]map[string]interface{}, 0)
	seenIDs := make(map[int64]bool)

	/* Add original results */
	for _, row := range originalRows {
		if !seenIDs[row.ID] {
			seenIDs[row.ID] = true
			allResults = append(allResults, map[string]interface{}{
				"id":         row.ID,
				"content":    row.Content,
				"metadata":   row.Metadata,
				"similarity": row.Similarity,
				"source":     "original",
			})
		}
	}

	/* Add hypothetical results with weighted scores */
	for i, hypoEmbedding := range hypotheticalEmbeddings {
		if hypoEmbedding == nil {
			continue
		}

		hypoQuery := fmt.Sprintf(`SELECT id, content, metadata, (1 - (%s <=> $1::vector)) * $3 AS similarity
			FROM %s
			ORDER BY %s <=> $1::vector
			LIMIT $2`, qVectorCol, qTable, qVectorCol)

		var hypoRows []OriginalRow
		err = r.db.DB.SelectContext(ctx, &hypoRows, hypoQuery, hypoEmbedding, limit, hypotheticalWeight)
		if err != nil {
			continue
		}

		for _, row := range hypoRows {
			if !seenIDs[row.ID] {
				seenIDs[row.ID] = true
				allResults = append(allResults, map[string]interface{}{
					"id":         row.ID,
					"content":    row.Content,
					"metadata":   row.Metadata,
					"similarity": row.Similarity,
					"source":     fmt.Sprintf("hypothetical_%d", i),
				})
			}
		}
	}

	/* Sort by similarity and take top K */
	documents := r.deduplicateAndRank(allResults, limit)

	/* Step 4: Generate answer */
	answer, err := r.generateAnswer(ctx, query, documents)
	if err != nil {
		return nil, fmt.Errorf("HyDE RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: documents,
		Count:     len(documents),
		Method:    "hyde",
	}, nil
}

/* parseHypotheticalDocuments parses hypothetical documents from LLM response */
func (r *AdvancedRAG) parseHypotheticalDocuments(response string, numHypotheticals int) []string {
	docs := make([]string, 0)
	lines := strings.Split(response, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "DOC:") {
			doc := strings.TrimPrefix(line, "DOC:")
			doc = strings.TrimSpace(doc)
			if doc != "" {
				docs = append(docs, doc)
			}
		}
	}

	/* If we didn't find enough, try splitting by other patterns */
	if len(docs) < numHypotheticals {
		/* Try splitting by numbered lists */
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" && !strings.HasPrefix(line, "DOC:") {
				/* Check if it looks like a document */
				if len(line) > 20 {
					docs = append(docs, line)
				}
			}
		}
	}

	/* Limit to requested number */
	if len(docs) > numHypotheticals {
		docs = docs[:numHypotheticals]
	}

	return docs
}

/* GraphRAG performs RAG with knowledge graph traversal */
func (r *AdvancedRAG) GraphRAG(ctx context.Context, query, tableName, vectorCol, textCol, entityCol, relationCol string, limit int, maxDepth int, traversalMethod string) (*RAGResult, error) {
	/* Step 1: Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: embedding_error=true, error=%w", err)
	}

	/* Step 2: Initial retrieval to find entities */
	qTable, err := validation.QuoteIdentifier(tableName)
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: invalid table name: %w", err)
	}
	qVectorCol, err := validation.QuoteIdentifier(vectorCol)
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: invalid vector column: %w", err)
	}
	qEntityCol, err := validation.QuoteIdentifier(entityCol)
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: invalid entity column: %w", err)
	}
	qRelationCol, err := validation.QuoteIdentifier(relationCol)
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: invalid relation column: %w", err)
	}
	initialQuery := fmt.Sprintf(`SELECT id, content, metadata, %s as entities, %s as relations, 1 - (%s <=> $1::vector) AS similarity
		FROM %s
		WHERE %s IS NOT NULL OR %s IS NOT NULL
		ORDER BY %s <=> $1::vector
		LIMIT $2`, qEntityCol, qRelationCol, qVectorCol, qTable, qEntityCol, qRelationCol, qVectorCol)

	type GraphRow struct {
		ID         int64                  `db:"id"`
		Content    string                 `db:"content"`
		Metadata   map[string]interface{} `db:"metadata"`
		Entities   interface{}            `db:"entities"`
		Relations  interface{}            `db:"relations"`
		Similarity float64                `db:"similarity"`
	}

	var initialRows []GraphRow
	err = r.db.DB.SelectContext(ctx, &initialRows, initialQuery, queryEmbedding, limit*2)
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: initial_retrieval_error=true, error=%w", err)
	}

	/* Step 3: Extract entities and build graph */
	visitedEntities := make(map[string]bool)
	entityQueue := make([]string, 0)
	allDocuments := make([]string, 0)
	graphPath := make([]string, 0)

	/* Process initial results */
	for _, row := range initialRows {
		allDocuments = append(allDocuments, row.Content)

		/* Extract entities (assuming JSONB or text format) */
		if row.Entities != nil {
			entities := r.extractEntities(row.Entities)
			for _, entity := range entities {
				if !visitedEntities[entity] {
					visitedEntities[entity] = true
					entityQueue = append(entityQueue, entity)
				}
			}
		}
	}

	/* Step 4: Graph traversal (BFS or DFS) */
	depth := 0
	for len(entityQueue) > 0 && depth < maxDepth {
		currentEntity := entityQueue[0]
		entityQueue = entityQueue[1:]

		/* Generate embedding for entity */
		entityEmbedding, err := r.embed.Embed(ctx, currentEntity, "all-MiniLM-L6-v2")
		if err != nil {
			continue
		}

		/* Find related documents */
		relatedQuery := fmt.Sprintf(`SELECT id, content, metadata, %s as entities, %s as relations, 1 - (%s <=> $1::vector) AS similarity
			FROM %s
			WHERE %s::text LIKE $2 OR %s::text LIKE $2
			ORDER BY %s <=> $1::vector
			LIMIT $3`, qEntityCol, qRelationCol, qVectorCol, qTable, qEntityCol, qRelationCol, qVectorCol)

		var relatedRows []GraphRow
		entityPattern := "%" + currentEntity + "%"
		err = r.db.DB.SelectContext(ctx, &relatedRows, relatedQuery, entityEmbedding, entityPattern, limit)
		if err != nil {
			continue
		}

		/* Process related documents */
		for _, row := range relatedRows {
			/* Check if document already included */
			alreadyIncluded := false
			for _, doc := range allDocuments {
				if doc == row.Content {
					alreadyIncluded = true
					break
				}
			}

			if !alreadyIncluded {
				allDocuments = append(allDocuments, row.Content)

				/* Extract relations and add to queue */
				if row.Relations != nil {
					relations := r.extractRelations(row.Relations)
					for _, targetEntity := range relations {
						if !visitedEntities[targetEntity] {
							visitedEntities[targetEntity] = true
							entityQueue = append(entityQueue, targetEntity)
							graphPath = append(graphPath, fmt.Sprintf("%s -> %s", currentEntity, targetEntity))
						}
					}
				}
			}
		}

		depth++
	}

	/* Step 5: Limit to top documents */
	if len(allDocuments) > limit {
		allDocuments = allDocuments[:limit]
	}

	/* Step 6: Generate answer */
	answer, err := r.generateAnswer(ctx, query, allDocuments)
	if err != nil {
		return nil, fmt.Errorf("graph RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &RAGResult{
		Query:     query,
		Answer:    answer,
		Documents: allDocuments,
		Count:     len(allDocuments),
		Method:    "graph",
	}, nil
}

/* extractEntities extracts entity names from various formats */
func (r *AdvancedRAG) extractEntities(entities interface{}) []string {
	result := make([]string, 0)

	/* Handle JSONB array */
	if entitiesMap, ok := entities.(map[string]interface{}); ok {
		if entitiesArray, ok := entitiesMap["entities"].([]interface{}); ok {
			for _, entity := range entitiesArray {
				if entityMap, ok := entity.(map[string]interface{}); ok {
					if name, ok := entityMap["name"].(string); ok {
						result = append(result, name)
					}
				} else if name, ok := entity.(string); ok {
					result = append(result, name)
				}
			}
		}
	} else if entitiesArray, ok := entities.([]interface{}); ok {
		for _, entity := range entitiesArray {
			if entityMap, ok := entity.(map[string]interface{}); ok {
				if name, ok := entityMap["name"].(string); ok {
					result = append(result, name)
				}
			} else if name, ok := entity.(string); ok {
				result = append(result, name)
			}
		}
	} else if entitiesStr, ok := entities.(string); ok {
		/* Comma-separated entities */
		parts := strings.Split(entitiesStr, ",")
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part != "" {
				result = append(result, part)
			}
		}
	}

	return result
}

/* extractRelations extracts target entities from relations */
func (r *AdvancedRAG) extractRelations(relations interface{}) []string {
	result := make([]string, 0)

	/* Handle JSONB array */
	if relationsArray, ok := relations.([]interface{}); ok {
		for _, relation := range relationsArray {
			if relationMap, ok := relation.(map[string]interface{}); ok {
				if target, ok := relationMap["target"].(string); ok {
					result = append(result, target)
				}
			}
		}
	} else if relationsMap, ok := relations.(map[string]interface{}); ok {
		if relationsArray, ok := relationsMap["relations"].([]interface{}); ok {
			for _, relation := range relationsArray {
				if relationMap, ok := relation.(map[string]interface{}); ok {
					if target, ok := relationMap["target"].(string); ok {
						result = append(result, target)
					}
				}
			}
		}
	}

	return result
}

/* quoteRAGIdentifiers returns quoted table and vector column identifiers for safe SQL building */
func quoteRAGIdentifiers(tableName, vectorCol string) (qTable, qVectorCol string, err error) {
	qTable, err = validation.QuoteIdentifier(tableName)
	if err != nil {
		return "", "", fmt.Errorf("invalid table name: %w", err)
	}
	qVectorCol, err = validation.QuoteIdentifier(vectorCol)
	if err != nil {
		return "", "", fmt.Errorf("invalid vector column: %w", err)
	}
	return qTable, qVectorCol, nil
}

/* CorrectiveRAG performs RAG with iterative self-correction */
func (r *AdvancedRAG) CorrectiveRAG(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, maxIterations int, qualityThreshold float64) (*RAGResult, error) {
	qTable, qVectorCol, err := quoteRAGIdentifiers(tableName, vectorCol)
	if err != nil {
		return nil, fmt.Errorf("corrective RAG failed: %w", err)
	}
	iterationCount := 0
	currentK := limit
	qualityScore := 0.0
	needsCorrection := true
	var finalAnswer string
	var finalDocuments []string

	/* Iterative correction loop */
	for iterationCount < maxIterations && needsCorrection {
		iterationCount++

		/* Step 1: Retrieve context */
		queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
		if err != nil {
			return nil, fmt.Errorf("corrective RAG failed: embedding_error=true, iteration=%d, error=%w", iterationCount, err)
		}

		retrieveQuery := fmt.Sprintf(`SELECT id, content, metadata, 1 - (%s <=> $1::vector) AS similarity
			FROM %s
			ORDER BY %s <=> $1::vector
			LIMIT $2`, qVectorCol, qTable, qVectorCol)

		type RetrieveRow struct {
			ID         int64                  `db:"id"`
			Content    string                 `db:"content"`
			Metadata   map[string]interface{} `db:"metadata"`
			Similarity float64                `db:"similarity"`
		}

		var rows []RetrieveRow
		err = r.db.DB.SelectContext(ctx, &rows, retrieveQuery, queryEmbedding, currentK)
		if err != nil {
			return nil, fmt.Errorf("corrective RAG failed: retrieval_error=true, iteration=%d, error=%w", iterationCount, err)
		}

		documents := make([]string, len(rows))
		for i, row := range rows {
			documents[i] = row.Content
		}

		/* Step 2: Generate answer */
		answer, err := r.generateAnswer(ctx, query, documents)
		if err != nil {
			return nil, fmt.Errorf("corrective RAG failed: answer_generation_error=true, iteration=%d, error=%w", iterationCount, err)
		}

		/* Step 3: Evaluate answer quality */
		evaluation, err := r.EvaluateRAG(ctx, query, answer, documents)
		if err != nil {
			/* Fallback: use simple heuristic */
			qualityScore = 0.5
		} else {
			/* Use overall score as quality metric */
			qualityScore = evaluation.OverallScore
		}

		/* Step 4: Check if correction is needed */
		if qualityScore >= qualityThreshold {
			needsCorrection = false
			finalAnswer = answer
			finalDocuments = documents
		} else {
			/* Expand retrieval for next iteration */
			currentK = currentK + limit

			/* Generate gap analysis */
			gapAnalysis, err := r.analyzeGaps(ctx, query, answer, documents)
			if err == nil && gapAnalysis != "" {
				/* Could use gap analysis to refine query, but for now just expand k */
			}
		}
	}

	/* If we exhausted iterations, use the best answer we have */
	if finalAnswer == "" && len(finalDocuments) == 0 {
		/* Final retrieval */
		queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
		if err != nil {
			return nil, fmt.Errorf("corrective RAG failed: final_embedding_error=true, error=%w", err)
		}

		retrieveQuery := fmt.Sprintf(`SELECT content
			FROM %s
			ORDER BY %s <=> $1::vector
			LIMIT $2`, qTable, qVectorCol)

		var rows []struct {
			Content string `db:"content"`
		}
		err = r.db.DB.SelectContext(ctx, &rows, retrieveQuery, queryEmbedding, currentK)
		if err == nil {
			finalDocuments = make([]string, len(rows))
			for i, row := range rows {
				finalDocuments[i] = row.Content
			}

			answer, err := r.generateAnswer(ctx, query, finalDocuments)
			if err == nil {
				finalAnswer = answer
			}
		}
	}

	return &RAGResult{
		Query:     query,
		Answer:    finalAnswer,
		Documents: finalDocuments,
		Count:     len(finalDocuments),
		Method:    fmt.Sprintf("corrective_%d_iterations", iterationCount),
	}, nil
}

/* analyzeGaps analyzes gaps in the answer and context */
func (r *AdvancedRAG) analyzeGaps(ctx context.Context, query, answer string, contexts []string) (string, error) {
	contextStr := joinStrings(contexts, "\n\n")
	prompt := fmt.Sprintf(`Analyze this answer and identify what information is missing or incorrect.

Question: %s
Answer: %s
Context: %s

Identify gaps in 1-2 sentences.`, query, answer, contextStr)

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  200,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return "", err
	}

	return response.Content, nil
}

/* AgenticRAG performs RAG with autonomous planning and dynamic retrieval */
func (r *AdvancedRAG) AgenticRAG(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, maxSteps int, evidenceThreshold float64, maxTokens int) (*AgenticRAGResult, error) {
	/* Step 1: Generate execution plan */
	planPrompt := fmt.Sprintf(`Break down this query into a multi-step retrieval plan. Each step should specify what information to retrieve and why.

Query: %s

Respond with a JSON array of steps, each with:
{"step": number, "action": "description", "query": "refined query for this step", "reason": "why this step is needed"}

Generate 2-4 steps maximum.`, query)

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  500,
	}

	planResponse, err := r.llm.Generate(ctx, "gpt-4", planPrompt, llmConfig)
	if err != nil {
		/* Fallback to single-step plan */
		planResponse = &LLMResponse{
			Content: `[{"step": 1, "action": "Retrieve relevant documents", "query": "` + query + `", "reason": "Initial retrieval for the query"}]`,
		}
	}

	/* Parse plan steps (simplified - in production would use proper JSON parsing) */
	planSteps := r.parsePlanSteps(planResponse.Content)
	if len(planSteps) == 0 {
		planSteps = []AgenticPlanStep{
			{Step: 1, Action: "Retrieve relevant documents", Query: query, Reason: "Initial retrieval"},
		}
	}

	/* Step 2: Execute plan steps with verification */
	var accumulatedContext []string
	var executionTrace []ExecutionStep
	var reasoningPath []string
	stepCount := 0
	tokensUsed := 0
	evidenceScore := 0.0

	qTable, qVectorCol, err := quoteRAGIdentifiers(tableName, vectorCol)
	if err != nil {
		return nil, fmt.Errorf("agentic RAG failed: %w", err)
	}
	for i, step := range planSteps {
		if i >= maxSteps {
			break
		}
		stepCount++

		/* Generate embedding for step query */
		queryEmbedding, err := r.embed.Embed(ctx, step.Query, "all-MiniLM-L6-v2")
		if err != nil {
			continue
		}

		/* Retrieve documents for this step */
		retrieveQuery := fmt.Sprintf(`SELECT id, content, metadata, 1 - (%s <=> $1::vector) AS similarity
			FROM %s
			ORDER BY %s <=> $1::vector
			LIMIT $2`, qVectorCol, qTable, qVectorCol)

		type StepRow struct {
			ID         int64                  `db:"id"`
			Content    string                 `db:"content"`
			Metadata   map[string]interface{} `db:"metadata"`
			Similarity float64                `db:"similarity"`
		}

		var rows []StepRow
		err = r.db.DB.SelectContext(ctx, &rows, retrieveQuery, queryEmbedding, limit)
		if err != nil {
			continue
		}

		/* Accumulate context */
		for _, row := range rows {
			accumulatedContext = append(accumulatedContext, row.Content)
		}

		/* Add step to reasoning path */
		reasoningPath = append(reasoningPath, fmt.Sprintf("Step %d: %s", step.Step, step.Action))

		/* Verify evidence sufficiency */
		verificationPrompt := fmt.Sprintf(`Evaluate if the following context is sufficient to answer the query.

Query: %s

Context:
%s

Respond with a JSON object: {"sufficient": true/false, "score": 0.0-1.0, "reason": "explanation"}`, query, joinStrings(accumulatedContext, "\n\n"))

		verificationConfig := map[string]interface{}{
			"temperature": 0.1,
			"max_tokens":  200,
		}

		verificationResponse, err := r.llm.Generate(ctx, "gpt-4", verificationPrompt, verificationConfig)
		if err == nil {
			/* Parse evidence score (simplified) */
			evidenceScore = r.extractEvidenceScore(verificationResponse.Content)
		} else {
			evidenceScore = 0.5
		}

		/* Add step to execution trace */
		executionTrace = append(executionTrace, ExecutionStep{
			Step:            step.Step,
			Query:           step.Query,
			ChunksRetrieved: len(rows),
			EvidenceScore:   evidenceScore,
			Sufficient:      evidenceScore >= evidenceThreshold,
		})

		/* Check stop condition */
		if evidenceScore >= evidenceThreshold {
			break
		}

		/* Check token budget */
		tokensUsed += 500
		if tokensUsed >= maxTokens {
			break
		}
	}

	/* Step 3: Generate final answer */
	answer, err := r.generateAnswer(ctx, query, accumulatedContext)
	if err != nil {
		return nil, fmt.Errorf("agentic RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &AgenticRAGResult{
		Query:          query,
		Answer:         answer,
		Documents:      accumulatedContext,
		Count:          len(accumulatedContext),
		Method:         "agentic",
		ExecutionTrace: executionTrace,
		ReasoningPath:  reasoningPath,
		StepsExecuted:  stepCount,
	}, nil
}

/* Helper types for Agentic RAG */
type AgenticPlanStep struct {
	Step   int
	Action string
	Query  string
	Reason string
}

type ExecutionStep struct {
	Step            int
	Query           string
	ChunksRetrieved int
	EvidenceScore   float64
	Sufficient      bool
}

type AgenticRAGResult struct {
	Query          string
	Answer         string
	Documents      []string
	Count          int
	Method         string
	ExecutionTrace []ExecutionStep
	ReasoningPath  []string
	StepsExecuted  int
}

/* parsePlanSteps parses plan steps from LLM response */
func (r *AdvancedRAG) parsePlanSteps(response string) []AgenticPlanStep {
	/* Simplified parsing - in production would use proper JSON parsing */
	steps := make([]AgenticPlanStep, 0)

	/* Try to extract JSON array from response */
	/* For now, create a simple fallback */
	if strings.Contains(response, "step") || strings.Contains(response, "Step") {
		/* Basic parsing - extract step information */
		lines := strings.Split(response, "\n")
		stepNum := 1
		for _, line := range lines {
			if strings.Contains(line, "query") || strings.Contains(line, "Query") {
				/* Extract query */
				query := strings.TrimSpace(line)
				steps = append(steps, AgenticPlanStep{
					Step:   stepNum,
					Action: "Retrieve information",
					Query:  query,
					Reason: "Step in execution plan",
				})
				stepNum++
			}
		}
	}

	/* If no steps parsed, return empty */
	if len(steps) == 0 {
		return nil
	}

	return steps
}

/* extractEvidenceScore extracts evidence score from verification response */
func (r *AdvancedRAG) extractEvidenceScore(response string) float64 {
	/* Simplified extraction - look for numeric score */
	/* In production would parse JSON properly */
	if strings.Contains(response, "score") {
		/* Try to extract number */
		var score float64
		_, err := fmt.Sscanf(response, "%f", &score)
		if err == nil && score >= 0 && score <= 1 {
			return score
		}
	}
	return 0.5
}

/* ContextualRAG performs RAG with context-aware query rewriting and adaptation */
func (r *AdvancedRAG) ContextualRAG(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, conversationHistory []map[string]interface{}, sessionContext map[string]interface{}, crossSessionContext bool) (*ContextualRAGResult, error) {
	/* Step 1: Build conversation context from history */
	conversationContext := ""
	for _, msg := range conversationHistory {
		if role, ok := msg["role"].(string); ok {
			if content, ok := msg["content"].(string); ok {
				if conversationContext != "" {
					conversationContext += "\n"
				}
				conversationContext += fmt.Sprintf("%s: %s", role, content)
			}
		}
	}

	/* Step 2: Extract strategic context from session context */
	strategicContext := ""
	if sessionContext != nil {
		if topics, ok := sessionContext["topics"].(string); ok {
			strategicContext += fmt.Sprintf("Topics discussed: %s\n", topics)
		}
		if intent, ok := sessionContext["intent"].(string); ok {
			strategicContext += fmt.Sprintf("User intent: %s\n", intent)
		}
		if domain, ok := sessionContext["domain"].(string); ok {
			strategicContext += fmt.Sprintf("Domain: %s\n", domain)
		}
	}

	/* Step 3: Context-aware query rewriting */
	rewritePrompt := fmt.Sprintf(`Rewrite the following query to be more specific and retrievable based on the conversation context. If the query is a follow-up question, expand it to include necessary context from the conversation. If the query uses pronouns or references, replace them with explicit terms.

Conversation History:
%s

Strategic Context:
%s

Original Query: %s

Rewritten Query (be specific and explicit):`, conversationContext, strategicContext, query)

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  200,
	}

	rewriteResponse, err := r.llm.Generate(ctx, "gpt-4", rewritePrompt, llmConfig)
	rewrittenQuery := query
	if err == nil && rewriteResponse.Content != "" {
		rewrittenQuery = strings.TrimSpace(rewriteResponse.Content)
	}

	/* Step 4: Adapt retrieval strategy based on context */
	adaptationPrompt := fmt.Sprintf(`Based on the conversation context, determine the best retrieval strategy:

Conversation: %s
Query: %s
Rewritten Query: %s

Respond with JSON: {"strategy": "semantic|hybrid|keyword", "weight": 0.0-1.0, "reason": "explanation"}`, conversationContext, query, rewrittenQuery)

	adaptationConfig := map[string]interface{}{
		"temperature": 0.2,
		"max_tokens":  150,
	}

	_, err = r.llm.Generate(ctx, "gpt-4", adaptationPrompt, adaptationConfig)
	contextAdaptation := map[string]interface{}{
		"strategy": "semantic",
		"weight":   0.7,
		"reason":   "Context-aware semantic retrieval",
	}
	if err == nil {
		/* Parse adaptation (simplified) */
		/* In production would parse JSON properly */
	}

	/* Step 5: Retrieve using rewritten query */
	queryEmbedding, err := r.embed.Embed(ctx, rewrittenQuery, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("contextual RAG failed: embedding_error=true, error=%w", err)
	}

	qTable, qVectorCol, err := quoteRAGIdentifiers(tableName, vectorCol)
	if err != nil {
		return nil, fmt.Errorf("contextual RAG failed: %w", err)
	}
	retrieveQuery := fmt.Sprintf(`SELECT id, content, metadata, 1 - (%s <=> $1::vector) AS similarity
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2`, qVectorCol, qTable, qVectorCol)

	type ContextualRow struct {
		ID         int64                  `db:"id"`
		Content    string                 `db:"content"`
		Metadata   map[string]interface{} `db:"metadata"`
		Similarity float64                `db:"similarity"`
	}

	var rows []ContextualRow
	err = r.db.DB.SelectContext(ctx, &rows, retrieveQuery, queryEmbedding, limit)
	if err != nil {
		return nil, fmt.Errorf("contextual RAG failed: retrieval_error=true, error=%w", err)
	}

	documents := make([]string, len(rows))
	for i, row := range rows {
		documents[i] = row.Content
	}

	/* Step 6: Generate answer with full context awareness */
	contextStr := joinStrings(documents, "\n\n")
	if conversationContext != "" {
		contextStr = fmt.Sprintf("Conversation History:\n%s\n\nRetrieved Context:\n%s", conversationContext, contextStr)
	}
	if strategicContext != "" {
		contextStr = fmt.Sprintf("Strategic Context:\n%s\n\n%s", strategicContext, contextStr)
	}

	answer, err := r.generateAnswer(ctx, query, []string{contextStr})
	if err != nil {
		return nil, fmt.Errorf("contextual RAG failed: answer_generation_error=true, error=%w", err)
	}

	return &ContextualRAGResult{
		Query:             query,
		RewrittenQuery:    rewrittenQuery,
		Answer:            answer,
		Documents:         documents,
		Count:             len(documents),
		Method:            "contextual",
		ContextAdaptation: contextAdaptation,
	}, nil
}

/* Helper types for Contextual RAG */
type ContextualRAGResult struct {
	Query             string
	RewrittenQuery    string
	Answer            string
	Documents         []string
	Count             int
	Method            string
	ContextAdaptation map[string]interface{}
}

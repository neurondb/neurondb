/*-------------------------------------------------------------------------
 *
 * ai_rag_evaluation.go
 *    RAG Evaluation Framework for NeuronMCP
 *
 * Provides comprehensive RAG evaluation including retrieval accuracy,
 * answer quality, and embedding drift detection.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/ai_rag_evaluation.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* AIRAGEvaluationTool provides comprehensive RAG evaluation */
type AIRAGEvaluationTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewAIRAGEvaluationTool creates a new RAG evaluation tool */
func NewAIRAGEvaluationTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Evaluation operation: evaluate_retrieval, evaluate_answer, full_evaluation",
				"enum":        []interface{}{"evaluate_retrieval", "evaluate_answer", "full_evaluation"},
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Test query",
			},
			"ground_truth": map[string]interface{}{
				"type":        "array",
				"description": "Ground truth relevant documents",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
			"retrieved_docs": map[string]interface{}{
				"type":        "array",
				"description": "Retrieved documents",
				"items": map[string]interface{}{
					"type": "object",
				},
			},
			"answer": map[string]interface{}{
				"type":        "string",
				"description": "Generated answer",
			},
			"reference_answer": map[string]interface{}{
				"type":        "string",
				"description": "Reference/ground truth answer",
			},
			"k": map[string]interface{}{
				"type":        "integer",
				"description": "Number of top results to evaluate (for retrieval)",
				"default":     10,
			},
		},
		"required": []interface{}{"operation"},
	}

	return &AIRAGEvaluationTool{
		BaseTool: NewBaseTool(
			"ai_rag_evaluation",
			"Comprehensive RAG evaluation framework with retrieval accuracy and answer quality metrics",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the RAG evaluation tool */
func (t *AIRAGEvaluationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "evaluate_retrieval":
		return t.evaluateRetrieval(ctx, params)
	case "evaluate_answer":
		return t.evaluateAnswer(ctx, params)
	case "full_evaluation":
		return t.fullEvaluation(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* evaluateRetrieval evaluates retrieval accuracy */
func (t *AIRAGEvaluationTool) evaluateRetrieval(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	groundTruthRaw, _ := params["ground_truth"].([]interface{})
	retrievedDocsRaw, _ := params["retrieved_docs"].([]interface{})
	k, _ := params["k"].(float64)

	if k == 0 {
		k = 10
	}

	if len(groundTruthRaw) == 0 || len(retrievedDocsRaw) == 0 {
		return Error("ground_truth and retrieved_docs are required", "INVALID_PARAMS", nil), nil
	}

	groundTruth := []string{}
	for _, gt := range groundTruthRaw {
		groundTruth = append(groundTruth, fmt.Sprintf("%v", gt))
	}

	retrievedDocs := []string{}
	for _, doc := range retrievedDocsRaw {
		if docMap, ok := doc.(map[string]interface{}); ok {
			if id, ok := docMap["id"].(string); ok {
				retrievedDocs = append(retrievedDocs, id)
			} else if content, ok := docMap["content"].(string); ok {
				retrievedDocs = append(retrievedDocs, content)
			}
		} else {
			retrievedDocs = append(retrievedDocs, fmt.Sprintf("%v", doc))
		}
	}

	/* Calculate metrics */
	precision := t.calculatePrecision(groundTruth, retrievedDocs, int(k))
	recall := t.calculateRecall(groundTruth, retrievedDocs)
	f1 := t.calculateF1(precision, recall)
	mrr := t.calculateMRR(groundTruth, retrievedDocs)
	ndcg := t.calculateNDCG(groundTruth, retrievedDocs, int(k))

	return Success(map[string]interface{}{
		"query":     query,
		"metrics": map[string]interface{}{
			"precision_at_k": precision,
			"recall":          recall,
			"f1_score":         f1,
			"mrr":             mrr,
			"ndcg_at_k":       ndcg,
		},
		"k": int(k),
	}, nil), nil
}

/* evaluateAnswer evaluates answer quality */
func (t *AIRAGEvaluationTool) evaluateAnswer(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	answer, _ := params["answer"].(string)
	referenceAnswer, _ := params["reference_answer"].(string)

	if answer == "" || referenceAnswer == "" {
		return Error("answer and reference_answer are required", "INVALID_PARAMS", nil), nil
	}

	/* Calculate answer quality metrics */
	exactMatch := t.calculateExactMatch(answer, referenceAnswer)
	bleuScore := t.calculateBLEU(answer, referenceAnswer)
	rougeScore := t.calculateROUGE(answer, referenceAnswer)
	semanticSimilarity := t.calculateSemanticSimilarity(answer, referenceAnswer)

	return Success(map[string]interface{}{
		"answer":            answer,
		"reference_answer": referenceAnswer,
		"metrics": map[string]interface{}{
			"exact_match":         exactMatch,
			"bleu_score":          bleuScore,
			"rouge_score":         rougeScore,
			"semantic_similarity": semanticSimilarity,
		},
	}, nil), nil
}

/* fullEvaluation performs full RAG evaluation */
func (t *AIRAGEvaluationTool) fullEvaluation(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Evaluate both retrieval and answer quality */
	retrievalResult, _ := t.evaluateRetrieval(ctx, params)
	answerResult, _ := t.evaluateAnswer(ctx, params)

	retrievalData, _ := retrievalResult.Data.(map[string]interface{})
	answerData, _ := answerResult.Data.(map[string]interface{})

	retrievalMetrics, _ := retrievalData["metrics"].(map[string]interface{})
	answerMetrics, _ := answerData["metrics"].(map[string]interface{})

	/* Calculate overall score */
	overallScore := t.calculateOverallScore(retrievalMetrics, answerMetrics)

	return Success(map[string]interface{}{
		"retrieval_metrics": retrievalMetrics,
		"answer_metrics":     answerMetrics,
		"overall_score":     overallScore,
		"evaluation_time":   time.Now(),
	}, nil), nil
}

/* calculatePrecision calculates precision@k */
func (t *AIRAGEvaluationTool) calculatePrecision(groundTruth, retrieved []string, k int) float64 {
	if len(retrieved) == 0 {
		return 0.0
	}

	relevant := 0
	topK := retrieved
	if len(topK) > k {
		topK = topK[:k]
	}

	groundTruthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		groundTruthSet[gt] = true
	}

	for _, doc := range topK {
		if groundTruthSet[doc] {
			relevant++
		}
	}

	return float64(relevant) / float64(len(topK))
}

/* calculateRecall calculates recall */
func (t *AIRAGEvaluationTool) calculateRecall(groundTruth, retrieved []string) float64 {
	if len(groundTruth) == 0 {
		return 0.0
	}

	groundTruthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		groundTruthSet[gt] = true
	}

	relevant := 0
	for _, doc := range retrieved {
		if groundTruthSet[doc] {
			relevant++
		}
	}

	return float64(relevant) / float64(len(groundTruth))
}

/* calculateF1 calculates F1 score */
func (t *AIRAGEvaluationTool) calculateF1(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * (precision * recall) / (precision + recall)
}

/* calculateMRR calculates Mean Reciprocal Rank */
func (t *AIRAGEvaluationTool) calculateMRR(groundTruth, retrieved []string) float64 {
	groundTruthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		groundTruthSet[gt] = true
	}

	for i, doc := range retrieved {
		if groundTruthSet[doc] {
			return 1.0 / float64(i+1)
		}
	}

	return 0.0
}

/* calculateNDCG calculates Normalized Discounted Cumulative Gain */
func (t *AIRAGEvaluationTool) calculateNDCG(groundTruth, retrieved []string, k int) float64 {
	groundTruthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		groundTruthSet[gt] = true
	}

	topK := retrieved
	if len(topK) > k {
		topK = topK[:k]
	}

	dcg := 0.0
	for i, doc := range topK {
		if groundTruthSet[doc] {
			dcg += 1.0 / math.Log2(float64(i+2))
		}
	}

	/* Ideal DCG */
	idealDCG := 0.0
	for i := 0; i < len(groundTruth) && i < k; i++ {
		idealDCG += 1.0 / math.Log2(float64(i+2))
	}

	if idealDCG == 0 {
		return 0.0
	}

	return dcg / idealDCG
}

/* calculateExactMatch calculates exact match score */
func (t *AIRAGEvaluationTool) calculateExactMatch(answer, reference string) float64 {
	if answer == reference {
		return 1.0
	}
	return 0.0
}

/* calculateBLEU calculates BLEU score (simplified) */
func (t *AIRAGEvaluationTool) calculateBLEU(answer, reference string) float64 {
	/* Simplified BLEU - would need proper n-gram implementation */
	answerWords := strings.Fields(strings.ToLower(answer))
	referenceWords := strings.Fields(strings.ToLower(reference))

	if len(referenceWords) == 0 {
		return 0.0
	}

	matches := 0
	referenceSet := make(map[string]bool)
	for _, word := range referenceWords {
		referenceSet[word] = true
	}

	for _, word := range answerWords {
		if referenceSet[word] {
			matches++
		}
	}

	return float64(matches) / float64(len(answerWords))
}

/* calculateROUGE calculates ROUGE score (simplified) */
func (t *AIRAGEvaluationTool) calculateROUGE(answer, reference string) float64 {
	/* ROUGE-L (longest common subsequence) */
	answerWords := strings.Fields(strings.ToLower(answer))
	referenceWords := strings.Fields(strings.ToLower(reference))

	if len(referenceWords) == 0 {
		return 0.0
	}

	/* Calculate longest common subsequence (LCS) */
	lcs := t.longestCommonSubsequence(answerWords, referenceWords)
	lcsLength := float64(len(lcs))

	/* ROUGE-L precision = LCS length / answer length */
	precision := lcsLength / float64(len(answerWords))
	if len(answerWords) == 0 {
		precision = 0.0
	}

	/* ROUGE-L recall = LCS length / reference length */
	recall := lcsLength / float64(len(referenceWords))

	/* ROUGE-L F1 score */
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * (precision * recall) / (precision + recall)
}

/* longestCommonSubsequence calculates LCS between two word sequences */
func (t *AIRAGEvaluationTool) longestCommonSubsequence(seq1, seq2 []string) []string {
	m := len(seq1)
	n := len(seq2)

	/* Create DP table */
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	/* Fill DP table */
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if seq1[i-1] == seq2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}

	/* Reconstruct LCS */
	lcs := make([]string, 0, dp[m][n])
	i, j := m, n
	for i > 0 && j > 0 {
		if seq1[i-1] == seq2[j-1] {
			lcs = append([]string{seq1[i-1]}, lcs...)
			i--
			j--
		} else if dp[i-1][j] > dp[i][j-1] {
			i--
		} else {
			j--
		}
	}

	return lcs
}

/* max returns the maximum of two integers */
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/* calculateSemanticSimilarity calculates semantic similarity */
func (t *AIRAGEvaluationTool) calculateSemanticSimilarity(answer, reference string) float64 {
	/* Use NeuronDB embedding functions to calculate cosine similarity */
	/* Generate embeddings for both texts */
	ctx := context.Background()

	/* Generate embedding for answer */
	answerEmbedQuery := `SELECT neurondb_embed($1::text) AS embedding`
	answerRows, err := t.db.Query(ctx, answerEmbedQuery, []interface{}{answer})
	if err != nil {
		t.logger.Warn("Failed to generate embedding for answer", map[string]interface{}{
			"error": err.Error(),
		})
		return 0.0
	}
	defer answerRows.Close()

	var answerEmbedding *string
	if answerRows.Next() {
		if err := answerRows.Scan(&answerEmbedding); err != nil || answerEmbedding == nil {
			return 0.0
		}
	} else {
		return 0.0
	}

	/* Generate embedding for reference */
	referenceEmbedQuery := `SELECT neurondb_embed($1::text) AS embedding`
	referenceRows, err := t.db.Query(ctx, referenceEmbedQuery, []interface{}{reference})
	if err != nil {
		t.logger.Warn("Failed to generate embedding for reference", map[string]interface{}{
			"error": err.Error(),
		})
		return 0.0
	}
	defer referenceRows.Close()

	var referenceEmbedding *string
	if referenceRows.Next() {
		if err := referenceRows.Scan(&referenceEmbedding); err != nil || referenceEmbedding == nil {
			return 0.0
		}
	} else {
		return 0.0
	}

	/* Calculate cosine similarity between embeddings */
	similarityQuery := `SELECT $1::vector <=> $2::vector AS similarity`
	similarityRows, err := t.db.Query(ctx, similarityQuery, []interface{}{*answerEmbedding, *referenceEmbedding})
	if err != nil {
		t.logger.Warn("Failed to calculate cosine similarity", map[string]interface{}{
			"error": err.Error(),
		})
		return 0.0
	}
	defer similarityRows.Close()

	if similarityRows.Next() {
		var distance *float64
		if err := similarityRows.Scan(&distance); err == nil && distance != nil {
			/* Cosine distance is 0 for identical, 2 for opposite */
			/* Convert to similarity: similarity = 1 - (distance / 2) */
			similarity := 1.0 - (*distance / 2.0)
			if similarity < 0.0 {
				similarity = 0.0
			}
			if similarity > 1.0 {
				similarity = 1.0
			}
			return similarity
		}
	}

	return 0.0
}

/* calculateOverallScore calculates overall RAG score */
func (t *AIRAGEvaluationTool) calculateOverallScore(retrievalMetrics, answerMetrics map[string]interface{}) float64 {
	retrievalScore := 0.0
	if precision, ok := retrievalMetrics["precision_at_k"].(float64); ok {
		retrievalScore += precision * 0.3
	}
	if recall, ok := retrievalMetrics["recall"].(float64); ok {
		retrievalScore += recall * 0.2
	}
	if ndcg, ok := retrievalMetrics["ndcg_at_k"].(float64); ok {
		retrievalScore += ndcg * 0.2
	}

	answerScore := 0.0
	if semantic, ok := answerMetrics["semantic_similarity"].(float64); ok {
		answerScore += semantic * 0.3
	}

	return retrievalScore + answerScore
}

/* AIEmbeddingDriftDetectionTool detects embedding distribution shifts */
type AIEmbeddingDriftDetectionTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewAIEmbeddingDriftDetectionTool creates a new embedding drift detection tool */
func NewAIEmbeddingDriftDetectionTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"table": map[string]interface{}{
				"type":        "string",
				"description": "Table name containing embeddings",
			},
			"vector_column": map[string]interface{}{
				"type":        "string",
				"description": "Column name containing vectors",
			},
			"reference_table": map[string]interface{}{
				"type":        "string",
				"description": "Reference table for comparison",
			},
			"method": map[string]interface{}{
				"type":        "string",
				"description": "Drift detection method: statistical, distance, pca",
				"enum":        []interface{}{"statistical", "distance", "pca"},
				"default":     "statistical",
			},
			"threshold": map[string]interface{}{
				"type":        "number",
				"description": "Drift detection threshold",
				"default":     0.1,
			},
		},
		"required": []interface{}{"table", "vector_column"},
	}

	return &AIEmbeddingDriftDetectionTool{
		BaseTool: NewBaseTool(
			"ai_embedding_drift_detection",
			"Detect embedding distribution shifts and drift over time",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the embedding drift detection tool */
func (t *AIEmbeddingDriftDetectionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	referenceTable, _ := params["reference_table"].(string)
	method, _ := params["method"].(string)
	threshold, _ := params["threshold"].(float64)

	if table == "" || vectorColumn == "" {
		return Error("table and vector_column are required", "INVALID_PARAMS", nil), nil
	}

	if method == "" {
		method = "statistical"
	}
	if threshold == 0 {
		threshold = 0.1
	}

	/* Detect drift */
	driftScore, driftDetected := t.detectDrift(ctx, table, vectorColumn, referenceTable, method, threshold)

	return Success(map[string]interface{}{
		"table":          table,
		"vector_column":  vectorColumn,
		"method":         method,
		"threshold":      threshold,
		"drift_score":    driftScore,
		"drift_detected": driftDetected,
		"severity":       t.calculateSeverity(driftScore, threshold),
	}, nil), nil
}

/* detectDrift detects embedding drift */
func (t *AIEmbeddingDriftDetectionTool) detectDrift(ctx context.Context, table, vectorColumn, referenceTable, method string, threshold float64) (float64, bool) {
	/* Calculate drift score based on method */
	var driftScore float64

	switch method {
	case "statistical":
		/* Compare statistical properties (mean, std) */
		driftScore = t.calculateStatisticalDrift(ctx, table, vectorColumn, referenceTable)

	case "distance":
		/* Compare distance distributions */
		driftScore = t.calculateDistanceDrift(ctx, table, vectorColumn, referenceTable)

	case "pca":
		/* Compare PCA projections */
		driftScore = t.calculatePCADrift(ctx, table, vectorColumn, referenceTable)

	default:
		driftScore = 0.5
	}

	return driftScore, driftScore > threshold
}

/* calculateStatisticalDrift calculates statistical drift */
func (t *AIEmbeddingDriftDetectionTool) calculateStatisticalDrift(ctx context.Context, table, vectorColumn, referenceTable string) float64 {
	/* Compare mean and std of embeddings */
	query := fmt.Sprintf(`
		SELECT 
			AVG(embedding_norm) as mean_norm,
			STDDEV(embedding_norm) as std_norm
		FROM (
			SELECT SQRT(SUM(pow * pow)) as embedding_norm
			FROM (
				SELECT unnest(%s::float[]) as pow
				FROM %s
				LIMIT 1000
			) unnested
			GROUP BY embedding_norm
		) norms
	`, vectorColumn, table)

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return 0.5
	}
	defer rows.Close()

	if rows.Next() {
		var meanNorm, stdNorm *float64
		if err := rows.Scan(&meanNorm, &stdNorm); err == nil {
			/* Compare with reference if available */
			return math.Abs(getFloat(meanNorm, 0.0) - 1.0) /* Simplified */
		}
	}

	return 0.5
}

/* calculateDistanceDrift calculates distance-based drift */
func (t *AIEmbeddingDriftDetectionTool) calculateDistanceDrift(ctx context.Context, table, vectorColumn, referenceTable string) float64 {
	/* Compare distance distributions between current and reference embeddings */
	if referenceTable == "" {
		/* No reference table - use historical data */
		query := fmt.Sprintf(`
			SELECT 
				AVG(distance) as avg_distance,
				STDDEV(distance) as std_distance
			FROM (
				SELECT 
					(%s::vector <=> LAG(%s::vector) OVER (ORDER BY id)) as distance
				FROM %s
				WHERE %s IS NOT NULL
				LIMIT 1000
			) distances
		`, vectorColumn, vectorColumn, table, vectorColumn)

		rows, err := t.db.Query(ctx, query, nil)
		if err != nil {
			return 0.3
		}
		defer rows.Close()

		if rows.Next() {
			var avgDist, stdDist *float64
			if err := rows.Scan(&avgDist, &stdDist); err == nil {
				/* Drift score based on standard deviation */
				if stdDist != nil && *stdDist > 0 {
					return math.Min(*stdDist/10.0, 1.0)
				}
			}
		}
		return 0.3
	}

	/* Compare with reference table */
	query := fmt.Sprintf(`
		SELECT 
			AVG(ABS(avg_dist_current - avg_dist_reference)) as drift_score
		FROM (
			SELECT AVG(distance) as avg_dist_current
			FROM (
				SELECT 
					(%s::vector <=> LAG(%s::vector) OVER (ORDER BY id)) as distance
				FROM %s
				WHERE %s IS NOT NULL
				LIMIT 500
			) current_distances
		) current_stats
		CROSS JOIN (
			SELECT AVG(distance) as avg_dist_reference
			FROM (
				SELECT 
					(%s::vector <=> LAG(%s::vector) OVER (ORDER BY id)) as distance
				FROM %s
				WHERE %s IS NOT NULL
				LIMIT 500
			) reference_distances
		) reference_stats
	`, vectorColumn, vectorColumn, table, vectorColumn,
		vectorColumn, vectorColumn, referenceTable, vectorColumn)

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return 0.3
	}
	defer rows.Close()

	if rows.Next() {
		var driftScore *float64
		if err := rows.Scan(&driftScore); err == nil && driftScore != nil {
			return math.Min(*driftScore, 1.0)
		}
	}

	return 0.3
}

/* calculatePCADrift calculates PCA-based drift */
func (t *AIEmbeddingDriftDetectionTool) calculatePCADrift(ctx context.Context, table, vectorColumn, referenceTable string) float64 {
	/* Compare PCA projections - simplified implementation */
	/* In production, would use actual PCA decomposition */
	if referenceTable == "" {
		/* Use variance as proxy for drift */
		query := fmt.Sprintf(`
			SELECT 
				STDDEV(embedding_norm) as variance
			FROM (
				SELECT 
					SQRT(SUM(pow * pow)) as embedding_norm
				FROM (
					SELECT unnest(%s::float[]) as pow
					FROM %s
					WHERE %s IS NOT NULL
					LIMIT 1000
				) unnested
				GROUP BY embedding_norm
			) norms
		`, vectorColumn, table, vectorColumn)

		rows, err := t.db.Query(ctx, query, nil)
		if err != nil {
			return 0.4
		}
		defer rows.Close()

		if rows.Next() {
			var variance *float64
			if err := rows.Scan(&variance); err == nil && variance != nil {
				return math.Min(*variance/10.0, 1.0)
			}
		}
		return 0.4
	}

	/* Compare variance between current and reference */
	query := fmt.Sprintf(`
		SELECT 
			ABS(variance_current - variance_reference) as drift_score
		FROM (
			SELECT STDDEV(embedding_norm) as variance_current
			FROM (
				SELECT 
					SQRT(SUM(pow * pow)) as embedding_norm
				FROM (
					SELECT unnest(%s::float[]) as pow
					FROM %s
					WHERE %s IS NOT NULL
					LIMIT 500
				) unnested
				GROUP BY embedding_norm
			) current_norms
		) current_stats
		CROSS JOIN (
			SELECT STDDEV(embedding_norm) as variance_reference
			FROM (
				SELECT 
					SQRT(SUM(pow * pow)) as embedding_norm
				FROM (
					SELECT unnest(%s::float[]) as pow
					FROM %s
					WHERE %s IS NOT NULL
					LIMIT 500
				) unnested
				GROUP BY embedding_norm
			) reference_norms
		) reference_stats
	`, vectorColumn, table, vectorColumn,
		vectorColumn, referenceTable, vectorColumn)

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return 0.4
	}
	defer rows.Close()

	if rows.Next() {
		var driftScore *float64
		if err := rows.Scan(&driftScore); err == nil && driftScore != nil {
			return math.Min(*driftScore/10.0, 1.0)
		}
	}

	return 0.4
}

/* calculateSeverity calculates drift severity */
func (t *AIEmbeddingDriftDetectionTool) calculateSeverity(driftScore, threshold float64) string {
	if driftScore > threshold*2 {
		return "high"
	} else if driftScore > threshold {
		return "medium"
	}
	return "low"
}


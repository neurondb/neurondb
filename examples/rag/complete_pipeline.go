package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

/* Complete RAG Pipeline Example in Go
 *
 * This example demonstrates:
 * 1. Document ingestion (chunking + embedding)
 * 2. RAG query execution
 * 3. Evaluation metrics
 */

func main() {
	/* Get database connection string from environment */
	dsn := os.Getenv("NEURONDB_DSN")
	if dsn == "" {
		dsn = "postgres://postgres:postgres@localhost:5432/neurondb?sslmode=disable"
	}

	db, err := sqlx.Connect("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	/* Step 1: Document Ingestion */
	fmt.Println("=== Step 1: Document Ingestion ===")
	documentText := `Machine learning is a subset of artificial intelligence that focuses on 
	the development of algorithms and statistical models that enable computer systems to 
	improve their performance on a specific task through experience. Unlike traditional 
	programming, where explicit instructions are provided, machine learning systems learn 
	from data patterns and make predictions or decisions based on that learning.`

	tableName := "documents"
	
	/* Create table if it doesn't exist */
	_, err = db.ExecContext(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id SERIAL PRIMARY KEY,
			content TEXT,
			embedding vector(384),
			metadata JSONB,
			created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
		)
	`, tableName))
	if err != nil {
		log.Fatalf("Failed to create table: %v", err)
	}

	/* Ingest document using rag_ingest_document */
	ingestQuery := `SELECT * FROM neurondb.rag_ingest_document($1, $2, 'content', 'embedding', 'default', 512, 128, '{}'::jsonb)`
	rows, err := db.QueryContext(ctx, ingestQuery, documentText, tableName)
	if err != nil {
		log.Fatalf("Failed to ingest document: %v", err)
	}
	defer rows.Close()

	var chunkIDs []int64
	for rows.Next() {
		var chunkID int64
		var chunkText string
		var embedding string
		if err := rows.Scan(&chunkID, &chunkText, &embedding); err != nil {
			log.Printf("Error scanning chunk: %v", err)
			continue
		}
		chunkIDs = append(chunkIDs, chunkID)
		fmt.Printf("  Ingested chunk %d: %s...\n", chunkID, chunkText[:min(50, len(chunkText))])
	}
	fmt.Printf("Successfully ingested %d chunks\n\n", len(chunkIDs))

	/* Step 2: RAG Query */
	fmt.Println("=== Step 2: RAG Query ===")
	query := "What is machine learning?"

	/* Use rag_query function */
	ragQuery := `SELECT chunk_text, relevance_score FROM neurondb.rag_query($1, $2, 'embedding', 'content', 'default', 3)`
	rows, err = db.QueryContext(ctx, ragQuery, query, tableName)
	if err != nil {
		log.Fatalf("Failed to execute RAG query: %v", err)
	}
	defer rows.Close()

	var contexts []string
	for rows.Next() {
		var chunkText string
		var relevanceScore float64
		if err := rows.Scan(&chunkText, &relevanceScore); err != nil {
			log.Printf("Error scanning result: %v", err)
			continue
		}
		contexts = append(contexts, chunkText)
		fmt.Printf("  Retrieved context (relevance: %.3f): %s...\n", relevanceScore, chunkText[:min(50, len(chunkText))])
	}
	fmt.Printf("Retrieved %d context chunks\n\n", len(contexts))

	/* Step 3: Generate Answer (using LLM if available) */
	fmt.Println("=== Step 3: Generate Answer ===")
	if len(contexts) > 0 {
		contextText := ""
		for i, ctx := range contexts {
			if i > 0 {
				contextText += "\n\n"
			}
			contextText += ctx
		}
		prompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s\n\nAnswer:", contextText, query)
		
		var answer string
		llmQuery := `SELECT ndb_llm_complete($1, '{"temperature": 0.7, "max_tokens": 500}') AS answer`
		err = db.GetContext(ctx, &answer, llmQuery, prompt)
		if err != nil {
			fmt.Printf("  LLM not available, using simple answer\n")
			answer = "Machine learning is a subset of AI that enables systems to learn from data."
		} else {
			fmt.Printf("  Generated answer: %s\n\n", answer)
		}

		/* Step 4: Evaluation */
		fmt.Println("=== Step 4: RAG Evaluation ===")
		evalQuery := `SELECT neurondb.rag_evaluate($1, $2, $3::text[], 'basic') AS evaluation`
		var evaluationJSON string
		err = db.GetContext(ctx, &evaluationJSON, evalQuery, query, answer, contexts)
		if err != nil {
			log.Printf("Failed to evaluate: %v", err)
		} else {
			var evaluation map[string]interface{}
			if err := json.Unmarshal([]byte(evaluationJSON), &evaluation); err == nil {
				fmt.Printf("  Relevancy: %.3f\n", getFloat64(evaluation, "relevancy", 0.0))
				fmt.Printf("  Semantic Similarity: %.3f\n", getFloat64(evaluation, "semantic_similarity", 0.0))
				if stats, ok := evaluation["similarity_stats"].(map[string]interface{}); ok {
					fmt.Printf("  Average Similarity: %.3f\n", getFloat64(stats, "avg", 0.0))
				}
			}
		}
	}

	fmt.Println("\n=== RAG Pipeline Complete ===")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func getFloat64(m map[string]interface{}, key string, defaultValue float64) float64 {
	if val, ok := m[key]; ok {
		if f, ok := val.(float64); ok {
			return f
		}
	}
	return defaultValue
}

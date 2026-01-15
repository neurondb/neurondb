/*-------------------------------------------------------------------------
 *
 * embedding.go
 *    Embedding client for NeuronDB integration
 *
 * Provides client functionality for generating text embeddings through
 * NeuronDB using various embedding models.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/embedding.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"context"
	"fmt"
	"strings"

	"github.com/jmoiron/sqlx"
)

/* EmbeddingClient handles embedding generation via NeuronDB */
type EmbeddingClient struct {
	db *sqlx.DB
}

/* NewEmbeddingClient creates a new embedding client */
func NewEmbeddingClient(db *sqlx.DB) *EmbeddingClient {
	return &EmbeddingClient{db: db}
}

/* Embed generates an embedding for the given text using the specified model */
func (c *EmbeddingClient) Embed(ctx context.Context, text string, model string) (Vector, error) {
	var embeddingStr string
	query := `SELECT neurondb_embed($1, $2)::text AS embedding`

	err := c.db.GetContext(ctx, &embeddingStr, query, text, model)
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed via NeuronDB: model_name='%s', text_length=%d, function='neurondb_embed', error=%w",
			model, len(text), err)
	}

	/* Parse vector string format [1.0, 2.0, 3.0] to []float32 */
	embedding, err := parseVector(embeddingStr)
	if err != nil {
		embeddingStrPreview := embeddingStr
		if len(embeddingStrPreview) > 200 {
			embeddingStrPreview = embeddingStrPreview[:200] + "..."
		}
		return nil, fmt.Errorf("embedding parsing failed: model_name='%s', text_length=%d, embedding_string_length=%d, embedding_string_preview='%s', function='neurondb_embed', error=%w",
			model, len(text), len(embeddingStr), embeddingStrPreview, err)
	}

	return embedding, nil
}

/* EmbedBatch generates embeddings for multiple texts */
func (c *EmbeddingClient) EmbedBatch(ctx context.Context, texts []string, model string) ([]Vector, error) {
	/* Use array format for batch embedding if available */
	query := `SELECT neurondb_embed_batch($1::text[], $2) AS embeddings`

	var embeddingsStr string
	err := c.db.GetContext(ctx, &embeddingsStr, query, texts, model)
	if err != nil {
		/* Fallback to individual embeddings if batch function not available */
		return c.embedBatchFallback(ctx, texts, model)
	}

	/* Parse array of vectors */
	embeddings, err := parseVectorArray(embeddingsStr)
	if err != nil {
		embeddingsStrPreview := embeddingsStr
		if len(embeddingsStrPreview) > 200 {
			embeddingsStrPreview = embeddingsStrPreview[:200] + "..."
		}
		return nil, fmt.Errorf("batch embedding parsing failed via NeuronDB: model_name='%s', text_count=%d, embeddings_string_length=%d, embeddings_string_preview='%s', function='neurondb_embed_batch', error=%w",
			model, len(texts), len(embeddingsStr), embeddingsStrPreview, err)
	}

	return embeddings, nil
}

/* embedBatchFallback generates embeddings one by one */
func (c *EmbeddingClient) embedBatchFallback(ctx context.Context, texts []string, model string) ([]Vector, error) {
	embeddings := make([]Vector, len(texts))
	for i, text := range texts {
		emb, err := c.Embed(ctx, text, model)
		if err != nil {
			return nil, fmt.Errorf("batch embedding fallback failed: model_name='%s', text_index=%d, text_count=%d, text_length=%d, function='neurondb_embed' (fallback), error=%w",
				model, i, len(texts), len(text), err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

/* parseVector parses a vector string like "[1.0, 2.0, 3.0]" into a Vector */
/* Handles formats: [1,2,3], [1.0, 2.0, 3.0], {1,2,3}, etc. */
func parseVector(s string) (Vector, error) {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return nil, fmt.Errorf("empty vector string")
	}

	/* Remove brackets or braces */
	if (s[0] == '[' && s[len(s)-1] == ']') || (s[0] == '{' && s[len(s)-1] == '}') {
		s = s[1 : len(s)-1]
	} else {
		return nil, fmt.Errorf("invalid vector format: expected brackets or braces, got: %s", s)
	}

	s = strings.TrimSpace(s)
	if len(s) == 0 {
		/* Empty vector */
		return Vector([]float32{}), nil
	}

	/* Split by comma, handling whitespace */
	var values []float32
	start := 0
	inNumber := false
	
	for i := 0; i <= len(s); i++ {
		if i == len(s) {
			/* End of string - parse last number */
			if inNumber || start < i {
				valStr := strings.TrimSpace(s[start:i])
				if len(valStr) > 0 {
					var val float32
					_, err := fmt.Sscanf(valStr, "%f", &val)
					if err != nil {
						return nil, fmt.Errorf("failed to parse float at position %d-%d: '%s', error: %w", start, i, valStr, err)
					}
					values = append(values, val)
				}
			}
			break
		}
		
		char := s[i]
		if char == ',' {
			/* Comma found - parse number before it */
			if inNumber || start < i {
				valStr := strings.TrimSpace(s[start:i])
				if len(valStr) > 0 {
					var val float32
					_, err := fmt.Sscanf(valStr, "%f", &val)
					if err != nil {
						return nil, fmt.Errorf("failed to parse float at position %d-%d: '%s', error: %w", start, i, valStr, err)
					}
					values = append(values, val)
				}
			}
			start = i + 1
			inNumber = false
		} else if !strings.ContainsRune(" \t\n\r", rune(char)) {
			/* Non-whitespace character - part of a number */
			if !inNumber {
				inNumber = true
			}
		}
	}

	if len(values) == 0 {
		return nil, fmt.Errorf("no values found in vector string: %s", s)
	}

	return Vector(values), nil
}

/* parseVectorArray parses an array of vectors from PostgreSQL array format */
/* Formats: "{[1.0,2.0],[3.0,4.0]}", "[1.0,2.0],[3.0,4.0]", "{vector(1,2),vector(3,4)}", etc. */
func parseVectorArray(s string) ([]Vector, error) {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return []Vector{}, nil
	}

	/* Remove outer braces if present (PostgreSQL array format) */
	if len(s) > 0 && s[0] == '{' && s[len(s)-1] == '}' {
		s = s[1 : len(s)-1]
		s = strings.TrimSpace(s)
	}

	if len(s) == 0 {
		return []Vector{}, nil
	}

	/* Handle PostgreSQL array format with escaped quotes: {"[1,2]","[3,4]"} */
	if s[0] == '"' {
		/* Parse quoted array format */
		var vectors []Vector
		i := 0
		for i < len(s) {
			/* Skip to next quote */
			if s[i] != '"' {
				i++
				continue
			}
			i++ // Skip opening quote
			start := i
			
			/* Find closing quote (handle escaped quotes) */
			for i < len(s) {
				if s[i] == '"' && (i == 0 || s[i-1] != '\\') {
					break
				}
				i++
			}
			
			if i >= len(s) {
				return nil, fmt.Errorf("unclosed quote in vector array at position %d", start)
			}
			
			/* Extract vector string (may have escaped quotes) */
			vectorStr := s[start:i]
			/* Unescape quotes */
			vectorStr = strings.ReplaceAll(vectorStr, "\\\"", "\"")
			vectorStr = strings.ReplaceAll(vectorStr, "\\\\", "\\")
			
			vec, err := parseVector(vectorStr)
			if err != nil {
				return nil, fmt.Errorf("failed to parse vector at position %d-%d: %w", start, i, err)
			}
			vectors = append(vectors, vec)
			
			i++ // Skip closing quote
			/* Skip comma and whitespace */
			for i < len(s) && (s[i] == ',' || s[i] == ' ' || s[i] == '\t') {
				i++
			}
		}
		return vectors, nil
	}

	/* Handle comma-separated vector format: [1,2],[3,4] */
	/* Split by "],[" or "]," to separate vectors */
	var vectors []Vector
	start := 0
	depth := 0
	
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				/* Found complete vector */
				vectorStr := strings.TrimSpace(s[start : i+1])
				if len(vectorStr) > 0 {
					vec, err := parseVector(vectorStr)
					if err != nil {
						return nil, fmt.Errorf("failed to parse vector at position %d-%d: %w", start, i+1, err)
					}
					vectors = append(vectors, vec)
				}
				/* Skip to next vector (after comma) */
				i++
				for i < len(s) && (s[i] == ',' || s[i] == ' ' || s[i] == '\t') {
					i++
				}
				start = i
				i-- // Adjust for loop increment
			}
		}
	}
	
	/* Handle last vector if not terminated by comma */
	if start < len(s) {
		vectorStr := strings.TrimSpace(s[start:])
		if len(vectorStr) > 0 {
			vec, err := parseVector(vectorStr)
			if err != nil {
				return nil, fmt.Errorf("failed to parse last vector: %w", err)
			}
			vectors = append(vectors, vec)
		}
	}

	return vectors, nil
}

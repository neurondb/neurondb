package neurondb

import (
	"context"
	"fmt"
	"strings"

	"github.com/jmoiron/sqlx"
)

// EmbeddingClient handles embedding generation via NeuronDB
type EmbeddingClient struct {
	db *sqlx.DB
}

// NewEmbeddingClient creates a new embedding client
func NewEmbeddingClient(db *sqlx.DB) *EmbeddingClient {
	return &EmbeddingClient{db: db}
}

// Embed generates an embedding for the given text using the specified model
func (c *EmbeddingClient) Embed(ctx context.Context, text string, model string) (Vector, error) {
	var embeddingStr string
	query := `SELECT neurondb_embed($1, $2)::text AS embedding`
	
	err := c.db.GetContext(ctx, &embeddingStr, query, text, model)
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed via NeuronDB: model_name='%s', text_length=%d, function='neurondb_embed', error=%w",
			model, len(text), err)
	}

	// Parse vector string format [1.0, 2.0, 3.0] to []float32
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

// EmbedBatch generates embeddings for multiple texts
func (c *EmbeddingClient) EmbedBatch(ctx context.Context, texts []string, model string) ([]Vector, error) {
	// Use array format for batch embedding if available
	query := `SELECT neurondb_embed_batch($1::text[], $2) AS embeddings`
	
	var embeddingsStr string
	err := c.db.GetContext(ctx, &embeddingsStr, query, texts, model)
	if err != nil {
		// Fallback to individual embeddings if batch function not available
		return c.embedBatchFallback(ctx, texts, model)
	}

	// Parse array of vectors
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

// embedBatchFallback generates embeddings one by one
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

// parseVector parses a vector string like "[1.0, 2.0, 3.0]" into a Vector
func parseVector(s string) (Vector, error) {
	// Remove brackets
	if len(s) < 2 || s[0] != '[' || s[len(s)-1] != ']' {
		return nil, fmt.Errorf("invalid vector format: %s", s)
	}
	s = s[1 : len(s)-1]

	// Split by comma
	var values []float32
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			if i > start {
				var val float32
				_, err := fmt.Sscanf(s[start:i], "%f", &val)
				if err != nil {
					return nil, fmt.Errorf("failed to parse float: %w", err)
				}
				values = append(values, val)
			}
			start = i + 1
		}
	}

	return Vector(values), nil
}

// parseVectorArray parses an array of vectors from PostgreSQL array format
// Format: "{[1.0,2.0],[3.0,4.0]}" or "[1.0,2.0],[3.0,4.0]"
func parseVectorArray(s string) ([]Vector, error) {
	s = strings.TrimSpace(s)
	
	// Remove outer braces if present
	if len(s) > 0 && s[0] == '{' && s[len(s)-1] == '}' {
		s = s[1 : len(s)-1]
	}
	
	if len(s) == 0 {
		return []Vector{}, nil
	}
	
	// Split by "],[" to separate vectors
	// Handle both "],[ and ], [" patterns
	parts := strings.Split(s, "],[")
	var vectors []Vector
	
	for _, part := range parts {
		// Clean up brackets
		part = strings.TrimSpace(part)
		if len(part) == 0 {
			continue
		}
		
		// Remove leading [ if present
		if len(part) > 0 && part[0] == '[' {
			part = part[1:]
		}
		// Remove trailing ] if present
		if len(part) > 0 && part[len(part)-1] == ']' {
			part = part[:len(part)-1]
		}
		
		// Add brackets back for parseVector
		vectorStr := "[" + part + "]"
		vec, err := parseVector(vectorStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse vector in array: %w", err)
		}
		vectors = append(vectors, vec)
	}
	
	return vectors, nil
}


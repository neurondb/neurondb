/*-------------------------------------------------------------------------
 *
 * client.go
 *    NeuronDB client package for NeuronAgent
 *
 * Provides a unified client interface for accessing NeuronDB functionality
 * including embeddings, LLM, ML, vector operations, RAG, analytics, and more.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/client.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"github.com/jmoiron/sqlx"
)

/* Client provides a unified interface to NeuronDB functions */
type Client struct {
	Embedding    *EmbeddingClient
	LLM          *LLMClient
	ML           *MLClient
	Vector       *VectorClient
	RAG          *RAGClient
	Analytics    *AnalyticsClient
	HybridSearch *HybridSearchClient
	Reranking    *RerankingClient
}

/* NewClient creates a new NeuronDB client */
func NewClient(db *sqlx.DB) *Client {
	return &Client{
		Embedding:    NewEmbeddingClient(db),
		LLM:          NewLLMClient(db),
		ML:           NewMLClient(db),
		Vector:       NewVectorClient(db),
		RAG:          NewRAGClient(db),
		Analytics:    NewAnalyticsClient(db),
		HybridSearch: NewHybridSearchClient(db),
		Reranking:    NewRerankingClient(db),
	}
}

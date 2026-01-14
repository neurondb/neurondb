/*-------------------------------------------------------------------------
 *
 * types.go
 *    Type definitions for NeuronDB client package
 *
 * Defines data structures, interfaces, and type aliases used throughout
 * the NeuronDB client package for embeddings, LLM, ML, and vector operations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/types.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

/* Vector represents a NeuronDB vector type */
type Vector []float32

/* EmbeddingResult contains the result of an embedding operation */
type EmbeddingResult struct {
	Embedding Vector
	Dimension int
}

/* LLMGenerateResult contains the result of an LLM generation */
type LLMGenerateResult struct {
	Output       string
	TokensUsed   int
	FinishReason string
}

/* LLMConfig contains configuration for LLM generation */
type LLMConfig struct {
	Model       string
	Temperature *float64
	MaxTokens   *int
	TopP        *float64
	Stream      bool
}

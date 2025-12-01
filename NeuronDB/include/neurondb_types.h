/*-------------------------------------------------------------------------
 *
 * neurondb_types.h
 *		Core data type definitions for NeuronDB
 *
 * Defines enterprise-grade data types including vectorp (packed SIMD),
 * vecmap (sparse high-dimensional), rtext (retrievable text with tokens),
 * and vgraph (compact graph storage).
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_types.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_TYPES_H
#define NEURONDB_TYPES_H

#include "postgres.h"

/*
 * vectorp: Packed SIMD vector with metadata
 * - Dimension fingerprint for validation
 * - Endianness guard for portability
 * - Version tag for schema evolution
 */
typedef struct VectorPacked
{
	int32		vl_len_;		/* varlena header */
	uint32		fingerprint;	/* CRC32 of dimensions for validation */
	uint16		version;		/* Schema version */
	uint16		dim;			/* Number of dimensions */
	uint8		endian_guard;	/* 0x01 for little, 0x10 for big */
	uint8		flags;			/* Reserved for future use */
	uint16		unused;			/* Alignment padding */
	float4		data[FLEXIBLE_ARRAY_MEMBER];
}			VectorPacked;

/*
 * vecmap: Sparse high-dimensional map
 * - Stores only non-zero values
 * - Mixed float and int keys
 * - Efficient for very high dimensions (>10K)
 */
typedef struct VectorMap
{
	int32		vl_len_;
	int32		total_dim;		/* Total dimensionality */
	int32		nnz;			/* Number of non-zero entries */
	/* Followed by parallel arrays: int32 indices[], float4 values[] */
}			VectorMap;

/*
 * rtext: Retrievable text with token metadata
 * - Token offsets for highlighting
 * - Section IDs for structured documents
 * - Language tag for multilingual support
 */
typedef struct RetrievableText
{
	int32		vl_len_;
	uint32		text_len;		/* Length of text in bytes */
	uint16		num_tokens;		/* Number of tokens */
	uint8		lang_tag;		/* Language code (ISO 639-1) */
	uint8		flags;			/* Encoding, case sensitivity, etc */

	/*
	 * Layout: - char text[text_len] - uint32 token_offsets[num_tokens] -
	 * uint16 section_ids[num_tokens]
	 */
}			RetrievableText;

/*
 * vgraph: Compact graph storage
 * - Node IDs as int64
 * - Typed edges with labels
 * - Adjacency list format
 */
typedef struct VectorGraph
{
	int32		vl_len_;
	int32		num_nodes;
	int32		num_edges;
	uint16		edge_types;		/* Number of edge type labels */
	uint16		unused;

	/*
	 * Layout: - int64 node_ids[num_nodes] - Edge edges[num_edges] (src, dst,
	 * type, weight) - char type_labels[][16]
	 */
}			VectorGraph;

typedef struct GraphEdge
{
	int32		src_idx;		/* Index into node_ids */
	int32		dst_idx;
	uint16		edge_type;
	uint16		unused;
	float4		weight;
}			GraphEdge;

#define VECTORP_SIZE(dim) \
	(offsetof(VectorPacked, data) + sizeof(float4) * (dim))
#define VECMAP_INDICES(vm) ((int32 *)(((char *)(vm)) + sizeof(VectorMap)))
#define VECMAP_VALUES(vm) ((float4 *)(VECMAP_INDICES(vm) + (vm)->nnz))

#define RTEXT_DATA(rt) ((char *)(((char *)(rt)) + sizeof(RetrievableText)))
#define RTEXT_OFFSETS(rt) ((uint32 *)(RTEXT_DATA(rt) + (rt)->text_len))
#define RTEXT_SECTIONS(rt) ((uint16 *)(RTEXT_OFFSETS(rt) + (rt)->num_tokens))

#define VGRAPH_NODES(vg) ((int64 *)(((char *)(vg)) + sizeof(VectorGraph)))
#define VGRAPH_EDGES(vg) ((GraphEdge *)(VGRAPH_NODES(vg) + (vg)->num_nodes))

#endif							/* NEURONDB_TYPES_H */

/*-------------------------------------------------------------------------
 *
 * types_core.c
 *		Core enterprise data types implementation
 *
 * Implements vectorp (packed SIMD), vecmap (sparse), rtext (retrievable
 * text), and vgraph (compact graph) data types with I/O functions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/types_core.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include <zlib.h>
#include <ctype.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * vectorp_in: Parse vectorp from text
 * Format: "[1.0,2.0,3.0]"
 */
PG_FUNCTION_INFO_V1(vectorp_in);
Datum
vectorp_in(PG_FUNCTION_ARGS)
{
	char	   *endptr = NULL;
	char	   *ptr = NULL;
	char	   *str = NULL;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);
	float4	   *temp_data = NULL;
	int			capacity;
	int			dim;
	int			size;
	uint32		fingerprint;
	VectorPacked *result = NULL;

	dim = 0;
	capacity = 16;

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr == '[')
		ptr++;

	nalloc(temp_data, float4, capacity);

	while (*ptr && *ptr != ']')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == ']' || *ptr == '\0')
			break;

		if (dim >= capacity)
		{
			capacity *= 2;
			temp_data = (float4 *) repalloc(
											temp_data, sizeof(float4) * capacity);
		}

		temp_data[dim] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input for vectorp")));

		ptr = endptr;
		dim++;
	}

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vectorp must have at least 1 "
						"dimension")));

	size = VECTORP_SIZE(dim);
		{
			char *tmp = NULL;
			nalloc(tmp, char, size);
			result = (VectorPacked *) tmp;
			MemSet(result, 0, size);
		}
	SET_VARSIZE(result, size);

	/* Compute fingerprint (CRC32 of dimension count) */
	fingerprint = crc32(0L, Z_NULL, 0);
	fingerprint = crc32(fingerprint, (unsigned char *) &dim, sizeof(dim));

	result->fingerprint = fingerprint;
	result->version = 1;
	result->dim = dim;
	result->endian_guard = 0x01;	/* Little endian */
	result->flags = 0;

	memcpy(result->data, temp_data, sizeof(float4) * dim);
	pfree(temp_data);

	PG_RETURN_POINTER(result);
}

/*
 * vectorp_out: Convert vectorp to text
 */
PG_FUNCTION_INFO_V1(vectorp_out);
Datum
vectorp_out(PG_FUNCTION_ARGS)
{
	int			i;
	StringInfoData buf;
	VectorPacked *vec = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_out requires 1 argument")));

	vec = (VectorPacked *) PG_GETARG_POINTER(0);

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (i = 0; i < vec->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", vec->data[i]);
	}

	appendStringInfoChar(&buf, ']');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * vecmap_in: Parse sparse vector map
 * Format: "{dim:1000,nnz:5,indices:[0,10,20],values:[1.0,2.0,3.0]}"
 */
PG_FUNCTION_INFO_V1(vecmap_in);
Datum
vecmap_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vecmap_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);
	VectorMap *result = NULL;
	int32		dim;
	int32		nnz;
	int32 *indices = NULL;
	float4 *values = NULL;
	char *ptr = NULL;
	char *endptr = NULL;
	int			i;
	int			size;

	dim = 0;
	nnz = 0;

	(void) i;					/* Suppress unused warning */

	/* Parse JSON-like format */
	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vecmap must start with '{'")));
	ptr++;

	/* Parse dim */
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (strncmp(ptr, "dim:", 4) == 0)
	{
		ptr += 4;
		dim = strtol(ptr, &endptr, 10);
		if (ptr == endptr || dim <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid dim value in vecmap")));
		ptr = endptr;
	}

	/* Parse nnz */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "nnz:", 4) == 0)
	{
		ptr += 4;
		nnz = strtol(ptr, &endptr, 10);
		if (ptr == endptr || nnz < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid nnz value in vecmap")));
		ptr = endptr;
	}

	if (dim == 0 || nnz == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vecmap must specify dim and nnz")));

	if (nnz > dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("nnz cannot exceed dim")));

	nalloc(indices, int32, nnz);
	nalloc(values, float4, nnz);

	/* Parse indices array */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "indices:", 8) == 0)
	{
		ptr += 8;
		while (isspace((unsigned char) *ptr))
			ptr++;

		if (*ptr != '[')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("indices must be an array")));
		ptr++;

		for (i = 0; i < nnz; i++)
		{
			while (isspace((unsigned char) *ptr) || *ptr == ',')
				ptr++;

			indices[i] = strtol(ptr, &endptr, 10);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid index value")));

			if (indices[i] < 0 || indices[i] >= dim)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("index %d out of range "
								"[0, %d)",
								indices[i],
								dim)));

			ptr = endptr;
		}

		while (isspace((unsigned char) *ptr))
			ptr++;
		if (*ptr != ']')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("expected ']' after indices")));
		ptr++;
	}

	/* Parse values array */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "values:", 7) == 0)
	{
		ptr += 7;
		while (isspace((unsigned char) *ptr))
			ptr++;

		if (*ptr != '[')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("values must be an array")));
		ptr++;

		for (i = 0; i < nnz; i++)
		{
			while (isspace((unsigned char) *ptr) || *ptr == ',')
				ptr++;

			values[i] = strtof(ptr, &endptr);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid value")));

			ptr = endptr;
		}

		while (isspace((unsigned char) *ptr))
			ptr++;
		if (*ptr != ']')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("expected ']' after values")));
		ptr++;
	}

	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
		{
			char *tmp = NULL;
			nalloc(tmp, char, size);
			result = (VectorMap *) tmp;
			MemSet(result, 0, size);
		}
	SET_VARSIZE(result, size);

	result->total_dim = dim;
	result->nnz = nnz;

	/* Validate macro results before memcpy */
	{
		int32 *result_indices = VECMAP_INDICES(result);
		float4 *result_values = VECMAP_VALUES(result);
		
		if (result_indices == NULL || result_values == NULL)
		{
			pfree(result);
			pfree(indices);
			pfree(values);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neurondb: VECMAP macro returned NULL pointer")));
		}
		
		/* Check for overflow in memcpy size calculation */
		if (nnz > 0 && (size_t) nnz > SIZE_MAX / sizeof(int32))
		{
			pfree(result);
			pfree(indices);
			pfree(values);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb: nnz %d too large for memcpy", nnz)));
		}
		if (nnz > 0 && (size_t) nnz > SIZE_MAX / sizeof(float4))
		{
			pfree(result);
			pfree(indices);
			pfree(values);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb: nnz %d too large for memcpy", nnz)));
		}
		
		memcpy(result_indices, indices, sizeof(int32) * nnz);
		memcpy(result_values, values, sizeof(float4) * nnz);
	}

	pfree(indices);
	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * vecmap_out: Convert sparse vector map to text
 */
PG_FUNCTION_INFO_V1(vecmap_out);
Datum
vecmap_out(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vecmap_out requires 1 argument")));

	vec = (VectorMap *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32	   *indices = NULL;
	float4	   *values = NULL;
	int			i;

	indices = VECMAP_INDICES(vec);
	values = VECMAP_VALUES(vec);

	initStringInfo(&buf);

	appendStringInfo(
					 &buf, "{dim:%d,nnz:%d,indices:[", vec->total_dim, vec->nnz);

	for (i = 0; i < vec->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%d", indices[i]);
	}

	appendStringInfoString(&buf, "],values:[");

	for (i = 0; i < vec->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", values[i]);
	}

	appendStringInfoString(&buf, "]}");

	PG_RETURN_CSTRING(buf.data);
}

/*
 * rtext_in: Parse retrievable text
 */
PG_FUNCTION_INFO_V1(rtext_in);
Datum
rtext_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: rtext_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);
	RetrievableText *result = NULL;
	int			text_len;
	int			size;

	text_len = strlen(str);

	/* Basic implementation: store text, tokenize later */
	size = sizeof(RetrievableText) + text_len + 1;
		{
			char *tmp = NULL;
			nalloc(tmp, char, size);
			result = (RetrievableText *) tmp;
			MemSet(result, 0, size);
		}
	SET_VARSIZE(result, size);

	result->text_len = text_len;
	result->num_tokens = 0;		/* Will be computed on first access */
	result->lang_tag = 0;		/* Auto-detect */
	result->flags = 0;

	memcpy(RTEXT_DATA(result), str, text_len);

	PG_RETURN_POINTER(result);
}

/*
 * rtext_out: Convert retrievable text to string
 */
PG_FUNCTION_INFO_V1(rtext_out);
Datum
rtext_out(PG_FUNCTION_ARGS)
{
	RetrievableText *rt = NULL;
	char *result = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: rtext_out requires 1 argument")));

	rt = (RetrievableText *) PG_GETARG_POINTER(0);

	nalloc(result, char, rt->text_len + 1);
	memcpy(result, RTEXT_DATA(rt), rt->text_len);
	result[rt->text_len] = '\0';

	PG_RETURN_CSTRING(result);
}

/*
 * vgraph_in: Parse graph structure
 * Format: "{nodes:5,edges:[[0,1],[1,2],[2,3]]}"
 */
PG_FUNCTION_INFO_V1(vgraph_in);
Datum
vgraph_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vgraph_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);
	VectorGraph *result = NULL;
	int32		num_nodes;
	int32		num_edges;
	GraphEdge *edges = NULL;
	char *ptr = NULL;
	char *endptr = NULL;
	int			size;
	int			edge_capacity;

	num_nodes = 0;
	num_edges = 0;
	edge_capacity = 32;

	/* Parse JSON-like format */
	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vgraph must start with '{'")));
	ptr++;

	/* Parse nodes */
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (strncmp(ptr, "nodes:", 6) == 0)
	{
		ptr += 6;
		num_nodes = strtol(ptr, &endptr, 10);
		if (ptr == endptr || num_nodes <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid nodes value in "
							"vgraph")));
		ptr = endptr;
	}

	if (num_nodes == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vgraph must specify nodes")));

	nalloc(edges, GraphEdge, edge_capacity);

	/* Parse edges array */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "edges:", 6) == 0)
	{
		ptr += 6;
		while (isspace((unsigned char) *ptr))
			ptr++;

		if (*ptr != '[')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("edges must be an array")));
		ptr++;

		/* Parse edge pairs */
		while (*ptr && *ptr != ']')
		{
			int32		from_node,
						to_node;

			while (isspace((unsigned char) *ptr) || *ptr == ',')
				ptr++;

			if (*ptr == ']')
				break;

			if (*ptr != '[')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("each edge must be an "
								"array [from,to]")));
			ptr++;

			/* Parse from node */
			while (isspace((unsigned char) *ptr))
				ptr++;

			from_node = strtol(ptr, &endptr, 10);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid from node")));

			if (from_node < 0 || from_node >= num_nodes)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("from node %d out of "
								"range [0, %d)",
								from_node,
								num_nodes)));

			ptr = endptr;

			/* Parse comma */
			while (isspace((unsigned char) *ptr))
				ptr++;
			if (*ptr != ',')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected ',' between "
								"edge nodes")));
			ptr++;

			/* Parse to node */
			while (isspace((unsigned char) *ptr))
				ptr++;

			to_node = strtol(ptr, &endptr, 10);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid to node")));

			if (to_node < 0 || to_node >= num_nodes)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("to node %d out of "
								"range [0, %d)",
								to_node,
								num_nodes)));

			ptr = endptr;

			/* Close edge array */
			while (isspace((unsigned char) *ptr))
				ptr++;
			if (*ptr != ']')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected ']' after "
								"edge pair")));
			ptr++;

			/* Store edge */
			if (num_edges >= edge_capacity)
			{
				edge_capacity *= 2;
				edges = (GraphEdge *) repalloc(edges,
											   sizeof(GraphEdge) * edge_capacity);
			}

			edges[num_edges].src_idx = from_node;
			edges[num_edges].dst_idx = to_node;
			edges[num_edges].edge_type = 0; /* Default edge type */
			edges[num_edges].weight = 1.0;	/* Default weight */
			num_edges++;
		}

		if (*ptr == ']')
			ptr++;
	}

	/* Build result - simplified: no node IDs, just edges */
	size = sizeof(VectorGraph) + sizeof(GraphEdge) * num_edges;
		{
			char *tmp = NULL;
			nalloc(tmp, char, size);
			result = (VectorGraph *) tmp;
			MemSet(result, 0, size);
		}
	SET_VARSIZE(result, size);

	result->num_nodes = num_nodes;
	result->num_edges = num_edges;
	result->edge_types = 0;		/* No labeled edge types in simple format */

	memcpy(VGRAPH_EDGES(result), edges, sizeof(GraphEdge) * num_edges);

	pfree(edges);

	PG_RETURN_POINTER(result);
}

/*
 * vgraph_out: Convert graph to text
 */
PG_FUNCTION_INFO_V1(vgraph_out);
Datum
vgraph_out(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = NULL;
	GraphEdge  *edges = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vgraph_out requires 1 argument")));

	graph = (VectorGraph *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int			i;

	edges = VGRAPH_EDGES(graph);

	initStringInfo(&buf);

	appendStringInfo(&buf, "{nodes:%d,edges:[", graph->num_nodes);

	for (i = 0; i < graph->num_edges; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(
						 &buf, "[%d,%d]", edges[i].src_idx, edges[i].dst_idx);
	}

	appendStringInfoString(&buf, "]}");

	PG_RETURN_CSTRING(buf.data);
}

/*
 * vectorp_dims: Get dimensions of packed vector
 */
PG_FUNCTION_INFO_V1(vectorp_dims);
Datum
vectorp_dims(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_dims requires 1 argument")));

	vec = (VectorPacked *) PG_GETARG_POINTER(0);

	PG_RETURN_INT32(vec->dim);
}

/*
 * vectorp_validate: Validate fingerprint and endianness
 */
PG_FUNCTION_INFO_V1(vectorp_validate);
Datum
vectorp_validate(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_validate requires 1 argument")));

	vec = (VectorPacked *) PG_GETARG_POINTER(0);
	uint32		expected_fp;
	uint32		dim;

	dim = vec->dim;

	/* Recompute fingerprint */
	expected_fp = crc32(0L, Z_NULL, 0);
	expected_fp = crc32(expected_fp, (unsigned char *) &dim, sizeof(dim));

	if (vec->fingerprint != expected_fp)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vectorp fingerprint mismatch: "
						"corrupted data")));

	if (vec->endian_guard != 0x01 && vec->endian_guard != 0x10)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vectorp endianness guard invalid")));

	PG_RETURN_BOOL(true);
}

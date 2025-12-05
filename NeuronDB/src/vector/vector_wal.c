/*
 * vector_wal.c
 *     Vector WAL Compression for NeuronDB
 *
 * Provides delta encoding and compression for vector updates in WAL
 * to reduce replication bandwidth and storage overhead.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 */

#include "postgres.h"
#include "neurondb_compat.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "lib/stringinfo.h"
#include <float.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Note: PG_MODULE_MAGIC is defined in src/core/neurondb.c (only once per extension) */

/* Helper for safe text->cstring conversion: always NOT NULL, panics otherwise */
static char *
safe_text_to_cstring(const text * txt)
{
	if (!txt)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("NULL vector or base_vector argument")));
	return text_to_cstring(txt);
}

/* Helper to split a vector string into a float array, crash-proof, Postgres style */
static float8 *
parse_vector_str(const char *vec, int *out_dim)
{
	char	   *copy;
	int			capacity = 32;
	int			count = 0;

	NDB_DECLARE(float8 *, result);
	char	   *token;
	char	   *ptr;
	char	   *saveptr = NULL;

	copy = pstrdup(vec);
	NDB_ALLOC(result, float8, capacity);

	*out_dim = 0;

	/* Strip whitespace and brackets */
	while (isspace((unsigned char) *copy))
		copy++;
	if (copy[0] == '[')
		copy++;
	ptr = copy + strlen(copy) - 1;
	while (ptr > copy && isspace((unsigned char) *ptr))
		ptr--;
	if (ptr >= copy && *ptr == ']')
		*ptr = '\0';

	/* Tokenize by comma */
	token = strtok_r(copy, ",", &saveptr);
	while (token)
	{
		double		val;
		char	   *endptr = NULL;

		/* Allow whitespace within token */
		while (isspace((unsigned char) *token))
			token++;
		val = strtod(token, &endptr);
		if (endptr == token || isnan(val) || isinf(val))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("vector contains invalid float8 "
							"value: \"%s\"",
							token)));
		}

		if (count >= capacity)
		{
			capacity *= 2;

			/*
			 * For reallocation, we need to manually handle since NDB_ALLOC
			 * doesn't support repalloc
			 */
			/* This is acceptable for dynamic growth patterns */
			result = (float8 *) repalloc(
										 result, capacity * sizeof(float8));
		}
		result[count++] = (float8) val;
		token = strtok_r(NULL, ",", &saveptr);
	}

	NDB_FREE(copy);
	*out_dim = count;

	if (count == 0)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("vector representation is empty")));

	return result;
}

/*
 * Delta encoding: output run-length and delta-encoded compact form in simple text.
 * For crashproofness, every error must elog/ereport with a detailed msg.
 *
 * Format example: dimension:int;[rle|delta,varint|...]
 * This is not a production binary encoding, just safe, round-trip, and traceable.
 */
PG_FUNCTION_INFO_V1(vector_wal_compress);
Datum
vector_wal_compress(PG_FUNCTION_ARGS)
{
	text	   *vector = PG_GETARG_TEXT_PP(0);
	text	   *base_vector = PG_GETARG_TEXT_PP(1);
	char	   *vec_str = NULL,
			   *base_str = NULL;
	float8	   *cur_values = NULL,
			   *base_values = NULL;
	int			dim = 0,
				base_dim = 0;
	int			i,
				run_len;
	float8		delta = 0;
	StringInfoData compressed;

	vec_str = safe_text_to_cstring(vector);
	base_str = safe_text_to_cstring(base_vector);

	cur_values = parse_vector_str(vec_str, &dim);
	base_values = parse_vector_str(base_str, &base_dim);

	if (dim != base_dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector and base_vector dimensions do "
						"not match (%d vs %d)",
						dim,
						base_dim)));

	initStringInfo(&compressed);

	elog(DEBUG1,
		 "neurondb: Compressing vector using delta encoding, dim=%d",
		 dim);

	/*
	 * Algorithm: For each element: delta = cur - base. Run-length encode
	 * sequences of repeated deltas. For demonstration, deltas stored as text
	 * and rle length. Example: 3:0.5;2:-0.1;1:0.0
	 */
	appendStringInfo(&compressed, "D%d:", dim);
	i = 0;
	while (i < dim)
	{
		delta = cur_values[i] - base_values[i];
		run_len = 1;

		/* Find repeated consecutive identical deltas */
		while ((i + run_len < dim)
			   && fabs((cur_values[i + run_len]
						- base_values[i + run_len])
					   - delta)
			   < DBL_EPSILON)
		{
			run_len++;
		}

		/* Write run-length, delta pair */
		appendStringInfo(&compressed,
						 "%d:%.17g%s",
						 run_len,
						 delta,
						 (i + run_len < dim) ? ";" : "");
		i += run_len;
	}

	PG_RETURN_TEXT_P(cstring_to_text(compressed.data));
}

/*
 * vector_wal_decompress: Decompress a delta-encoded vector, robust and crash-proof.
 */
PG_FUNCTION_INFO_V1(vector_wal_decompress);
Datum
vector_wal_decompress(PG_FUNCTION_ARGS)
{
	text	   *compressed = PG_GETARG_TEXT_PP(0);
	text	   *base_vector = PG_GETARG_TEXT_PP(1);
	char	   *comp_str = NULL,
			   *base_str = NULL;
	int			dim = 0,
				base_dim = 0;
	float8	   *base_values = NULL,
			   *output = NULL;
	char	   *p,
			   *endptr;
	int			out_idx = 0,
				run,
				found_dim = 0;
	float8		delta = 0;
	StringInfoData decompressed;

	comp_str = safe_text_to_cstring(compressed);
	base_str = safe_text_to_cstring(base_vector);

	elog(DEBUG1,
		 "neurondb: Decompressing vector from delta encoding: \"%s\"",
		 comp_str);

	/*
	 * Parse header: D<dim>:
	 */
	p = comp_str;
	if (*p != 'D')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Compressed vector format error: "
						"missing dimension header")));

	p++;
	found_dim = strtol(p, &endptr, 10);
	if (found_dim <= 0 || *endptr != ':')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Compressed vector format error: "
						"malformed dimension: \"%s\"",
						p)));
	dim = found_dim;
	p = endptr + 1;

	base_values = parse_vector_str(base_str, &base_dim);
	if (base_dim != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("base_vector dim %d does not match "
						"compressed dim %d",
						base_dim,
						dim)));

	output = NULL;
	NDB_ALLOC(output, float8, dim);

	/*
	 * Parse run-length encoding: <run>:<delta>;... (semi-colon separated)
	 */
	out_idx = 0;
	while (*p && out_idx < dim)
	{
		/* read run-length */
		run = strtol(p, &endptr, 10);
		if (run <= 0 || *endptr != ':')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Compressed vector format error "
							"at position %d: expected "
							"<run>:<delta>",
							(int) (p - comp_str))));
		p = endptr + 1;
		/* read delta */
		delta = strtod(p, &endptr);
		if ((endptr == p) || isnan(delta) || isinf(delta))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("Compressed encoding contains "
							"invalid float8 delta: \"%s\"",
							p)));
		p = endptr;
		/* apply run times */
		for (int j = 0; j < run; ++j)
		{
			if (out_idx >= dim)
				ereport(ERROR,
						(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
						 errmsg("Decoded vector length "
								"exceeds expected %d",
								dim)));
			output[out_idx] = base_values[out_idx] + delta;
			out_idx++;
		}
		if (*p == ';')
			p++;
		else if (*p && *p != ';')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("Compressed vector format error "
							"after delta at position %d: "
							"got '%c'",
							(int) (p - comp_str),
							*p)));
	}
	if (out_idx != dim)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("Decompression failed: output vector "
						"length %d does not match dimension %d",
						out_idx,
						dim)));

	/* render back to string [x,y,z,...], max precision */
	initStringInfo(&decompressed);
	appendStringInfoChar(&decompressed, '[');
	for (int i = 0; i < dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&decompressed, ',');
		appendStringInfo(&decompressed, "%.17g", output[i]);
	}
	appendStringInfoChar(&decompressed, ']');

	PG_RETURN_TEXT_P(cstring_to_text(decompressed.data));
}

/*
 * vector_wal_estimate_size: Estimate compressed size robustly.
 * Handles NULL and crazy input thoughtfully. Never crashes.
 */
PG_FUNCTION_INFO_V1(vector_wal_estimate_size);
Datum
vector_wal_estimate_size(PG_FUNCTION_ARGS)
{
	text	   *vector = PG_GETARG_TEXT_PP(0);
	text	   *base_vector = PG_GETARG_TEXT_PP(1);
	char	   *vec_str;
	int32		original_size;
	int32		estimated_compressed_size;
	float4		compression_ratio;

	/* NULL-safe: treat NULL as empty vector */
	if (!vector)
		PG_RETURN_INT32(0);

	/* base_vector is accepted for API consistency but not currently used */
	(void) base_vector;

	vec_str = text_to_cstring(
							  vector);	/* Postgres returns "" for zero-length
										 * text */
	original_size = strlen(vec_str);

	/* Compression ratio: guard against divide by zero, crazy size */
	if (original_size == 0)
		compression_ratio = 1.0f;
	else
		compression_ratio = 2.5f;

	if (compression_ratio < 1.0f)
		compression_ratio = 1.0f;

	estimated_compressed_size =
		(int32) ((float4) original_size / compression_ratio);

	if (estimated_compressed_size < 0)
		estimated_compressed_size = 0;

	elog(DEBUG1,
		 "neurondb: Estimated compression: %d -> %d bytes (%.1fx)",
		 original_size,
		 estimated_compressed_size,
		 compression_ratio);

	PG_RETURN_INT32(estimated_compressed_size);
}

/*
 * vector_wal_set_compression: Simulate enable/disable WAL compression feature.
 * Always crash-proof, always returns true (for now).
 */
PG_FUNCTION_INFO_V1(vector_wal_set_compression);
Datum
vector_wal_set_compression(PG_FUNCTION_ARGS)
{
	bool		enable;

	/* Strict in PostgreSQL ensures not called with NULL, but double check */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("enable/disable argument must not be "
						"NULL")));
	enable = PG_GETARG_BOOL(0);

	if (enable)
	{
	}
	else
	{
	}

	/*
	 * In real system: would set neurondb.wal_compression GUC, and
	 * (un)register hooks here.
	 */

	PG_RETURN_BOOL(true);
}

/*
 * vector_wal_get_stats: Return current WAL compression statistics.
 * For now, returns simulated numbers, never crashes.
 */
PG_FUNCTION_INFO_V1(vector_wal_get_stats);
Datum
vector_wal_get_stats(PG_FUNCTION_ARGS)
{
	StringInfoData stats;
	int64		total_bytes_original = 1024000;
	int64		total_bytes_compressed = 409600;
	float4		compression_ratio;

	/* In production, would be extracted from shared memory/GUC, for now fake. */
	if (total_bytes_compressed == 0)
		compression_ratio = 0.0;
	else
		compression_ratio = (float4) total_bytes_original
			/ (float4) total_bytes_compressed;

	initStringInfo(&stats);
	appendStringInfo(&stats,
					 "{\"original_bytes\":" NDB_INT64_FMT
					 ",\"compressed_bytes\":" NDB_INT64_FMT
					 ",\"compression_ratio\":%.2f}",
					 NDB_INT64_CAST(total_bytes_original),
					 NDB_INT64_CAST(total_bytes_compressed),
					 compression_ratio);

	elog(DEBUG1,
		 "neurondb: WAL compression stats: %.2fx ratio "
		 "(original=" NDB_INT64_FMT ", compressed=" NDB_INT64_FMT ")",
		 compression_ratio,
		 NDB_INT64_CAST(total_bytes_original),
		 NDB_INT64_CAST(total_bytes_compressed));

	PG_RETURN_TEXT_P(cstring_to_text(stats.data));
}

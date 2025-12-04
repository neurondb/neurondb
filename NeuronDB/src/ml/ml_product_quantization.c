/*-------------------------------------------------------------------------
 *
 * ml_product_quantization.c
 *    Product quantization for vector compression.
 *
 * This module implements product quantization for efficient vector compression
 * and approximate nearest neighbor search.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_product_quantization.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "utils/jsonb.h"
#include "executor/spi.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_simd.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "ml_gpu_registry.h"
#include "ml_catalog.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/*
 * Product Quantization Codebook Data Structure
 * --------------------------------------------
 * This structure describes the learned centroids for each subquantizer (subspace).
 *  - m:      the number of subspaces (subquantizers)
 *  - ksub:   number of centroids for each subspace (each code is in [0, ksub-1])
 *  - dsub:   subspace dimensionality, i.e., dsub = total_dim / m (must divide evenly)
 *  - centroids: a 3-level array: centroids[m][ksub][dsub]
 */
typedef struct PQCodebook
{
	int			m;				/* Number of subspaces/subquantizers */
	int			ksub;			/* Number of centroids for each subspace */
	int			dsub;			/* Subspace dimension (must be d/m) */
	float	 ***centroids;		/* Shape: [m][ksub][dsub] */
}			PQCodebook;

/*
 * train_subspace_kmeans
 * ---------------------
 * Internal function for fitting a k-means codebook on a batch of subspace vectors.
 *
 * Args:
 *   subspace_data: 2D array of nvec vectors, each of dimension dsub, for this subspace.
 *   nvec:          Number of training vectors.
 *   dsub:          Dimension of each vector in this subspace.
 *   k:             Number of clusters/centroids for k-means (ksub).
 *   centroids:     Output: centroids[k][dsub].
 *   max_iters:     Maximum number of Lloyd iterations.
 *
 * Returns:
 *   void (centroids array is modified in-place)
 */
static void
train_subspace_kmeans(float **subspace_data,
					  int nvec,
					  int dsub,
					  int k,
					  float **centroids,
					  int max_iters)
{
	int		   *assignments =
		NULL;					/* Assignment for each vector (index of
								 * centroid) */
	NDB_DECLARE(int *, counts);	/* Number of points assigned to each centroid */
	bool		changed = true;
	int			iter,
				i,
				c,
				d;

	/*
	 * Allocate assignment array, initialize to zeros (or later random if
	 * desired)
	 */
	assignments = (int *) palloc0(sizeof(int) * nvec);

	/* Random initialization of centroids from dataset */
	for (c = 0; c < k; c++)
	{
		int			idx = rand() % nvec;

		memcpy(centroids[c], subspace_data[idx], sizeof(float) * dsub);
	}

	/* Lloyd's k-means main loop */
	for (iter = 0; iter < max_iters; iter++)
	{
		changed = false;

		/* Assignment step: assign each vector to the nearest centroid */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best = -1;

			for (c = 0; c < k; c++)
			{
				double		dist = 0.0;

				/* Compute squared L2 distance */
				for (d = 0; d < dsub; d++)
				{
					double		diff =
						(double) subspace_data[i][d]
						- (double) centroids[c][d];

					dist += diff * diff;
				}
				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			/* Only record if assignment has changed */
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		if (!changed)
			break;				/* Converged */

		/* Update step: recompute centroids as mean of assigned points */
		counts = (int *) palloc0(sizeof(int) * k);
		for (c = 0; c < k; c++)
			memset(centroids[c], 0, sizeof(float) * dsub);

		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dsub; d++)
				centroids[c][d] += subspace_data[i][d];
			counts[c]++;
		}

		for (c = 0; c < k; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dsub; d++)
					centroids[c][d] /= counts[c];
			}
			/* If a centroid got no points, it remains from old pass */
		}
		NDB_FREE(counts);
	}

	NDB_FREE(assignments);
}

/*
 * train_pq_codebook
 * -----------------
 * Trains a PQ codebook from a table/column containing real-valued vector data.
 *
 * Args (SQL):
 *    table_name  - Name of source table containing the training vectors (text)
 *    column_name - Name of the vector column (text)
 *    m           - Number of subspaces/subquantizers (typically a divisor of dim)
 *    ksub        - Number of clusters per subspace (usually 256 for 8-bit PQ)
 *
 * Returns: PQ codebook as a bytea object (serialized in host order).
 *
 * Validation:
 *    - Number of subspaces (m) must be >=1, <=128, and divide dimensionality.
 *    - Number of centroids per subspace (ksub) must be >=2 and <=65536.
 *
 * Serialization format (bytea):
 *    - int m
 *    - int ksub
 *    - int dsub
 *    - float centroids[m][ksub][dsub] (in order)
 */
PG_FUNCTION_INFO_V1(train_pq_codebook);

Datum
train_pq_codebook(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			m,
				ksub;
	char	   *tbl_str,
			   *col_str;
	float	  **data = NULL;	/* Training data array [nvec][dim] */
	int			nvec = 0,
				dim = 0;
	PQCodebook	codebook;
	int			sub,
				i;
	bytea	   *result;
	int			result_size;
	char	   *result_ptr;

	/* Argument parsing and validation */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	m = PG_GETARG_INT32(2);
	ksub = PG_GETARG_INT32(3);

	if (m < 1 || m > 128)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("m (number of subspaces) must be "
						"between 1 and 128")));
	if (ksub < 2 || ksub > 65536)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ksub (centroids per subspace) must be "
						"between 2 and 65536")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Starting PQ training for table = %s.%s, m=%d, ksub=%d",
		 tbl_str,
		 col_str,
		 m,
		 ksub);

	/*
	 * Pull training vectors from the table: neurondb_fetch_vectors_from_table
	 * is expected to allocate a 2D array [nvec][dim] and fill nvec, dim
	 */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (data == NULL || nvec <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No training vectors found in table '%s' column '%s'",
						tbl_str,
						col_str)));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (int i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (dim % m != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Vector dimension %d must be divisible by number of subspaces m=%d",
						dim,
						m)));

	/* Prepare codebook structure */
	codebook.m = m;
	codebook.ksub = ksub;
	codebook.dsub = dim / m;

	/*
	 * Allocate centroids 3D array: [m][ksub][dsub], each float* properly
	 * allocated
	 */
	codebook.centroids = (float ***) palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		codebook.centroids[sub] =
			(float **) palloc(sizeof(float *) * ksub);
		for (i = 0; i < ksub; i++)
			codebook.centroids[sub][i] =
				(float *) palloc(sizeof(float) * codebook.dsub);
	}

	/*
	 * Begin PQ codebook training: fit a k-means on each subspace
	 * independently
	 */
	for (sub = 0; sub < m; sub++)
	{
		float	  **subspace_data = NULL;
		int			start_dim = sub * codebook.dsub;

		elog(DEBUG1,
			 "neurondb: Training PQ subspace %d of %d (dims %d-%d)",
			 sub + 1,
			 m,
			 start_dim,
			 start_dim + codebook.dsub - 1);

		/*
		 * Extract subspace vectors from data (copy block of contiguous
		 * floats)
		 */
		subspace_data = (float **) palloc(sizeof(float *) * nvec);
		for (i = 0; i < nvec; i++)
		{
			subspace_data[i] =
				(float *) palloc(sizeof(float) * codebook.dsub);
			memcpy(subspace_data[i],
				   &data[i][start_dim],
				   sizeof(float) * codebook.dsub);
		}

		/* Train k-means for this subspace. 100 iterations is typical. */
		train_subspace_kmeans(subspace_data,
							  nvec,
							  codebook.dsub,
							  ksub,
							  codebook.centroids[sub],
							  100);

		/* Release subspace data for this subspace */
		for (i = 0; i < nvec; i++)
			NDB_FREE(subspace_data[i]);
		NDB_FREE(subspace_data);
	}

	/* Serialization step: compute total serialized length for bytea */
	result_size = sizeof(int) * 3 + /* header: m, ksub, dsub */
		m * ksub * codebook.dsub * sizeof(float);	/* centroids */

	result = (bytea *) palloc(VARHDRSZ + result_size);
	SET_VARSIZE(result, VARHDRSZ + result_size);
	result_ptr = VARDATA(result);

	/* Write codebook header fields (all host order: m, ksub, dsub) */
	memcpy(result_ptr, &m, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &ksub, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &codebook.dsub, sizeof(int));
	result_ptr += sizeof(int);

	/*
	 * Write centroids sequentially: for each subspace, for each centroid,
	 * write full dsub floats
	 */
	for (sub = 0; sub < m; sub++)
		for (i = 0; i < ksub; i++)
		{
			memcpy(result_ptr,
				   codebook.centroids[sub][i],
				   sizeof(float) * codebook.dsub);
			result_ptr += sizeof(float) * codebook.dsub;
		}

	/* Clean up all allocated memory for centroids and input data */
	for (sub = 0; sub < m; sub++)
	{
		for (i = 0; i < ksub; i++)
			NDB_FREE(codebook.centroids[sub][i]);
		NDB_FREE(codebook.centroids[sub]);
	}
	NDB_FREE(codebook.centroids);

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);

	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_BYTEA_P(result);
}

/*
 * pq_encode_vector
 * ----------------
 * Encode a query vector using a learned PQ codebook.
 *
 * Args (SQL):
 *   vector   - Input vector (float4[])
 *   codebook - PQ codebook (bytea serialization format, see train_pq_codebook).
 *
 * Returns:
 *   Smallint array (int2[]) of length m, one code [0, ksub-1] per subspace.
 *   Each code identifies the closest centroid in the corresponding codebook subspace.
 */
PG_FUNCTION_INFO_V1(pq_encode_vector);
PG_FUNCTION_INFO_V1(predict_pq_codebook);
PG_FUNCTION_INFO_V1(evaluate_pq_codebook_by_model_id);

Datum
pq_encode_vector(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array;
	bytea	   *codebook_bytea;
	float4	   *vec_data;
	int			dim,
				m,
				ksub,
				dsub;
	char	   *cb_ptr;
	float	 ***centroids = NULL;
	NDB_DECLARE(int16 *, codes);	/* array of length m */
	ArrayType  *result;
	Datum	   *result_datums;
	int			sub,
				c,
				d;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* ----- Parse Inputs ----- */
	vec_array = PG_GETARG_ARRAYTYPE_P(0);
	codebook_bytea = PG_GETARG_BYTEA_PP(1);

	dim = ARR_DIMS(
				   vec_array)[0];	/* Expect one-dimensional packed float4[] */
	vec_data = (float4 *) ARR_DATA_PTR(vec_array);

	/* ----- Deserialize Codebook ----- */
	cb_ptr = VARDATA(codebook_bytea);
	memcpy(&m, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&ksub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&dsub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);

	if (dim != m * dsub)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Vector dimension (%d) does not match codebook definition (m=%d * dsub=%d).",
						dim,
						m,
						dsub)));

	/* ----- Reconstruct centroids 3D array from codebook bytea ----- */
	centroids = (float ***) palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		centroids[sub] = (float **) palloc(sizeof(float *) * ksub);
		for (c = 0; c < ksub; c++)
		{
			centroids[sub][c] =
				(float *) palloc(sizeof(float) * dsub);
			memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
			cb_ptr += sizeof(float) * dsub;
		}
	}

	/* ----- Compute PQ codes for each subspace ----- */
	codes = (int16 *) palloc(sizeof(int16) * m);
	for (sub = 0; sub < m; sub++)
	{
		int			start_dim = sub * dsub;
		double		min_dist = DBL_MAX;
		int			best = -1;

		for (c = 0; c < ksub; c++)
		{
			double		dist = 0.0;

			for (d = 0; d < dsub; d++)
			{
				double		diff = (double) vec_data[start_dim + d]
					- (double) centroids[sub][c][d];

				dist += diff * diff;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		codes[sub] = (int16) best;
	}

	/* ----- Build int2[] result array ----- */
	result_datums = (Datum *) palloc(sizeof(Datum) * m);
	for (sub = 0; sub < m; sub++)
		result_datums[sub] = Int16GetDatum(codes[sub]);

	get_typlenbyvalalign(INT2OID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, m, INT2OID, typlen, typbyval, typalign);

	/* ----- Clean up c-allocated memory ----- */
	for (sub = 0; sub < m; sub++)
	{
		for (c = 0; c < ksub; c++)
			NDB_FREE(centroids[sub][c]);
		NDB_FREE(centroids[sub]);
	}
	NDB_FREE(centroids);
	NDB_FREE(codes);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * predict_pq_codebook
 *      Encodes a vector using a trained PQ codebook.
 *      Arguments: int4 model_id, float8[] vector
 *      Returns: int2[] quantization codes
 */
Datum
predict_pq_codebook(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *vector_array;
	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, parameters);
	float4	   *vec_data;
	int			dim;
	int			m,
				ksub,
				dsub;
	char	   *cb_ptr;
	float	 ***centroids = NULL;
	NDB_DECLARE(int16 *, codes);
	ArrayType  *result;
	Datum	   *result_datums;
	int			sub,
				c,
				d;
	int			start_dim;
	double		min_dist;
	int			best;

	model_id = PG_GETARG_INT32(0);
	vector_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Validate input vector */
	if (ARR_NDIM(vector_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_pq_codebook: vector must be 1-dimensional")));

	dim = ARR_DIMS(vector_array)[0];
	vec_data = (float4 *) ARR_DATA_PTR(vector_array);

	/* Load model data from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, &parameters, NULL))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("predict_pq_codebook: model %d not found", model_id)));

	/* Extract codebook from model_data */

	/*
	 * Codebook format: int m, int ksub, int dsub, float
	 * centroids[m][ksub][dsub]
	 */
	if (model_data == NULL || VARSIZE(model_data) < VARHDRSZ + sizeof(int) * 3)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("predict_pq_codebook: model %d has invalid codebook data",
						model_id)));

	cb_ptr = VARDATA(model_data);
	memcpy(&m, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&ksub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&dsub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);

	/* Validate dimensions */
	if (dim != m * dsub)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_pq_codebook: vector dimension (%d) does not match "
						"codebook (m=%d * dsub=%d = %d)",
						dim,
						m,
						dsub,
						m * dsub)));

	/* Reconstruct centroids 3D array from codebook */
	centroids = (float ***) palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		centroids[sub] = (float **) palloc(sizeof(float *) * ksub);
		for (c = 0; c < ksub; c++)
		{
			centroids[sub][c] = (float *) palloc(sizeof(float) * dsub);
			memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
			cb_ptr += sizeof(float) * dsub;
		}
	}

	/* Compute PQ codes for each subspace */
	codes = (int16 *) palloc(sizeof(int16) * m);
	for (sub = 0; sub < m; sub++)
	{
		start_dim = sub * dsub;
		min_dist = DBL_MAX;
		best = -1;

		/* Find nearest centroid in this subspace */
		for (c = 0; c < ksub; c++)
		{
			double		dist = 0.0;

			for (d = 0; d < dsub; d++)
			{
				double		diff = (double) vec_data[start_dim + d]
					- (double) centroids[sub][c][d];

				dist += diff * diff;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		codes[sub] = (int16) best;
	}

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * m);
	for (sub = 0; sub < m; sub++)
		result_datums[sub] = Int16GetDatum(codes[sub]);

	result = construct_array(result_datums, m, INT2OID, 2, true, 's');

	for (sub = 0; sub < m; sub++)
	{
		for (c = 0; c < ksub; c++)
			NDB_FREE(centroids[sub][c]);
		NDB_FREE(centroids[sub]);
	}
	NDB_FREE(centroids);
	NDB_FREE(codes);
	NDB_FREE(result_datums);
	if (model_data)
		NDB_FREE(model_data);
	if (parameters)
		NDB_FREE(parameters);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * evaluate_pq_codebook_by_model_id
 *      Evaluates PQ codebook quality on a dataset.
 *      Arguments: int4 model_id, text table_name, text vector_col
 *      Returns: jsonb with quantization metrics
 */
Datum
evaluate_pq_codebook_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *vector_col;
	char	   *tbl_str;
	char	   *vec_str;
	StringInfoData query;
	int			ret;
	int			n_points = 0;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext;
	double		avg_quantization_error;
	int			subquantizers;
	int			codebook_size;
	int			bits_per_code;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext_spi;

	/* Validate arguments */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_pq_codebook_by_model_id: 3 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_pq_codebook_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_pq_codebook_by_model_id: table_name and vector_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	vector_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	vec_str = text_to_cstring(vector_col);

	oldcontext = CurrentMemoryContext;
	oldcontext_spi = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	/* Build query */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL",
					 vec_str, tbl_str, vec_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(vec_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_pq_codebook_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(vec_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_pq_codebook_by_model_id: need at least 2 vectors, got %d",
						n_points)));
	}

	/* Load model and compute quantization metrics */
	{
		NDB_DECLARE(bytea *, model_payload);
		NDB_DECLARE(Jsonb *, parameters);
		NDB_DECLARE(Jsonb *, metrics);
		char	   *cb_ptr;
		int			m,
					ksub,
					dsub;
		int			i,
					sub,
					c,
					d;
		double		total_error = 0.0;
		int			valid_vectors = 0;
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		float	  ***centroids = NULL;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id,
										   &model_payload,
										   &parameters,
										   &metrics))
		{
			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				/* Deserialize codebook */
				cb_ptr = VARDATA(model_payload);
				memcpy(&m, cb_ptr, sizeof(int));
				cb_ptr += sizeof(int);
				memcpy(&ksub, cb_ptr, sizeof(int));
				cb_ptr += sizeof(int);
				memcpy(&dsub, cb_ptr, sizeof(int));
				cb_ptr += sizeof(int);

				subquantizers = m;
				codebook_size = ksub;
				bits_per_code = (int) (log2((double) ksub) + 0.5);
				if (bits_per_code < 1)
					bits_per_code = 1;

				/* Reconstruct centroids */
				NDB_ALLOC(centroids, float **, m);
				for (sub = 0; sub < m; sub++)
				{
					NDB_DECLARE(float **, centroids_sub);
					NDB_ALLOC(centroids_sub, float *, ksub);
					centroids[sub] = centroids_sub;
					for (c = 0; c < ksub; c++)
					{
						NDB_DECLARE(float *, centroids_sub_c);
						NDB_ALLOC(centroids_sub_c, float, dsub);
						centroids[sub][c] = centroids_sub_c;
						memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
						cb_ptr += sizeof(float) * dsub;
					}
				}

				/* Compute quantization error for each vector */
				for (i = 0; i < n_points; i++)
				{
					/* Safe access to SPI_tuptable - validate before access */
					if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
						i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
					{
						continue;
					}
					{
						HeapTuple	tuple = SPI_tuptable->vals[i];
						Datum		vec_datum;
						bool		vec_null;
						Vector	   *vec;
						float	   *vec_data;
						double		vector_error = 0.0;
						int			start_dim;

						if (tupdesc == NULL)
						{
							continue;
						}

					vec_datum = SPI_getbinval(tuple, tupdesc, 1, &vec_null);
					if (vec_null)
						continue;

					vec = DatumGetVector(vec_datum);
					if (vec == NULL || vec->dim != m * dsub)
						continue;

					vec_data = vec->data;
					valid_vectors++;

					/*
					 * For each subspace, find nearest centroid and compute
					 * error
					 */
					for (sub = 0; sub < m; sub++)
					{
						double		min_dist;
						int			best_c;
						double		sub_error;

						start_dim = sub * dsub;
						min_dist = DBL_MAX;
						best_c = -1;
						sub_error = 0.0;

						/* Find nearest centroid */
						for (c = 0; c < ksub; c++)
						{
							double		dist = 0.0;

							for (d = 0; d < dsub; d++)
							{
								double		diff = (double) vec_data[start_dim + d]
									- (double) centroids[sub][c][d];

								dist += diff * diff;
							}
							if (dist < min_dist)
							{
								min_dist = dist;
								best_c = c;
							}
						}

						/* Compute reconstruction error for this subspace */
						if (best_c >= 0)
						{
							for (d = 0; d < dsub; d++)
							{
								double		diff = (double) vec_data[start_dim + d]
									- (double) centroids[sub][best_c][d];

								sub_error += diff * diff;
							}
						}

						vector_error += sub_error;
					}

					total_error += vector_error;
					}
				}

				/* Free centroids */
				if (centroids != NULL)
				{
					for (sub = 0; sub < m; sub++)
					{
						if (centroids[sub] != NULL)
						{
							for (c = 0; c < ksub; c++)
							{
								if (centroids[sub][c] != NULL)
									NDB_FREE(centroids[sub][c]);
							}
							NDB_FREE(centroids[sub]);
						}
					}
					NDB_FREE(centroids);
				}

				/* Calculate average quantization error */
				if (valid_vectors > 0)
				{
					avg_quantization_error = total_error / valid_vectors;
				}
				else
				{
					avg_quantization_error = 0.0;
				}

				if (model_payload != NULL)
					NDB_FREE(model_payload);
				if (parameters != NULL)
					NDB_FREE(parameters);
				if (metrics != NULL)
					NDB_FREE(metrics);
			}
			else
			{
				/* Model not found or invalid - use defaults */
				avg_quantization_error = 0.0;
				subquantizers = 8;
				codebook_size = 256;
				bits_per_code = 8;
			}
		}
		else
		{
			/* Model not found - use defaults */
			avg_quantization_error = 0.0;
			subquantizers = 8;
			codebook_size = 256;
			bits_per_code = 8;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"quantization_error\":%.6f,\"subquantizers\":%d,\"codebook_size\":%d,\"bits_per_code\":%d,\"n_vectors\":%d}",
					 avg_quantization_error, subquantizers, codebook_size, bits_per_code, n_points);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	NDB_FREE(jsonbuf.data);

	NDB_FREE(tbl_str);
	NDB_FREE(vec_str);

	PG_RETURN_JSONB_P(result);
}

/*
 * pq_asymmetric_distance
 * ----------------------
 * Compute approximate (asymmetric) L2 distance between a query (float4[])
 * and a PQ-compressed vector (codes), given a codebook.
 *
 * Args (SQL):
 *   query     - Query vector (float4[]), must match PQ codebook dim
 *   pq_codes  - PQ codes as int2[] (length m, one code per subspace)
 *   codebook  - PQ codebook (bytea)
 *
 * Returns:
 *   Float4: Approximate Euclidean distance between query and reconstructed database vector.
 *
 * Safety: raises error if dimensions mismatch or if a PQ code is out of valid range.
 */
PG_FUNCTION_INFO_V1(pq_asymmetric_distance);

Datum
pq_asymmetric_distance(PG_FUNCTION_ARGS)
{
	ArrayType  *query_array,
			   *codes_array;
	bytea	   *codebook_bytea;
	float4	   *query_data;
	int16	   *codes;
	int			dim,
				m,
				ksub,
				dsub;
	char	   *cb_ptr;
	float	 ***centroids = NULL;
	double		total_dist = 0.0;
	int			sub,
				d;
	int16		code;

	/* ----- Parse arguments and extract data ----- */
	query_array = PG_GETARG_ARRAYTYPE_P(0);
	codes_array = PG_GETARG_ARRAYTYPE_P(1);
	codebook_bytea = PG_GETARG_BYTEA_PP(2);

	dim = ARR_DIMS(query_array)[0];
	query_data = (float4 *) ARR_DATA_PTR(query_array);
	codes = (int16 *) ARR_DATA_PTR(codes_array);

	/* ----- Deserialize codebook header and centroids ----- */
	cb_ptr = VARDATA(codebook_bytea);
	memcpy(&m, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&ksub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);
	memcpy(&dsub, cb_ptr, sizeof(int));
	cb_ptr += sizeof(int);

	if (dim != m * dsub)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Query vector dimension (%d) does not match codebook (m=%d * dsub=%d)",
						dim,
						m,
						dsub)));

	centroids = (float ***) palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		centroids[sub] = (float **) palloc(sizeof(float *) * ksub);
		for (int c = 0; c < ksub; c++)
		{
			centroids[sub][c] =
				(float *) palloc(sizeof(float) * dsub);
			memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
			cb_ptr += sizeof(float) * dsub;
		}
	}

	/* ----- Compute asymmetric (query-to-product) distance ----- */
	total_dist = 0.0;
	for (sub = 0; sub < m; sub++)
	{
		int			start_dim = sub * dsub;

		code = codes[sub];

		if (code < 0 || code >= ksub)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("Invalid PQ code %d at subspace %d (valid: 0-%d)",
							code,
							sub,
							ksub - 1)));

		for (d = 0; d < dsub; d++)
		{
			double		diff = (double) query_data[start_dim + d]
				- (double) centroids[sub][code][d];

			total_dist += diff * diff;
		}
	}

	/* ----- Free memory for reconstructed centroids ----- */
	for (sub = 0; sub < m; sub++)
	{
		for (int c = 0; c < ksub; c++)
			NDB_FREE(centroids[sub][c]);
		NDB_FREE(centroids[sub]);
	}
	NDB_FREE(centroids);

	PG_RETURN_FLOAT4((float4) sqrt(total_dist));
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Product Quantization
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"

typedef struct ProductQuantizationGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	PQCodebook *codebook;
	int			m;
	int			ksub;
	int			dsub;
	int			dim;
	int			n_samples;
}			ProductQuantizationGpuModelState;

static bytea *
pq_codebook_serialize_to_bytea(const PQCodebook * codebook)
{
	int			result_size;
	bytea	   *result;
	char	   *result_ptr;
	int			sub,
				i;

	result_size = sizeof(int) * 3 + codebook->m * codebook->ksub * codebook->dsub * sizeof(float);
	result = (bytea *) palloc(VARHDRSZ + result_size);
	SET_VARSIZE(result, VARHDRSZ + result_size);
	result_ptr = VARDATA(result);

	memcpy(result_ptr, &codebook->m, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &codebook->ksub, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &codebook->dsub, sizeof(int));
	result_ptr += sizeof(int);

	for (sub = 0; sub < codebook->m; sub++)
		for (i = 0; i < codebook->ksub; i++)
		{
			memcpy(result_ptr, codebook->centroids[sub][i], sizeof(float) * codebook->dsub);
			result_ptr += sizeof(float) * codebook->dsub;
		}

	return result;
}

static int
pq_codebook_deserialize_from_bytea(const bytea * data, PQCodebook * codebook)
{
	const char *buf;
	int			offset = 0;
	int			sub,
				i;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 3)
		return -1;

	buf = VARDATA(data);
	memcpy(&codebook->m, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&codebook->ksub, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&codebook->dsub, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (codebook->m < 1 || codebook->m > 128 || codebook->ksub < 2 || codebook->ksub > 65536 || codebook->dsub <= 0)
		return -1;

	codebook->centroids = (float ***) palloc(sizeof(float **) * codebook->m);
	for (sub = 0; sub < codebook->m; sub++)
	{
		codebook->centroids[sub] = (float **) palloc(sizeof(float *) * codebook->ksub);
		for (i = 0; i < codebook->ksub; i++)
		{
			codebook->centroids[sub][i] = (float *) palloc(sizeof(float) * codebook->dsub);
			memcpy(codebook->centroids[sub][i], buf + offset, sizeof(float) * codebook->dsub);
			offset += sizeof(float) * codebook->dsub;
		}
	}

	return 0;
}

static void
pq_codebook_free(PQCodebook * codebook)
{
	int			sub,
				i;

	if (codebook == NULL || codebook->centroids == NULL)
		return;

	for (sub = 0; sub < codebook->m; sub++)
	{
		if (codebook->centroids[sub] != NULL)
		{
			for (i = 0; i < codebook->ksub; i++)
				if (codebook->centroids[sub][i] != NULL)
					NDB_FREE(codebook->centroids[sub][i]);
			NDB_FREE(codebook->centroids[sub]);
		}
	}
	NDB_FREE(codebook->centroids);
}

static bool
product_quantization_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	ProductQuantizationGpuModelState *state;
	float	  **data = NULL;
	PQCodebook	codebook;
	int			m = 8;
	int			ksub = 256;
	int			nvec = 0;
	int			dim = 0;
	int			sub,
				i;
	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, metrics);
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_train: invalid parameters");
		return false;
	}

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		it = JsonbIteratorInit((JsonbContainer *) & spec->hyperparameters->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "m") == 0 && v.type == jbvNumeric)
					m = DatumGetInt32(DirectFunctionCall1(numeric_int4,
														  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "ksub") == 0 && v.type == jbvNumeric)
					ksub = DatumGetInt32(DirectFunctionCall1(numeric_int4,
															 NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (m < 1 || m > 128)
		m = 8;
	if (ksub < 2 || ksub > 65536)
		ksub = 256;

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	if (dim % m != 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_train: dimension must be divisible by m");
		return false;
	}

	data = (float **) palloc(sizeof(float *) * nvec);
	for (i = 0; i < nvec; i++)
	{
		data[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	/* Prepare codebook structure */
	codebook.m = m;
	codebook.ksub = ksub;
	codebook.dsub = dim / m;

	/* Allocate centroids */
	codebook.centroids = (float ***) palloc(sizeof(float **) * m);
	for (sub = 0; sub < m; sub++)
	{
		codebook.centroids[sub] = (float **) palloc(sizeof(float *) * ksub);
		for (i = 0; i < ksub; i++)
			codebook.centroids[sub][i] = (float *) palloc(sizeof(float) * codebook.dsub);
	}

	/* Train PQ codebook */
	for (sub = 0; sub < m; sub++)
	{
		float	  **subspace_data = NULL;
		int			start_dim = sub * codebook.dsub;

		subspace_data = (float **) palloc(sizeof(float *) * nvec);
		for (i = 0; i < nvec; i++)
		{
			subspace_data[i] = (float *) palloc(sizeof(float) * codebook.dsub);
			memcpy(subspace_data[i], &data[i][start_dim], sizeof(float) * codebook.dsub);
		}

		train_subspace_kmeans(subspace_data, nvec, codebook.dsub, ksub, codebook.centroids[sub], 100);

		for (i = 0; i < nvec; i++)
			NDB_FREE(subspace_data[i]);
		NDB_FREE(subspace_data);
	}

	/* Serialize model */
	model_data = pq_codebook_serialize_to_bytea(&codebook);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"m\":%d,\"ksub\":%d,\"dsub\":%d,\"dim\":%d,\"n_samples\":%d}",
					 m, ksub, codebook.dsub, dim, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (ProductQuantizationGpuModelState *) palloc0(sizeof(ProductQuantizationGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->codebook = (PQCodebook *) palloc(sizeof(PQCodebook));
	*state->codebook = codebook;
	state->m = m;
	state->ksub = ksub;
	state->dsub = codebook.dsub;
	state->dim = dim;
	state->n_samples = nvec;

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	/* Cleanup temp data */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);

	return true;
}

static bool
product_quantization_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
								 float *output, int output_dim, char **errstr)
{
	const		ProductQuantizationGpuModelState *state;
	PQCodebook *codebook;
	int			sub;
	int			start_dim;
	double		min_dist;
	int			best_code;
	float	   *reconstructed;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		memset(output, 0, output_dim * sizeof(float));
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_predict: invalid parameters");
		return false;
	}
	if (model->backend_state == NULL || output_dim < ((ProductQuantizationGpuModelState *) model->backend_state)->dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_predict: model not ready");
		return false;
	}

	state = (const ProductQuantizationGpuModelState *) model->backend_state;
	if (state->codebook == NULL && state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_predict: codebook is NULL");
		return false;
	}

	if (input_dim != state->dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_predict: dimension mismatch");
		return false;
	}

	/* Deserialize codebook if needed */
	if (state->codebook == NULL)
	{
		PQCodebook	temp_codebook;

		if (pq_codebook_deserialize_from_bytea(state->model_blob, &temp_codebook) != 0)
		{
			if (errstr != NULL)
				*errstr = pstrdup("product_quantization_gpu_predict: failed to deserialize");
			return false;
		}
		((ProductQuantizationGpuModelState *) state)->codebook = (PQCodebook *) palloc(sizeof(PQCodebook));
		*((ProductQuantizationGpuModelState *) state)->codebook = temp_codebook;
	}

	codebook = state->codebook;
	reconstructed = (float *) palloc(sizeof(float) * state->dim);

	/* Encode and reconstruct */
	for (sub = 0; sub < codebook->m; sub++)
	{
		start_dim = sub * codebook->dsub;
		min_dist = DBL_MAX;
		best_code = 0;

		for (int c = 0; c < codebook->ksub; c++)
		{
			double		dist = sqrt(neurondb_l2_distance_squared(
																 &input[start_dim], codebook->centroids[sub][c], codebook->dsub));

			if (dist < min_dist)
			{
				min_dist = dist;
				best_code = c;
			}
		}

		memcpy(&reconstructed[start_dim], codebook->centroids[sub][best_code],
			   sizeof(float) * codebook->dsub);
	}

	memcpy(output, reconstructed, sizeof(float) * state->dim);
	NDB_FREE(reconstructed);

	return true;
}

static bool
product_quantization_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
								  MLGpuMetrics * out, char **errstr)
{
	const		ProductQuantizationGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;
	double		avg_error = 0.0;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_evaluate: invalid model");
		return false;
	}

	state = (const ProductQuantizationGpuModelState *) model->backend_state;

	/*
	 * Note: MLGpuEvalSpec does not provide feature_matrix/sample_count.
	 * Evaluation metrics would need to be computed from evaluation_table via
	 * SPI if needed. For now, avg_error remains 0.0.
	 */

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"product_quantization\",\"storage\":\"cpu\","
					 "\"m\":%d,\"ksub\":%d,\"dsub\":%d,\"dim\":%d,\"avg_error\":%.6f,\"n_samples\":%d}",
					 state->m > 0 ? state->m : 8,
					 state->ksub > 0 ? state->ksub : 256,
					 state->dsub > 0 ? state->dsub : 0,
					 state->dim > 0 ? state->dim : 0,
					 avg_error,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
product_quantization_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
								   Jsonb * *metadata_out, char **errstr)
{
	const		ProductQuantizationGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_serialize: invalid model");
		return false;
	}

	state = (const ProductQuantizationGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
product_quantization_gpu_deserialize(MLGpuModel * model, const bytea * payload,
									 const Jsonb * metadata, char **errstr)
{
	ProductQuantizationGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	PQCodebook	codebook;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	if (pq_codebook_deserialize_from_bytea(payload_copy, &codebook) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("product_quantization_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (ProductQuantizationGpuModelState *) palloc0(sizeof(ProductQuantizationGpuModelState));
	state->model_blob = payload_copy;
	state->codebook = (PQCodebook *) palloc(sizeof(PQCodebook));
	*state->codebook = codebook;
	state->m = codebook.m;
	state->ksub = codebook.ksub;
	state->dsub = codebook.dsub;
	state->dim = codebook.m * codebook.dsub;
	state->n_samples = 0;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		Jsonb	   *metadata_copy = (Jsonb *) palloc(metadata_size);

		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					state->n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		 NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "dim") == 0 && v.type == jbvNumeric)
					state->dim = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																   NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
product_quantization_gpu_destroy(MLGpuModel * model)
{
	ProductQuantizationGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (ProductQuantizationGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		if (state->codebook != NULL)
		{
			pq_codebook_free(state->codebook);
			NDB_FREE(state->codebook);
		}
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps product_quantization_gpu_model_ops = {
	.algorithm = "product_quantization",
	.train = product_quantization_gpu_train,
	.predict = product_quantization_gpu_predict,
	.evaluate = product_quantization_gpu_evaluate,
	.serialize = product_quantization_gpu_serialize,
	.deserialize = product_quantization_gpu_deserialize,
	.destroy = product_quantization_gpu_destroy,
};

void
neurondb_gpu_register_product_quantization_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&product_quantization_gpu_model_ops);
	registered = true;
}

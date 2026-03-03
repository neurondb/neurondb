/*-------------------------------------------------------------------------
 *
 * ml_dbscan.c
 *    DBSCAN density-based clustering.
 *
 * This module implements DBSCAN for density-based spatial clustering of
 * closely packed points.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_dbscan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include <math.h>
#include <float.h>

#define DBSCAN_NOISE			-1
#define DBSCAN_UNDEFINED		-2

/* Forward declaration */
static int	dbscan_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *n_clusters_out, int *dim_out, double *eps_out, int *min_pts_out, uint8 * training_backend_out);

typedef struct DBSCANState
{
	float	  **data;
	int			nvec;
	int			dim;
	double		eps;
	int			min_pts;
	int		   *labels;
	int			next_cluster;
}			DBSCANState;

/*
 * dbscan_region_query - Find neighbors within epsilon distance
 *
 * Finds all data points within epsilon distance of a given point for DBSCAN
 * clustering. Uses L2 distance squared for efficiency, then takes square root
 * for comparison.
 *
 * Parameters:
 *   state - DBSCAN state containing data and parameters
 *   idx - Index of the query point
 *   neighbor_count - Output parameter to receive number of neighbors found
 *
 * Returns:
 *   Dynamically allocated array of neighbor indices, caller must free
 *
 * Notes:
 *   The function dynamically grows the neighbors array as needed. Memory
 *   is allocated in CurrentMemoryContext and must be freed by the caller.
 */
static int *
dbscan_region_query(const DBSCANState * state, int idx, int *neighbor_count)
{
	double		dist_sq;
	int			capacity;
	int			count;
	int			i;
	int		   *neighbors = NULL;

	capacity = 16;
	nalloc(neighbors, int, capacity);
	count = 0;

	for (i = 0; i < state->nvec; i++)
	{
		dist_sq = neurondb_l2_distance_squared(state->data[idx], state->data[i], state->dim);
		if (sqrt(dist_sq) <= state->eps)
		{
			if (count >= capacity)
			{
				capacity *= 2;
				neighbors = (int *) repalloc(neighbors, capacity * sizeof(int));
			}
			neighbors[count++] = i;
		}
	}

	*neighbor_count = count;
	return neighbors;
}

/*
 * dbscan_expand_cluster - Expand cluster from seed point
 *
 * Recursively expands a DBSCAN cluster starting from a seed point by finding
 * all density-reachable points. Marks points as belonging to the cluster and
 * continues expansion from newly found core points.
 *
 * Parameters:
 *   state - DBSCAN state containing data, labels, and parameters
 *   point_idx - Index of seed point to start expansion from
 *   neighbors - Array of neighbor indices
 *   neighbor_count - Number of neighbors
 *   cluster_id - Cluster ID to assign to points
 *
 * Notes:
 *   This function implements the core DBSCAN cluster expansion algorithm.
 *   It recursively processes neighbors and expands the cluster until no
 *   more density-reachable points are found.
 */
static void
dbscan_expand_cluster(DBSCANState * state,
					  int point_idx,
					  int *neighbors,
					  int neighbor_count,
					  int cluster_id)
{
	int			current;
	int			current_neighbor_count;
	int			j;
	int			neighbor;
	int		   *current_neighbors = NULL;
	int			seed_count;
	int			seed_idx;
	int		   *seeds = NULL;

	state->labels[point_idx] = cluster_id;

	nalloc(seeds, int, neighbor_count);
	memcpy(seeds, neighbors, neighbor_count * sizeof(int));
	seed_count = neighbor_count;
	seed_idx = 0;

	while (seed_idx < seed_count)
	{
		current = seeds[seed_idx];
		seed_idx++;

		if (state->labels[current] == cluster_id)
			continue;

		if (state->labels[current] == DBSCAN_NOISE)
		{
			state->labels[current] = cluster_id;
			continue;
		}

		state->labels[current] = cluster_id;

		current_neighbors = dbscan_region_query(state, current, &current_neighbor_count);

		if (current_neighbor_count >= state->min_pts)
		{
			for (j = 0; j < current_neighbor_count; j++)
			{
				neighbor = current_neighbors[j];

				if (state->labels[neighbor] == DBSCAN_UNDEFINED)
				{
					seeds = (int *) repalloc(seeds, (seed_count + 1) * sizeof(int));
					seeds[seed_count++] = neighbor;
				}
			}
		}

		nfree(current_neighbors);
	}

	nfree(seeds);
}

PG_FUNCTION_INFO_V1(cluster_dbscan);
PG_FUNCTION_INFO_V1(predict_dbscan);
PG_FUNCTION_INFO_V1(evaluate_dbscan_by_model_id);

Datum
cluster_dbscan(PG_FUNCTION_ARGS)
{
	ArrayType  *out_array = NULL;
	char	   *col_str = NULL;
	char	   *tbl_str = NULL;
	DBSCANState state;
	double		eps;
	int			i;
	int			min_pts;
	text	   *column_name = NULL;
	text	   *table_name = NULL;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Datum *out_datums = NULL;

	/* Validate minimum argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: cluster_dbscan requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	eps = PG_GETARG_FLOAT8(2);

	if (PG_NARGS() >= 4)
		min_pts = PG_GETARG_INT32(3);
	else
		min_pts = 5;

	if (eps <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("eps must be positive")));

	if (min_pts < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("min_pts must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);


	/* Initialize state structure to zero */
	memset(&state, 0, sizeof(state));

	state.data = neurondb_fetch_vectors_from_table(
												   tbl_str, col_str, &state.nvec, &state.dim);

	if (state.data == NULL || state.nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found in table")));
	}

	if (state.dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		/* Free data array and rows if data is not NULL */
		if (state.data != NULL && state.nvec > 0)
		{
			neurondb_free_vectors(state.data, state.nvec);
		}
		else if (state.data != NULL)
		{
			nfree(state.data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", state.dim)));
	}

	state.eps = eps;
	state.min_pts = min_pts;
	state.next_cluster = 0;

	/* Check memory allocation size before palloc */
	{
		size_t		labels_size = (size_t) state.nvec * sizeof(int);

		if (labels_size > MaxAllocSize)
		{
			for (i = 0; i < state.nvec; i++)
				nfree(state.data[i]);
			nfree(state.data);
			nfree(tbl_str);
			nfree(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("cluster_dbscan: labels array size (%zu bytes) exceeds MaxAllocSize (%zu bytes)",
							labels_size, (size_t) MaxAllocSize),
					 errhint("Reduce dataset size or use a different clustering algorithm")));
		}
	}

	/* Ensure state.labels is NULL before allocation */
	if (state.labels != NULL)
		nfree(state.labels);
	state.labels = NULL;
	{
		int *labels_tmp = NULL;
		nalloc(labels_tmp, int, state.nvec);
		state.labels = labels_tmp;
	}

	for (i = 0; i < state.nvec; i++)
		state.labels[i] = DBSCAN_UNDEFINED;

	for (i = 0; i < state.nvec; i++)
	{
		int *neighbors = NULL;
		int			neighbor_count;

		if (state.labels[i] != DBSCAN_UNDEFINED)
			continue;

		neighbors = dbscan_region_query(&state, i, &neighbor_count);

		if (neighbor_count < state.min_pts)
			state.labels[i] = DBSCAN_NOISE;
		else
		{
			dbscan_expand_cluster(&state, i, neighbors, neighbor_count, state.next_cluster);
			state.next_cluster++;
		}

		nfree(neighbors);
	}


	nalloc(out_datums, Datum, state.nvec);
	for (i = 0; i < state.nvec; i++)
		out_datums[i] = Int32GetDatum(state.labels[i]);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);

	out_array = construct_array(out_datums, state.nvec, INT4OID, typlen, typbyval, typalign);

	for (i = 0; i < state.nvec; i++)
		nfree(state.data[i]);
	nfree(state.data);
	nfree(state.labels);
	nfree(out_datums);
	nfree(tbl_str);
	nfree(col_str);

	PG_RETURN_ARRAYTYPE_P(out_array);
}

/*
 * predict_dbscan
 *      Predicts cluster assignment for new data points using trained DBSCAN model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: int4 cluster_id (-1 for noise)
 */
Datum
predict_dbscan(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *features_array = NULL;
	int			cluster_id = DBSCAN_NOISE;
	int			n_elems = 0;
	float *features = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_dbscan requires 2 arguments")));

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Extract features from array */
	{
		Oid			elmtype = ARR_ELEMTYPE(features_array);
		int16		typlen;
		bool		typbyval;
		char		typalign;
		Datum *elems = NULL;
		bool *nulls = NULL;
		int			i;

		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);
		deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
						  &elems, &nulls, &n_elems);

		nalloc(features, float, n_elems);

		for (i = 0; i < n_elems; i++)
			features[i] = DatumGetFloat4(elems[i]);
	}

	/* Load model from catalog and perform proper DBSCAN prediction */
	{
		bytea *model_payload = NULL;
		Jsonb *model_parameters = NULL;
		float **cluster_centers = NULL;
		int			n_clusters = 0;
		int			dim = 0;
		double		eps = 0.0;
		int			min_pts = 0;
		int			i;
		double		min_distance = DBL_MAX;
		int			nearest_cluster = -1;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id, &model_payload, &model_parameters, NULL))
		{
			if (model_payload != NULL)
			{
				/* Deserialize model to get cluster centers, eps, and min_pts */
				{
					uint8		training_backend = 0;

					if (dbscan_model_deserialize_from_bytea(model_payload,
															&cluster_centers, &n_clusters, &dim, &eps, &min_pts,
															&training_backend) == 0)
					{
					/* Validate feature dimension matches model dimension */
					if (n_elems == dim && n_clusters > 0)
					{
						/* Find nearest cluster center */
						for (i = 0; i < n_clusters; i++)
						{
							double		dist_sq = neurondb_l2_distance_squared(features, cluster_centers[i], dim);
							double		dist = sqrt(dist_sq);

							if (dist < min_distance)
							{
								min_distance = dist;
								nearest_cluster = i;
							}
						}

						/*
						 * If point is within eps of a cluster center, assign
						 * to that cluster
						 */

						/*
						 * Note: In full DBSCAN, we'd also check if cluster
						 * has >= min_pts points, but for prediction we use
						 * the cluster centers as representatives
						 */
						if (min_distance <= eps && nearest_cluster >= 0)
						{
							cluster_id = nearest_cluster;
						}
						else
						{
							/*
							 * Point is too far from any cluster center,
							 * classify as noise
							 */
							cluster_id = DBSCAN_NOISE;
						}
					}
					else
					{
						/* Dimension mismatch or no clusters */
						cluster_id = DBSCAN_NOISE;
					}

					/* Free cluster centers */
					if (cluster_centers != NULL)
					{
						for (i = 0; i < n_clusters; i++)
						{
							if (cluster_centers[i] != NULL)
								nfree(cluster_centers[i]);
						}
						nfree(cluster_centers);
					}
				}
				else
				{
					/* Failed to deserialize model */
					cluster_id = DBSCAN_NOISE;
				}
				}
			}
			else
			{
				/* Model payload not found */
				cluster_id = DBSCAN_NOISE;
			}
		}
		else
		{
			/* Failed to load model from catalog */
			cluster_id = DBSCAN_NOISE;
		}
	}

	nfree(features);

	PG_RETURN_INT32(cluster_id);
}

/*
 * evaluate_dbscan_by_model_id
 *      Evaluates DBSCAN clustering quality on a dataset.
 *      Arguments: int4 model_id, text table_name, text feature_col
 *      Returns: jsonb with clustering metrics
 */
Datum
evaluate_dbscan_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	StringInfoData query;
	int			ret;
	int			n_points = 0;
	StringInfoData jsonbuf;
	Jsonb *result = NULL;
	MemoryContext oldcontext;
	int			n_clusters;
	int			n_noise;
	double		eps;
	int			min_pts;

	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext_spi;

	/* Validate arguments */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_dbscan_by_model_id: 3 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_dbscan_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_dbscan_by_model_id: table_name and feature_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);

	oldcontext = CurrentMemoryContext;
	oldcontext_spi = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	/* Build query */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL",
					 feat_str, tbl_str, feat_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_dbscan_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_dbscan_by_model_id: need at least 2 points, got %d",
						n_points)));
	}

	/* Load model and compute actual metrics */
	{
		bytea *model_payload = NULL;
		Jsonb *model_parameters = NULL;
		float **centers = NULL;
		float **data = NULL;
		int			model_dim = 0;
		int			i,
					c;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id, &model_payload, &model_parameters, NULL))
		{
			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				{
					uint8		training_backend = 0;

					if (dbscan_model_deserialize_from_bytea(model_payload,
															&centers,
															&n_clusters,
															&model_dim,
															&eps,
															&min_pts,
															&training_backend) == 0)
					{
					/* Fetch data points */
					data = neurondb_fetch_vectors_from_table(tbl_str, feat_str, &n_points, &model_dim);

					if (data != NULL && n_points > 0)
					{
						/* Allocate assignments */
						int *assignments = NULL;
						nalloc(assignments, int, n_points);

						/*
						 * Assign points to nearest cluster centers or mark as
						 * noise
						 */
						n_noise = 0;
						for (i = 0; i < n_points; i++)
						{
							double		min_dist = DBL_MAX;
							int			best = -1;

							for (c = 0; c < n_clusters; c++)
							{
								double		dist = 0.0;
								int			d;

								for (d = 0; d < model_dim; d++)
								{
									double		diff = (double) data[i][d] - (double) centers[c][d];

									dist += diff * diff;
								}
								dist = sqrt(dist);
								if (dist < min_dist)
								{
									min_dist = dist;
									best = c;
								}
							}

							/*
							 * If point is within eps of a cluster, assign it;
							 * otherwise mark as noise
							 */
							if (best >= 0 && min_dist <= eps)
							{
								assignments[i] = best;
							}
							else
							{
								assignments[i] = DBSCAN_NOISE;
								n_noise++;
							}
						}

						/* Cleanup data */
						for (i = 0; i < n_points; i++)
							nfree(data[i]);
						nfree(data);
						nfree(assignments);
					}
					else
					{
						n_clusters = 0;
						n_noise = 0;
						eps = 0.5;
						min_pts = 5;
						if (data != NULL)
						{
							for (i = 0; i < n_points; i++)
								nfree(data[i]);
							nfree(data);
						}
					}

					/* Cleanup centers */
					if (centers != NULL)
					{
						for (c = 0; c < n_clusters; c++)
							nfree(centers[c]);
						nfree(centers);
					}
				}
				else
				{
					n_clusters = 0;
					n_noise = 0;
					eps = 0.5;
					min_pts = 5;
				}
				}
			}
			else
			{
				n_clusters = 0;
				n_noise = 0;
				eps = 0.5;
				min_pts = 5;
			}
			if (model_payload)
				nfree(model_payload);
			if (model_parameters)
				nfree(model_parameters);
		}
		else
		{
			/* Model not found */
			n_clusters = 0;
			n_noise = 0;
			eps = 0.5;
			min_pts = 5;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"n_clusters\":%d,\"n_noise\":%d,\"noise_ratio\":%.6f,\"eps\":%.6f,\"min_pts\":%d,\"n_points\":%d}",
					 n_clusters, n_noise, (double) n_noise / n_points, eps, min_pts, n_points);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	nfree(jsonbuf.data);

	nfree(tbl_str);
	nfree(feat_str);

	PG_RETURN_JSONB_P(result);
}

#include "neurondb_gpu_model.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct DBSCANGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	double		eps;
	int			min_pts;
	int			n_clusters;
	int			dim;
	int			n_samples;
}			DBSCANGpuModelState;

static bytea *
dbscan_model_serialize_to_bytea(float **cluster_centers, int n_clusters, int dim, double eps, int min_pts, uint8 training_backend)
{
	StringInfoData buf;
	int			i,
				j;
	int			total_size;

	char *result_bytes = NULL;
	bytea *result = NULL;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dbscan_model_serialize_to_bytea: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	initStringInfo(&buf);
	/* Write training_backend first (0=CPU, 1=GPU) - unified storage format */
	appendBinaryStringInfo(&buf, (char *) &training_backend, sizeof(uint8));
	appendBinaryStringInfo(&buf, (char *) &n_clusters, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &dim, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &eps, sizeof(double));
	appendBinaryStringInfo(&buf, (char *) &min_pts, sizeof(int));

	for (i = 0; i < n_clusters; i++)
		for (j = 0; j < dim; j++)
			appendBinaryStringInfo(&buf, (char *) &cluster_centers[i][j], sizeof(float));

	total_size = VARHDRSZ + buf.len;
	nalloc(result_bytes, char, total_size);
	result = (bytea *) result_bytes;
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	nfree(buf.data);

	return result;
}

static int
dbscan_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *n_clusters_out, int *dim_out, double *eps_out, int *min_pts_out, uint8 * training_backend_out)
{
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	float **centers = NULL;
	uint8		training_backend = 0;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(uint8) + sizeof(int) * 2 + sizeof(double) + sizeof(int))
		return -1;

	buf = VARDATA(data);
	/* Read training_backend first (unified storage format) */
	training_backend = (uint8) buf[offset];
	offset += sizeof(uint8);
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;
	memcpy(n_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(eps_out, buf + offset, sizeof(double));
	offset += sizeof(double);
	memcpy(min_pts_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (*n_clusters_out < 0 || *n_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	nalloc(centers, float *, *n_clusters_out);
	for (i = 0; i < *n_clusters_out; i++)
	{
		nalloc(centers[i], float, *dim_out);
		for (j = 0; j < *dim_out; j++)
		{
			memcpy(&centers[i][j], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*centers_out = centers;
	return 0;
}

static bool
dbscan_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	float **data = NULL;
	int *labels = NULL;
	float **cluster_centers = NULL;
	int *cluster_sizes = NULL;
	double		eps = 0.5;
	int			min_pts = 5;
	int			nvec = 0;
	int			dim = 0;
	int			i,
				c,
				d;
	int			n_clusters = 0;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	StringInfoData metrics_json;
	DBSCANState dbscan_state;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	DBSCANGpuModelState *state = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_train: invalid parameters");
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
				if (strcmp(key, "eps") == 0 && v.type == jbvNumeric)
					eps = DatumGetFloat8(DirectFunctionCall1(numeric_float8,
															 NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "min_pts") == 0 && v.type == jbvNumeric)
					min_pts = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																NumericGetDatum(v.val.numeric)));
				nfree(key);
			}
		}
	}

	if (eps <= 0.0)
		eps = 0.5;
	if (min_pts < 1)
		min_pts = 5;

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	nalloc(data, float *, nvec);
	for (i = 0; i < nvec; i++)
	{
		nalloc(data[i], float, dim);
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	/* Run DBSCAN clustering */
	dbscan_state.data = data;
	dbscan_state.nvec = nvec;
	dbscan_state.dim = dim;
	dbscan_state.eps = eps;
	dbscan_state.min_pts = min_pts;
	dbscan_state.next_cluster = 0;
	nalloc(labels, int, nvec);

	for (i = 0; i < nvec; i++)
		labels[i] = DBSCAN_UNDEFINED;

	for (i = 0; i < nvec; i++)
	{
		int *neighbors = NULL;
		int			neighbor_count;

		if (labels[i] != DBSCAN_UNDEFINED)
			continue;

		neighbors = dbscan_region_query(&dbscan_state, i, &neighbor_count);

		if (neighbor_count < min_pts)
			labels[i] = DBSCAN_NOISE;
		else
		{
			dbscan_expand_cluster(&dbscan_state, i, neighbors, neighbor_count, dbscan_state.next_cluster);
			dbscan_state.next_cluster++;
		}

		nfree(neighbors);
	}

	n_clusters = dbscan_state.next_cluster;

	/* Compute cluster centroids */
	if (n_clusters > 0)
	{
		nalloc(cluster_centers, float *, n_clusters);
		nalloc(cluster_sizes, int, n_clusters);

		for (c = 0; c < n_clusters; c++)
		{
			nalloc(cluster_centers[c], float, dim);
		}

		for (i = 0; i < nvec; i++)
		{
			if (labels[i] >= 0 && labels[i] < n_clusters)
			{
				for (d = 0; d < dim; d++)
					cluster_centers[labels[i]][d] += data[i][d];
				cluster_sizes[labels[i]]++;
			}
		}

		for (c = 0; c < n_clusters; c++)
		{
			if (cluster_sizes[c] > 0)
			{
				for (d = 0; d < dim; d++)
					cluster_centers[c][d] /= cluster_sizes[c];
			}
		}

		nfree(cluster_sizes);
	}
	else
	{
		/* No clusters found, create empty model */
		nalloc(cluster_centers, float *, 1);
		nalloc(cluster_centers[0], float, dim);
		n_clusters = 0;
	}

	/* Serialize model */
	model_data = dbscan_model_serialize_to_bytea(cluster_centers, n_clusters, dim, eps, min_pts, 0); /* training_backend=0 for CPU */

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"training_backend\":0,\"n_clusters\":%d,\"eps\":%.6f,\"min_pts\":%d,\"dim\":%d,\"n_samples\":%d}",
					 n_clusters, eps, min_pts, dim, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	nfree(metrics_json.data);

	nalloc(state, DBSCANGpuModelState, 1);
	state->model_blob = model_data;
	state->metrics = metrics;
	state->eps = eps;
	state->min_pts = min_pts;
	state->n_clusters = n_clusters;
	state->dim = dim;
	state->n_samples = nvec;

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	for (c = 0; c < n_clusters; c++)
		nfree(cluster_centers[c]);
	nfree(data);
	nfree(cluster_centers);
	nfree(labels);

	return true;
}

static bool
dbscan_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
				   float *output, int output_dim, char **errstr)
{
	const		DBSCANGpuModelState *state;
	float **centers = NULL;
	int			n_clusters = 0;
	int			dim = 0;
	double		eps = 0.0;
	int			min_pts = 0;
	int			c;
	double		min_dist = DBL_MAX;
	int			best_cluster = DBSCAN_NOISE;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = (float) DBSCAN_NOISE;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_predict: model not ready");
		return false;
	}

	state = (const DBSCANGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_predict: model blob is NULL");
		return false;
	}

	{
		uint8		training_backend = 0;

		if (dbscan_model_deserialize_from_bytea(state->model_blob,
												&centers, &n_clusters, &dim, &eps, &min_pts,
												&training_backend) != 0)
		{
			if (errstr != NULL)
				*errstr = pstrdup("dbscan_gpu_predict: failed to deserialize");
			return false;
		}
	}

	if (input_dim != dim)
	{
		for (c = 0; c < n_clusters; c++)
			nfree(centers[c]);
		nfree(centers);
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_predict: dimension mismatch");
		return false;
	}

	/* Find nearest cluster centroid */
	for (c = 0; c < n_clusters; c++)
	{
		double		dist = sqrt(neurondb_l2_distance_squared(input, centers[c], dim));

		if (dist < min_dist)
		{
			min_dist = dist;
			best_cluster = c;
		}
	}

	/* If distance to nearest cluster > eps, mark as noise */
	if (min_dist > eps)
		best_cluster = DBSCAN_NOISE;

	output[0] = (float) best_cluster;

	for (c = 0; c < n_clusters; c++)
		nfree(centers[c]);
	nfree(centers);

	return true;
}

static bool
dbscan_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
					MLGpuMetrics *out, char **errstr)
{
	const		DBSCANGpuModelState *state;
	Jsonb	   *metrics_json = NULL;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_evaluate: invalid model");
		return false;
	}

	state = (const DBSCANGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"dbscan\",\"storage\":\"cpu\","
					 "\"n_clusters\":%d,\"eps\":%.6f,\"min_pts\":%d,\"dim\":%d,\"n_samples\":%d}",
					 state->n_clusters > 0 ? state->n_clusters : 0,
					 state->eps > 0.0 ? state->eps : 0.5,
					 state->min_pts > 0 ? state->min_pts : 5,
					 state->dim > 0 ? state->dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	nfree(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
dbscan_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
					 Jsonb * *metadata_out, char **errstr)
{
	const		DBSCANGpuModelState *state;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	char *payload_bytes = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_serialize: invalid model");
		return false;
	}

	state = (const DBSCANGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	nalloc(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
dbscan_gpu_deserialize(MLGpuModel *model, const bytea * payload,
					   const Jsonb * metadata, char **errstr)
{
	bytea	   *payload_copy = NULL;
	int			payload_size;
	float **centers = NULL;
	int			n_clusters = 0;
	int			dim = 0;
	double		eps = 0.0;
	int			min_pts = 0;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	Jsonb *metadata_copy = NULL;
	char *payload_bytes = NULL;
	DBSCANGpuModelState *state = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("dbscan_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	nalloc(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, payload, payload_size);

	{
		uint8		training_backend = 0;

		if (dbscan_model_deserialize_from_bytea(payload_copy,
												&centers, &n_clusters, &dim, &eps, &min_pts,
												&training_backend) != 0)
		{
			nfree(payload_copy);
			if (errstr != NULL)
				*errstr = pstrdup("dbscan_gpu_deserialize: failed to deserialize");
			return false;
		}
	}

	{
		int			c;

		for (c = 0; c < n_clusters; c++)
			nfree(centers[c]);
		nfree(centers);
	}

	nalloc(state, DBSCANGpuModelState, 1);
	state->model_blob = payload_copy;
	state->eps = eps;
	state->min_pts = min_pts;
	state->n_clusters = n_clusters;
	state->dim = dim;
	state->n_samples = 0;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		char *metadata_bytes = NULL;

		nalloc(metadata_bytes, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_bytes;

		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		it = JsonbIteratorInit((JsonbContainer *) & metadata_copy->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					state->n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		 NumericGetDatum(v.val.numeric)));
				nfree(key);
			}
		}
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
dbscan_gpu_destroy(MLGpuModel *model)
{
	DBSCANGpuModelState *state = NULL;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (DBSCANGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			nfree(state->model_blob);
		if (state->metrics != NULL)
			nfree(state->metrics);
		nfree(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps dbscan_gpu_model_ops = {
	.algorithm = "dbscan",
	.train = dbscan_gpu_train,
	.predict = dbscan_gpu_predict,
	.evaluate = dbscan_gpu_evaluate,
	.serialize = dbscan_gpu_serialize,
	.deserialize = dbscan_gpu_deserialize,
	.destroy = dbscan_gpu_destroy,
};

/* Forward declaration to avoid missing prototype warning */
extern void neurondb_gpu_register_dbscan_model(void);

void
neurondb_gpu_register_dbscan_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&dbscan_gpu_model_ops);
	registered = true;
}

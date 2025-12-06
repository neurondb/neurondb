/*-------------------------------------------------------------------------
 *
 * ml_davies_bouldin.c
 *    Davies-Bouldin index metric implementation.
 *
 * The Davies-Bouldin index is a metric for evaluating clustering quality.
 * Lower values indicate better clustering. The index measures the average
 * similarity ratio of each cluster with its most similar cluster.
 *
 * Formula: DB = (1/k) * sum(max((σi + σj) / d(ci, cj)))
 * where:
 *   - k is the number of clusters
 *   - σi is the average distance from points in cluster i to its centroid
 *   - d(ci, cj) is the distance between centroids of clusters i and j
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_davies_bouldin.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include <math.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_spi_safe.h"
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

/* Forward declarations */
static inline double euclidean_distance(const float *a, const float *b, int dim);
static void compute_cluster_centroids(float **data, int *assignments, int nvec,
									  int dim, int num_clusters, int *cluster_sizes,
									  float **centroids);

/*
 * Compute Euclidean distance between two vectors
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - (double) b[i];

		sum += diff * diff;
	}
	return sqrt(sum);
}

/*
 * Compute cluster centroids from data and assignments
 */
static void
compute_cluster_centroids(float **data, int *assignments, int nvec,
						  int dim, int num_clusters, int *cluster_sizes,
						  float **centroids)
{
	int			i,
				c,
				d;

	/* Initialize centroids to zero */
	for (c = 0; c < num_clusters; c++)
	{
		for (d = 0; d < dim; d++)
			centroids[c][d] = 0.0f;
		cluster_sizes[c] = 0;
	}

	/* Sum vectors by cluster */
	for (i = 0; i < nvec; i++)
	{
		c = assignments[i];
		if (c >= 0 && c < num_clusters)
		{
			cluster_sizes[c]++;
			for (d = 0; d < dim; d++)
				centroids[c][d] += data[i][d];
		}
	}

	/* Average to get centroids */
	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
		{
			for (d = 0; d < dim; d++)
				centroids[c][d] /= (float) cluster_sizes[c];
		}
	}
}

/*
 * davies_bouldin_index
 *
 * Computes the Davies-Bouldin index for a clustering result.
 *
 * Parameters:
 *   table_name: Name of the table containing vectors and cluster assignments
 *   vector_col: Name of the column containing vector data
 *   cluster_col: Name of the column containing cluster assignments (integer)
 *
 * Returns:
 *   double precision: Davies-Bouldin index (lower is better)
 */
PG_FUNCTION_INFO_V1(davies_bouldin_index);

Datum
davies_bouldin_index(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_col;
	text	   *cluster_col;
	char	   *tbl_str;
	char	   *col_str;
	char	   *cluster_str;
	int			nvec = 0;
	int			i,
				j,
				c;
	float	  **data = NULL;
	int			dim = 0;
	int			*assignments = NULL;
	int			*cluster_sizes = NULL;
	float	  **centroids = NULL;
	double		davies_bouldin = 0.0;
	double		*cluster_scatter = NULL;
	int			num_clusters = 0;
	int			max_cluster_id = -1;
	MemoryContext oldcontext;
	NdbSpiSession *spi_session = NULL;
	StringInfoData sql;
	int			ret;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: all parameters are required")));

	table_name = PG_GETARG_TEXT_PP(0);
	vector_col = PG_GETARG_TEXT_PP(1);
	cluster_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);
	cluster_str = text_to_cstring(cluster_col);

	oldcontext = CurrentMemoryContext;

	/* Begin SPI session */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Fetch vectors from table */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < 1)
	{
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: no valid vectors found in table")));
	}

	/* Fetch cluster assignments */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d", cluster_str, tbl_str, 500000);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: davies_bouldin_index: failed to fetch cluster assignments")));
	}

	if (SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: vector count (%d) does not match cluster count (%d)",
						nvec, (int) SPI_processed)));
	}

	/* Extract cluster assignments and find max cluster ID */
	NDB_ALLOC(assignments, int, nvec);
	for (i = 0; i < nvec; i++)
	{
		int32		cluster_id;
		bool		isnull;
		Datum		cluster_datum;

		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL ||
			SPI_tuptable->tupdesc == NULL)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			for (j = 0; j < nvec; j++)
				NDB_FREE(data[j]);
			NDB_FREE(data);
			NDB_FREE(assignments);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			NDB_FREE(cluster_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: davies_bouldin_index: invalid SPI result at row %d", i)));
		}

		cluster_datum = SPI_getbinval(SPI_tuptable->vals[i],
									  SPI_tuptable->tupdesc,
									  1,
									  &isnull);

		if (isnull)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			for (j = 0; j < nvec; j++)
				NDB_FREE(data[j]);
			NDB_FREE(data);
			NDB_FREE(assignments);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			NDB_FREE(cluster_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: davies_bouldin_index: NULL cluster assignment at row %d", i)));
		}

		cluster_id = DatumGetInt32(cluster_datum);
		assignments[i] = cluster_id;
		if (cluster_id > max_cluster_id)
			max_cluster_id = cluster_id;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	if (max_cluster_id < 0)
	{
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
		NDB_FREE(assignments);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: no valid cluster assignments found")));
	}

	num_clusters = max_cluster_id + 1;

	/* Allocate cluster structures */
	NDB_ALLOC(cluster_sizes, int, num_clusters);
	NDB_ALLOC(cluster_scatter, double, num_clusters);
	NDB_ALLOC(centroids, float *, num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		NDB_ALLOC(centroids[c], float, dim);
	}

	/* Compute cluster centroids */
	compute_cluster_centroids(data, assignments, nvec, dim, num_clusters,
							  cluster_sizes, centroids);

	/* Compute cluster scatter (average distance from points to centroid) */
	for (c = 0; c < num_clusters; c++)
		cluster_scatter[c] = 0.0;

	for (i = 0; i < nvec; i++)
	{
		c = assignments[i];
		if (c >= 0 && c < num_clusters && cluster_sizes[c] > 0)
			cluster_scatter[c] += euclidean_distance(data[i], centroids[c], dim);
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
			cluster_scatter[c] /= (double) cluster_sizes[c];
	}

	/* Compute Davies-Bouldin index */
	{
		int			valid_clusters = 0;
		double		sum_dbi = 0.0;

		for (i = 0; i < num_clusters; i++)
		{
			double		max_ratio = 0.0;

			/* Skip clusters with less than 2 points */
			if (cluster_sizes[i] < 2)
				continue;

			for (j = 0; j < num_clusters; j++)
			{
				double		centroid_dist;
				double		ratio;

				if (i == j || cluster_sizes[j] < 2)
					continue;

				centroid_dist = euclidean_distance(centroids[i], centroids[j], dim);
				if (centroid_dist < 1e-10)
					continue;

				ratio = (cluster_scatter[i] + cluster_scatter[j]) / centroid_dist;
				if (ratio > max_ratio)
					max_ratio = ratio;
			}
			sum_dbi += max_ratio;
			valid_clusters++;
		}

		if (valid_clusters > 0)
			davies_bouldin = sum_dbi / (double) valid_clusters;
		else
			davies_bouldin = 0.0;	/* No valid clusters */
	}

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(assignments);
	NDB_FREE(cluster_sizes);
	NDB_FREE(cluster_scatter);
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centroids[c]);
	NDB_FREE(centroids);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);
	NDB_FREE(cluster_str);

	PG_RETURN_FLOAT8(davies_bouldin);
}

/*
 * GPU registration stub for Davies-Bouldin metric.
 * Since this is an evaluation metric rather than a trainable model,
 * we provide a minimal stub that satisfies the registration requirement.
 */
void
neurondb_gpu_register_davies_bouldin_model(void)
{
	/* Davies-Bouldin is a metric, not a trainable model.
	 * This stub exists to satisfy the registration call.
	 * Actual metric computation is handled in the clustering algorithms
	 * and the standalone davies_bouldin_index function above.
	 */
	static bool registered = false;

	if (registered)
		return;

	/* No GPU model ops needed for metrics - they're computed during evaluation */
	registered = true;
}

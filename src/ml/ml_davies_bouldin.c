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
 * Copyright (c) 2024-2026, neurondb, Inc.
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
	double		diff;
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		diff = (double) a[i] - (double) b[i];

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
	bool		isnull;
	char	   *cluster_str = NULL;
	char	   *col_str = NULL;
	char	   *tbl_str = NULL;
	double		centroid_dist;
	double		cluster_scatter_val;
	double		davies_bouldin = 0.0;
	double		diff;
	double		max_ratio;
	double		ratio;
	double		sum_dbi;
	double	   *cluster_scatter = NULL;
	Datum		cluster_datum;
	float	   **centroids = NULL;
	float	   **data = NULL;
	int			c;
	int			dim = 0;
	int			i;
	int			j;
	int			max_cluster_id = -1;
	int			nvec = 0;
	int			num_clusters = 0;
	int			ret;
	int			valid_clusters;
	int		   *assignments = NULL;
	int		   *cluster_sizes = NULL;
	int32		cluster_id;
	MemoryContext oldcontext;
	NdbSpiSession *spi_session = NULL;
	StringInfoData sql;
	text	   *cluster_col = NULL;
	text	   *table_name = NULL;
	text	   *vector_col = NULL;

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
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: no valid vectors found in table")));
	}

	/* Fetch cluster assignments. Use quote_identifier to prevent SQL injection. */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d",
					quote_identifier(cluster_str),
					quote_identifier(tbl_str),
					500000);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		for (i = 0; i < nvec; i++)
			nfree(data[i]);
		nfree(data);
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: davies_bouldin_index: failed to fetch cluster assignments")));
	}

	if (SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		for (i = 0; i < nvec; i++)
			nfree(data[i]);
		nfree(data);
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: vector count (%d) does not match cluster count (%d)",
						nvec, (int) SPI_processed)));
	}

	/* Extract cluster assignments and find max cluster ID */
	nalloc(assignments, int, nvec);
	for (i = 0; i < nvec; i++)
	{

		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL ||
			SPI_tuptable->tupdesc == NULL)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			for (j = 0; j < nvec; j++)
				nfree(data[j]);
			nfree(data);
			nfree(assignments);
			nfree(tbl_str);
			nfree(col_str);
			nfree(cluster_str);
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
				nfree(data[j]);
			nfree(data);
			nfree(assignments);
			nfree(tbl_str);
			nfree(col_str);
			nfree(cluster_str);
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
			nfree(data[i]);
		nfree(data);
		nfree(assignments);
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: davies_bouldin_index: no valid cluster assignments found")));
	}

	num_clusters = max_cluster_id + 1;

	/* Allocate cluster structures */
	nalloc(cluster_sizes, int, num_clusters);
	nalloc(cluster_scatter, double, num_clusters);
	nalloc(centroids, float *, num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		nalloc(centroids[c], float, dim);
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
		{
			diff = euclidean_distance(data[i], centroids[c], dim);
			cluster_scatter[c] += diff;
		}
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
		{
			cluster_scatter_val = cluster_scatter[c];
			cluster_scatter[c] = cluster_scatter_val / (double) cluster_sizes[c];
		}
	}

	/* Compute Davies-Bouldin index */
	valid_clusters = 0;
	sum_dbi = 0.0;

	for (i = 0; i < num_clusters; i++)
	{
		max_ratio = 0.0;

		/* Skip clusters with less than 2 points */
		if (cluster_sizes[i] < 2)
			continue;

		for (j = 0; j < num_clusters; j++)
		{
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

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(assignments);
	nfree(cluster_sizes);
	nfree(cluster_scatter);
	for (c = 0; c < num_clusters; c++)
		nfree(centroids[c]);
	nfree(centroids);
	nfree(tbl_str);
	nfree(col_str);
	nfree(cluster_str);

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

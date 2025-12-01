/*-------------------------------------------------------------------------
 *
 * ml_hierarchical.c
 *    Hierarchical agglomerative clustering.
 *
 * This module implements hierarchical clustering that builds a dendrogram
 * by iteratively merging the closest pair of clusters.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_hierarchical.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "ml_catalog.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <math.h>
#include <float.h>

/* Forward declaration */
static int	hierarchical_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *n_clusters_out, int *dim_out, char *linkage_out, int linkage_max);

/*--------------------
 * ClusterNode - Represents a cluster in the hierarchy.
 *--------------------
 */
typedef struct ClusterNode
{
	int			id;				/* Cluster ID */
	int		   *members;		/* Indices of data points in this cluster */
	int			size;			/* Number of points */
	double	   *centroid;		/* Centroid coordinates */
}			ClusterNode;

static inline double
euclidean_dist(const float *a, const float *b, int dim)
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

static void
compute_centroid(float **data, int *members, int size, int dim, double *centroid)
{
	int			i;
	int			d;

	for (d = 0; d < dim; d++)
		centroid[d] = 0.0;

	for (i = 0; i < size; i++)
	{
		for (d = 0; d < dim; d++)
			centroid[d] += (double) data[members[i]][d];
	}

	for (d = 0; d < dim; d++)
		centroid[d] /= (double) size;
}

static double
cluster_distance_average(float **data, ClusterNode * c1, ClusterNode * c2, int dim)
{
	double		sum = 0.0;
	int			i;
	int			j;
#include "ml_gpu_registry.h"
	for (i = 0; i < c1->size; i++)
	{
		for (j = 0; j < c2->size; j++)
			sum += euclidean_dist(data[c1->members[i]],
								  data[c2->members[j]],
								  dim);
	}
	return sum / ((double) c1->size * (double) c2->size);
}

static double
cluster_distance_complete(float **data, ClusterNode * c1, ClusterNode * c2, int dim)
{
	double		max_dist = 0.0;
	int			i;
	int			j;

	for (i = 0; i < c1->size; i++)
	{
		for (j = 0; j < c2->size; j++)
		{
			double		dist = euclidean_dist(data[c1->members[i]],
											  data[c2->members[j]],
											  dim);

			if (dist > max_dist)
				max_dist = dist;
		}
	}
	return max_dist;
}

static double
cluster_distance_single(float **data, ClusterNode * c1, ClusterNode * c2, int dim)
{
	double		min_dist = DBL_MAX;
	int			i;
	int			j;

	for (i = 0; i < c1->size; i++)
	{
		for (j = 0; j < c2->size; j++)
		{
			double		dist = euclidean_dist(data[c1->members[i]],
											  data[c2->members[j]],
											  dim);

			if (dist < min_dist)
				min_dist = dist;
		}
	}
	return min_dist;
}

PG_FUNCTION_INFO_V1(cluster_hierarchical);
PG_FUNCTION_INFO_V1(predict_hierarchical_cluster);
PG_FUNCTION_INFO_V1(evaluate_hierarchical_by_model_id);

Datum
cluster_hierarchical(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_clusters;
	text	   *linkage_text;
	char	   *tbl_str;
	char	   *col_str;
	char	   *linkage;
	float	  **data;
	int			nvec;
	int			dim;
	ClusterNode *clusters;
	int		   *cluster_assignments;
	int			n_active_clusters;
	int			iter,
				i,
				j,
				k,
				d;
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse input arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_clusters = PG_GETARG_INT32(2);
	linkage_text = PG_ARGISNULL(3) ? cstring_to_text("average")
		: PG_GETARG_TEXT_PP(3);

	if (num_clusters < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_clusters must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);
	linkage = text_to_cstring(linkage_text);

	/* Validate linkage */
	if (strcmp(linkage, "average") != 0 &&
		strcmp(linkage, "complete") != 0 &&
		strcmp(linkage, "single") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("linkage must be 'average', 'complete', or 'single'")));

	elog(DEBUG1,
		 "neurondb: hierarchical clustering (k=%d, linkage=%s)",
		 num_clusters, linkage);

	/* Fetch data from table */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_clusters)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("not enough vectors (%d) for %d clusters",
						nvec, num_clusters)));

	/*
	 * Hierarchical clustering has O(n²) complexity - limit to reasonable
	 * size
	 */
	{
		int			max_hierarchical_points = 10000;	/* Hard limit for
														 * hierarchical
														 * clustering */
		int			original_nvec = nvec;

		if (nvec > max_hierarchical_points)
		{
			elog(WARNING,
				 "hierarchical clustering on %d points is computationally infeasible (O(n²) complexity), limiting to %d points. Consider using K-means or Mini-batch K-means for large datasets",
				 nvec, max_hierarchical_points);

			/* Sample the data */
			{
				NDB_DECLARE(float **, sampled_data);
				int			sample_step;
				int			sampled_idx = 0;
				int			sample_j;

				NDB_ALLOC(sampled_data, float *, max_hierarchical_points);
				sample_step = original_nvec / max_hierarchical_points;

				for (sample_j = 0; sample_j < original_nvec && sampled_idx < max_hierarchical_points; sample_j += sample_step)
				{
					NDB_DECLARE(float *, sampled_row);
					NDB_ALLOC(sampled_row, float, dim);
					sampled_data[sampled_idx] = sampled_row;
					memcpy(sampled_data[sampled_idx], data[sample_j], sizeof(float) * dim);
					sampled_idx++;
				}

				/* Free original data */
				for (sample_j = 0; sample_j < original_nvec; sample_j++)
					NDB_FREE(data[sample_j]);
				NDB_FREE(data);

				/* Use sampled data */
				data = sampled_data;
				nvec = sampled_idx;
			}

			elog(INFO,
				 "hierarchical clustering: using %d sampled points from %d total points",
				 nvec, original_nvec);
		}
		else if (nvec > 5000)
		{
			elog(WARNING,
				 "hierarchical clustering on %d points may be slow, consider K-means", nvec);
		}
	}

	/* Initialize clusters: each point as singleton cluster */
	NDB_ALLOC(clusters, ClusterNode, nvec);
	for (i = 0; i < nvec; i++)
	{
		clusters[i].id = i;
		clusters[i].size = 1;
		{
			NDB_DECLARE(int *, members);
			NDB_ALLOC(members, int, 1);
			clusters[i].members = members;
			clusters[i].members[0] = i;
		}
		{
			NDB_DECLARE(double *, centroid);
			NDB_ALLOC(centroid, double, dim);
			clusters[i].centroid = centroid;
			for (d = 0; d < dim; d++)
				clusters[i].centroid[d] = (double) data[i][d];
		}
	}

	n_active_clusters = nvec;

	/* Agglomerative merge */
	for (iter = 0; iter < nvec - num_clusters; iter++)
	{
		double		min_dist = DBL_MAX;
		int			merge_i = -1;
		int			merge_j = -1;

		for (i = 0; i < nvec; i++)
		{
			double		dist;

			if (clusters[i].size == 0)
				continue;

			for (j = i + 1; j < nvec; j++)
			{
				if (clusters[j].size == 0)
					continue;

				if (strcmp(linkage, "average") == 0)
					dist = cluster_distance_average(data, &clusters[i], &clusters[j], dim);
				else if (strcmp(linkage, "complete") == 0)
					dist = cluster_distance_complete(data, &clusters[i], &clusters[j], dim);
				else
					dist = cluster_distance_single(data, &clusters[i], &clusters[j], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					merge_i = i;
					merge_j = j;
				}
			}
		}

		if (merge_i < 0 || merge_j < 0)
			break;

		{
			int			new_size;
			NDB_DECLARE(int *, new_members);

			new_size = clusters[merge_i].size + clusters[merge_j].size;
			NDB_ALLOC(new_members, int, new_size);

			for (k = 0; k < clusters[merge_i].size; k++)
				new_members[k] = clusters[merge_i].members[k];
			for (k = 0; k < clusters[merge_j].size; k++)
				new_members[clusters[merge_i].size + k] = clusters[merge_j].members[k];

			NDB_FREE(clusters[merge_i].members);
			clusters[merge_i].members = new_members;
			clusters[merge_i].size = new_size;

			compute_centroid(data,
							 clusters[merge_i].members,
							 clusters[merge_i].size,
							 dim,
							 clusters[merge_i].centroid);

			NDB_FREE(clusters[merge_j].members);
			NDB_FREE(clusters[merge_j].centroid);
			clusters[merge_j].size = 0;
			clusters[merge_j].members = NULL;
			clusters[merge_j].centroid = NULL;

			n_active_clusters--;
		}

		if ((iter + 1) % 100 == 0)
			elog(DEBUG1,
				 "neurondb: hierarchical merge iteration %d (%d clusters remain)",
				 iter + 1, n_active_clusters);
	}

	/* Assign cluster labels: 1-based */
	NDB_ALLOC(cluster_assignments, int, nvec);
	{
		int			cluster_id = 1;

		for (i = 0; i < nvec; i++)
		{
			if (clusters[i].size > 0)
			{
				for (k = 0; k < clusters[i].size; k++)
					cluster_assignments[clusters[i].members[k]] = cluster_id;
				cluster_id++;
			}
		}
	}

	/* Build result array */
	NDB_ALLOC(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Int32GetDatum(cluster_assignments[i]);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(result_datums, nvec, INT4OID, typlen, typbyval, typalign);

	/* Free memory */
	for (i = 0; i < nvec; i++)
	{
		NDB_FREE(data[i]);
		if (clusters[i].size > 0)
		{
			NDB_FREE(clusters[i].members);
			NDB_FREE(clusters[i].centroid);
		}
	}
	NDB_FREE(data);
	NDB_FREE(clusters);
	NDB_FREE(cluster_assignments);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);
	NDB_FREE(linkage);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * predict_hierarchical_cluster
 *      Predicts cluster assignment for new data points using a trained hierarchical model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: int4 cluster_id
 */
Datum
predict_hierarchical_cluster(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *features_array;
	int			n_features;
	int			cluster_id = -1;
	NDB_DECLARE(float *, features);

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);
	(void) model_id;			/* Not used in simplified implementation */

	/* Extract features from array */
	{
		Oid			elmtype = ARR_ELEMTYPE(features_array);
		int16		typlen;
		bool		typbyval;
		char		typalign;
		Datum	   *elems;
		bool	   *nulls;
		int			n_elems;
		int			i;

		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);
		deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
						  &elems, &nulls, &n_elems);

		NDB_ALLOC(features, float, n_elems);
		n_features = n_elems;
		(void) n_features;		/* Not used in simplified implementation */

		for (i = 0; i < n_elems; i++)
			features[i] = DatumGetFloat4(elems[i]);
	}

	/* Load model from catalog and find closest cluster */
	{
		bytea	   *model_data = NULL;
		Jsonb	   *parameters = NULL;
		Jsonb	   *metrics = NULL;
		float	  **cluster_centers = NULL;
		int			n_clusters = 0;
		int			model_dim = 0;
		char		linkage[16] = "average";
		double		min_distance = DBL_MAX;
		int			closest_cluster = 0;
		int			c;

		/* Fetch model from catalog */
		if (!ml_catalog_fetch_model_payload(model_id, &model_data, &parameters, &metrics))
		{
			NDB_FREE(features);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_hierarchical_by_model_id: model %d not found", model_id)));
		}

		if (model_data == NULL)
		{
			NDB_FREE(features);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_hierarchical_by_model_id: model %d has no data", model_id)));
		}

		/* Deserialize model to get cluster centers */
		if (hierarchical_model_deserialize_from_bytea(model_data,
													  &cluster_centers,
													  &n_clusters,
													  &model_dim,
													  linkage,
													  sizeof(linkage)) != 0)
		{
			NDB_FREE(features);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_hierarchical_by_model_id: failed to deserialize model %d", model_id)));
		}

		/* Validate dimension match */
		if (model_dim != n_features)
		{
			/* Cleanup cluster centers */
			if (cluster_centers != NULL)
			{
				for (c = 0; c < n_clusters; c++)
					NDB_FREE(cluster_centers[c]);
				NDB_FREE(cluster_centers);
			}
			NDB_FREE(features);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_hierarchical_by_model_id: feature dimension mismatch: model expects %d, got %d",
							model_dim, n_features)));
		}

		/* Find closest cluster center */
		for (c = 0; c < n_clusters; c++)
		{
			double		distance = euclidean_dist(features, cluster_centers[c], n_features);

			if (distance < min_distance)
			{
				min_distance = distance;
				closest_cluster = c;
			}
		}

		cluster_id = closest_cluster;

		/* Cleanup cluster centers */
		if (cluster_centers != NULL)
		{
			for (c = 0; c < n_clusters; c++)
				NDB_FREE(cluster_centers[c]);
			NDB_FREE(cluster_centers);
		}
	}

	NDB_FREE(features);

	PG_RETURN_INT32(cluster_id);
}

/*
 * evaluate_hierarchical_by_model_id
 *      Evaluates hierarchical clustering quality on a dataset.
 *      Arguments: int4 model_id, text table_name, text feature_col, int4 n_clusters
 *      Returns: jsonb with clustering metrics
 */
Datum
evaluate_hierarchical_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	int32		n_clusters;
	char	   *tbl_str;
	char	   *feat_str;
	StringInfoData query;
	int			ret;
	int			n_points = 0;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext;
	double		silhouette_score;
	double		calinski_harabasz;
	int			n_clusters_found;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext_spi;

	/* Validate arguments */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_hierarchical_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_hierarchical_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);
	(void) model_id;			/* Not used in simplified implementation */

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_hierarchical_by_model_id: table_name, feature_col, and n_clusters are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	n_clusters = PG_GETARG_INT32(3);

	if (n_clusters < 2 || n_clusters > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_clusters must be between 2 and 100")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
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
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_hierarchical_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < n_clusters)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_hierarchical_by_model_id: need at least %d points for %d clusters, got %d",
						n_clusters, n_clusters, n_points)));
	}

	/* Load model to get centroids */
	{
		bytea	   *model_payload = NULL;
		Jsonb	   *model_parameters = NULL;
		float	  **centers = NULL;
		int			model_dim = 0;
		char		linkage_str[16] = "average";
		float	  **data = NULL;
		int			i,
					j,
					c,
					d;
		int			dim;
		double		global_mean[1000];	/* Max dim support */
		double		wss = 0.0;	/* Within-cluster sum of squares */
		double		bss = 0.0;	/* Between-cluster sum of squares */
		int			valid_count = 0;
		double		sum_silhouette = 0.0;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id, &model_payload, &model_parameters, NULL))
		{
			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				if (hierarchical_model_deserialize_from_bytea(model_payload,
															  &centers,
															  &n_clusters_found,
															  &model_dim,
															  linkage_str,
															  sizeof(linkage_str)) == 0)
				{
					/* Fetch data points */
					data = neurondb_fetch_vectors_from_table(tbl_str, feat_str, &n_points, &dim);

					if (data != NULL && n_points > 0 && dim == model_dim)
					{
						/* Allocate arrays */
						NDB_DECLARE(int *, assignments);
						NDB_DECLARE(int *, cluster_sizes);
						NDB_DECLARE(double *, a_scores);
						NDB_DECLARE(double *, b_scores);
						NDB_DECLARE(double *, cluster_means);
						NDB_ALLOC(assignments, int, n_points);
						NDB_ALLOC(cluster_sizes, int, n_clusters_found);
						NDB_ALLOC(a_scores, double, n_points);
						NDB_ALLOC(b_scores, double, n_points);
						NDB_ALLOC(cluster_means, double, n_clusters_found * dim);

						/* Assign points to nearest centroids */
						for (i = 0; i < n_points; i++)
						{
							double		min_dist = DBL_MAX;
							int			best = 0;

							for (c = 0; c < n_clusters_found; c++)
							{
								double		dist = euclidean_dist(data[i], centers[c], dim);

								if (dist < min_dist)
								{
									min_dist = dist;
									best = c;
								}
							}
							assignments[i] = best;
							cluster_sizes[best]++;
							wss += min_dist * min_dist; /* Sum of squared
														 * distances */
						}

						/* Compute global mean */
						for (d = 0; d < dim; d++)
						{
							global_mean[d] = 0.0;
							for (i = 0; i < n_points; i++)
								global_mean[d] += (double) data[i][d];
							global_mean[d] /= (double) n_points;
						}

						/* Compute cluster means */
						for (c = 0; c < n_clusters_found; c++)
						{
							if (cluster_sizes[c] > 0)
							{
								for (d = 0; d < dim; d++)
								{
									cluster_means[c * dim + d] = 0.0;
									for (i = 0; i < n_points; i++)
									{
										if (assignments[i] == c)
											cluster_means[c * dim + d] += (double) data[i][d];
									}
									cluster_means[c * dim + d] /= (double) cluster_sizes[c];
								}
							}
						}

						/* Compute between-cluster sum of squares */
						for (c = 0; c < n_clusters_found; c++)
						{
							if (cluster_sizes[c] > 0)
							{
								double		cluster_ss = 0.0;

								for (d = 0; d < dim; d++)
								{
									double		diff = cluster_means[c * dim + d] - global_mean[d];

									cluster_ss += diff * diff;
								}
								bss += (double) cluster_sizes[c] * cluster_ss;
							}
						}

						/* Compute silhouette score */
						for (i = 0; i < n_points; i++)
						{
							int			my_cluster = assignments[i];
							int			same_count = 0;
							double		same_dist = 0.0;
							double		min_other_dist = DBL_MAX;

							if (cluster_sizes[my_cluster] <= 1)
							{
								a_scores[i] = 0.0;
								b_scores[i] = 0.0;
								continue;
							}

							/* Average distance to same cluster */
							for (j = 0; j < n_points; j++)
							{
								if (i == j)
									continue;
								if (assignments[j] == my_cluster)
								{
									same_dist += euclidean_dist(data[i], data[j], dim);
									same_count++;
								}
							}
							if (same_count > 0)
								a_scores[i] = same_dist / (double) same_count;
							else
								a_scores[i] = 0.0;

							/* Minimum average distance to other clusters */
							for (c = 0; c < n_clusters_found; c++)
							{
								if (c == my_cluster || cluster_sizes[c] == 0)
									continue;
								{
									double		other_dist = 0.0;
									int			other_count = 0;

									for (j = 0; j < n_points; j++)
									{
										if (assignments[j] == c)
										{
											other_dist += euclidean_dist(data[i], data[j], dim);
											other_count++;
										}
									}
									if (other_count > 0)
									{
										other_dist /= (double) other_count;
										if (other_dist < min_other_dist)
											min_other_dist = other_dist;
									}
								}
							}
							b_scores[i] = min_other_dist;
						}

						/* Compute average silhouette */
						for (i = 0; i < n_points; i++)
						{
							double		max_ab = (a_scores[i] > b_scores[i]) ? a_scores[i] : b_scores[i];

							if (max_ab > 0.0)
							{
								double		s = (b_scores[i] - a_scores[i]) / max_ab;

								sum_silhouette += s;
								valid_count++;
							}
						}
						if (valid_count > 0)
							silhouette_score = sum_silhouette / (double) valid_count;
						else
							silhouette_score = 0.0;

						/*
						 * Compute Calinski-Harabasz index: CH = (BSS/(k-1)) /
						 * (WSS/(n-k))
						 */
						if (n_clusters_found > 1 && n_points > n_clusters_found && wss > 0.0)
						{
							double		bss_norm = bss / (double) (n_clusters_found - 1);
							double		wss_norm = wss / (double) (n_points - n_clusters_found);

							calinski_harabasz = bss_norm / wss_norm;
						}
						else
						{
							calinski_harabasz = 0.0;
						}

						/* Cleanup */
						for (i = 0; i < n_points; i++)
							NDB_FREE(data[i]);
						NDB_FREE(data);
						for (c = 0; c < n_clusters_found; c++)
							NDB_FREE(centers[c]);
						NDB_FREE(centers);
						NDB_FREE(assignments);
						NDB_FREE(cluster_sizes);
						NDB_FREE(a_scores);
						NDB_FREE(b_scores);
						NDB_FREE(cluster_means);
					}
					else
					{
						/* Failed to load data or dimension mismatch */
						if (data != NULL)
						{
							for (i = 0; i < n_points; i++)
								NDB_FREE(data[i]);
							NDB_FREE(data);
						}
						for (c = 0; c < n_clusters_found; c++)
							NDB_FREE(centers[c]);
						NDB_FREE(centers);
						silhouette_score = 0.0;
						calinski_harabasz = 0.0;
						n_clusters_found = n_clusters;
					}
				}
				else
				{
					silhouette_score = 0.0;
					calinski_harabasz = 0.0;
					n_clusters_found = n_clusters;
				}
			}
			else
			{
				silhouette_score = 0.0;
				calinski_harabasz = 0.0;
				n_clusters_found = n_clusters;
			}
			if (model_payload)
				NDB_FREE(model_payload);
			if (model_parameters)
				NDB_FREE(model_parameters);
		}
		else
		{
			/* Model not found - use defaults */
			silhouette_score = 0.0;
			calinski_harabasz = 0.0;
			n_clusters_found = n_clusters;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"silhouette_score\":%.6f,\"calinski_harabasz\":%.6f,\"n_clusters\":%d,\"n_points\":%d}",
					 silhouette_score, calinski_harabasz, n_clusters_found, n_points);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	NDB_FREE(jsonbuf.data);

	/* Cleanup */
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);

	PG_RETURN_JSONB_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Hierarchical Clustering
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct HierarchicalGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			n_clusters;
	int			dim;
	int			n_samples;
	char		linkage[16];
}			HierarchicalGpuModelState;

static bytea *
hierarchical_model_serialize_to_bytea(float **cluster_centers, int n_clusters, int dim, const char *linkage)
{
	StringInfoData buf;
	int			i,
				j;
	int			total_size;
	bytea	   *result;
	int			linkage_len;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &n_clusters, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &dim, sizeof(int));
	linkage_len = strlen(linkage);
	appendBinaryStringInfo(&buf, (char *) &linkage_len, sizeof(int));
	appendBinaryStringInfo(&buf, linkage, linkage_len);

	for (i = 0; i < n_clusters; i++)
		for (j = 0; j < dim; j++)
			appendBinaryStringInfo(&buf, (char *) &cluster_centers[i][j], sizeof(float));

	total_size = VARHDRSZ + buf.len;
	{
		NDB_DECLARE(char *, result_bytes);
		NDB_ALLOC(result_bytes, char, total_size);
		result = (bytea *) result_bytes;
		SET_VARSIZE(result, total_size);
		memcpy(VARDATA(result), buf.data, buf.len);
		NDB_FREE(buf.data);
	}

	return result;
}

static int
hierarchical_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *n_clusters_out, int *dim_out, char *linkage_out, int linkage_max)
{
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	float	  **centers;
	int			linkage_len;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 3)
		return -1;

	buf = VARDATA(data);
	memcpy(n_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&linkage_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (linkage_len >= linkage_max)
		return -1;
	memcpy(linkage_out, buf + offset, linkage_len);
	linkage_out[linkage_len] = '\0';
	offset += linkage_len;

	if (*n_clusters_out < 0 || *n_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	NDB_ALLOC(centers, float *, *n_clusters_out);
	for (i = 0; i < *n_clusters_out; i++)
	{
		NDB_DECLARE(float *, center_row);
		NDB_ALLOC(center_row, float, *dim_out);
		centers[i] = center_row;
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
hierarchical_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	HierarchicalGpuModelState *state;
	float	  **data = NULL;
	int		   *cluster_assignments = NULL;
	float	  **cluster_centers = NULL;
	int		   *cluster_sizes = NULL;
	int			num_clusters = 8;
	char		linkage[16] = "average";
	int			nvec = 0;
	int			dim = 0;
	int			i,
				c,
				d;
	ClusterNode *clusters = NULL;
	int			n_active_clusters = 0;
	int			iter,
				j,
				k;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_train: invalid parameters");
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
				if (strcmp(key, "n_clusters") == 0 && v.type == jbvNumeric)
					num_clusters = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	 NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "linkage") == 0 && v.type == jbvString)
					strncpy(linkage, v.val.string.val, sizeof(linkage) - 1);
				NDB_FREE(key);
			}
		}
	}

	if (num_clusters < 1)
		num_clusters = 8;
	if (strlen(linkage) == 0)
		strcpy(linkage, "average");

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	if (nvec < num_clusters)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_train: not enough samples");
		return false;
	}

	/* Limit size for performance */
	if (nvec > 10000)
		nvec = 10000;

	NDB_ALLOC(data, float *, nvec);
	for (i = 0; i < nvec; i++)
	{
		NDB_DECLARE(float *, data_row);
		NDB_ALLOC(data_row, float, dim);
		data[i] = data_row;
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	/* Initialize clusters: each point as singleton */
	NDB_ALLOC(clusters, ClusterNode, nvec);
	for (i = 0; i < nvec; i++)
	{
		int		   *members;
		double	   *centroid;
		clusters[i].id = i;
		clusters[i].size = 1;
		{
			NDB_ALLOC(members, int, 1);
			clusters[i].members = members;
			clusters[i].members[0] = i;
		}
		{
			NDB_ALLOC(centroid, double, dim);
			NDB_ALLOC(centroid, double, dim);
			clusters[i].centroid = centroid;
			for (d = 0; d < dim; d++)
				clusters[i].centroid[d] = (double) data[i][d];
		}
	}

	n_active_clusters = nvec;

	/* Agglomerative merge */
	for (iter = 0; iter < nvec - num_clusters && n_active_clusters > num_clusters; iter++)
	{
		double		min_dist = DBL_MAX;
		int			merge_i = -1;
		int			merge_j = -1;

		for (i = 0; i < nvec; i++)
		{
			if (clusters[i].size == 0)
				continue;

			for (j = i + 1; j < nvec; j++)
			{
				double		dist;

				if (clusters[j].size == 0)
					continue;

				if (strcmp(linkage, "average") == 0)
					dist = cluster_distance_average(data, &clusters[i], &clusters[j], dim);
				else if (strcmp(linkage, "complete") == 0)
					dist = cluster_distance_complete(data, &clusters[i], &clusters[j], dim);
				else
					dist = cluster_distance_single(data, &clusters[i], &clusters[j], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					merge_i = i;
					merge_j = j;
				}
			}
		}

		if (merge_i < 0 || merge_j < 0)
			break;

		{
			int			new_size = clusters[merge_i].size + clusters[merge_j].size;
			NDB_DECLARE(int *, new_members);
			NDB_ALLOC(new_members, int, new_size);

			for (k = 0; k < clusters[merge_i].size; k++)
				new_members[k] = clusters[merge_i].members[k];
			for (k = 0; k < clusters[merge_j].size; k++)
				new_members[clusters[merge_i].size + k] = clusters[merge_j].members[k];

			NDB_FREE(clusters[merge_i].members);
			clusters[merge_i].members = new_members;
			clusters[merge_i].size = new_size;

			compute_centroid(data, clusters[merge_i].members, clusters[merge_i].size, dim, clusters[merge_i].centroid);

			NDB_FREE(clusters[merge_j].members);
			NDB_FREE(clusters[merge_j].centroid);
			clusters[merge_j].size = 0;
			clusters[merge_j].members = NULL;
			clusters[merge_j].centroid = NULL;

			n_active_clusters--;
		}
	}

	/* Extract cluster centroids */
	NDB_ALLOC(cluster_assignments, int, nvec);
	NDB_ALLOC(cluster_centers, float *, num_clusters);
	NDB_ALLOC(cluster_sizes, int, num_clusters);

	c = 0;
	for (i = 0; i < nvec && c < num_clusters; i++)
	{
		if (clusters[i].size > 0)
		{
			NDB_DECLARE(float *, center_row);
			NDB_ALLOC(center_row, float, dim);
			cluster_centers[c] = center_row;
			for (d = 0; d < dim; d++)
				cluster_centers[c][d] = (float) clusters[i].centroid[d];

			for (j = 0; j < clusters[i].size; j++)
			{
				int			point_idx = clusters[i].members[j];

				cluster_assignments[point_idx] = c;
				cluster_sizes[c]++;
			}
			c++;
		}
	}

	/* Serialize model */
	model_data = hierarchical_model_serialize_to_bytea(cluster_centers, num_clusters, dim, linkage);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"n_clusters\":%d,\"linkage\":\"%s\",\"dim\":%d,\"n_samples\":%d}",
					 num_clusters, linkage, dim, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	NDB_ALLOC(state, HierarchicalGpuModelState, 1);
	state->model_blob = model_data;
	state->metrics = metrics;
	state->n_clusters = num_clusters;
	state->dim = dim;
	state->n_samples = nvec;
	strncpy(state->linkage, linkage, sizeof(state->linkage) - 1);

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	/* Cleanup */
	for (i = 0; i < nvec; i++)
	{
		NDB_FREE(data[i]);
		if (clusters[i].members != NULL)
			NDB_FREE(clusters[i].members);
		if (clusters[i].centroid != NULL)
			NDB_FREE(clusters[i].centroid);
	}
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(cluster_centers[c]);
	NDB_FREE(data);
	NDB_FREE(clusters);
	NDB_FREE(cluster_centers);
	NDB_FREE(cluster_sizes);
	NDB_FREE(cluster_assignments);

	return true;
}

static bool
hierarchical_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
						 float *output, int output_dim, char **errstr)
{
	const		HierarchicalGpuModelState *state;
	float	  **centers = NULL;
	int			n_clusters = 0;
	int			dim = 0;
	char		linkage[16];
	int			c;
	double		min_dist = DBL_MAX;
	int			best_cluster = -1;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_predict: model not ready");
		return false;
	}

	state = (const HierarchicalGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_predict: model blob is NULL");
		return false;
	}

	if (hierarchical_model_deserialize_from_bytea(state->model_blob,
												  &centers, &n_clusters, &dim, linkage, sizeof(linkage)) != 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_predict: failed to deserialize");
		return false;
	}

	if (input_dim != dim)
	{
		for (c = 0; c < n_clusters; c++)
			NDB_FREE(centers[c]);
		NDB_FREE(centers);
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_predict: dimension mismatch");
		return false;
	}

	for (c = 0; c < n_clusters; c++)
	{
		double		dist = sqrt(neurondb_l2_distance_squared(input, centers[c], dim));

		if (dist < min_dist)
		{
			min_dist = dist;
			best_cluster = c;
		}
	}

	output[0] = (float) best_cluster;

	for (c = 0; c < n_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(centers);

	return true;
}

static bool
hierarchical_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
						  MLGpuMetrics * out, char **errstr)
{
	const		HierarchicalGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_evaluate: invalid model");
		return false;
	}

	state = (const HierarchicalGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"hierarchical\",\"storage\":\"cpu\","
					 "\"n_clusters\":%d,\"linkage\":\"%s\",\"dim\":%d,\"n_samples\":%d}",
					 state->n_clusters > 0 ? state->n_clusters : 0,
					 state->linkage[0] ? state->linkage : "average",
					 state->dim > 0 ? state->dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
hierarchical_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
						   Jsonb * *metadata_out, char **errstr)
{
	const		HierarchicalGpuModelState *state;
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
			*errstr = pstrdup("hierarchical_gpu_serialize: invalid model");
		return false;
	}

	state = (const HierarchicalGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_serialize: model blob is NULL");
		return false;
	}

	{
		char	   *payload_bytes;
		payload_size = VARSIZE(state->model_blob);
		NDB_ALLOC(payload_bytes, char, payload_size);
		payload_copy = (bytea *) payload_bytes;
		memcpy(payload_copy, state->model_blob, payload_size);
	}

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
hierarchical_gpu_deserialize(MLGpuModel * model, const bytea * payload,
							 const Jsonb * metadata, char **errstr)
{
	HierarchicalGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	float	  **centers = NULL;
	int			n_clusters = 0;
	int			dim = 0;
	char		linkage[16];
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_deserialize: invalid parameters");
		return false;
	}

	{
		char	   *payload_bytes;
		payload_size = VARSIZE(payload);
		NDB_ALLOC(payload_bytes, char, payload_size);
		payload_copy = (bytea *) payload_bytes;
		memcpy(payload_copy, payload, payload_size);
	}

	if (hierarchical_model_deserialize_from_bytea(payload_copy,
												  &centers, &n_clusters, &dim, linkage, sizeof(linkage)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("hierarchical_gpu_deserialize: failed to deserialize");
		return false;
	}

	for (int c = 0; c < n_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(centers);

	NDB_ALLOC(state, HierarchicalGpuModelState, 1);
	state->model_blob = payload_copy;
	state->n_clusters = n_clusters;
	state->dim = dim;
	state->n_samples = 0;
	strncpy(state->linkage, linkage, sizeof(state->linkage) - 1);

	if (metadata != NULL)
	{
		int			metadata_size;
		char	   *metadata_bytes;
		Jsonb	   *metadata_copy;

		metadata_size = VARSIZE(metadata);
		NDB_ALLOC(metadata_bytes, char, metadata_size);
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
hierarchical_gpu_destroy(MLGpuModel * model)
{
	HierarchicalGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (HierarchicalGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps hierarchical_gpu_model_ops = {
	.algorithm = "hierarchical",
	.train = hierarchical_gpu_train,
	.predict = hierarchical_gpu_predict,
	.evaluate = hierarchical_gpu_evaluate,
	.serialize = hierarchical_gpu_serialize,
	.deserialize = hierarchical_gpu_deserialize,
	.destroy = hierarchical_gpu_destroy,
};

/* Forward declaration to avoid missing prototype warning */
extern void neurondb_gpu_register_hierarchical_model(void);

void
neurondb_gpu_register_hierarchical_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&hierarchical_gpu_model_ops);
	registered = true;
}

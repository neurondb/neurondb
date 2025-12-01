/*-------------------------------------------------------------------------
 *
 * ml_gmm.c
 *    Gaussian mixture model for soft clustering.
 *
 * This module implements GMM using expectation-maximization for probabilistic
 * cluster assignments and density estimation.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_gmm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_json.h"
#include "ml_catalog.h"
#include "lib/stringinfo.h"
#include "utils/jsonb.h"
#include "common/jsonapi.h"
#include "vector/vector_types.h"
#include "neurondb_cuda_gmm.h"
#include "ml_gpu_registry.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

#include <math.h>
#include <float.h>

#define GMM_EPSILON		1e-6	/* Regularization for covariance */
#define GMM_MIN_PROB	1e-10	/* Minimum probability to avoid log(0) */

/* GMM model structure */
typedef struct GMMModel
{
	int			k;
	int			dim;
	double	   *mixing_coeffs;
	double	  **means;
	double	  **variances;
}			GMMModel;

/*
 * Compute Gaussian probability density (diagonal covariance)
 */
static double
gaussian_pdf(const float *x, const double *mean, const double *variance, int dim)
{
	double		log_likelihood = 0.0;
	double		log_det = 0.0;
	int			d;

	for (d = 0; d < dim; d++)
	{
		double		diff = (double) x[d] - mean[d];
		double		var = variance[d] + GMM_EPSILON;

		log_likelihood -= 0.5 * (diff * diff) / var;
		log_det += log(var);
	}

	log_likelihood -= 0.5 * (dim * log(2.0 * M_PI) + log_det);

	return exp(log_likelihood);
}

PG_FUNCTION_INFO_V1(cluster_gmm);

Datum
cluster_gmm(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_components;
	int			max_iters;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	GMMModel	model;
	double	  **responsibilities;
	double		log_likelihood,
				prev_log_likelihood;
	int			iter,
				i,
				k,
				d;
	ArrayType  *result;
	Datum	   *result_datums;
	int			dims[2];
	int			lbs[2];

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_components = PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	if (num_components < 1 || num_components > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_components must be between 1 and 100")));

	if (max_iters < 1)
		max_iters = 100;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_components)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough vectors (%d) for %d components",
						nvec, num_components)));

	model.k = num_components;
	model.dim = dim;
	model.mixing_coeffs = (double *) palloc(sizeof(double) * num_components);
	model.means = (double **) palloc(sizeof(double *) * num_components);
	model.variances = (double **) palloc(sizeof(double *) * num_components);

	for (k = 0; k < num_components; k++)
	{
		int			idx;

		model.means[k] = (double *) palloc(sizeof(double) * dim);
		model.variances[k] = (double *) palloc(sizeof(double) * dim);

		idx = rand() % nvec;

		for (d = 0; d < dim; d++)
			model.means[k][d] = (double) data[idx][d];

		for (d = 0; d < dim; d++)
			model.variances[k][d] = 1.0;

		model.mixing_coeffs[k] = 1.0 / num_components;
	}

	responsibilities = (double **) palloc(sizeof(double *) * nvec);
	for (i = 0; i < nvec; i++)
		responsibilities[i] = (double *) palloc(sizeof(double) * num_components);

	prev_log_likelihood = -DBL_MAX;

	for (iter = 0; iter < max_iters; iter++)
	{
		log_likelihood = 0.0;

		for (i = 0; i < nvec; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < num_components; k++)
			{
				double		pdf = gaussian_pdf(data[i],
											   model.means[k],
											   model.variances[k],
											   dim);

				responsibilities[i][k] = model.mixing_coeffs[k] * pdf;
				sum += responsibilities[i][k];
			}

			if (sum < GMM_MIN_PROB)
				sum = GMM_MIN_PROB;

			for (k = 0; k < num_components; k++)
			{
				responsibilities[i][k] /= sum;
				if (responsibilities[i][k] < GMM_MIN_PROB)
					responsibilities[i][k] = GMM_MIN_PROB;
			}

			log_likelihood += log(sum);
		}

		log_likelihood /= nvec;

		if (fabs(log_likelihood - prev_log_likelihood) < 1e-6)
			break;
		prev_log_likelihood = log_likelihood;

		{
			double	   *N_k = (double *) palloc0(sizeof(double) * num_components);

			for (k = 0; k < num_components; k++)
			{
				for (i = 0; i < nvec; i++)
					N_k[k] += responsibilities[i][k];

				if (N_k[k] < GMM_MIN_PROB)
					N_k[k] = GMM_MIN_PROB;
			}

			for (k = 0; k < num_components; k++)
				model.mixing_coeffs[k] = N_k[k] / nvec;

			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.means[k][d] = 0.0;

				for (i = 0; i < nvec; i++)
					for (d = 0; d < dim; d++)
						model.means[k][d] +=
							responsibilities[i][k] * data[i][d];

				for (d = 0; d < dim; d++)
					model.means[k][d] /= N_k[k];
			}

			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.variances[k][d] = 0.0;

				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < dim; d++)
					{
						double		diff = data[i][d] - model.means[k][d];

						model.variances[k][d] +=
							responsibilities[i][k] * diff * diff;
					}
				}

				for (d = 0; d < dim; d++)
					model.variances[k][d] =
						(model.variances[k][d] / N_k[k]) + GMM_EPSILON;
			}

			NDB_FREE(N_k);
		}

		if ((iter + 1) % 10 == 0)
			elog(DEBUG1,
				 "neurondb: GMM iteration %d, log_likelihood=%.6f",
				 iter + 1, log_likelihood);
	}

	result_datums = (Datum *) palloc(sizeof(Datum) * nvec * num_components);
	for (i = 0; i < nvec; i++)
	{
		for (k = 0; k < num_components; k++)
			result_datums[i * num_components + k] =
				Float8GetDatum(responsibilities[i][k]);
	}

	dims[0] = nvec;
	dims[1] = num_components;
	lbs[0] = 1;
	lbs[1] = 1;

	result = construct_md_array(result_datums,
								NULL,
								2,
								dims,
								lbs,
								FLOAT8OID,
								sizeof(float8),
								FLOAT8PASSBYVAL,
								'd');

	for (i = 0; i < nvec; i++)
	{
		NDB_FREE(data[i]);
		NDB_FREE(responsibilities[i]);
	}
	NDB_FREE(data);
	NDB_FREE(responsibilities);

	for (k = 0; k < num_components; k++)
	{
		NDB_FREE(model.means[k]);
		NDB_FREE(model.variances[k]);
	}
	NDB_FREE(model.means);
	NDB_FREE(model.variances);
	NDB_FREE(model.mixing_coeffs);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Serialize GMMModel to bytea for storage
 */
static bytea *
gmm_model_serialize_to_bytea(const GMMModel * model, uint8 training_backend)
{
	StringInfoData buf;
	int			i,
				j;
	int			total_size;
	bytea	   *result;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: gmm_model_serialize_to_bytea: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	initStringInfo(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	appendBinaryStringInfo(&buf, (char *) &training_backend, sizeof(uint8));

	appendBinaryStringInfo(&buf, (char *) &model->k, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &model->dim, sizeof(int));

	for (i = 0; i < model->k; i++)
		appendBinaryStringInfo(&buf, (char *) &model->mixing_coeffs[i], sizeof(double));

	for (i = 0; i < model->k; i++)
		for (j = 0; j < model->dim; j++)
			appendBinaryStringInfo(&buf, (char *) &model->means[i][j], sizeof(double));

	for (i = 0; i < model->k; i++)
		for (j = 0; j < model->dim; j++)
			appendBinaryStringInfo(&buf, (char *) &model->variances[i][j], sizeof(double));

	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

/*
 * Deserialize GMMModel from bytea
 */
static GMMModel *
gmm_model_deserialize_from_bytea(const bytea * data, uint8 *training_backend_out)
{
	GMMModel   *model;
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	uint8		training_backend = 0;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid GMM model data: too small")));

	buf = VARDATA(data);
	offset = 0;

	/* Read training_backend first */
	training_backend = (uint8) buf[offset];
	offset += sizeof(uint8);
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	model = (GMMModel *) palloc0(sizeof(GMMModel));

	memcpy(&model->k, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&model->dim, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (model->k <= 0 || model->k > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid model data: k=%d (expected 1-100)", model->k)));
	if (model->dim <= 0 || model->dim > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid model data: dim=%d (expected 1-100000)", model->dim)));

	model->mixing_coeffs = (double *) palloc0(sizeof(double) * model->k);
	model->means = (double **) palloc(sizeof(double *) * model->k);
	model->variances = (double **) palloc(sizeof(double *) * model->k);

	for (i = 0; i < model->k; i++)
	{
		memcpy(&model->mixing_coeffs[i], buf + offset, sizeof(double));
		offset += sizeof(double);
	}

	for (i = 0; i < model->k; i++)
	{
		model->means[i] = (double *) palloc(sizeof(double) * model->dim);
		for (j = 0; j < model->dim; j++)
		{
			memcpy(&model->means[i][j], buf + offset, sizeof(double));
			offset += sizeof(double);
		}
	}

	for (i = 0; i < model->k; i++)
	{
		model->variances[i] = (double *) palloc(sizeof(double) * model->dim);
		for (j = 0; j < model->dim; j++)
		{
			memcpy(&model->variances[i][j], buf + offset, sizeof(double));
			offset += sizeof(double);
		}
	}

	return model;
}

/*
 * train_gmm_model_id
 *
 * Trains GMM and stores model in catalog, returns model_id
 */
PG_FUNCTION_INFO_V1(train_gmm_model_id);

Datum
train_gmm_model_id(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_components;
	int			max_iters;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	GMMModel	model;
	double	  **responsibilities;
	double		log_likelihood,
				prev_log_likelihood;
	int			iter,
				i,
				k,
				d;
	bytea	   *model_data;
	MLCatalogModelSpec spec;
	Jsonb	   *metrics;
	StringInfoData metrics_json;
	int32		model_id;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_components = PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);

	if (num_components < 1 || num_components > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_components must be between 1 and 100")));

	if (max_iters < 1)
		max_iters = 100;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (nvec < num_components)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough vectors (%d) for %d components", nvec, num_components)));
	}

	model.k = num_components;
	model.dim = dim;
	model.mixing_coeffs = (double *) palloc(sizeof(double) * num_components);
	model.means = (double **) palloc(sizeof(double *) * num_components);
	model.variances = (double **) palloc(sizeof(double *) * num_components);

	for (k = 0; k < num_components; k++)
	{
		int			idx;

		model.means[k] = (double *) palloc(sizeof(double) * dim);
		model.variances[k] = (double *) palloc(sizeof(double) * dim);
		idx = rand() % nvec;
		for (d = 0; d < dim; d++)
			model.means[k][d] = (double) data[idx][d];
		for (d = 0; d < dim; d++)
			model.variances[k][d] = 1.0;
		model.mixing_coeffs[k] = 1.0 / num_components;
	}

	responsibilities = (double **) palloc(sizeof(double *) * nvec);
	for (i = 0; i < nvec; i++)
		responsibilities[i] = (double *) palloc(sizeof(double) * num_components);

	prev_log_likelihood = -DBL_MAX;

	for (iter = 0; iter < max_iters; iter++)
	{
		log_likelihood = 0.0;

		for (i = 0; i < nvec; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < num_components; k++)
			{
				double		pdf = gaussian_pdf(data[i], model.means[k], model.variances[k], dim);

				responsibilities[i][k] = model.mixing_coeffs[k] * pdf;
				sum += responsibilities[i][k];
			}
			if (sum > GMM_MIN_PROB)
			{
				double		point_likelihood = 0.0;

				for (k = 0; k < num_components; k++)
					responsibilities[i][k] /= sum;
				for (k = 0; k < num_components; k++)
					point_likelihood += model.mixing_coeffs[k] * gaussian_pdf(data[i], model.means[k], model.variances[k], dim);
				log_likelihood += log(point_likelihood + GMM_MIN_PROB);
			}
		}

		if (fabs(log_likelihood - prev_log_likelihood) < 1e-6)
			break;
		prev_log_likelihood = log_likelihood;

		{
			double	   *N_k = (double *) palloc0(sizeof(double) * num_components);

			for (k = 0; k < num_components; k++)
			{
				for (i = 0; i < nvec; i++)
					N_k[k] += responsibilities[i][k];
				if (N_k[k] < GMM_MIN_PROB)
					N_k[k] = GMM_MIN_PROB;
			}
			for (k = 0; k < num_components; k++)
				model.mixing_coeffs[k] = N_k[k] / nvec;
			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.means[k][d] = 0.0;
				for (i = 0; i < nvec; i++)
					for (d = 0; d < dim; d++)
						model.means[k][d] += responsibilities[i][k] * data[i][d];
				for (d = 0; d < dim; d++)
					model.means[k][d] /= N_k[k];
			}
			for (k = 0; k < num_components; k++)
			{
				for (d = 0; d < dim; d++)
					model.variances[k][d] = 0.0;
				for (i = 0; i < nvec; i++)
				{
					for (d = 0; d < dim; d++)
					{
						double		diff = data[i][d] - model.means[k][d];

						model.variances[k][d] += responsibilities[i][k] * diff * diff;
					}
				}
				for (d = 0; d < dim; d++)
					model.variances[k][d] = (model.variances[k][d] / N_k[k]) + GMM_EPSILON;
			}
			NDB_FREE(N_k);
		}
	}

	model_data = gmm_model_serialize_to_bytea(&model, 0);

	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"training_backend\":0, \"k\": %d, \"dim\": %d, \"max_iters\": %d}",
					 model.k, model.dim, max_iters);
	metrics = ndb_jsonb_in_cstring(metrics_json.data);
	NDB_FREE(metrics_json.data);

	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL;
	spec.algorithm = "gmm";
	spec.training_table = tbl_str;
	spec.training_column = NULL;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
	{
		NDB_FREE(data[i]);
		NDB_FREE(responsibilities[i]);
	}
	NDB_FREE(data);
	NDB_FREE(responsibilities);
	for (k = 0; k < num_components; k++)
	{
		NDB_FREE(model.means[k]);
		NDB_FREE(model.variances[k]);
	}
	NDB_FREE(model.means);
	NDB_FREE(model.variances);
	NDB_FREE(model.mixing_coeffs);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_INT32(model_id);
}

/*
 * predict_gmm_model_id
 *
 * Predict cluster assignment for a feature vector using GMM model
 */
PG_FUNCTION_INFO_V1(predict_gmm_model_id);

Datum
predict_gmm_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector	   *features;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	GMMModel   *model = NULL;
	double	   *probabilities;
	int			predicted_cluster = 0;
	double		max_prob = -DBL_MAX;
	int			i,
				k;

	model_id = PG_GETARG_INT32(0);
	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("GMM model %d not found", model_id)));
	}

	if (model_data == NULL)
	{
		if (metrics)
			NDB_FREE(metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("GMM model %d has no model data", model_id)));
	}

	/* Ensure bytea is in current function's memory context */
	if (model_data != NULL)
	{
		int			data_len = VARSIZE(model_data);
		bytea	   *copy = (bytea *) palloc(data_len);

		memcpy(copy, model_data, data_len);
		model_data = copy;
	}

	/* Deserialize model */
	model = gmm_model_deserialize_from_bytea(model_data, NULL);

	if (features->dim != model->dim)
	{
		/* Cleanup */
		for (i = 0; i < model->k; i++)
		{
			NDB_FREE(model->means[i]);
			NDB_FREE(model->variances[i]);
		}
		NDB_FREE(model->means);
		NDB_FREE(model->variances);
		NDB_FREE(model->mixing_coeffs);
		NDB_FREE(model);
		if (model_data)
			NDB_FREE(model_data);
		if (metrics)
			NDB_FREE(metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Feature dimension mismatch: expected %d, got %d",
						model->dim, features->dim)));
	}

	/* Compute probabilities for each component */
	probabilities = (double *) palloc(sizeof(double) * model->k);
	for (k = 0; k < model->k; k++)
	{
		double		pdf = gaussian_pdf(features->data, model->means[k], model->variances[k], model->dim);

		probabilities[k] = model->mixing_coeffs[k] * pdf;
		if (probabilities[k] > max_prob)
		{
			max_prob = probabilities[k];
			predicted_cluster = k;
		}
	}

	/* Cleanup */
	NDB_FREE(probabilities);
	for (i = 0; i < model->k; i++)
	{
		NDB_FREE(model->means[i]);
		NDB_FREE(model->variances[i]);
	}
	NDB_FREE(model->means);
	NDB_FREE(model->variances);
	NDB_FREE(model->mixing_coeffs);
	NDB_FREE(model);
	if (model_data)
		NDB_FREE(model_data);
	if (metrics)
		NDB_FREE(metrics);

	PG_RETURN_INT32(predicted_cluster);
}


/*
 * Compute Euclidean distance squared (float to double)
 */
static inline double
gmm_euclidean_distance_squared(const float *a, const double *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - b[i];

		sum += diff * diff;
	}
	return sum;
}

/*
 * Compute Euclidean distance (float to float)
 */
static inline double
gmm_euclidean_distance_ff(const float *a, const float *b, int dim)
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
 * evaluate_gmm_by_model_id
 *
 * Evaluates GMM clustering model by computing:
 * - Inertia (within-cluster sum of squares)
 * - Silhouette score
 * - Davies-Bouldin index
 */
PG_FUNCTION_INFO_V1(evaluate_gmm_by_model_id);

Datum
evaluate_gmm_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *vector_col;
	char	   *tbl_str;
	char	   *col_str;
	int			nvec = 0;
	int			i,
				j,
				c;
	float	  **data = NULL;
	int			dim = 0;
	GMMModel   *model = NULL;
	int		   *assignments = NULL;
	int		   *cluster_sizes = NULL;
	double		inertia = 0.0;
	double		silhouette = 0.0;
	double	   *a_scores = NULL;
	double	   *b_scores = NULL;
	MemoryContext oldcontext;
	Jsonb	   *result_jsonb = NULL;
	bytea	   *model_payload = NULL;
	Jsonb	   *model_metrics = NULL;
	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: table_name and vector_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	vector_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_payload, NULL, &model_metrics))
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model %d not found", model_id)));
	}

	if (model_payload == NULL)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model %d has no model_data", model_id)));
	}

	/* Deserialize model */
	model = gmm_model_deserialize_from_bytea(model_payload, NULL);
	if (model == NULL)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_payload)
			NDB_FREE(model_payload);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: failed to deserialize model")));
	}

	/* Connect to SPI */
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Fetch test data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < 1)
	{
		NDB_SPI_SESSION_END(spi_session);
		for (c = 0; c < model->k; c++)
		{
			NDB_FREE(model->means[c]);
			NDB_FREE(model->variances[c]);
		}
		NDB_FREE(model->means);
		NDB_FREE(model->variances);
		NDB_FREE(model->mixing_coeffs);
		NDB_FREE(model);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_payload)
			NDB_FREE(model_payload);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: no valid data found")));
	}

	if (dim != model->dim)
	{
		NDB_SPI_SESSION_END(spi_session);
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
		for (c = 0; c < model->k; c++)
		{
			NDB_FREE(model->means[c]);
			NDB_FREE(model->variances[c]);
		}
		NDB_FREE(model->means);
		NDB_FREE(model->variances);
		NDB_FREE(model->mixing_coeffs);
		NDB_FREE(model);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_payload)
			NDB_FREE(model_payload);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: dimension mismatch: model dim=%d, data dim=%d",
						model->dim, dim)));
	}

	/* Assign points to clusters (based on maximum probability) */
	assignments = (int *) palloc(sizeof(int) * nvec);
	cluster_sizes = (int *) palloc0(sizeof(int) * model->k);

	for (i = 0; i < nvec; i++)
	{
		double		max_prob = -DBL_MAX;
		int			best = 0;

		for (c = 0; c < model->k; c++)
		{
			double		pdf = gaussian_pdf(data[i], model->means[c], model->variances[c], dim);
			double		prob = model->mixing_coeffs[c] * pdf;

			if (prob > max_prob)
			{
				max_prob = prob;
				best = c;
			}
		}
		assignments[i] = best;
		cluster_sizes[best]++;
		inertia += gmm_euclidean_distance_squared(data[i], model->means[best], dim);
	}

	/* Compute silhouette score */
	a_scores = (double *) palloc0(sizeof(double) * nvec);
	b_scores = (double *) palloc0(sizeof(double) * nvec);

	for (i = 0; i < nvec; i++)
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
		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;
			if (assignments[j] == my_cluster)
			{
				double		dist = gmm_euclidean_distance_ff(data[i], data[j], dim);

				same_dist += dist;
				same_count++;
			}
		}
		if (same_count > 0)
			a_scores[i] = same_dist / (double) same_count;
		else
			a_scores[i] = 0.0;

		/* Minimum average distance to other clusters */
		{
			double		other_dist = 0.0;
			int			other_count = 0;
			int			other_cluster_loop;

			for (other_cluster_loop = 0; other_cluster_loop < model->k; other_cluster_loop++)
			{
				if (other_cluster_loop == my_cluster)
					continue;
				if (cluster_sizes[other_cluster_loop] == 0)
					continue;

				other_dist = 0.0;
				other_count = 0;

				for (j = 0; j < nvec; j++)
				{
					if (assignments[j] == other_cluster_loop)
					{
						other_dist += gmm_euclidean_distance_ff(data[i], data[j], dim);
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
		/* If no other clusters found, set to 0 to avoid DBL_MAX issues */
		if (min_other_dist >= DBL_MAX)
			b_scores[i] = 0.0;
		else
			b_scores[i] = min_other_dist;
	}

	/* Compute average silhouette */
	{
		int			valid_count = 0;
		double		sum_silhouette = 0.0;

		for (i = 0; i < nvec; i++)
		{
			/*
			 * Skip if no other clusters (b_scores is 0 and a_scores might be
			 * 0)
			 */
			double		max_ab;

			if (b_scores[i] <= 0.0 && a_scores[i] <= 0.0)
				continue;

			max_ab = (a_scores[i] > b_scores[i]) ? a_scores[i] : b_scores[i];
			if (max_ab > 0.0)
			{
				double		s = (b_scores[i] - a_scores[i]) / max_ab;

				sum_silhouette += s;
				valid_count++;
			}
		}
		if (valid_count > 0)
			silhouette = sum_silhouette / (double) valid_count;
	}

	/* Cleanup */
	if (data)
	{
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
	}
	if (assignments)
		NDB_FREE(assignments);
	if (cluster_sizes)
		NDB_FREE(cluster_sizes);
	if (a_scores)
		NDB_FREE(a_scores);
	if (b_scores)
		NDB_FREE(b_scores);
	for (c = 0; c < model->k; c++)
	{
		NDB_FREE(model->means[c]);
		NDB_FREE(model->variances[c]);
	}
	NDB_FREE(model->means);
	NDB_FREE(model->variances);
	NDB_FREE(model->mixing_coeffs);
	NDB_FREE(model);
	if (model_payload)
		NDB_FREE(model_payload);
	if (model_metrics)
		NDB_FREE(model_metrics);

	/* Build JSONB in SPI context first (like naive_bayes does) */
	/* This ensures the JSONB is accessible when called via SPI from neurondb.evaluate() */
	/* Use the same pattern as naive_bayes: build JSON string, then parse with jsonb_in */
	{
		StringInfoData jsonbuf;
		double		safe_inertia = inertia;
		double		safe_silhouette = silhouette;

		/* Ensure values are finite numbers for JSON */
		if (!isfinite(safe_inertia))
			safe_inertia = 0.0;
		if (!isfinite(safe_silhouette))
			safe_silhouette = 0.0;

		initStringInfo(&jsonbuf);
		appendStringInfo(&jsonbuf,
						 "{\"inertia\":%.6f,\"silhouette_score\":%.6f,\"n_samples\":%d}",
						 safe_inertia, safe_silhouette, nvec);

		/* Create JSONB in current context (SPI context) using ndb_jsonb_in_cstring */
		/* This wrapper handles text conversion and error handling properly */
		result_jsonb = ndb_jsonb_in_cstring(jsonbuf.data);
		if (result_jsonb == NULL)
		{
			NDB_FREE(jsonbuf.data);
			NDB_SPI_SESSION_END(spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("neurondb: evaluate_gmm_by_model_id: failed to parse metrics JSON"),
					 errdetail("JSON string: {\"inertia\":%.6f,\"silhouette_score\":%.6f,\"n_samples\":%d}",
							   safe_inertia, safe_silhouette, nvec)));
		}

		NDB_FREE(jsonbuf.data);
		jsonbuf.data = NULL;
	}

	if (result_jsonb == NULL)
	{
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_gmm_by_model_id: JSONB result is NULL")));
	}

	/*
	 * Copy result_jsonb to caller's context before SPI_finish().
	 * This ensures the JSONB remains valid after the SPI context is
	 * deleted. The JSONB is allocated in SPI context and will be
	 * invalid after SPI_finish() deletes that context.
	 * This pattern matches naive_bayes.c for consistency.
	 */
	MemoryContextSwitchTo(oldcontext);
	result_jsonb = (Jsonb *) PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));

	/* Now safe to finish SPI and free SPI-allocated memory */
	NDB_SPI_SESSION_END(spi_session);

	/* Free strings allocated before SPI_connect (in oldcontext) */
	/* Note: tbl_str and col_str were allocated in oldcontext, not SPI context */
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	/* Return result (already in oldcontext and properly copied) */
	/* PG_RETURN_JSONB_P will handle detoasting and copying */
	PG_RETURN_JSONB_P(result_jsonb);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for GMM
 *-------------------------------------------------------------------------*/
#include "ml_gpu_registry.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_gpu_model.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* GPU Model State */
typedef struct GmmGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
	int			n_components;
}			GmmGpuModelState;

static void
gmm_gpu_release_state(GmmGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		NDB_FREE(state->model_blob);
	if (state->metrics != NULL)
		NDB_FREE(state->metrics);
	NDB_FREE(state);
}

static bool
gmm_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	GmmGpuModelState *state;
	bytea	   *payload;
	Jsonb	   *metrics;
	int			rc;
	int			n_components = 2;
	int			max_iters = 100;
	const		ndb_gpu_backend *backend;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->gmm_train == NULL)
		return false;

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		Datum		n_components_datum;
		Datum		max_iters_datum;
		Datum		numeric_datum;
		Numeric		num;

		n_components_datum = DirectFunctionCall2(
												 jsonb_object_field,
												 JsonbPGetDatum(spec->hyperparameters),
												 CStringGetTextDatum("n_components"));
		if (DatumGetPointer(n_components_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, n_components_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				n_components = DatumGetInt32(
											 DirectFunctionCall1(numeric_int4,
																 NumericGetDatum(num)));
				if (n_components <= 0 || n_components > 1000)
					n_components = 2;
			}
		}

		max_iters_datum = DirectFunctionCall2(
											  jsonb_object_field,
											  JsonbPGetDatum(spec->hyperparameters),
											  CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_iters = DatumGetInt32(
										  DirectFunctionCall1(numeric_int4,
															  NumericGetDatum(num)));
				if (max_iters <= 0 || max_iters > 10000)
					max_iters = 100;
			}
		}
	}

	payload = NULL;
	metrics = NULL;

	rc = backend->gmm_train(spec->feature_matrix,
							spec->sample_count,
							spec->feature_dim,
							n_components,
							spec->hyperparameters,
							&payload,
							&metrics,
							errstr);
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
			NDB_FREE(payload);
		if (metrics != NULL)
			NDB_FREE(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		gmm_gpu_release_state((GmmGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	state = (GmmGpuModelState *) palloc0(sizeof(GmmGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;
	state->n_components = n_components;

	if (metrics != NULL)
	{
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(
														 PointerGetDatum(metrics));
	}
	else
	{
		state->metrics = NULL;
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
gmm_gpu_predict(const MLGpuModel * model,
				const float *input,
				int input_dim,
				float *output,
				int output_dim,
				char **errstr)
{
	const		GmmGpuModelState *state;
	int			cluster_out;
	double		probability_out;
	int			rc;
	const		ndb_gpu_backend *backend;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const GmmGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->gmm_predict == NULL)
		return false;

	rc = backend->gmm_predict(state->model_blob,
							  input,
							  state->feature_dim > 0 ? state->feature_dim : input_dim,
							  &cluster_out,
							  &probability_out,
							  errstr);
	if (rc != 0)
		return false;

	/* GMM prediction returns cluster ID as output */
	output[0] = (float) cluster_out;

	return true;
}

static bool
gmm_gpu_evaluate(const MLGpuModel * model,
				 const MLGpuEvalSpec * spec,
				 MLGpuMetrics * out,
				 char **errstr)
{
	const		GmmGpuModelState *state;
	Jsonb	   *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const GmmGpuModelState *) model->backend_state;

	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"gmm\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"n_components\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0,
						 state->n_components);

		metrics_json = ndb_jsonb_in_cstring(buf.data);
		NDB_FREE(buf.data);
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
gmm_gpu_serialize(const MLGpuModel * model,
				  bytea * *payload_out,
				  Jsonb * *metadata_out,
				  char **errstr)
{
	const		GmmGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const GmmGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
	}

	return true;
}

static bool
gmm_gpu_deserialize(MLGpuModel * model,
					const bytea * payload,
					const Jsonb * metadata,
					char **errstr)
{
	GmmGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	state = (GmmGpuModelState *) palloc0(sizeof(GmmGpuModelState));
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->n_samples = -1;
	state->n_components = -1;

	if (model->backend_state != NULL)
		gmm_gpu_release_state((GmmGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
gmm_gpu_destroy(MLGpuModel * model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		gmm_gpu_release_state((GmmGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps gmm_gpu_model_ops = {
	.algorithm = "gmm",
	.train = gmm_gpu_train,
	.predict = gmm_gpu_predict,
	.evaluate = gmm_gpu_evaluate,
	.serialize = gmm_gpu_serialize,
	.deserialize = gmm_gpu_deserialize,
	.destroy = gmm_gpu_destroy,
};

#endif							/* NDB_GPU_CUDA */

void
neurondb_gpu_register_gmm_model(void)
{
#ifdef NDB_GPU_CUDA
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&gmm_gpu_model_ops);
	registered = true;
#else
	/* GPU not available - registration is a no-op */
	return;
#endif
}

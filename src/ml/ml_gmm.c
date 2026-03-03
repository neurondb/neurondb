/*-------------------------------------------------------------------------
 *
 * ml_gmm.c
 *    Gaussian mixture model for soft clustering.
 *
 * This module implements GMM using expectation-maximization for probabilistic
 * cluster assignments and density estimation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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
#include "utils/memutils.h"
#include "common/jsonapi.h"
#include "vector/vector_types.h"
#include "common/pg_prng.h"
#include "neurondb_cuda_gmm.h"
#include "ml_gpu_registry.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif
#include "neurondb_gpu_model.h"
#include "neurondb_gpu.h"
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
 * gaussian_pdf - Compute Gaussian probability density with diagonal covariance
 *
 * Computes the probability density of a point under a multivariate Gaussian
 * distribution with diagonal covariance matrix. Uses log-space computation
 * for numerical stability.
 *
 * Parameters:
 *   x - Input point (feature vector)
 *   mean - Mean vector of the Gaussian distribution
 *   variance - Variance vector (diagonal of covariance matrix)
 *   dim - Dimension of all vectors
 *
 * Returns:
 *   Probability density as double
 *
 * Notes:
 *   The function adds GMM_EPSILON to variances for regularization to avoid
 *   numerical issues. Computes in log-space and exponentiates at the end
 *   for better numerical stability.
 */
static double
gaussian_pdf(const float *x, const double *mean, const double *variance, int dim)
{
	double		diff;
	double		log_det = 0.0;
	double		log_likelihood = 0.0;
	double		var;
	int			d;

	for (d = 0; d < dim; d++)
	{
		diff = (double) x[d] - mean[d];
		var = variance[d] + GMM_EPSILON;

		log_likelihood -= 0.5 * (diff * diff) / var;
		log_det += log(var);
	}

	log_likelihood -= 0.5 * (dim * log(2.0 * M_PI) + log_det);

	/* Clamp to avoid exp overflow and return 0 for -inf */
	if (log_likelihood <= -700.0)
		return 0.0;
	if (log_likelihood >= 700.0)
		return exp(700.0);
	return exp(log_likelihood);
}

/*
 * cluster_gmm - Perform Gaussian mixture model clustering
 *
 * User-facing function that performs soft clustering using Gaussian mixture
 * models with expectation-maximization algorithm. Returns cluster assignments
 * and probabilities for each data point.
 *
 * Parameters:
 *   table_name - Name of table containing vector data (text)
 *   vector_column - Name of vector column to cluster (text)
 *   num_components - Number of Gaussian components/clusters (int32)
 *   max_iters - Maximum EM iterations (int32, optional, default 100)
 *
 * Returns:
 *   Array of cluster assignments as float8[]
 *
 * Notes:
 *   The function uses the EM algorithm to fit GMM parameters. Supports both
 *   CPU and GPU backends. Returns probabilistic cluster assignments where
 *   each point has a probability distribution over all clusters.
 */
PG_FUNCTION_INFO_V1(cluster_gmm);

Datum
cluster_gmm(PG_FUNCTION_ARGS)
{
	ArrayType  *result = NULL;
	char	   *col_str = NULL;
	char	   *tbl_str = NULL;
	Datum	   *result_datums = NULL;
	double		log_likelihood = 0.0;
	double		prev_log_likelihood = -DBL_MAX;
	double	   **responsibilities = NULL;
	float	   **data = NULL;
	GMMModel	model = {0};
	int			d = 0;
	int			dim = 0;
	int			dims[2] = {0, 0};
	int			i = 0;
	int			iter = 0;
	int			k = 0;
	int			lbs[2] = {0, 0};
	int			max_iters = 100;
	int			num_components = 0;
	int			nvec = 0;
	text	   *table_name = NULL;
	text	   *vector_column = NULL;

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

	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL && nvec > 0)
		{
			neurondb_free_vectors(data, nvec);
		}
		else if (data != NULL)
		{
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < num_components)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough vectors (%d) for %d components",
						nvec, num_components)));

	model.k = num_components;
	model.dim = dim;
	{
		double	   *mixing_coeffs_tmp = NULL;
		double	   **means_tmp = NULL;
		double	   **variances_tmp = NULL;
		double	   *means_row = NULL;
		double	   *variances_row = NULL;
		int			idx = 0;

		nalloc(mixing_coeffs_tmp, double, num_components);
		nalloc(means_tmp, double *, num_components);
		nalloc(variances_tmp, double *, num_components);

		{
			pg_prng_state rng;

			pg_prng_seed(&rng, (uint64) nvec ^ (uint64) dim);
			for (k = 0; k < num_components; k++)
			{
				nalloc(means_row, double, dim);
				nalloc(variances_row, double, dim);
				means_tmp[k] = means_row;
				variances_tmp[k] = variances_row;

				idx = (nvec > 1)
					? (int) pg_prng_uint64_range_inclusive(&rng, 0, (uint64) (nvec - 1))
					: 0;

				for (d = 0; d < dim; d++)
					means_tmp[k][d] = (double) data[idx][d];

				for (d = 0; d < dim; d++)
					variances_tmp[k][d] = 1.0;

				mixing_coeffs_tmp[k] = 1.0 / num_components;
			}
		}
		model.mixing_coeffs = mixing_coeffs_tmp;
		model.means = means_tmp;
		model.variances = variances_tmp;
	}

	nalloc(responsibilities, double *, nvec);
	/* Explicitly ensure all pointers are NULL before allocation (palloc0 should do this, but be defensive) */
	for (i = 0; i < nvec; i++)
	{
		responsibilities[i] = NULL;
		nalloc(responsibilities[i], double, num_components);
	}

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
			double *N_k = NULL;

			nalloc(N_k, double, num_components);

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

		nfree(N_k);
	}
}

nalloc(result_datums, Datum, nvec * num_components);
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
		nfree(data[i]);
		nfree(responsibilities[i]);
	}
	nfree(data);
	nfree(responsibilities);

	for (k = 0; k < num_components; k++)
	{
		nfree(model.means[k]);
		nfree(model.variances[k]);
	}
	nfree(model.means);
	nfree(model.variances);
	nfree(model.mixing_coeffs);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Serialize GMMModel to bytea for storage
 */
static bytea *
gmm_model_serialize_to_bytea(const GMMModel * model, uint8 training_backend)
{
	StringInfoData buf = {0};
	int			i = 0;
	int			j = 0;
	char	   *result_raw = NULL;
	int			total_size = 0;
	bytea	   *result = NULL;

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

	/* Convert to bytea - match naive_bayes pattern exactly */
	{
		total_size = VARHDRSZ + buf.len;
		nalloc(result_raw, char, total_size);
		result = (bytea *) result_raw;
		SET_VARSIZE(result, total_size);
		memcpy(VARDATA(result), buf.data, buf.len);
		/* Don't free buf.data - StringInfo manages its own memory and will
		 * be cleaned up when the memory context is reset. Freeing it manually
		 * can cause "invalid pointer" errors if the buffer was reallocated. */
	}

	return result;
}

/*
 * Deserialize GMMModel from bytea
 */
static GMMModel *
gmm_model_deserialize_from_bytea(const bytea * data, uint8 * training_backend_out)
{
	const char *buf = NULL;
	GMMModel   *model = NULL;
	int			i = 0;
	int			j = 0;
	int			offset = 0;
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

	nalloc(model, GMMModel, 1);

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

	{
		double *mixing_coeffs_tmp = NULL;
		double **means_tmp = NULL;
		double **variances_tmp = NULL;
		nalloc(mixing_coeffs_tmp, double, model->k);
		nalloc(means_tmp, double *, model->k);
		nalloc(variances_tmp, double *, model->k);

		for (i = 0; i < model->k; i++)
		{
			memcpy(&mixing_coeffs_tmp[i], buf + offset, sizeof(double));
			offset += sizeof(double);
		}

		for (i = 0; i < model->k; i++)
		{
			double	   *means_row = NULL;

			nalloc(means_row, double, model->dim);
			for (j = 0; j < model->dim; j++)
			{
				memcpy(&means_row[j], buf + offset, sizeof(double));
				offset += sizeof(double);
			}
			means_tmp[i] = means_row;
		}

		for (i = 0; i < model->k; i++)
		{
			double	   *variances_row = NULL;

			nalloc(variances_row, double, model->dim);
			for (j = 0; j < model->dim; j++)
			{
				memcpy(&variances_row[j], buf + offset, sizeof(double));
				offset += sizeof(double);
			}
			variances_tmp[i] = variances_row;
		}
		model->mixing_coeffs = mixing_coeffs_tmp;
		model->means = means_tmp;
		model->variances = variances_tmp;
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
	bytea	   *model_data = NULL;
	char	   *col_str = NULL;
	char	   *tbl_str = NULL;
	double		log_likelihood = 0.0;
	double		prev_log_likelihood = -DBL_MAX;
	double	   **responsibilities = NULL;
	float	   **data = NULL;
	GMMModel	model = {0};
	int			d = 0;
	int			dim = 0;
	int			i = 0;
	int			iter = 0;
	int			k = 0;
	int			max_iters = 100;
	int			num_components = 0;
	int			nvec = 0;
	int32		model_id = 0;
	Jsonb	   *metrics = NULL;
	MLCatalogModelSpec spec = {0};
	Jsonb	   *params_jsonb = NULL;
	text	   *table_name = NULL;
	text	   *vector_column = NULL;

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

	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL && nvec > 0)
		{
			neurondb_free_vectors(data, nvec);
		}
		else if (data != NULL)
		{
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < num_components)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough vectors (%d) for %d components", nvec, num_components)));
	}

	model.k = num_components;
	model.dim = dim;
	/* Ensure model fields are NULL before allocation */
	if (model.mixing_coeffs != NULL)
		nfree(model.mixing_coeffs);
	if (model.means != NULL)
		nfree(model.means);
	if (model.variances != NULL)
		nfree(model.variances);
	{
		double *mixing_coeffs_tmp = NULL;
		double **means_tmp = NULL;
		double **variances_tmp = NULL;
		nalloc(mixing_coeffs_tmp, double, num_components);
		nalloc(means_tmp, double *, num_components);
		nalloc(variances_tmp, double *, num_components);

		{
			pg_prng_state rng;
			int			idx;

			pg_prng_seed(&rng, (uint64) nvec ^ (uint64) dim);
			for (k = 0; k < num_components; k++)
			{
				double	   *means_row = NULL;
				double	   *variances_row = NULL;

				nalloc(means_row, double, dim);
				nalloc(variances_row, double, dim);
				means_tmp[k] = means_row;
				variances_tmp[k] = variances_row;
				idx = (nvec > 1)
					? (int) pg_prng_uint64_range_inclusive(&rng, 0, (uint64) (nvec - 1))
					: 0;
				for (d = 0; d < dim; d++)
					means_tmp[k][d] = (double) data[idx][d];
				for (d = 0; d < dim; d++)
					variances_tmp[k][d] = 1.0;
				mixing_coeffs_tmp[k] = 1.0 / num_components;
			}
		}
		model.mixing_coeffs = mixing_coeffs_tmp;
		model.means = means_tmp;
		model.variances = variances_tmp;
	}

	nalloc(responsibilities, double *, nvec);
	/* Explicitly ensure all pointers are NULL before allocation (palloc0 should do this, but be defensive) */
	for (i = 0; i < nvec; i++)
	{
		responsibilities[i] = NULL;
		nalloc(responsibilities[i], double, num_components);
	}

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
			else
			{
				/* Normalize to uniform when sum is too small to avoid unnormalized responsibilities */
				for (k = 0; k < num_components; k++)
					responsibilities[i][k] = 1.0 / (double) num_components;
				log_likelihood += log(1.0 / (double) num_components + GMM_MIN_PROB);
			}
		}

		if (fabs(log_likelihood - prev_log_likelihood) < 1e-6)
			break;
		prev_log_likelihood = log_likelihood;

		{
			double *N_k = NULL;
			nalloc(N_k, double, num_components);

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
			nfree(N_k);
		}
	}

	model_data = gmm_model_serialize_to_bytea(&model, 0);

	/* Build parameters JSON using JSONB API */
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;

		JsonbValue *final_value = NULL;
		Numeric		k_num,
					max_iters_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			/* Add k */
			jkey.type = jbvString;
			jkey.val.string.val = "k";
			jkey.val.string.len = strlen("k");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			k_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model.k)));
			jval.type = jbvNumeric;
			jval.val.numeric = k_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add max_iters */
			jkey.val.string.val = "max_iters";
			jkey.val.string.len = strlen("max_iters");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_iters)));
			jval.type = jbvNumeric;
			jval.val.numeric = max_iters_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: train_gmm: pushJsonbValue(WJB_END_OBJECT) returned NULL for parameters");
			}

			params_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: train_gmm: parameters JSONB construction failed: %s", edata->message);
			FlushErrorState();
			params_jsonb = NULL;
		}
		PG_END_TRY();
	}

	/* Build metrics JSON using JSONB API */
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;

		JsonbValue *final_value = NULL;
		Numeric		training_backend_num,
					k_num,
					dim_num,
					max_iters_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			/* Add training_backend */
			jkey.type = jbvString;
			jkey.val.string.val = "training_backend";
			jkey.val.string.len = strlen("training_backend");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			training_backend_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(0)));
			jval.type = jbvNumeric;
			jval.val.numeric = training_backend_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add k */
			jkey.val.string.val = "k";
			jkey.val.string.len = strlen("k");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			k_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model.k)));
			jval.type = jbvNumeric;
			jval.val.numeric = k_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add dim */
			jkey.val.string.val = "dim";
			jkey.val.string.len = strlen("dim");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			dim_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model.dim)));
			jval.type = jbvNumeric;
			jval.val.numeric = dim_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add max_iters */
			jkey.val.string.val = "max_iters";
			jkey.val.string.len = strlen("max_iters");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_iters)));
			jval.type = jbvNumeric;
			jval.val.numeric = max_iters_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: train_gmm: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
			}

			metrics = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: train_gmm: metrics JSONB construction failed: %s", edata->message);
			FlushErrorState();
			metrics = NULL;
		}
		PG_END_TRY();
	}

	if (metrics == NULL)
	{
	}

	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL;
	spec.algorithm = "gmm";
	spec.training_table = tbl_str;
	spec.training_column = NULL;
	spec.model_data = model_data;
	spec.parameters = params_jsonb;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	for (i = 0; i < nvec; i++)
	{
		nfree(data[i]);
		nfree(responsibilities[i]);
	}
	nfree(data);
	nfree(responsibilities);
	for (k = 0; k < num_components; k++)
	{
		nfree(model.means[k]);
		nfree(model.variances[k]);
	}
	nfree(model.means);
	nfree(model.variances);
	nfree(model.mixing_coeffs);
	nfree(tbl_str);
	nfree(col_str);

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
	Vector *features = NULL;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	GMMModel *model = NULL;
	double *probabilities = NULL;
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
			nfree(metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("GMM model %d has no model data", model_id)));
	}

	/* Ensure bytea is in current function's memory context */
	if (model_data != NULL)
	{
		int			data_len = VARSIZE(model_data);

		char *copy_bytes = NULL;
		bytea *copy = NULL;

		nalloc(copy_bytes, char, data_len);
		copy = (bytea *) copy_bytes;

		memcpy(copy, model_data, data_len);
		model_data = copy;
	}

	/* Deserialize model */
	model = gmm_model_deserialize_from_bytea(model_data, NULL);

	if (features->dim != model->dim)
	{
		for (i = 0; i < model->k; i++)
		{
			nfree(model->means[i]);
			nfree(model->variances[i]);
		}
		nfree(model->means);
		nfree(model->variances);
		nfree(model->mixing_coeffs);
		nfree(model);
		if (model_data)
			nfree(model_data);
		if (metrics)
			nfree(metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Feature dimension mismatch: expected %d, got %d",
						model->dim, features->dim)));
	}

	/* Compute probabilities for each component */
	nalloc(probabilities, double, model->k);
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

	nfree(probabilities);
	for (i = 0; i < model->k; i++)
	{
		nfree(model->means[i]);
		nfree(model->variances[i]);
	}
	nfree(model->means);
	nfree(model->variances);
	nfree(model->mixing_coeffs);
	nfree(model);
	if (model_data)
		nfree(model_data);
	if (metrics)
		nfree(metrics);

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
	text *table_name = NULL;
	text *vector_col = NULL;
	char *tbl_str = NULL;
	char *col_str = NULL;
	int			nvec = 0;
	int			i,
				j,
				c;
	float **data = NULL;
	int			dim = 0;

	GMMModel *model = NULL;
	int *assignments = NULL;
	int *cluster_sizes = NULL;
	double		inertia = 0.0;
	double		silhouette = 0.0;

	double *a_scores = NULL;
	double *b_scores = NULL;
	MemoryContext oldcontext;

	Jsonb *result_jsonb = NULL;
	bytea *model_payload = NULL;
	Jsonb *model_metrics = NULL;
	NdbSpiSession *spi_session = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Validate model_id before attempting to load */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model_id must be positive, got %d", model_id),
				 errdetail("Invalid model_id: %d", model_id),
				 errhint("Provide a valid model_id from neurondb.ml_models catalog.")));

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
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model %d not found", model_id)));
	}

	if (model_payload == NULL)
	{
		nfree(tbl_str);
		nfree(col_str);
		if (model_metrics)
			nfree(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: model %d has no model_data", model_id)));
	}

	/* Deserialize model */
	model = gmm_model_deserialize_from_bytea(model_payload, NULL);
	if (model == NULL)
	{
		nfree(tbl_str);
		nfree(col_str);
		if (model_payload)
			nfree(model_payload);
		if (model_metrics)
			nfree(model_metrics);
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
			nfree(model->means[c]);
			nfree(model->variances[c]);
		}
		nfree(model->means);
		nfree(model->variances);
		nfree(model->mixing_coeffs);
		nfree(model);
		nfree(tbl_str);
		nfree(col_str);
		if (model_payload)
			nfree(model_payload);
		if (model_metrics)
			nfree(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: no valid data found")));
	}

	if (dim != model->dim)
	{
		NDB_SPI_SESSION_END(spi_session);
		for (i = 0; i < nvec; i++)
			nfree(data[i]);
		nfree(data);
		for (c = 0; c < model->k; c++)
		{
			nfree(model->means[c]);
			nfree(model->variances[c]);
		}
		nfree(model->means);
		nfree(model->variances);
		nfree(model->mixing_coeffs);
		nfree(model);
		nfree(tbl_str);
		nfree(col_str);
		if (model_payload)
			nfree(model_payload);
		if (model_metrics)
			nfree(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_gmm_by_model_id: dimension mismatch: model dim=%d, data dim=%d",
						model->dim, dim)));
	}

	/* Assign points to clusters (based on maximum probability) */
	nalloc(assignments, int, nvec);
	nalloc(cluster_sizes, int, model->k);

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
	nalloc(a_scores, double, nvec);
	nalloc(b_scores, double, nvec);

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

	if (data)
	{
		for (i = 0; i < nvec; i++)
			nfree(data[i]);
		nfree(data);
	}
	if (assignments)
		nfree(assignments);
	if (cluster_sizes)
		nfree(cluster_sizes);
	if (a_scores)
		nfree(a_scores);
	if (b_scores)
		nfree(b_scores);
	for (c = 0; c < model->k; c++)
	{
		nfree(model->means[c]);
		nfree(model->variances[c]);
	}
	nfree(model->means);
	nfree(model->variances);
	nfree(model->mixing_coeffs);
	nfree(model);
	if (model_payload)
		nfree(model_payload);
	if (model_metrics)
		nfree(model_metrics);

	/* Build JSONB in SPI context first (like naive_bayes does) */

	/*
	 * This ensures the JSONB is accessible when called via SPI from
	 * neurondb.evaluate()
	 */

	/*
	 * Use the same pattern as naive_bayes: build JSON string, then parse with
	 * jsonb_in
	 */
	{
		StringInfoData jsonbuf = {0};
		double		safe_inertia = inertia;
		double		safe_silhouette = silhouette;

		/* Ensure values are finite numbers for JSON */
		if (!isfinite(safe_inertia))
			safe_inertia = 0.0;
		if (!isfinite(safe_silhouette))
			safe_silhouette = 0.0;

		/*
		 * End SPI session BEFORE creating JSONB to avoid memory context
		 * issues
		 */
		NDB_SPI_SESSION_END(spi_session);

		/* Switch to oldcontext to create JSONB */
		MemoryContextSwitchTo(oldcontext);

		initStringInfo(&jsonbuf);
		appendStringInfo(&jsonbuf,
						 "{\"inertia\":%.6f,\"silhouette_score\":%.6f,\"n_samples\":%d}",
						 safe_inertia, safe_silhouette, nvec);

		/* Create JSONB in oldcontext using ndb_jsonb_in_cstring */
		result_jsonb = ndb_jsonb_in_cstring(jsonbuf.data);
		/* Don't free jsonbuf.data - let PostgreSQL's memory context handle it */

		if (result_jsonb == NULL)
		{
			nfree(tbl_str);
			nfree(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("neurondb: evaluate_gmm_by_model_id: failed to parse metrics JSON"),
					 errdetail("JSON string: {\"inertia\":%.6f,\"silhouette_score\":%.6f,\"n_samples\":%d}",
							   safe_inertia, safe_silhouette, nvec)));
		}
	}

	/* Free strings allocated before SPI_connect (in oldcontext) */
	/* Note: tbl_str and col_str were allocated in oldcontext, not SPI context */
	nfree(tbl_str);
	nfree(col_str);

	/* Return result (already in oldcontext and properly copied) */
	/* PG_RETURN_JSONB_P will handle detoasting and copying */
	PG_RETURN_JSONB_P(result_jsonb);
}

#include "ml_gpu_registry.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#include "neurondb_gpu_model.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif

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
		nfree(state->model_blob);
	if (state->metrics != NULL)
		nfree(state->metrics);
	nfree(state);
}

static bool
gmm_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	GmmGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	int			rc;
	int			n_components = 2;
	int			max_iters = 100;
	const ndb_gpu_backend *backend;

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
		/* On error, GPU backend should have cleaned up - don't free here */
		/* Ensure error message is set if backend didn't set one */
		if (errstr && *errstr == NULL)
		{
			if (rc != 0)
				*errstr = pstrdup(psprintf("GMM GPU training failed with return code %d", rc));
			else if (payload == NULL)
				*errstr = pstrdup("GMM GPU training returned NULL model data");
			else
				*errstr = pstrdup("GMM GPU training failed for unknown reason");
		}
		return false;
	}	if (model->backend_state != NULL)
	{
		gmm_gpu_release_state((GmmGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	nalloc(state, GmmGpuModelState, 1);
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
gmm_gpu_predict(const MLGpuModel *model,
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
	const ndb_gpu_backend *backend;

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
gmm_gpu_evaluate(const MLGpuModel *model,
				 const MLGpuEvalSpec *spec,
				 MLGpuMetrics *out,
				 char **errstr)
{
	const		GmmGpuModelState *state;
	Jsonb	   *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const GmmGpuModelState *) model->backend_state;

	{
		StringInfoData buf = {0};

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
		/* Don't free buf.data - let PostgreSQL's memory context handle it */
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
gmm_gpu_serialize(const MLGpuModel *model,
				  bytea * *payload_out,
				  Jsonb * *metadata_out,
				  char **errstr)
{
	const		GmmGpuModelState *state;
	char *payload_bytes = NULL;
	bytea *payload_copy = NULL;
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
	nalloc(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

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
gmm_gpu_deserialize(MLGpuModel *model,
					const bytea * payload,
					const Jsonb * metadata,
					char **errstr)
{
	GmmGpuModelState *state = NULL;
	char *payload_bytes = NULL;
	bytea *payload_copy = NULL;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	nalloc(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, payload, payload_size);

	nalloc(state, GmmGpuModelState, 1);
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
gmm_gpu_destroy(MLGpuModel *model)
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

#endif							/* NDB_GPU_CUDA || NDB_GPU_METAL */

void
neurondb_gpu_register_gmm_model(void)
{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
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

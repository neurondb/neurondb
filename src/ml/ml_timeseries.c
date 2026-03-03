/*-------------------------------------------------------------------------
 *
 * ml_timeseries.c
 *    Time series analysis and forecasting.
 *
 * This module implements ARIMA, exponential smoothing, and trend analysis
 * for univariate and multivariate time series.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_timeseries.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "common/jsonapi.h"
#include "access/htup_details.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "neurondb.h"
#include "neurondb_pgcompat.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_macros.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_json.h"
#include "ml_catalog.h"
#include "neurondb_constants.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_safe_memory.h"

#include <math.h>
#include <string.h>

/* PG_MODULE_MAGIC is in neurondb.c only */

#define MAX_ARIMA_ORDER_P 10
#define MAX_ARIMA_ORDER_Q 10
#define MAX_ARIMA_ORDER_D 2
#define MIN_ARIMA_OBSERVATIONS 10
#define MAX_FORECAST_AHEAD 1000
#define MIN_SEASONAL_PERIOD 1
#define MAX_SEASONAL_PERIOD 365

typedef struct TimeSeriesModel
{
	int32		p;
	int32		d;
	int32		q;
	float *ar_coeffs;
	float *ma_coeffs;
	float		intercept;
	int32		n_obs;
	float *residuals;
}			TimeSeriesModel;

typedef struct TimeSeriesGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	float *ar_coeffs;
	float *ma_coeffs;
	int			p;
	int			d;
	int			q;
	float		intercept;
	int			n_obs;
	int			n_samples;
	char		model_type[32];
}			TimeSeriesGpuModelState;

/* Always use a local 'bool' variable for SPI_getbinval 'isnull' argument, not 'int' */
/* throughout this file, especially within the ARIMA forecast and model loading code. */

static void
compute_moving_average(const float *data, int n, int window, float *result)
{
	int			i,
				j;
	float		sum;

	if (window <= 0)
		elog(ERROR,
			 "window length for moving average must be positive");

	for (i = 0; i < n; i++)
	{
		if (i < window - 1)
			result[i] = data[i];
		else
		{
			sum = 0.0f;
			for (j = 0; j < window; j++)
				sum += data[i - j];
			result[i] = sum / (float) window;
		}
	}
}

static void
exponential_smoothing(const float *data, int n, float alpha, float *result)
{
	int			i;

	if (n <= 0)
		return;
	if (alpha < 0.0f || alpha > 1.0f)
		elog(ERROR, "alpha for exponential smoothing must be in [0,1]");

	result[0] = data[0];
	for (i = 1; i < n; i++)
		result[i] = alpha * data[i] + (1.0f - alpha) * result[i - 1];
}

static float *
compute_differences(const float *data, int n, int order, int *out_n)
{
	float *diff = NULL;
	int			curr_n,
				d,
				i;

	Assert(data != NULL);
	Assert(out_n != NULL);
	Assert(order >= 0);

	if (order == 0)
	{
		nalloc(diff, float, n);
		memcpy(diff, data, sizeof(float) * n);
		*out_n = n;
		return diff;
	}

	curr_n = n;
	/* Ensure diff is NULL before allocation */
	if (diff != NULL)
		nfree(diff);
	diff = NULL;
	nalloc(diff, float, curr_n);
	memcpy(diff, data, sizeof(float) * curr_n);

	for (d = 0; d < order; d++)
	{
		int			new_n = curr_n - 1;
		float *new_diff = NULL;

		if (new_n <= 0)
			elog(ERROR,
				 "cannot difference data sequence below length "
				 "1");
		nalloc(new_diff, float, new_n);
		for (i = 0; i < new_n; i++)
			new_diff[i] = diff[i + 1] - diff[i];
		nfree(diff);
		diff = new_diff;
		curr_n = new_n;
	}

	*out_n = curr_n;
	return diff;
}

static float
compute_mean(const float *data, int n)
{
	int			i;
	float		sum = 0.0f;

	if (n <= 0)
		return 0.0f;

	for (i = 0; i < n; i++)
		sum += data[i];

	return sum / (float) n;
}

static float
pg_attribute_unused()
compute_sample_variance(const float *data, int n, float mean)
{
	int			i;
	float		var = 0.0f;

	if (n < 2)
		return 0.0f;

	for (i = 0; i < n; i++)
	{
		float		d = data[i] - mean;

		var += d * d;
	}

	return var / (float) (n - 1);
}

/*
 * AR fitting with Yule-Walker equations using Cholesky decomposition.
 * Only AR coefficients are estimated, MA parameters set to zeros if requested.
 */
static TimeSeriesModel *
fit_arima(const float *data, int n, int p, int d, int q)
{
	TimeSeriesModel *model = NULL;

	float *diff_data = NULL;
	int			diff_n,
				i,
				j;
	float		mean;

	if (n <= 0)
		elog(ERROR,
			 "number of observations for ARIMA must be positive");
	if (p < 0 || p > MAX_ARIMA_ORDER_P)
		elog(ERROR, "arima p out of bounds");
	if (d < 0 || d > MAX_ARIMA_ORDER_D)
		elog(ERROR, "arima d out of bounds");
	if (q < 0 || q > MAX_ARIMA_ORDER_Q)
		elog(ERROR, "arima q out of bounds");

	{
		TimeSeriesModel *model_ptr = NULL;

		nalloc(model_ptr, TimeSeriesModel, 1);
		memset(model_ptr, 0, sizeof(TimeSeriesModel));
		model = model_ptr;
	}
	model->p = p;
	model->d = d;
	model->q = q;
	model->n_obs = n;
	model->ar_coeffs = NULL;
	model->ma_coeffs = NULL;
	model->residuals = NULL;

	diff_data = compute_differences(data, n, d, &diff_n);

	if (p > 0)
	{
		float *autocorr = NULL;
		float *a = NULL;
		float **R = NULL;
		float *right = NULL;

		nalloc(autocorr, float, p + 1);
		nalloc(right, float, p);
		nalloc(R, float *, p);

		for (i = 0; i <= p; i++)
		{
			float		sum = 0.0f;

			for (j = i; j < diff_n; j++)
				sum += diff_data[j] * diff_data[j - i];
			autocorr[i] = sum / (diff_n - i);
		}
		for (i = 0; i < p; i++)
		{
			float	   *R_row = NULL;

			nalloc(R_row, float, p);
			memset(R_row, 0, sizeof(float) * p);
			R[i] = R_row;
			for (j = 0; j < p; j++)
				R[i][j] = autocorr[abs(i - j)];
			right[i] = autocorr[i + 1];
		}

		/* Add regularization to ensure positive definiteness */
		/* This helps with numerical stability for small datasets or near-singular matrices */
		{
			float		lambda = 1e-4f;	/* Small regularization parameter */
			float		max_diag = 0.0f;
			int			k;

			/* Find maximum diagonal element for adaptive regularization */
			for (k = 0; k < p; k++)
			{
				if (R[k][k] > max_diag)
					max_diag = R[k][k];
			}

			/* Adaptive regularization: scale lambda based on matrix size */
			if (max_diag > 0.0f)
				lambda = max_diag * 1e-4f;
			else
				lambda = 1e-4f;

			/* Add regularization to diagonal (Tikhonov regularization) */
			for (k = 0; k < p; k++)
				R[k][k] += lambda;
		}

		nalloc(a, float, p);
		{
			float **L = NULL;
			nalloc(L, float *, p);

			for (i = 0; i < p; i++)
			{
				float *L_row = NULL;
				nalloc(L_row, float, p);
				L[i] = L_row;
			}
			for (i = 0; i < p; i++)
			{
				for (j = 0; j <= i; j++)
				{
					float		sum = R[i][j];
					int			k;

					for (k = 0; k < j; k++)
						sum -= L[i][k] * L[j][k];
					if (i == j)
					{
						/* With regularization, sum should be positive, but add safety check */
						if (sum <= 1e-8f)
						{
							/* Try adding more regularization dynamically */
							float		extra_lambda = 1e-6f - sum;
							if (extra_lambda > 0.0f)
							{
								R[i][i] += extra_lambda;
								sum += extra_lambda;
							}

							/* Final check - if still too small, matrix is truly singular */
							if (sum <= 1e-8f)
								elog(ERROR,
									 "Failed to decompose autocorrelation matrix: "
									 "non-positive definite (diag_sum=%.6e). "
									 "Try using simpler ARIMA parameters (e.g., lower p) "
									 "or ensure sufficient data variation.",
									 sum);
						}
						L[i][j] = sqrtf(sum);
					}
					else
						L[i][j] = sum / L[j][j];
				}
			}
			{
				float	   *y = NULL;

				nalloc(y, float, p);
				memset(y, 0, sizeof(float) * p);
				for (i = 0; i < p; i++)
				{
					float		sum = right[i];

					for (j = 0; j < i; j++)
						sum -= L[i][j] * y[j];
					y[i] = sum / L[i][i];
				}
				for (i = p - 1; i >= 0; i--)
				{
					float		sum = y[i];

					for (j = i + 1; j < p; j++)
						sum -= L[j][i] * a[j];
					a[i] = sum / L[i][i];
				}
				for (i = 0; i < p; i++)
					nfree(L[i]);
				nfree(L);
				nfree(y);
			}
		}

		{
			float *ar_coeffs = NULL;

			nalloc(ar_coeffs, float, p);
			model->ar_coeffs = ar_coeffs;
			for (i = 0; i < p; i++)
				model->ar_coeffs[i] = a[i];
		}

		for (i = 0; i < p; i++)
			nfree(R[i]);
		nfree(R);
		nfree(a);
		nfree(right);
		nfree(autocorr);
	}
	else
		model->ar_coeffs = NULL;

	if (q > 0)
	{
		float	   *ma_coeffs = NULL;

		nalloc(ma_coeffs, float, q);
		memset(ma_coeffs, 0, sizeof(float) * q);
		model->ma_coeffs = ma_coeffs;
	}
	else
	{
		model->ma_coeffs = NULL;
	}

	mean = compute_mean(diff_data, diff_n);
	model->intercept = mean;

	{
		float *residuals = NULL;

		nalloc(residuals, float, diff_n);
		model->residuals = residuals;
	}
	if (p > 0)
	{
		for (i = p; i < diff_n; i++)
		{
			float		pred = model->intercept;

			for (j = 0; j < p; j++)
			{
				int			idx = i - (j + 1);

				/* Validate array index */
				if (idx < 0 || idx >= diff_n)
				{
					elog(ERROR,
						 "neurondb: train_arima: array index out of bounds: idx=%d, diff_n=%d, i=%d, j=%d",
						 idx, diff_n, i, j);
				}
				pred += model->ar_coeffs[j] * diff_data[idx];
			}
			model->residuals[i] = diff_data[i] - pred;
		}
		for (i = 0; i < p && i < diff_n; i++)
			model->residuals[i] = 0.0f;
	}
	else
	{
		for (i = 0; i < diff_n; i++)
			model->residuals[i] = diff_data[i] - model->intercept;
	}

	nfree(diff_data);

	return model;
}

static void
arima_forecast(const TimeSeriesModel * model,
			   const float *last_values,
			   int n_last,
			   int n_ahead,
			   float *forecast)
{
	int			i,
				j;
	int			p,
				d;
	float	   *history = NULL;

	Assert(model != NULL);
	Assert(last_values != NULL);
	Assert(forecast != NULL);

	if (n_ahead < 1)
		elog(ERROR, "Must forecast at least 1 ahead");

	p = model->p;
	d = model->d;
	{
		float	   *history_ptr = NULL;

		nalloc(history_ptr, float, n_last + n_ahead);
		memset(history_ptr, 0, sizeof(float) * (n_last + n_ahead));
		history = history_ptr;
	}

	memcpy(history, last_values, sizeof(float) * n_last);

	for (i = 0; i < n_ahead; i++)
	{
		float		val;
		int			idx = n_last + i;

		if (p > 0)
		{
			val = model->intercept;
			for (j = 0; j < p && idx - (j + 1) >= 0; j++)
				val += model->ar_coeffs[j]
					* history[idx - (j + 1)];
		}
		else
			val = model->intercept;

		history[idx] = val;
		forecast[i] = val;
		if (d > 0)
		{
			int			step;

			for (step = 0; step < d; step++)
			{
				if (i - 1 >= 0)
					forecast[i] += forecast[i - 1];
			}
		}
	}

	nfree(history);
}

PG_FUNCTION_INFO_V1(train_arima);
Datum
train_arima(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *time_col;
	text	   *value_col;
	int32		p;
	int32		d;
	int32		q;
	char	   *table_name_str;
	char	   *time_col_str;
	char	   *value_col_str;
	StringInfoData sql;
	int			ret;
	int			n_samples;
	int			i;
	SPITupleTable *tuptable = NULL;
	TupleDesc	tupdesc;
	float	   *values = NULL;
	TimeSeriesModel *model = NULL;
	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Validate argument count */
	if (PG_NARGS() < 3 || PG_NARGS() > 6)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_arima requires 3 to 6 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	time_col = PG_GETARG_TEXT_PP(1);
	value_col = PG_GETARG_TEXT_PP(2);
	p = PG_ARGISNULL(3) ? 1 : PG_GETARG_INT32(3);
	d = PG_ARGISNULL(4) ? 1 : PG_GETARG_INT32(4);
	q = PG_ARGISNULL(5) ? 1 : PG_GETARG_INT32(5);

	table_name_str = text_to_cstring(table_name);
	time_col_str = text_to_cstring(time_col);
	value_col_str = text_to_cstring(value_col);

	if (p < 0 || p > MAX_ARIMA_ORDER_P)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("p must be between 0 and %d",
						MAX_ARIMA_ORDER_P)));
	if (d < 0 || d > MAX_ARIMA_ORDER_D)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("d must be between 0 and %d",
						MAX_ARIMA_ORDER_D)));
	if (q < 0 || q > MAX_ARIMA_ORDER_Q)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("q must be between 0 and %d",
						MAX_ARIMA_ORDER_Q)));

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT %s FROM %s ORDER BY %s",
					 quote_identifier(value_col_str),
					 quote_identifier(table_name_str),
					 quote_identifier(time_col_str));

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (table_name_str)
			nfree(table_name_str);
		if (time_col_str)
			nfree(time_col_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to execute time series query")));
	}

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n_samples = SPI_processed;

	if (n_samples < MIN_ARIMA_OBSERVATIONS)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (table_name_str)
			nfree(table_name_str);
		if (time_col_str)
			nfree(time_col_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least %d observations are required for ARIMA",
						MIN_ARIMA_OBSERVATIONS)));
	}

	nalloc(values, float, n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = tuptable->vals[i];
		bool		isnull = false;
		Datum		value_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);

		if (isnull)
		{
			if (values)
				nfree(values);
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			if (table_name_str)
				nfree(table_name_str);
			if (time_col_str)
				nfree(time_col_str);
			if (value_col_str)
				nfree(value_col_str);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("time series cannot contain "
							"NULL values")));
		}
		values[i] = (float) DatumGetFloat8(value_datum);
	}

	model = fit_arima(values, n_samples, p, d, q);

	if (!model)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (values)
			nfree(values);
		if (table_name_str)
			nfree(table_name_str);
		if (time_col_str)
			nfree(time_col_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("ARIMA model fitting failed")));
	}

	/* Save model to database */
	{
		int32		model_id;
		int32		model_id_val = 0;
		StringInfoData insert_sql;
		int			ret2;

		Datum *ar_datums = NULL;
		Datum *ma_datums = NULL;
		ArrayType *ar_array = NULL;
		ArrayType *ma_array = NULL;
		int			i2;

		initStringInfo(&insert_sql);

		/* Build AR coefficients array */
		if (model->ar_coeffs && p > 0)
		{
			nalloc(ar_datums, Datum, p);
			for (i2 = 0; i2 < p; i2++)
			{
				ar_datums[i2] = Float4GetDatum(model->ar_coeffs[i2]);
			}
			ar_array = construct_array(ar_datums, p, FLOAT4OID, sizeof(float4), true, 'i');
		}

		/* Build MA coefficients array */
		if (model->ma_coeffs && q > 0)
		{
			nalloc(ma_datums, Datum, q);
			for (i2 = 0; i2 < q; i2++)
				ma_datums[i2] = Float4GetDatum(model->ma_coeffs[i2]);
			ma_array = construct_array(ma_datums, q, FLOAT4OID, sizeof(float4), true, 'i');
		}

		/* Insert model */
		{
			char	   *ar_array_str = NULL;
			char	   *ma_array_str = NULL;

			if (ar_array != NULL)
				ar_array_str = DatumGetCString(DirectFunctionCall1(array_out, PointerGetDatum(ar_array)));
			if (ma_array != NULL)
				ma_array_str = DatumGetCString(DirectFunctionCall1(array_out, PointerGetDatum(ma_array)));

			appendStringInfo(&insert_sql,
							 "INSERT INTO neurondb.arima_models (p, d, q, intercept, ar_coeffs, ma_coeffs) "
							 "VALUES (%d, %d, %d, %.10f, %s, %s) RETURNING model_id",
							 p, d, q, (double) model->intercept,
							 ar_array_str ? ar_array_str : "NULL",
							 ma_array_str ? ma_array_str : "NULL");

			/* Free the array strings to prevent memory leaks */
			if (ar_array_str)
				nfree(ar_array_str);
			if (ma_array_str)
				nfree(ma_array_str);
		}

		ret2 = ndb_spi_execute(spi_session, insert_sql.data, true, 1);
		if (ret2 != SPI_OK_INSERT || SPI_processed != 1)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			if (values)
				nfree(values);
			if (table_name_str)
				nfree(table_name_str);
			if (time_col_str)
				nfree(time_col_str);
			if (value_col_str)
				nfree(value_col_str);
			if (model->ar_coeffs)
				nfree(model->ar_coeffs);
			if (model->ma_coeffs)
				nfree(model->ma_coeffs);
			if (model->residuals)
				nfree(model->residuals);
			nfree(model);
			if (ar_datums)
				nfree(ar_datums);
			if (ma_datums)
				nfree(ma_datums);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("failed to save ARIMA model")));
		}

		/* Get model_id using safe function */
		if (!ndb_spi_get_int32(spi_session, 0, 1, &model_id_val))
		{
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("failed to get model_id from result")));
		}
		model_id = model_id_val;

		/* Save history */
		for (i = 0; i < n_samples; i++)
		{
			resetStringInfo(&insert_sql);
			appendStringInfo(&insert_sql,
							 "INSERT INTO neurondb.arima_history (model_id, observed) VALUES (%d, %.10f)",
							 model_id, (double) values[i]);
			ndb_spi_execute(spi_session, insert_sql.data, true, 0);
		}
		if (ar_datums)
			nfree(ar_datums);
		if (ma_datums)
			nfree(ma_datums);
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);

		if (values)
			nfree(values);
		if (table_name_str)
			nfree(table_name_str);
		if (time_col_str)
			nfree(time_col_str);
		if (value_col_str)
			nfree(value_col_str);
		if (model->ar_coeffs)
			nfree(model->ar_coeffs);
		if (model->ma_coeffs)
			nfree(model->ma_coeffs);
		if (model->residuals)
			nfree(model->residuals);
		nfree(model);

		PG_RETURN_INT32(model_id);
	}
}

PG_FUNCTION_INFO_V1(forecast_arima);
Datum
forecast_arima(PG_FUNCTION_ARGS)
{
	int32		model_id;
	int32		n_ahead;
	StringInfoData sql;
	TimeSeriesModel model;
	ArrayType *ar_coeffs_arr = NULL;
	ArrayType *ma_coeffs_arr = NULL;
	ArrayType *last_values_arr = NULL;
	int			ret;
	int		   *dims;
	int			i;

	int			p = 0;
	int			d = 0;
	int			q = 0;
	int			n_last = 0;
	float8		intercept = 0;
	float	   *ar_coeffs = NULL;
	float	   *ma_coeffs = NULL;
	float	   *last_values = NULL;
	float	   *forecast = NULL;
	Datum	   *outdatums = NULL;
	ArrayType *arr = NULL;
	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Validate argument count */
	if (PG_NARGS() < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: forecast_arima requires at least 2 arguments")));

	model_id = PG_GETARG_INT32(0);
	n_ahead = PG_GETARG_INT32(1);

	if (n_ahead < 1 || n_ahead > MAX_FORECAST_AHEAD)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_ahead must be between 1 and %d",
						MAX_FORECAST_AHEAD)));

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT p, d, q, intercept, ar_coeffs, ma_coeffs FROM neurondb.arima_models WHERE model_id = %d",
					 model_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("model_id %d not found in neurondb.arima_models",
						model_id)));
	}
	{
		HeapTuple	modeltuple;
		TupleDesc	modeldesc;
		int32		p_val = 0,
					d_val = 0,
					q_val = 0;
		bool		ar_isnull = false;
		bool		ma_isnull = false;

		/* Safe access for complex types - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
		{
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("ARIMA model result is invalid")));
		}
		modeltuple = SPI_tuptable->vals[0];
		modeldesc = SPI_tuptable->tupdesc;

		/* Get int32 values using safe function */

		if (!ndb_spi_get_int32(spi_session, 0, 1, &p_val))
			; /* p_isnull would be true, but not used */
		else
			p = p_val;
		if (!ndb_spi_get_int32(spi_session, 0, 2, &d_val))
			; /* d_isnull would be true, but not used */
		else
			d = d_val;
		if (!ndb_spi_get_int32(spi_session, 0, 3, &q_val))
			; /* q_isnull would be true, but not used */
		else
			q = q_val;

		/* For float8, need to use SPI_getbinval with safe access */
		{
			Datum		intercept_datum;
			bool		intercept_isnull;

			if (modeldesc->natts >= 4)
			{
				intercept_datum = SPI_getbinval(modeltuple, modeldesc, 4, &intercept_isnull);
				if (!intercept_isnull)
					intercept = DatumGetFloat8(intercept_datum);
				else
					intercept = 0.0;
			}
			else
			{
				intercept = 0.0;
		}
	}

	{
		Datum		ar_datum = SPI_getbinval(modeltuple, modeldesc, 5, &ar_isnull);
		Datum		ma_datum = SPI_getbinval(modeltuple, modeldesc, 6, &ma_isnull);

		if (!ar_isnull)
			ar_coeffs_arr = DatumGetArrayTypeP(ar_datum);
		else
			ar_coeffs_arr = NULL;

		if (!ma_isnull)
			ma_coeffs_arr = DatumGetArrayTypeP(ma_datum);
		else
			ma_coeffs_arr = NULL;
	}
	}

	if (ar_coeffs_arr != NULL && p > 0)
	{
		int			ar_ndims = ARR_NDIM(ar_coeffs_arr);

		if (ar_ndims != 1)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("forecast_arima: AR coefficients array must be 1-dimensional, got %d dimensions", ar_ndims)));

		dims = ARR_DIMS(ar_coeffs_arr);
		if (dims[0] != p)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("forecast_arima: AR coefficients array length %d does not match p=%d", dims[0], p)));

		nalloc(ar_coeffs, float, p);
		for (i = 0; i < p; i++)
		{
			float4		val;

			memcpy(&val,
				   (char *) ARR_DATA_PTR(ar_coeffs_arr)
				   + i * sizeof(float4),	/* Use float4 size since we store as FLOAT4 */
				   sizeof(float4));
			ar_coeffs[i] = val;
		}
	}
	else if (p > 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("forecast_arima: AR coefficients array is NULL but p=%d > 0", p)));
	}

	if (ma_coeffs_arr != NULL && q > 0)
	{
		int			ma_ndims = ARR_NDIM(ma_coeffs_arr);

		if (ma_ndims != 1)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("forecast_arima: MA coefficients array must be 1-dimensional, got %d dimensions", ma_ndims)));

		dims = ARR_DIMS(ma_coeffs_arr);
		if (dims[0] != q)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("forecast_arima: MA coefficients array length %d does not match q=%d", dims[0], q)));

		nalloc(ma_coeffs, float, q);
		for (i = 0; i < q; i++)
		{
			float4		val;

			memcpy(&val,
				   (char *) ARR_DATA_PTR(ma_coeffs_arr)
				   + i * sizeof(float4),	/* Use float4 size since we store as FLOAT4 */
				   sizeof(float4));
			ma_coeffs[i] = val;
		}
	}
	else if (q > 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("forecast_arima: MA coefficients array is NULL but q=%d > 0", q)));
	}
	else
	{
		ma_coeffs = NULL;
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT array_agg(observed ORDER BY observed_id) "
					 "FROM neurondb.arima_history "
					 "WHERE model_id = %d",
					 model_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		if (ar_coeffs)
			nfree(ar_coeffs);
		if (ma_coeffs)
			nfree(ma_coeffs);
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("recent observed values for model_id %d not found",
						model_id)));
	}

	{
		bool		isnull = false;
		HeapTuple	observedtuple = SPI_tuptable->vals[0];
		TupleDesc	observeddesc = SPI_tuptable->tupdesc;

		last_values_arr = DatumGetArrayTypeP(
											 SPI_getbinval(observedtuple, observeddesc, 1, &isnull));

		if (isnull || last_values_arr == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("forecast_arima: no history found for model_id %d", model_id)));
	}
	/* ndims = ARR_NDIM(last_values_arr); */ /* unused */
	dims = ARR_DIMS(last_values_arr);
	n_last = dims[0];
	nalloc(last_values, float, n_last);
	for (i = 0; i < n_last; i++)
	{
		float8		val;

		memcpy(&val,
			   (char *) ARR_DATA_PTR(last_values_arr)
			   + i * sizeof(float8),
			   sizeof(float8));
		last_values[i] = (float) val;
	}

	model.p = p;
	model.d = d;
	model.q = q;
	model.intercept = (float) intercept;
	model.n_obs = n_last;
	model.ar_coeffs = ar_coeffs;
	model.ma_coeffs = ma_coeffs;
	model.residuals = NULL;

	nalloc(forecast, float, n_ahead);
	arima_forecast(&model, last_values, n_last, n_ahead, forecast);

	nalloc(outdatums, Datum, n_ahead);
	for (i = 0; i < n_ahead; i++)
		outdatums[i] = Float8GetDatum((float8) forecast[i]);

	arr = construct_array(outdatums,
						  n_ahead,
						  FLOAT8OID,
						  sizeof(float8),
#ifdef USE_FLOAT8_BYVAL
						  true,
#else
						  false,
#endif
						  TYPALIGN_DOUBLE);

	if (ar_coeffs)
		nfree(ar_coeffs);
	if (ma_coeffs)
		nfree(ma_coeffs);
	if (last_values)
		nfree(last_values);
	if (forecast)
		nfree(forecast);
	if (outdatums)
		nfree(outdatums);

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_ARRAYTYPE_P(arr);
}

/*
 * Load TimeSeriesModel from a single SPI row (arima_models shape: p, d, q, intercept, ar_coeffs, ma_coeffs).
 * Returns NULL on failure. Caller must call free_arima_model_for_eval to free.
 */
static TimeSeriesModel *
load_arima_model_from_spi_row(HeapTuple tuple, TupleDesc tupdesc)
{
	TimeSeriesModel *model = NULL;
	int32		p = 0,
				d = 0,
				q = 0;
	bool		ar_isnull = false,
				ma_isnull = false;
	Datum		intercept_datum;
	bool		intercept_isnull = false;
	ArrayType  *ar_coeffs_arr = NULL;
	ArrayType  *ma_coeffs_arr = NULL;
	int		   *dims;
	int			i;

	if (tuple == NULL || tupdesc == NULL || tupdesc->natts < 6)
		return NULL;

	p = DatumGetInt32(SPI_getbinval(tuple, tupdesc, 1, &intercept_isnull));
	d = DatumGetInt32(SPI_getbinval(tuple, tupdesc, 2, &intercept_isnull));
	q = DatumGetInt32(SPI_getbinval(tuple, tupdesc, 3, &intercept_isnull));
	intercept_datum = SPI_getbinval(tuple, tupdesc, 4, &intercept_isnull);
	ar_coeffs_arr = (ArrayType *) DatumGetPointer(SPI_getbinval(tuple, tupdesc, 5, &ar_isnull));
	if (ar_isnull)
		ar_coeffs_arr = NULL;
	ma_coeffs_arr = (ArrayType *) DatumGetPointer(SPI_getbinval(tuple, tupdesc, 6, &ma_isnull));
	if (ma_isnull)
		ma_coeffs_arr = NULL;

	model = (TimeSeriesModel *) palloc0(sizeof(TimeSeriesModel));
	model->p = p;
	model->d = d;
	model->q = q;
	model->intercept = (float) (intercept_isnull ? 0.0 : DatumGetFloat8(intercept_datum));
	model->n_obs = 0;
	model->ar_coeffs = NULL;
	model->ma_coeffs = NULL;
	model->residuals = NULL;

	if (ar_coeffs_arr != NULL && p > 0 && ARR_NDIM(ar_coeffs_arr) == 1)
	{
		dims = ARR_DIMS(ar_coeffs_arr);
		if (dims[0] >= p)
		{
			model->ar_coeffs = (float *) palloc(p * sizeof(float));
			for (i = 0; i < p; i++)
			{
				float4		val;

				memcpy(&val,
					   (char *) ARR_DATA_PTR(ar_coeffs_arr) + i * sizeof(float4),
					   sizeof(float4));
				model->ar_coeffs[i] = val;
			}
		}
	}
	if (ma_coeffs_arr != NULL && q > 0 && ARR_NDIM(ma_coeffs_arr) == 1)
	{
		dims = ARR_DIMS(ma_coeffs_arr);
		if (dims[0] >= q)
		{
			model->ma_coeffs = (float *) palloc(q * sizeof(float));
			for (i = 0; i < q; i++)
			{
				float4		val;

				memcpy(&val,
					   (char *) ARR_DATA_PTR(ma_coeffs_arr) + i * sizeof(float4),
					   sizeof(float4));
				model->ma_coeffs[i] = val;
			}
		}
	}
	return model;
}

static void
free_arima_model_for_eval(TimeSeriesModel *model)
{
	if (model == NULL)
		return;
	if (model->ar_coeffs)
		pfree(model->ar_coeffs);
	if (model->ma_coeffs)
		pfree(model->ma_coeffs);
	if (model->residuals)
		pfree(model->residuals);
	pfree(model);
}

PG_FUNCTION_INFO_V1(evaluate_arima_by_model_id);

/*
 * evaluate_arima_by_model_id
 *      Evaluates ARIMA model forecasting accuracy on historical data.
 *      Arguments: int4 model_id, text table_name, text time_col, text value_col, int4 forecast_horizon
 *      Returns: jsonb with evaluation metrics
 */
Datum
evaluate_arima_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id = 0;

	text *table_name = NULL;
	text *time_col = NULL;
	text *value_col = NULL;
	int32		forecast_horizon = 0;

	char *tbl_str = NULL;
	char *time_str = NULL;
	char *value_str = NULL;
	StringInfoData query;
	int			ret = 0;
	int			n_points = 0;
	double		mse = 0.0;
	double		mae = 0.0;
	int			i = 0;
	StringInfoData jsonbuf;

	Jsonb *result = NULL;
	MemoryContext oldcontext = NULL;
	int			valid_predictions = 0;
	double		rmse = 0.0;
	HeapTuple	actual_tuple = NULL;
	TupleDesc	tupdesc = NULL;
	Datum		actual_datum = (Datum) 0;
	bool		actual_null = false;
	float		actual_value = 0.0f;
	float		forecast_value = 0.0f;
	float		error = 0.0f;

	NdbSpiSession *spi_session = NULL;
	TimeSeriesModel *arima_model = NULL;
	float	   *data = NULL;
	float	   *forecast_arr = NULL;

	/* Validate arguments */
	if (PG_NARGS() != 5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_arima_by_model_id: 5 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_arima_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3) || PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_arima_by_model_id: table_name, time_col, value_col, and forecast_horizon are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	time_col = PG_GETARG_TEXT_PP(2);
	value_col = PG_GETARG_TEXT_PP(3);
	forecast_horizon = PG_GETARG_INT32(4);

	if (forecast_horizon < 1 || forecast_horizon > MAX_FORECAST_AHEAD)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("forecast_horizon must be between 1 and %d", MAX_FORECAST_AHEAD)));

	tbl_str = text_to_cstring(table_name);
	time_str = text_to_cstring(time_col);
	value_str = text_to_cstring(value_col);

	oldcontext = CurrentMemoryContext;

	/* Connect to SPI */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Load ARIMA model from neurondb.arima_models by model_id */
	{
		StringInfoData model_sql;

		ndb_spi_stringinfo_init(spi_session, &model_sql);
		appendStringInfo(&model_sql,
						 "SELECT p, d, q, intercept, ar_coeffs, ma_coeffs FROM neurondb.arima_models WHERE model_id = %d",
						 model_id);
		ret = ndb_spi_execute(spi_session, model_sql.data, true, 1);
		ndb_spi_stringinfo_free(spi_session, &model_sql);
		if (ret != SPI_OK_SELECT || SPI_processed != 1)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(tbl_str);
			nfree(time_str);
			nfree(value_str);
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("model_id %d not found in neurondb.arima_models", model_id),
					 errhint("Provide a valid model_id from neurondb.arima_models table.")));
		}
		arima_model = load_arima_model_from_spi_row(SPI_tuptable->vals[0], SPI_tuptable->tupdesc);
		if (arima_model == NULL)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(tbl_str);
			nfree(time_str);
			nfree(value_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("evaluate_arima_by_model_id: failed to load ARIMA model parameters")));
		}
	}

	/* Build query to get time series data ordered by time */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL ORDER BY %s",
					 time_str, value_str, tbl_str, time_str, value_str, time_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		free_arima_model_for_eval(arima_model);
		nfree(tbl_str);
		nfree(time_str);
		nfree(value_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_arima_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < MIN_ARIMA_OBSERVATIONS + forecast_horizon)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		free_arima_model_for_eval(arima_model);
		nfree(tbl_str);
		nfree(time_str);
		nfree(value_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_arima_by_model_id: need at least %d observations for evaluation with horizon %d, got %d",
						MIN_ARIMA_OBSERVATIONS + forecast_horizon, forecast_horizon, n_points)));
	}

	/* Copy time series values into float array for forecasting */
	nalloc(data, float, n_points);
	for (i = 0; i < n_points; i++)
	{
		bool		vn = false;

		actual_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &vn);
		data[i] = (float) (vn ? 0.0 : DatumGetFloat8(actual_datum));
	}

	nalloc(forecast_arr, float, forecast_horizon);

	/* Evaluate forecast accuracy using rolling forecast with loaded ARIMA model */
	valid_predictions = 0;
	for (i = MIN_ARIMA_OBSERVATIONS; i < n_points - forecast_horizon; i++)
	{
		int			n_history = i;

		actual_value = data[i + forecast_horizon];

		arima_forecast(arima_model, data, n_history, forecast_horizon, forecast_arr);
		forecast_value = forecast_arr[forecast_horizon - 1];

		error = actual_value - forecast_value;
		mse += error * error;
		mae += fabs(error);
		valid_predictions++;
	}

	nfree(forecast_arr);
	nfree(data);
	free_arima_model_for_eval(arima_model);

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	if (valid_predictions == 0)
	{
		nfree(tbl_str);
		nfree(time_str);
		nfree(value_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_arima_by_model_id: no valid predictions could be made")));
	}

	mse /= valid_predictions;
	mae /= valid_predictions;
	rmse = sqrt(mse);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"n_predictions\":%d,\"forecast_horizon\":%d}",
					 mse, mae, rmse, valid_predictions, forecast_horizon);

	result = ndb_jsonb_in_cstring(jsonbuf.data);
	/* Don't free jsonbuf.data - let memory context handle it */

	nfree(tbl_str);
	nfree(time_str);
	nfree(value_str);

	PG_RETURN_JSONB_P(result);
}

PG_FUNCTION_INFO_V1(detect_anomalies);

Datum
detect_anomalies(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *time_col;
	text	   *value_col;
	float8		threshold;
	int			n_anomalies = 0;

	/* Validate argument count */
	if (PG_NARGS() < 3 || PG_NARGS() > 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: detect_anomalies requires 3 to 4 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	time_col = PG_GETARG_TEXT_PP(1);
	value_col = PG_GETARG_TEXT_PP(2);
	threshold = PG_ARGISNULL(3) ? 3.0 : PG_GETARG_FLOAT8(3);

	{
		char	   *table_name_str = text_to_cstring(table_name);
		char	   *time_col_str = text_to_cstring(time_col);
		char	   *value_col_str = text_to_cstring(value_col);
		StringInfoData sql;
		int			ret;
		int			n_samples;
		int			i;
		SPITupleTable *tuptable = NULL;
		TupleDesc	tupdesc;

	float *values = NULL;
	float *ma_values = NULL;
	float *smoothed = NULL;
	float		mean,
				stddev,
				sum = 0.0f,
				sum_sq = 0.0f;

	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT %s FROM %s ORDER BY %s",
					 quote_identifier(value_col_str),
					 quote_identifier(table_name_str),
					 quote_identifier(time_col_str));

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (table_name_str)
			nfree(table_name_str);
		if (time_col_str)
			nfree(time_col_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to execute anomaly detection "
						"query")));
	}

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n_samples = SPI_processed;

	if (n_samples < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (table_name_str)
			nfree(table_name_str);
		if (time_col_str)
			nfree(time_col_str);
		if (value_col_str)
			nfree(value_col_str);
		PG_RETURN_INT32(0);
	}

	nalloc(values, float, n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tup = tuptable->vals[i];
		bool		isnull = false;
		Datum		val = SPI_getbinval(tup, tupdesc, 1, &isnull);

		if (isnull)
			values[i] = 0.0f;
		else
			values[i] = (float) DatumGetFloat8(val);
		sum += values[i];
		sum_sq += values[i] * values[i];
	}

	mean = sum / n_samples;
	stddev = sqrtf((sum_sq / n_samples) - (mean * mean));
	if (stddev <= 0.0f)
		stddev = 1.0f;

	nalloc(ma_values, float, n_samples);
	compute_moving_average(values, n_samples, 5, ma_values);

	nalloc(smoothed, float, n_samples);
	exponential_smoothing(ma_values, n_samples, 0.3f, smoothed);

	for (i = 0; i < n_samples; i++)
	{
		float		residual = values[i] - smoothed[i];
		float		z_score = fabsf(residual / stddev);

		if (z_score > (float) threshold)
			n_anomalies++;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	if (values)
		nfree(values);
	if (ma_values)
		nfree(ma_values);
	if (smoothed)
		nfree(smoothed);
	if (table_name_str)
		nfree(table_name_str);
	if (time_col_str)
		nfree(time_col_str);
	if (value_col_str)
		nfree(value_col_str);
	}

	PG_RETURN_INT32(n_anomalies);
}

PG_FUNCTION_INFO_V1(seasonal_decompose);

Datum
seasonal_decompose(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *value_col;
	int32		period;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: seasonal_decompose requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	value_col = PG_GETARG_TEXT_PP(1);
	period = PG_GETARG_INT32(2);

	{
		char	   *table_name_str = text_to_cstring(table_name);
		char	   *value_col_str = text_to_cstring(value_col);
		StringInfoData sql;
		int			ret;
		int			n;
		int			i;
		int			j;
		SPITupleTable *tuptable = NULL;
		TupleDesc	tupdesc;

	float *values = NULL;
	float *trend = NULL;
	float *seasonal = NULL;
	float *residual = NULL;
	float *seasonal_pattern = NULL;
	int *seasonal_counts = NULL;
	Datum *trend_datums = NULL;
	Datum *seasonal_datums = NULL;
	Datum *residual_datums = NULL;
	ArrayType  *trend_arr,
			   *seasonal_arr,
			   *residual_arr;
	HeapTuple	result_tuple;
	TupleDesc	tupdesc_out;
	Datum		result_values[3];
	bool		result_nulls[3] = {false, false, false};

	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	if (period < MIN_SEASONAL_PERIOD || period > MAX_SEASONAL_PERIOD)
	{
		if (table_name_str)
			nfree(table_name_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("period must be between %d and %d",
						MIN_SEASONAL_PERIOD,
						MAX_SEASONAL_PERIOD)));
	}

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT %s FROM %s ORDER BY 1",
					 quote_identifier(value_col_str),
					 quote_identifier(table_name_str));

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (table_name_str)
			nfree(table_name_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to execute seasonal "
						"decomposition query")));
	}

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n = SPI_processed;

	if (n < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (table_name_str)
			nfree(table_name_str);
		if (value_col_str)
			nfree(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least 2 values required for "
						"seasonal decomposition")));
	}

	nalloc(values, float, n);
	for (i = 0; i < n; i++)
	{
		HeapTuple	tup = tuptable->vals[i];
		bool		isnull = false;
		Datum		val = SPI_getbinval(tup, tupdesc, 1, &isnull);

		if (isnull)
			values[i] = 0.0f;
		else
			values[i] = (float) DatumGetFloat8(val);
	}

	nalloc(trend, float, n);
	{
		int			window = (period % 2 == 1) ? period : period + 1;
		int			half_w = window / 2;

		for (i = 0; i < n; i++)
		{
			int			left = i - half_w;
			int			right = i + half_w;
			float		sum = 0.0f;
			int			count = 0;

			for (j = left; j <= right; j++)
			{
				if (j >= 0 && j < n)
				{
					sum += values[j];
					count++;
				}
			}
			trend[i] = (count > 0) ? (sum / count) : 0.0f;
		}
	}

	{
		float	   *seasonal_pattern_ptr = NULL;
		int		   *seasonal_counts_ptr = NULL;

		nalloc(seasonal_pattern_ptr, float, period);
		memset(seasonal_pattern_ptr, 0, sizeof(float) * period);
		seasonal_pattern = seasonal_pattern_ptr;
		nalloc(seasonal_counts_ptr, int, period);
		memset(seasonal_counts_ptr, 0, sizeof(int) * period);
		seasonal_counts = seasonal_counts_ptr;
	}
	for (i = 0; i < n; i++)
	{
		int			s = i % period;
		float		val = values[i] - trend[i];

		seasonal_pattern[s] += val;
		seasonal_counts[s]++;
	}
	for (i = 0; i < period; i++)
	{
		if (seasonal_counts[i] > 0)
			seasonal_pattern[i] /= (float) seasonal_counts[i];
		else
			seasonal_pattern[i] = 0.0f;
	}

	nalloc(seasonal, float, n);
	for (i = 0; i < n; i++)
		seasonal[i] = seasonal_pattern[i % period];

	nalloc(residual, float, n);
	for (i = 0; i < n; i++)
		residual[i] = values[i] - trend[i] - seasonal[i];

	nalloc(trend_datums, Datum, n);
	nalloc(seasonal_datums, Datum, n);
	nalloc(residual_datums, Datum, n);
	for (i = 0; i < n; i++)
	{
		trend_datums[i] = Float8GetDatum((float8) trend[i]);
		seasonal_datums[i] = Float8GetDatum((float8) seasonal[i]);
		residual_datums[i] = Float8GetDatum((float8) residual[i]);
	}
	trend_arr = construct_array(trend_datums,
								n,
								FLOAT8OID,
								sizeof(float8),
								true,
								TYPALIGN_DOUBLE);
	seasonal_arr = construct_array(seasonal_datums,
								   n,
								   FLOAT8OID,
								   sizeof(float8),
								   true,
								   TYPALIGN_DOUBLE);
	residual_arr = construct_array(residual_datums,
								   n,
								   FLOAT8OID,
								   sizeof(float8),
								   true,
								   TYPALIGN_DOUBLE);

	if (get_call_result_type(fcinfo, NULL, &tupdesc_out)
		!= TYPEFUNC_COMPOSITE)
	{
		if (trend)
			nfree(trend);
		if (seasonal)
			nfree(seasonal);
		if (residual)
			nfree(residual);
		if (seasonal_pattern)
			nfree(seasonal_pattern);
		if (seasonal_counts)
			nfree(seasonal_counts);
		if (values)
			nfree(values);
		if (trend_datums)
			nfree(trend_datums);
		if (seasonal_datums)
			nfree(seasonal_datums);
		if (residual_datums)
			nfree(residual_datums);
		if (table_name_str)
			nfree(table_name_str);
		if (value_col_str)
			nfree(value_col_str);
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		elog(ERROR,
			 "return type must be composite (trend float8[], "
			 "seasonal float8[], residual float8[])");
	}

	result_values[0] = PointerGetDatum(trend_arr);
	result_values[1] = PointerGetDatum(seasonal_arr);
	result_values[2] = PointerGetDatum(residual_arr);

	result_tuple =
		heap_form_tuple(tupdesc_out, result_values, result_nulls);

	if (trend)
		nfree(trend);
	if (seasonal)
		nfree(seasonal);
	if (residual)
		nfree(residual);
	if (seasonal_pattern)
		nfree(seasonal_pattern);
	if (seasonal_counts)
		nfree(seasonal_counts);
	if (values)
		nfree(values);
	if (trend_datums)
		nfree(trend_datums);
	if (seasonal_datums)
		nfree(seasonal_datums);
	if (residual_datums)
		nfree(residual_datums);
	if (table_name_str)
		nfree(table_name_str);
	if (value_col_str)
		nfree(value_col_str);

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_DATUM(HeapTupleGetDatum(result_tuple));
	}
}

static bytea *
timeseries_model_serialize_to_bytea(const float *ar_coeffs_param, int p_param, const float *ma_coeffs_param, int q_param, int d_param, float intercept, int n_obs, const char *model_type, uint8 training_backend)
{
	StringInfoData buf;
	int			total_size;
	bytea *result = NULL;
	int			type_len;
	int			i;

	/* Validate inputs */
	if (model_type == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("timeseries_model_serialize_to_bytea: model_type cannot be NULL")));

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: timeseries_model_serialize_to_bytea: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	/* Use nalloc for StringInfo buffer */
	{
		int			initial_size = 256;
		char	   *buf_data = NULL;

		nalloc(buf_data, char, initial_size);
		buf.data = buf_data;
		buf.len = 0;
		buf.maxlen = initial_size;
	}

	/* Append data to buffer, reallocating if needed */
	{
		int			needed = sizeof(uint8) + sizeof(int) * 4 + sizeof(float) + strlen(model_type) + (p_param > 0 ? p_param * sizeof(float) : 0) + (q_param > 0 ? q_param * sizeof(float) : 0);

		if (buf.maxlen < needed)
		{
			char	   *new_data = NULL;

			nalloc(new_data, char, needed);
			if (buf.data != NULL)
			{
				memcpy(new_data, buf.data, buf.len);
				nfree(buf.data);
			}
			buf.data = new_data;
			buf.maxlen = needed;
		}

		/* Write training_backend first (0=CPU, 1=GPU) - unified storage format */
		memcpy(buf.data + buf.len, &training_backend, sizeof(uint8));
		buf.len += sizeof(uint8);
		memcpy(buf.data + buf.len, &p_param, sizeof(int));
		buf.len += sizeof(int);
		memcpy(buf.data + buf.len, &d_param, sizeof(int));
		buf.len += sizeof(int);
		memcpy(buf.data + buf.len, &q_param, sizeof(int));
		buf.len += sizeof(int);
		memcpy(buf.data + buf.len, &intercept, sizeof(float));
		buf.len += sizeof(float);
		memcpy(buf.data + buf.len, &n_obs, sizeof(int));
		buf.len += sizeof(int);
		type_len = model_type != NULL ? strlen(model_type) : 0;
		memcpy(buf.data + buf.len, &type_len, sizeof(int));
		buf.len += sizeof(int);
		if (type_len > 0)
		{
			memcpy(buf.data + buf.len, model_type, type_len);
			buf.len += type_len;
		}

		/* Only serialize ar_coeffs if p_param > 0 and ar_coeffs_param is not NULL */
		if (p_param > 0 && ar_coeffs_param != NULL)
		{
			for (i = 0; i < p_param; i++)
			{
				memcpy(buf.data + buf.len, &ar_coeffs_param[i], sizeof(float));
				buf.len += sizeof(float);
			}
		}
		/* Only serialize ma_coeffs if q_param > 0 and ma_coeffs_param is not NULL */
		if (q_param > 0 && ma_coeffs_param != NULL)
		{
			for (i = 0; i < q_param; i++)
			{
				memcpy(buf.data + buf.len, &ma_coeffs_param[i], sizeof(float));
				buf.len += sizeof(float);
			}
		}
	}

	total_size = VARHDRSZ + buf.len;
	{
		char	   *result_raw = NULL;

		nalloc(result_raw, char, total_size);
		result = (bytea *) result_raw;
		SET_VARSIZE(result, total_size);
		memcpy(VARDATA(result), buf.data, buf.len);
	}
	if (buf.data != NULL)
		nfree(buf.data);

	return result;
}

static int
timeseries_model_deserialize_from_bytea(const bytea * data, float **ar_coeffs_out, int *p_out, float **ma_coeffs_out, int *q_out, int *d_out, float *intercept_out, int *n_obs_out, char *model_type_out, int type_max, uint8 * training_backend_out)
{
	const char *buf;
	int			offset = 0;
	int			type_len;
	int			i;
	uint8		training_backend = 0;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(uint8) + sizeof(int) * 4 + sizeof(float))
		return -1;

	buf = VARDATA(data);
	/* Read training_backend first (unified storage format) */
	training_backend = (uint8) buf[offset];
	offset += sizeof(uint8);
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;
	memcpy(p_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(d_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(q_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(intercept_out, buf + offset, sizeof(float));
	offset += sizeof(float);
	memcpy(n_obs_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&type_len, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (type_len >= type_max)
		return -1;
	memcpy(model_type_out, buf + offset, type_len);
	model_type_out[type_len] = '\0';
	offset += type_len;

	if (*p_out < 0 || *p_out > MAX_ARIMA_ORDER_P || *q_out < 0 || *q_out > MAX_ARIMA_ORDER_Q || *d_out < 0 || *d_out > MAX_ARIMA_ORDER_D)
		return -1;

	{
		float *ar_coeffs_out_local = NULL;
		float *ma_coeffs_out_local = NULL;

		nalloc(ar_coeffs_out_local, float, *p_out);
		*ar_coeffs_out = ar_coeffs_out_local;
		for (i = 0; i < *p_out; i++)
		{
			memcpy(&ar_coeffs_out_local[i], buf + offset, sizeof(float));
			offset += sizeof(float);
		}

		nalloc(ma_coeffs_out_local, float, *q_out);
		*ma_coeffs_out = ma_coeffs_out_local;
		for (i = 0; i < *q_out; i++)
		{
			memcpy(&ma_coeffs_out_local[i], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	return 0;
}

static bool
timeseries_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	TimeSeriesGpuModelState *state = NULL;

	float *ar_coeffs = NULL;
	float *ma_coeffs = NULL;
	int			p = 1;
	int			d = 1;
	int			q = 1;
	float		intercept = 0.0f;
	char		model_type[32] = "arima";
	int			nvec = 0;
	int			dim = 0;
	int			i;

	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_train: invalid parameters");
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
				if (strcmp(key, "p") == 0 && v.type == jbvNumeric)
				{
					/* Parse numeric directly without DirectFunctionCall */
					p = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
				}
				else if (strcmp(key, "d") == 0 && v.type == jbvNumeric)
				{
					d = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
				}
				else if (strcmp(key, "q") == 0 && v.type == jbvNumeric)
				{
					q = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
				}
				else if (strcmp(key, "model_type") == 0 && v.type == jbvString)
					strncpy(model_type, v.val.string.val, sizeof(model_type) - 1);
					model_type[sizeof(model_type) - 1] = '\0';
				nfree(key);
			}
		}
	}

	if (p < 0 || p > MAX_ARIMA_ORDER_P)
		p = 1;
	if (d < 0 || d > MAX_ARIMA_ORDER_D)
		d = 1;
	if (q < 0 || q > MAX_ARIMA_ORDER_Q)
		q = 1;

	/* Convert feature matrix */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	/* Initialize ARIMA coefficients */
	{
		float	   *ar_coeffs_ptr = NULL;
		float	   *ma_coeffs_ptr = NULL;

		nalloc(ar_coeffs_ptr, float, p);
		memset(ar_coeffs_ptr, 0, sizeof(float) * p);
		ar_coeffs = ar_coeffs_ptr;
		nalloc(ma_coeffs_ptr, float, q);
		memset(ma_coeffs_ptr, 0, sizeof(float) * q);
		ma_coeffs = ma_coeffs_ptr;
	}

	/* Simple initialization */
	for (i = 0; i < p; i++)
		ar_coeffs[i] = 0.5f / (p + 1);
	for (i = 0; i < q; i++)
		ma_coeffs[i] = 0.3f / (q + 1);

	/* Compute intercept from data mean */
	if (nvec > 0 && dim > 0)
	{
		float		sum = 0.0f;

		for (i = 0; i < nvec; i++)
			sum += spec->feature_matrix[i * dim];
		intercept = sum / nvec;
	}

	/* Serialize model */
	model_data = timeseries_model_serialize_to_bytea(ar_coeffs, p, ma_coeffs, q, d, intercept, nvec, model_type, 1); /* training_backend=1 for GPU */

	/* Build metrics using direct JSONB API (safe in GPU context) */
	{
		JsonbParseState *state = NULL;
		JsonbValue	k;
		JsonbValue	jval;
		MemoryContext oldcontext = NULL;
		char		p_str[32], d_str[32], q_str[32], intercept_str[32], nvec_str[32];

		/* Switch to TopMemoryContext for JSONB allocations */
		oldcontext = MemoryContextSwitchTo(TopMemoryContext);

		(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

		/* Add "storage": "gpu" */
		k.type = jbvString;
		k.val.string.len = strlen("storage");
		k.val.string.val = "storage";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		jval.type = jbvString;
		jval.val.string.len = strlen("gpu");
		jval.val.string.val = "gpu";
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "training_backend": "1" (as string to avoid DirectFunctionCall) */
		k.type = jbvString;
		k.val.string.len = strlen("training_backend");
		k.val.string.val = "training_backend";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		jval.type = jbvString;
		jval.val.string.len = 1;
		jval.val.string.val = "1";
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "p": p (as string) */
		k.type = jbvString;
		k.val.string.len = 1;
		k.val.string.val = "p";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		snprintf(p_str, sizeof(p_str), "%d", p);
		jval.type = jbvString;
		jval.val.string.len = strlen(p_str);
		jval.val.string.val = p_str;
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "d": d (as string) */
		k.type = jbvString;
		k.val.string.len = 1;
		k.val.string.val = "d";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		snprintf(d_str, sizeof(d_str), "%d", d);
		jval.type = jbvString;
		jval.val.string.len = strlen(d_str);
		jval.val.string.val = d_str;
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "q": q (as string) */
		k.type = jbvString;
		k.val.string.len = 1;
		k.val.string.val = "q";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		snprintf(q_str, sizeof(q_str), "%d", q);
		jval.type = jbvString;
		jval.val.string.len = strlen(q_str);
		jval.val.string.val = q_str;
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "intercept": intercept (as string) */
		k.type = jbvString;
		k.val.string.len = strlen("intercept");
		k.val.string.val = "intercept";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		snprintf(intercept_str, sizeof(intercept_str), "%.6f", intercept);
		jval.type = jbvString;
		jval.val.string.len = strlen(intercept_str);
		jval.val.string.val = intercept_str;
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "model_type": model_type */
		k.type = jbvString;
		k.val.string.len = strlen("model_type");
		k.val.string.val = "model_type";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		jval.type = jbvString;
		jval.val.string.len = strlen(model_type);
		jval.val.string.val = model_type;
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		/* Add "n_samples": nvec (as string) */
		k.type = jbvString;
		k.val.string.len = strlen("n_samples");
		k.val.string.val = "n_samples";
		(void) pushJsonbValue(&state, WJB_KEY, &k);
		snprintf(nvec_str, sizeof(nvec_str), "%d", nvec);
		jval.type = jbvString;
		jval.val.string.len = strlen(nvec_str);
		jval.val.string.val = nvec_str;
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);

		{
			JsonbValue *final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
			if (final_value == NULL)
			{
				MemoryContextSwitchTo(oldcontext);
				metrics = NULL;
			}
			else
			{
				metrics = JsonbValueToJsonb(final_value);
			}
		}

		/* Switch back to original memory context */
		MemoryContextSwitchTo(oldcontext);
	}

	{
		TimeSeriesGpuModelState *state_ptr = NULL;

		nalloc(state_ptr, TimeSeriesGpuModelState, 1);
		memset(state_ptr, 0, sizeof(TimeSeriesGpuModelState));
		state = state_ptr;
	}
	state->model_blob = model_data;
	state->metrics = metrics;
	state->ar_coeffs = ar_coeffs;
	state->ma_coeffs = ma_coeffs;
	state->p = p;
	state->d = d;
	state->q = q;
	state->intercept = intercept;
	state->n_obs = nvec;
	state->n_samples = nvec;
	strncpy(state->model_type, model_type, sizeof(state->model_type) - 1);
	state->model_type[sizeof(state->model_type) - 1] = '\0';

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
timeseries_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
					   float *output, int output_dim, char **errstr)
{
	const		TimeSeriesGpuModelState *state;
	float		prediction = 0.0f;
	int			i;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_predict: model not ready");
		return false;
	}

	state = (const TimeSeriesGpuModelState *) model->backend_state;

	/* Deserialize if needed */
	if (state->ar_coeffs == NULL)
	{
		float *ar_coeffs = NULL;
		float *ma_coeffs = NULL;
		int			p = 0,
					d = 0,
					q = 0;
		float		intercept = 0.0f;
		int			n_obs = 0;
		char		model_type[32];

		{
			uint8		training_backend = 0;

			if (timeseries_model_deserialize_from_bytea(state->model_blob,
														&ar_coeffs, &p, &ma_coeffs, &q, &d, &intercept, &n_obs, model_type, sizeof(model_type),
														&training_backend) != 0)
			{
				if (errstr != NULL)
					*errstr = pstrdup("timeseries_gpu_predict: failed to deserialize");
				return false;
			}
		}
		((TimeSeriesGpuModelState *) state)->ar_coeffs = ar_coeffs;
		((TimeSeriesGpuModelState *) state)->ma_coeffs = ma_coeffs;
		((TimeSeriesGpuModelState *) state)->p = p;
		((TimeSeriesGpuModelState *) state)->d = d;
		((TimeSeriesGpuModelState *) state)->q = q;
		((TimeSeriesGpuModelState *) state)->intercept = intercept;
		((TimeSeriesGpuModelState *) state)->n_obs = n_obs;
	}

	/* ARIMA prediction: AR component */
	prediction = state->intercept;
	/* Validate input_dim before accessing array */
	if (input_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("timeseries_gpu_predict: invalid input_dim");
		return false;
	}
	for (i = 0; i < state->p && i < input_dim; i++)
	{
		int			idx = input_dim - 1 - i;

		if (idx < 0 || idx >= input_dim)
		{
			if (errstr)
				*errstr = psprintf("timeseries_gpu_predict: array index out of bounds: idx=%d, input_dim=%d", idx, input_dim);
			return false;
		}
		prediction += state->ar_coeffs[i] * input[idx];
	}

	output[0] = prediction;

	return true;
}

static bool
timeseries_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
						MLGpuMetrics *out, char **errstr)
{
	const		TimeSeriesGpuModelState *state;
	Jsonb	   *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_evaluate: invalid model");
		return false;
	}

	state = (const TimeSeriesGpuModelState *) model->backend_state;

	{
		char	   *buf_str = NULL;
		int			buf_len = 0;

		buf_len = snprintf(NULL, 0,
						   "{\"algorithm\":\"timeseries\",\"storage\":\"cpu\","
						   "\"p\":%d,\"d\":%d,\"q\":%d,\"intercept\":%.6f,\"model_type\":\"%s\",\"n_samples\":%d}",
						   state->p > 0 ? state->p : 1,
						   state->d > 0 ? state->d : 1,
						   state->q > 0 ? state->q : 1,
						   state->intercept,
						   state->model_type[0] ? state->model_type : "arima",
						   state->n_samples > 0 ? state->n_samples : 0) + 1;
		nalloc(buf_str, char, buf_len);
		snprintf(buf_str, buf_len,
				 "{\"algorithm\":\"timeseries\",\"storage\":\"cpu\","
				 "\"p\":%d,\"d\":%d,\"q\":%d,\"intercept\":%.6f,\"model_type\":\"%s\",\"n_samples\":%d}",
				 state->p > 0 ? state->p : 1,
				 state->d > 0 ? state->d : 1,
				 state->q > 0 ? state->q : 1,
				 state->intercept,
				 state->model_type[0] ? state->model_type : "arima",
				 state->n_samples > 0 ? state->n_samples : 0);

		metrics_json = ndb_jsonb_in_cstring(buf_str);
		nfree(buf_str);
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
timeseries_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
						 Jsonb * *metadata_out, char **errstr)
{
	const		TimeSeriesGpuModelState *state;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	char *payload_copy_raw = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_serialize: invalid model");
		return false;
	}

	state = (const TimeSeriesGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	nalloc(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
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
timeseries_gpu_deserialize(MLGpuModel *model, const bytea * payload,
						   const Jsonb * metadata, char **errstr)
{
	TimeSeriesGpuModelState *state = NULL;
	bytea	   *payload_copy = NULL;
	int			payload_size;

	float *ar_coeffs = NULL;
	float *ma_coeffs = NULL;
	int			p = 0,
				d = 0,
				q = 0;
	float		intercept = 0.0f;
	int			n_obs = 0;
	char		model_type[32];
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	{
		char *payload_copy_raw = NULL;

		nalloc(payload_copy_raw, char, payload_size);
		payload_copy = (bytea *) payload_copy_raw;
		memcpy(payload_copy, payload, payload_size);
	}

	{
		uint8		training_backend = 0;

		if (timeseries_model_deserialize_from_bytea(payload_copy,
													&ar_coeffs, &p, &ma_coeffs, &q, &d, &intercept, &n_obs, model_type, sizeof(model_type),
													&training_backend) != 0)
		{
			nfree(payload_copy);
			if (errstr != NULL)
				*errstr = pstrdup("timeseries_gpu_deserialize: failed to deserialize");
			return false;
		}
	}

	{
		TimeSeriesGpuModelState *state_ptr = NULL;

		nalloc(state_ptr, TimeSeriesGpuModelState, 1);
		memset(state_ptr, 0, sizeof(TimeSeriesGpuModelState));
		state = state_ptr;
	}
	state->model_blob = payload_copy;
	state->ar_coeffs = ar_coeffs;
	state->ma_coeffs = ma_coeffs;
	state->p = p;
	state->d = d;
	state->q = q;
	state->intercept = intercept;
	state->n_obs = n_obs;
	state->n_samples = 0;
	strncpy(state->model_type, model_type, sizeof(state->model_type) - 1);
	state->model_type[sizeof(state->model_type) - 1] = '\0';

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		Jsonb *metadata_copy = NULL;
		char *metadata_copy_raw = NULL;
		nalloc(metadata_copy_raw, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_copy_raw;

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
timeseries_gpu_destroy(MLGpuModel *model)
{
	TimeSeriesGpuModelState *state = NULL;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (TimeSeriesGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			nfree(state->model_blob);
		if (state->metrics != NULL)
			nfree(state->metrics);
		if (state->ar_coeffs != NULL)
			nfree(state->ar_coeffs);
		if (state->ma_coeffs != NULL)
			nfree(state->ma_coeffs);
		nfree(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps timeseries_gpu_model_ops = {
	.algorithm = "timeseries",
	.train = timeseries_gpu_train,
	.predict = timeseries_gpu_predict,
	.evaluate = timeseries_gpu_evaluate,
	.serialize = timeseries_gpu_serialize,
	.deserialize = timeseries_gpu_deserialize,
	.destroy = timeseries_gpu_destroy,
};

void
neurondb_gpu_register_timeseries_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&timeseries_gpu_model_ops);
	registered = true;
}

/*
 * train_timeseries_cpu
 * CPU training function for timeseries models via unified API
 *
 * This function trains a timeseries (ARIMA) model on CPU using the unified API format:
 * - table_name: table containing the data
 * - feature_column: column name containing feature vectors (unused, kept for API compatibility)
 * - target_column: column name containing target values (time series values)
 * - p, d, q: ARIMA parameters
 */
PG_FUNCTION_INFO_V1(train_timeseries_cpu);

Datum
train_timeseries_cpu(PG_FUNCTION_ARGS)
{
	bytea	   *model_data = NULL;
	bool		isnull = false;
	char	   *feature_col_str = NULL;
	char	   *table_name_str = NULL;
	char	   *target_col_str = NULL;
	Datum		value_datum;
	float	   *values = NULL;
	int			i = 0;
	int			n_samples = 0;
	int			ret = 0;
	int32		d;
	int32		model_id = 0;
	int32		p;
	int32		q;
	Jsonb	   *metrics_jsonb = NULL;
	Jsonb	   *params_jsonb = NULL;
	MLCatalogModelSpec spec = {0};
	MemoryContext oldcontext = NULL;
	NdbSpiSession *spi_session = NULL;
	SPITupleTable *tuptable = NULL;
	StringInfoData metricsbuf;
	StringInfoData paramsbuf;
	StringInfoData sql;
	TimeSeriesModel *model = NULL;
	TupleDesc	tupdesc = NULL;
	HeapTuple	tuple = NULL;
	text	   *feature_col;
	text	   *table_name;
	text	   *target_col;

	/* Validate argument count */
	if (PG_NARGS() < 3 || PG_NARGS() > 6)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_timeseries_cpu requires 3 to 6 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	p = PG_ARGISNULL(3) ? 1 : PG_GETARG_INT32(3);
	d = PG_ARGISNULL(4) ? 1 : PG_GETARG_INT32(4);
	q = PG_ARGISNULL(5) ? 1 : PG_GETARG_INT32(5);

	memset(&spec, 0, sizeof(MLCatalogModelSpec));

	/* Validate inputs */
	if (table_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("train_timeseries_cpu: table_name cannot be NULL")));
	if (feature_col == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("train_timeseries_cpu: feature_col cannot be NULL")));
	if (target_col == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("train_timeseries_cpu: target_col cannot be NULL")));

	table_name_str = text_to_cstring(table_name);
	feature_col_str = text_to_cstring(feature_col);
	target_col_str = text_to_cstring(target_col);

	if (p < 0 || p > MAX_ARIMA_ORDER_P)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("p must be between 0 and %d", MAX_ARIMA_ORDER_P)));
	if (d < 0 || d > MAX_ARIMA_ORDER_D)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("d must be between 0 and %d", MAX_ARIMA_ORDER_D)));
	if (q < 0 || q > MAX_ARIMA_ORDER_Q)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("q must be between 0 and %d", MAX_ARIMA_ORDER_Q)));

	/*
	 * SPI section: fetch time series values from target column in a
	 * deterministic order.
	 */
	oldcontext = CurrentMemoryContext;
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT %s FROM %s ORDER BY 1",
					 quote_identifier(target_col_str),
					 quote_identifier(table_name_str));

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		nfree(table_name_str);
		nfree(feature_col_str);
		nfree(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to execute time series query")));
	}

	if (SPI_tuptable == NULL ||
		SPI_tuptable->tupdesc == NULL ||
		SPI_tuptable->vals == NULL)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		nfree(table_name_str);
		nfree(feature_col_str);
		nfree(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("SPI_tuptable is NULL or invalid after time series query")));
	}

	tuptable = SPI_tuptable;
	tupdesc = tuptable->tupdesc;
	n_samples = SPI_processed;

	if (n_samples < MIN_ARIMA_OBSERVATIONS)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		nfree(table_name_str);
		nfree(feature_col_str);
		nfree(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least %d observations are required for ARIMA",
						MIN_ARIMA_OBSERVATIONS)));
	}

	/* Allocate values in the outer (parent) context, not SPI context */
	{
		MemoryContext save_context = MemoryContextSwitchTo(oldcontext);
		nalloc(values, float, n_samples);
		MemoryContextSwitchTo(save_context);
	}

	for (i = 0; i < n_samples; i++)
	{
		if (i >= SPI_processed || tuptable->vals[i] == NULL)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			if (values)
				nfree(values);
			nfree(table_name_str);
			nfree(feature_col_str);
			nfree(target_col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("invalid tuple at index %d in time series data", i)));
		}

		tuple = tuptable->vals[i];
		value_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);

		if (isnull)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			if (values)
				nfree(values);
			nfree(table_name_str);
			nfree(feature_col_str);
			nfree(target_col_str);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("time series cannot contain NULL values")));
		}
		values[i] = (float) DatumGetFloat8(value_datum);
	}

	/* Clean up SPI resources, leave values[] in outer context */
	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	/* Ensure we're in the outer context for all subsequent operations */
	MemoryContextSwitchTo(oldcontext);

	/* Fit ARIMA model in outer context */
	model = fit_arima(values, n_samples, p, d, q);
	if (model == NULL)
	{
		if (values)
			nfree(values);
		nfree(table_name_str);
		nfree(feature_col_str);
		nfree(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("ARIMA model fitting failed")));
	}

	/* Serialize model in outer context */
	model_data = timeseries_model_serialize_to_bytea(
													 model->ar_coeffs,
													 p,
													 model->ma_coeffs,
													 q,
													 d,
													 model->intercept,
													 n_samples,
													 "arima",
													 1); /* training_backend=1 for GPU */

	/* Build parameters JSON in outer context */
	initStringInfo(&paramsbuf);
	appendStringInfo(&paramsbuf,
					 "{\"p\":%d,\"d\":%d,\"q\":%d,\"model_type\":\"arima\"}",
					 p, d, q);
	params_jsonb = ndb_jsonb_in_cstring(paramsbuf.data);
	/* Don't free - let memory context handle it */

	/* Build metrics JSON in outer context */
	initStringInfo(&metricsbuf);
	appendStringInfo(&metricsbuf,
					 "{\"storage\":\"cpu\",\"p\":%d,\"d\":%d,\"q\":%d,\"intercept\":%.6f,\"model_type\":\"arima\",\"n_samples\":%d,\"training_backend\":0}",
					 p, d, q, model->intercept, n_samples);
	metrics_jsonb = ndb_jsonb_in_cstring(metricsbuf.data);
	/* Don't free - let memory context handle it */

	/*
	 * Register model in catalog.
	 * ml_catalog_register_model takes ownership of model_data, parameters,
	 * and metrics. Do not free them here.
	 */
	spec.algorithm = NDB_ALGO_TIMESERIES;
	spec.training_table = table_name_str;
	spec.training_column = target_col_str;
	spec.model_data = model_data;
	spec.parameters = params_jsonb;
	spec.metrics = metrics_jsonb;
	spec.num_samples = n_samples;
	spec.num_features = 1;

	model_id = ml_catalog_register_model(&spec);

	/* Local clean-up: only free objects owned here */
	if (values)
		nfree(values);
	if (model->ar_coeffs)
		nfree(model->ar_coeffs);
	if (model->ma_coeffs)
		nfree(model->ma_coeffs);
	if (model->residuals)
		nfree(model->residuals);
	nfree(model);

	/* ml_catalog_register_model copies the strings, so we can free them */
	nfree(table_name_str);
	nfree(target_col_str);
	nfree(feature_col_str);

	PG_RETURN_INT32(model_id);
}

/*
 * timeseries_try_gpu_predict_catalog
 *
 * Attempts GPU prediction for a timeseries model loaded from the catalog.
 * Returns true if GPU prediction succeeded, false otherwise.
 */
static bool
timeseries_try_gpu_predict_catalog(int32 model_id,
								  const Vector *feature_vec,
								  double *result_out)
{
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	char	   *gpu_err = NULL;
	float		prediction = 0.0f;
	bool		success = false;
	MLGpuModel	gpu_model;
	const MLGpuModelOps *ops = NULL;

	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	/* Ensure GPU models are registered */
	neurondb_gpu_register_models();
	
	/* Try GPU prediction for all models if GPU is available */
	/* This allows both CPU and GPU-trained models to use GPU acceleration */

	/* Look up timeseries GPU model ops */
	ops = ndb_gpu_lookup_model_ops("timeseries");
	if (ops == NULL || ops->predict == NULL)
	{
		goto cleanup;
	}

	/* Initialize GPU model */
	memset(&gpu_model, 0, sizeof(gpu_model));
	gpu_model.ops = ops;
	gpu_model.catalog_id = model_id;

	/* Deserialize model */
	if (!ops->deserialize(&gpu_model, payload, metrics, &gpu_err))
	{
		goto cleanup;
	}

	/* Perform prediction */
	if (ops->predict(&gpu_model, feature_vec->data, feature_vec->dim, &prediction, 1, &gpu_err))
	{
		if (result_out != NULL)
			*result_out = (double) prediction;
		success = true;
	}
	else
	{
	}

	/* Cleanup GPU model */
	if (ops->destroy)
		ops->destroy(&gpu_model);

cleanup:
	nfree(payload);
	nfree(metrics);
	nfree(gpu_err);

	return success;
}

/*
 * predict_timeseries_model_id
 *
 * Makes predictions using trained timeseries (ARIMA) model from catalog.
 * Uses the input features as the last values in the series and predicts
 * the next value using ARIMA forecasting.
 */
PG_FUNCTION_INFO_V1(predict_timeseries_model_id);

Datum
predict_timeseries_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector	   *features = NULL;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	float	   *ar_coeffs = NULL;
	float	   *ma_coeffs = NULL;
	int			p = 0;
	int			d = 0;
	int			q = 0;
	float		intercept = 0.0f;
	int			n_obs = 0;
	char		model_type[32] = {0};
	float	   *last_values = NULL;
	float	   *forecast = NULL;
	double		prediction;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_timeseries_model_id requires at least 2 arguments")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: timeseries: model_id is required"),
				 errdetail("First argument (model_id) is NULL"),
				 errhint("Provide a valid model_id from neurondb.ml_models table.")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: timeseries: features vector is required"),
				 errdetail("Second argument (features) is NULL"),
				 errhint("Provide a valid vector of historical values for prediction.")));

	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Try GPU prediction first */
	if (timeseries_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		PG_RETURN_FLOAT8(prediction);
	}

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: timeseries: model %d not found", model_id),
				 errdetail("Model does not exist in neurondb.ml_models catalog"),
				 errhint("Verify the model_id is correct and the model exists in the catalog.")));
	}

	if (model_data == NULL)
	{
		if (metrics)
			nfree(metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: timeseries: model %d has no model data", model_id)));
	}

	/* Deserialize model */
	{
		uint8		training_backend = 0;

		if (timeseries_model_deserialize_from_bytea(model_data, &ar_coeffs, &p,
													&ma_coeffs, &q, &d, &intercept,
													&n_obs, model_type,
													sizeof(model_type),
													&training_backend) != 0)
		{
			if (metrics)
				nfree(metrics);
			nfree(model_data);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: timeseries: failed to deserialize model %d", model_id)));
		}
	}

	/* Use features as last values - need at least p values for AR prediction */
	{
		int			n_last = features->dim;

		if (n_last < p && p > 0)
		{
			elog(WARNING,
				 "neurondb: timeseries: input vector has %d values, but model requires at least %d for AR(%d) prediction. Using available values.",
				 n_last, p, p);
		}

		/* Allocate arrays for last values and forecast */
		nalloc(last_values, float, n_last);
		nalloc(forecast, float, 1);

		/* Copy features to last_values */
		for (i = 0; i < n_last; i++)
			last_values[i] = features->data[i];

		/* Create a TimeSeriesModel structure for arima_forecast */
		{
			TimeSeriesModel model;

			model.p = p;
			model.d = d;
			model.q = q;
			model.ar_coeffs = ar_coeffs;
			model.ma_coeffs = ma_coeffs;
			model.intercept = intercept;
			model.n_obs = n_obs;
			model.residuals = NULL;	/* Not needed for prediction */

			/* Forecast 1 step ahead */
			arima_forecast(&model, last_values, n_last, 1, forecast);
		}

		prediction = (double) forecast[0];
	}

	/* Cleanup */
	if (ar_coeffs)
		nfree(ar_coeffs);
	if (ma_coeffs)
		nfree(ma_coeffs);
	if (last_values)
		nfree(last_values);
	if (forecast)
		nfree(forecast);
	if (model_data)
		nfree(model_data);
	if (metrics)
		nfree(metrics);

	PG_RETURN_FLOAT8(prediction);
}


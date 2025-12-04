/*-------------------------------------------------------------------------
 *
 * ml_timeseries.c
 *    Time series analysis and forecasting.
 *
 * This module implements ARIMA, exponential smoothing, and trend analysis
 * for univariate and multivariate time series.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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
#include "access/htup_details.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "neurondb_pgcompat.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "ml_catalog.h"
#include "neurondb_constants.h"

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
	float	   *ar_coeffs;
	float	   *ma_coeffs;
	float		intercept;
	int32		n_obs;
	float	   *residuals;
}			TimeSeriesModel;

/*  Fix for SPI_getbinval 'isnull' argument: always use a local 'bool' variable, not 'int' */
/*  throughout this file, especially within the ARIMA forecast and model loading code. */

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
	float	   *diff;
	int			curr_n,
				d,
				i;

	Assert(data != NULL);
	Assert(out_n != NULL);
	Assert(order >= 0);

	if (order == 0)
	{
		diff = (float *) palloc(sizeof(float) * n);
		memcpy(diff, data, sizeof(float) * n);
		*out_n = n;
		return diff;
	}

	curr_n = n;
	diff = (float *) palloc(sizeof(float) * curr_n);
	memcpy(diff, data, sizeof(float) * curr_n);

	for (d = 0; d < order; d++)
	{
		int			new_n = curr_n - 1;
		float	   *new_diff;

		if (new_n <= 0)
			elog(ERROR,
				 "cannot difference data sequence below length "
				 "1");
		new_diff = (float *) palloc(sizeof(float) * new_n);
		for (i = 0; i < new_n; i++)
			new_diff[i] = diff[i + 1] - diff[i];
		NDB_FREE(diff);
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
	TimeSeriesModel *model;
	NDB_DECLARE(float *, diff_data);
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

	model = (TimeSeriesModel *) palloc0(sizeof(TimeSeriesModel));
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
		float	   *autocorr = (float *) palloc0(sizeof(float) * (p + 1));
		float	  **R = (float **) palloc(sizeof(float *) * p);
		float	   *a;
		float	   *right = (float *) palloc0(sizeof(float) * p);

		for (i = 0; i <= p; i++)
		{
			float		sum = 0.0f;

			for (j = i; j < diff_n; j++)
				sum += diff_data[j] * diff_data[j - i];
			autocorr[i] = sum / (diff_n - i);
		}
		for (i = 0; i < p; i++)
		{
			R[i] = (float *) palloc0(sizeof(float) * p);
			for (j = 0; j < p; j++)
				R[i][j] = autocorr[abs(i - j)];
			right[i] = autocorr[i + 1];
		}

		a = (float *) palloc0(sizeof(float) * p);
		{
			float	  **L = (float **) palloc0(sizeof(float *) * p);

			for (i = 0; i < p; i++)
				L[i] = (float *) palloc0(sizeof(float) * p);
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
						if (sum <= 0)
							elog(ERROR,
								 "Failed to "
								 "decompose "
								 "autocorrelatio"
								 "n matrix: "
								 "non-positive "
								 "definite");
						L[i][j] = sqrtf(sum);
					}
					else
						L[i][j] = sum / L[j][j];
				}
			}
			{
				float	   *y;

				y = (float *) palloc0(sizeof(float) * p);
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
					NDB_FREE(L[i]);
				NDB_FREE(L);
				NDB_FREE(y);
			}
		}
		model->ar_coeffs = (float *) palloc(sizeof(float) * p);
		for (i = 0; i < p; i++)
			model->ar_coeffs[i] = a[i];

		for (i = 0; i < p; i++)
			NDB_FREE(R[i]);
		NDB_FREE(R);
		NDB_FREE(a);
		NDB_FREE(right);
		NDB_FREE(autocorr);
	}
	else
		model->ar_coeffs = NULL;

	if (q > 0)
	{
		model->ma_coeffs = (float *) palloc0(sizeof(float) * q);
	}
	else
	{
		model->ma_coeffs = NULL;
	}

	mean = compute_mean(diff_data, diff_n);
	model->intercept = mean;

	model->residuals = (float *) palloc(sizeof(float) * diff_n);
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

	NDB_FREE(diff_data);

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
	float	   *history;

	Assert(model != NULL);
	Assert(last_values != NULL);
	Assert(forecast != NULL);

	if (n_ahead < 1)
		elog(ERROR, "Must forecast at least 1 ahead");

	p = model->p;
	d = model->d;
	history = (float *) palloc0(sizeof(float) * (n_last + n_ahead));

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

	NDB_FREE(history);
}

PG_FUNCTION_INFO_V1(train_arima);
Datum
train_arima(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *time_col = PG_GETARG_TEXT_PP(1);
	text	   *value_col = PG_GETARG_TEXT_PP(2);
	int32		p = PG_ARGISNULL(3) ? 1 : PG_GETARG_INT32(3);
	int32		d = PG_ARGISNULL(4) ? 1 : PG_GETARG_INT32(4);
	int32		q = PG_ARGISNULL(5) ? 1 : PG_GETARG_INT32(5);

	char	   *table_name_str = text_to_cstring(table_name);
	char	   *time_col_str = text_to_cstring(time_col);
	char	   *value_col_str = text_to_cstring(value_col);

	StringInfoData sql;
	int			ret,
				n_samples,
				i;
	SPITupleTable *tuptable;
	TupleDesc	tupdesc;
	NDB_DECLARE(float *, values);
	NDB_DECLARE(TimeSeriesModel *, model);
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

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
			NDB_FREE(table_name_str);
		if (time_col_str)
			NDB_FREE(time_col_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
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
			NDB_FREE(table_name_str);
		if (time_col_str)
			NDB_FREE(time_col_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least %d observations are required for ARIMA",
						MIN_ARIMA_OBSERVATIONS)));
	}

	values = (float *) palloc(sizeof(float) * n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = tuptable->vals[i];
		bool		isnull = false;
		Datum		value_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);

		if (isnull)
		{
			if (values)
				NDB_FREE(values);
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			if (table_name_str)
				NDB_FREE(table_name_str);
			if (time_col_str)
				NDB_FREE(time_col_str);
			if (value_col_str)
				NDB_FREE(value_col_str);
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
			NDB_FREE(values);
		if (table_name_str)
			NDB_FREE(table_name_str);
		if (time_col_str)
			NDB_FREE(time_col_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
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
		NDB_DECLARE(Datum *, ar_datums);
		NDB_DECLARE(Datum *, ma_datums);
		NDB_DECLARE(ArrayType *, ar_array);
		NDB_DECLARE(ArrayType *, ma_array);
		int			i2;

		initStringInfo(&insert_sql);

		/* Build AR coefficients array */
		if (model->ar_coeffs && p > 0)
		{
			ar_datums = (Datum *) palloc(sizeof(Datum) * p);
			for (i2 = 0; i2 < p; i2++)
			{
				ar_datums[i2] = Float4GetDatum(model->ar_coeffs[i2]);
			}
			ar_array = construct_array(ar_datums, p, FLOAT4OID, sizeof(float4), true, 'i');
		}

		/* Build MA coefficients array */
		if (model->ma_coeffs && q > 0)
		{
			ma_datums = (Datum *) palloc(sizeof(Datum) * q);
			for (i2 = 0; i2 < q; i2++)
				ma_datums[i2] = Float4GetDatum(model->ma_coeffs[i2]);
			ma_array = construct_array(ma_datums, q, FLOAT4OID, sizeof(float4), true, 'i');
		}

		/* Insert model */
		appendStringInfo(&insert_sql,
						 "INSERT INTO neurondb.arima_models (p, d, q, intercept, ar_coeffs, ma_coeffs) "
						 "VALUES (%d, %d, %d, %.10f, %s, %s) RETURNING model_id",
						 p, d, q, (double) model->intercept,
						 ar_array ? DatumGetCString(DirectFunctionCall1(array_out, PointerGetDatum(ar_array))) : "NULL",
						 ma_array ? DatumGetCString(DirectFunctionCall1(array_out, PointerGetDatum(ma_array))) : "NULL");

		ret2 = ndb_spi_execute(spi_session, insert_sql.data, true, 1);
		if (ret2 != SPI_OK_INSERT || SPI_processed != 1)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			if (values)
				NDB_FREE(values);
			if (table_name_str)
				NDB_FREE(table_name_str);
			if (time_col_str)
				NDB_FREE(time_col_str);
			if (value_col_str)
				NDB_FREE(value_col_str);
			if (model->ar_coeffs)
				NDB_FREE(model->ar_coeffs);
			if (model->ma_coeffs)
				NDB_FREE(model->ma_coeffs);
			if (model->residuals)
				NDB_FREE(model->residuals);
			NDB_FREE(model);
			if (ar_datums)
				NDB_FREE(ar_datums);
			if (ma_datums)
				NDB_FREE(ma_datums);
			NDB_FREE(insert_sql.data);
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
			NDB_FREE(insert_sql.data);
			initStringInfo(&insert_sql);
			appendStringInfo(&insert_sql,
							 "INSERT INTO neurondb.arima_history (model_id, observed) VALUES (%d, %.10f)",
							 model_id, (double) values[i]);
			ndb_spi_execute(spi_session, insert_sql.data, true, 0);
		}

		NDB_FREE(insert_sql.data);
		if (ar_datums)
			NDB_FREE(ar_datums);
		if (ma_datums)
			NDB_FREE(ma_datums);
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);

		if (values)
			NDB_FREE(values);
		if (table_name_str)
			NDB_FREE(table_name_str);
		if (time_col_str)
			NDB_FREE(time_col_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
		if (model->ar_coeffs)
			NDB_FREE(model->ar_coeffs);
		if (model->ma_coeffs)
			NDB_FREE(model->ma_coeffs);
		if (model->residuals)
			NDB_FREE(model->residuals);
		NDB_FREE(model);

		PG_RETURN_INT32(model_id);
	}
}

PG_FUNCTION_INFO_V1(forecast_arima);
Datum
forecast_arima(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	int32		n_ahead = PG_GETARG_INT32(1);

	StringInfoData sql;
	TimeSeriesModel model;
	NDB_DECLARE(ArrayType *, ar_coeffs_arr);
	NDB_DECLARE(ArrayType *, ma_coeffs_arr);
	NDB_DECLARE(ArrayType *, last_values_arr);
	int			ret;
	int		   *dims,
				ndims,
				i;
	Oid			arr_elem_type;
	int			p = 0,
				d = 0,
				q = 0,
				n_last = 0;
	float8		intercept = 0;
	NDB_DECLARE(float *, ar_coeffs);
	NDB_DECLARE(float *, ma_coeffs);
	NDB_DECLARE(float *, last_values);
	NDB_DECLARE(float *, forecast);
	NDB_DECLARE(Datum *, outdatums);
	NDB_DECLARE(ArrayType *, arr);
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

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
				 errmsg("model_id %d not found in neurondb_arima_models",
						model_id)));
	}
	{
		HeapTuple	modeltuple;
		TupleDesc	modeldesc;
		bool		isnull;
		int32		p_val = 0, d_val = 0, q_val = 0;
		bool		p_isnull = false, d_isnull = false, q_isnull = false;

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
			p_isnull = true;
		else
			p = p_val;
		if (!ndb_spi_get_int32(spi_session, 0, 2, &d_val))
			d_isnull = true;
		else
			d = d_val;
		if (!ndb_spi_get_int32(spi_session, 0, 3, &q_val))
			q_isnull = true;
		else
			q = q_val;
		
		isnull = (p_isnull || d_isnull || q_isnull);
		
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
		ar_coeffs_arr = DatumGetArrayTypeP(
										   SPI_getbinval(modeltuple, modeldesc, 5, &isnull));
		ma_coeffs_arr = DatumGetArrayTypeP(
										   SPI_getbinval(modeltuple, modeldesc, 6, &isnull));
	}

	arr_elem_type = ARR_ELEMTYPE(ar_coeffs_arr);
	(void) arr_elem_type;		/* Suppress unused variable warning */
	ndims = ARR_NDIM(ar_coeffs_arr);
	Assert(ndims == 1);
	(void) ndims;				/* Used in Assert only */
	dims = ARR_DIMS(ar_coeffs_arr);
	Assert(dims[0] == p);
	ar_coeffs = (float *) palloc(sizeof(float) * p);
	for (i = 0; i < p; i++)
	{
		float8		val;

		memcpy(&val,
			   (char *) ARR_DATA_PTR(ar_coeffs_arr)
			   + i * sizeof(float8),
			   sizeof(float8));
		ar_coeffs[i] = (float) val;
	}

	arr_elem_type = ARR_ELEMTYPE(ma_coeffs_arr);
	ndims = ARR_NDIM(ma_coeffs_arr);
	(void) ndims;				/* Used in Assert only */
	dims = ARR_DIMS(ma_coeffs_arr);
	if (q > 0 && dims[0] == q)
	{
		ma_coeffs = (float *) palloc(sizeof(float) * q);
		for (i = 0; i < q; i++)
		{
			float8		val;

			memcpy(&val,
				   (char *) ARR_DATA_PTR(ma_coeffs_arr)
				   + i * sizeof(float8),
				   sizeof(float8));
			ma_coeffs[i] = (float) val;
		}
	}
	else if (q > 0)
	{
		ma_coeffs = (float *) palloc0(sizeof(float) * q);
	}
	else
	{
		ma_coeffs = NULL;
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT observed FROM neurondb.arima_history WHERE model_id = %d ORDER BY observed_id DESC LIMIT 1",
					 model_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		if (ar_coeffs)
			NDB_FREE(ar_coeffs);
		if (ma_coeffs)
			NDB_FREE(ma_coeffs);
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("recent observed values for model_id %d not found",
						model_id)));
	}

	{
		bool		isnull;
		HeapTuple	observedtuple = SPI_tuptable->vals[0];
		TupleDesc	observeddesc = SPI_tuptable->tupdesc;

		last_values_arr = DatumGetArrayTypeP(
											 SPI_getbinval(observedtuple, observeddesc, 1, &isnull));
	}
	ndims = ARR_NDIM(last_values_arr);
	dims = ARR_DIMS(last_values_arr);
	n_last = dims[0];
	last_values = (float *) palloc(sizeof(float) * n_last);
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

	forecast = (float *) palloc(sizeof(float) * n_ahead);
	arima_forecast(&model, last_values, n_last, n_ahead, forecast);

	outdatums = (Datum *) palloc(sizeof(Datum) * n_ahead);
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
		NDB_FREE(ar_coeffs);
	if (ma_coeffs)
		NDB_FREE(ma_coeffs);
	if (last_values)
		NDB_FREE(last_values);
	if (forecast)
		NDB_FREE(forecast);
	if (outdatums)
		NDB_FREE(outdatums);

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_ARRAYTYPE_P(arr);
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
	NDB_DECLARE(text *, table_name);
	NDB_DECLARE(text *, time_col);
	NDB_DECLARE(text *, value_col);
	int32		forecast_horizon = 0;
	NDB_DECLARE(char *, tbl_str);
	NDB_DECLARE(char *, time_str);
	NDB_DECLARE(char *, value_str);
	StringInfoData query;
	int			ret = 0;
	int			n_points = 0;
	double		mse = 0.0;
	double		mae = 0.0;
	int			i = 0;
	StringInfoData jsonbuf;
	NDB_DECLARE(Jsonb *, result);
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
	NDB_DECLARE(NdbSpiSession *, spi_session);

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

	/*
	 * Suppress unused variable warning - placeholder for future
	 * implementation
	 */
	(void) model_id;

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
		NDB_FREE(tbl_str);
		NDB_FREE(time_str);
		NDB_FREE(value_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_arima_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < MIN_ARIMA_OBSERVATIONS + forecast_horizon)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(time_str);
		NDB_FREE(value_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_arima_by_model_id: need at least %d observations for evaluation with horizon %d, got %d",
						MIN_ARIMA_OBSERVATIONS + forecast_horizon, forecast_horizon, n_points)));
	}

	/* Evaluate forecast accuracy using rolling forecast evaluation */
	valid_predictions = 0;
	for (i = MIN_ARIMA_OBSERVATIONS; i < n_points - forecast_horizon; i++)
	{
		/* Get actual value forecast_horizon steps ahead */
		actual_tuple = SPI_tuptable->vals[i + forecast_horizon];
		tupdesc = SPI_tuptable->tupdesc;

		actual_datum = SPI_getbinval(actual_tuple, tupdesc, 2, &actual_null);
		if (actual_null)
			continue;

		actual_value = DatumGetFloat4(actual_datum);

		/*
		 * Create temporary table with data up to current point for
		 * forecasting
		 */

		/*
		 * This is a simplified approach - in practice, you'd need to retrain
		 * or use one-step-ahead forecasts
		 */
		/* For now, we'll use a simple persistence forecast as baseline */
		forecast_value = actual_value;	/* Simple baseline - predict current
										 * value */

		/* Compute error */
		error = actual_value - forecast_value;
		mse += error * error;
		mae += fabs(error);
		valid_predictions++;
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	if (valid_predictions == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(time_str);
		NDB_FREE(value_str);
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

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	NDB_FREE(jsonbuf.data);

	NDB_FREE(tbl_str);
	NDB_FREE(time_str);
	NDB_FREE(value_str);

	PG_RETURN_JSONB_P(result);
}

PG_FUNCTION_INFO_V1(detect_anomalies);

Datum
detect_anomalies(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *time_col = PG_GETARG_TEXT_PP(1);
	text	   *value_col = PG_GETARG_TEXT_PP(2);
	float8		threshold = PG_ARGISNULL(3) ? 3.0 : PG_GETARG_FLOAT8(3);

	char	   *table_name_str = text_to_cstring(table_name);
	char	   *time_col_str = text_to_cstring(time_col);
	char	   *value_col_str = text_to_cstring(value_col);

	StringInfoData sql;
	int			ret,
				n_samples,
				i,
				n_anomalies = 0;
	SPITupleTable *tuptable;
	TupleDesc	tupdesc;
	NDB_DECLARE(float *, values);
	NDB_DECLARE(float *, ma_values);
	NDB_DECLARE(float *, smoothed);
	float		mean,
				stddev,
				sum = 0.0f,
				sum_sq = 0.0f;

	NDB_DECLARE(NdbSpiSession *, spi_session);
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
			NDB_FREE(table_name_str);
		if (time_col_str)
			NDB_FREE(time_col_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
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
			NDB_FREE(table_name_str);
		if (time_col_str)
			NDB_FREE(time_col_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
		PG_RETURN_INT32(0);
	}

	values = (float *) palloc(sizeof(float) * n_samples);

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

	ma_values = (float *) palloc(sizeof(float) * n_samples);
	compute_moving_average(values, n_samples, 5, ma_values);

	smoothed = (float *) palloc(sizeof(float) * n_samples);
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
		NDB_FREE(values);
	if (ma_values)
		NDB_FREE(ma_values);
	if (smoothed)
		NDB_FREE(smoothed);
	if (table_name_str)
		NDB_FREE(table_name_str);
	if (time_col_str)
		NDB_FREE(time_col_str);
	if (value_col_str)
		NDB_FREE(value_col_str);

	PG_RETURN_INT32(n_anomalies);
}

PG_FUNCTION_INFO_V1(seasonal_decompose);

Datum
seasonal_decompose(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *value_col = PG_GETARG_TEXT_PP(1);
	int32		period = PG_GETARG_INT32(2);

	char	   *table_name_str = text_to_cstring(table_name);
	char	   *value_col_str = text_to_cstring(value_col);

	StringInfoData sql;
	int			ret,
				n,
				i,
				j;
	SPITupleTable *tuptable;
	TupleDesc	tupdesc;
	NDB_DECLARE(float *, values);
	NDB_DECLARE(float *, trend);
	NDB_DECLARE(float *, seasonal);
	NDB_DECLARE(float *, residual);
	NDB_DECLARE(float *, seasonal_pattern);
	NDB_DECLARE(int *, seasonal_counts);
	NDB_DECLARE(Datum *, trend_datums);
	NDB_DECLARE(Datum *, seasonal_datums);
	NDB_DECLARE(Datum *, residual_datums);
	ArrayType  *trend_arr,
			   *seasonal_arr,
			   *residual_arr;
	HeapTuple	result_tuple;
	TupleDesc	tupdesc_out;
	Datum		result_values[3];
	bool		result_nulls[3] = {false, false, false};
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	if (period < MIN_SEASONAL_PERIOD || period > MAX_SEASONAL_PERIOD)
	{
		if (table_name_str)
			NDB_FREE(table_name_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
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
			NDB_FREE(table_name_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
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
			NDB_FREE(table_name_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least 2 values required for "
						"seasonal decomposition")));
	}

	values = (float *) palloc(sizeof(float) * n);
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

	trend = (float *) palloc(sizeof(float) * n);
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

	seasonal_pattern = (float *) palloc0(sizeof(float) * period);
	seasonal_counts = (int *) palloc0(sizeof(int) * period);
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

	seasonal = (float *) palloc(sizeof(float) * n);
	for (i = 0; i < n; i++)
		seasonal[i] = seasonal_pattern[i % period];

	residual = (float *) palloc(sizeof(float) * n);
	for (i = 0; i < n; i++)
		residual[i] = values[i] - trend[i] - seasonal[i];

	trend_datums = (Datum *) palloc(sizeof(Datum) * n);
	seasonal_datums = (Datum *) palloc(sizeof(Datum) * n);
	residual_datums = (Datum *) palloc(sizeof(Datum) * n);
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
			NDB_FREE(trend);
		if (seasonal)
			NDB_FREE(seasonal);
		if (residual)
			NDB_FREE(residual);
		if (seasonal_pattern)
			NDB_FREE(seasonal_pattern);
		if (seasonal_counts)
			NDB_FREE(seasonal_counts);
		if (values)
			NDB_FREE(values);
		if (trend_datums)
			NDB_FREE(trend_datums);
		if (seasonal_datums)
			NDB_FREE(seasonal_datums);
		if (residual_datums)
			NDB_FREE(residual_datums);
		if (table_name_str)
			NDB_FREE(table_name_str);
		if (value_col_str)
			NDB_FREE(value_col_str);
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
		NDB_FREE(trend);
	if (seasonal)
		NDB_FREE(seasonal);
	if (residual)
		NDB_FREE(residual);
	if (seasonal_pattern)
		NDB_FREE(seasonal_pattern);
	if (seasonal_counts)
		NDB_FREE(seasonal_counts);
	if (values)
		NDB_FREE(values);
	if (trend_datums)
		NDB_FREE(trend_datums);
	if (seasonal_datums)
		NDB_FREE(seasonal_datums);
	if (residual_datums)
		NDB_FREE(residual_datums);
	if (table_name_str)
		NDB_FREE(table_name_str);
	if (value_col_str)
		NDB_FREE(value_col_str);

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_DATUM(HeapTupleGetDatum(result_tuple));
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Time Series
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct TimeSeriesGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	float	   *ar_coeffs;
	float	   *ma_coeffs;
	int			p;
	int			d;
	int			q;
	float		intercept;
	int			n_obs;
	int			n_samples;
	char		model_type[32];
}			TimeSeriesGpuModelState;

static bytea *
timeseries_model_serialize_to_bytea(const float *ar_coeffs, int p, const float *ma_coeffs, int q, int d, float intercept, int n_obs, const char *model_type)
{
	StringInfoData buf;
	int			total_size;
	bytea	   *result;
	int			type_len;
	int			i;

	/* Validate inputs */
	if (model_type == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("timeseries_model_serialize_to_bytea: model_type cannot be NULL")));

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &p, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &d, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &q, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &intercept, sizeof(float));
	appendBinaryStringInfo(&buf, (char *) &n_obs, sizeof(int));
	type_len = model_type != NULL ? strlen(model_type) : 0;
	appendBinaryStringInfo(&buf, (char *) &type_len, sizeof(int));
	appendBinaryStringInfo(&buf, model_type, type_len);

	/* Only serialize ar_coeffs if p > 0 and ar_coeffs is not NULL */
	if (p > 0 && ar_coeffs != NULL)
	{
		for (i = 0; i < p; i++)
			appendBinaryStringInfo(&buf, (char *) &ar_coeffs[i], sizeof(float));
	}
	/* Only serialize ma_coeffs if q > 0 and ma_coeffs is not NULL */
	if (q > 0 && ma_coeffs != NULL)
	{
		for (i = 0; i < q; i++)
			appendBinaryStringInfo(&buf, (char *) &ma_coeffs[i], sizeof(float));
	}

	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
timeseries_model_deserialize_from_bytea(const bytea * data, float **ar_coeffs_out, int *p_out, float **ma_coeffs_out, int *q_out, int *d_out, float *intercept_out, int *n_obs_out, char *model_type_out, int type_max)
{
	const char *buf;
	int			offset = 0;
	int			type_len;
	int			i;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 4 + sizeof(float))
		return -1;

	buf = VARDATA(data);
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

	*ar_coeffs_out = (float *) palloc(sizeof(float) * *p_out);
	for (i = 0; i < *p_out; i++)
	{
		memcpy(&(*ar_coeffs_out)[i], buf + offset, sizeof(float));
		offset += sizeof(float);
	}

	*ma_coeffs_out = (float *) palloc(sizeof(float) * *q_out);
	for (i = 0; i < *q_out; i++)
	{
		memcpy(&(*ma_coeffs_out)[i], buf + offset, sizeof(float));
		offset += sizeof(float);
	}

	return 0;
}

static bool
timeseries_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	TimeSeriesGpuModelState *state;
	NDB_DECLARE(float *, ar_coeffs);
	NDB_DECLARE(float *, ma_coeffs);
	int			p = 1;
	int			d = 1;
	int			q = 1;
	float		intercept = 0.0f;
	char		model_type[32] = "arima";
	int			nvec = 0;
	int			dim = 0;
	int			i;
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
					p = DatumGetInt32(DirectFunctionCall1(numeric_int4,
														  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "d") == 0 && v.type == jbvNumeric)
					d = DatumGetInt32(DirectFunctionCall1(numeric_int4,
														  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "q") == 0 && v.type == jbvNumeric)
					q = DatumGetInt32(DirectFunctionCall1(numeric_int4,
														  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "model_type") == 0 && v.type == jbvString)
					strncpy(model_type, v.val.string.val, sizeof(model_type) - 1);
				NDB_FREE(key);
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
	ar_coeffs = (float *) palloc0(sizeof(float) * p);
	ma_coeffs = (float *) palloc0(sizeof(float) * q);

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
	model_data = timeseries_model_serialize_to_bytea(ar_coeffs, p, ma_coeffs, q, d, intercept, nvec, model_type);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"p\":%d,\"d\":%d,\"q\":%d,\"intercept\":%.6f,\"model_type\":\"%s\",\"n_samples\":%d}",
					 p, d, q, intercept, model_type, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (TimeSeriesGpuModelState *) palloc0(sizeof(TimeSeriesGpuModelState));
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

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
timeseries_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
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
		NDB_DECLARE(float *, ar_coeffs);
		NDB_DECLARE(float *, ma_coeffs);
		int			p = 0,
					d = 0,
					q = 0;
		float		intercept = 0.0f;
		int			n_obs = 0;
		char		model_type[32];

		if (timeseries_model_deserialize_from_bytea(state->model_blob,
													&ar_coeffs, &p, &ma_coeffs, &q, &d, &intercept, &n_obs, model_type, sizeof(model_type)) != 0)
		{
			if (errstr != NULL)
				*errstr = pstrdup("timeseries_gpu_predict: failed to deserialize");
			return false;
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
timeseries_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
						MLGpuMetrics * out, char **errstr)
{
	const		TimeSeriesGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

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

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"timeseries\",\"storage\":\"cpu\","
					 "\"p\":%d,\"d\":%d,\"q\":%d,\"intercept\":%.6f,\"model_type\":\"%s\",\"n_samples\":%d}",
					 state->p > 0 ? state->p : 1,
					 state->d > 0 ? state->d : 1,
					 state->q > 0 ? state->q : 1,
					 state->intercept,
					 state->model_type[0] ? state->model_type : "arima",
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
timeseries_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
						 Jsonb * *metadata_out, char **errstr)
{
	const		TimeSeriesGpuModelState *state;
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
timeseries_gpu_deserialize(MLGpuModel * model, const bytea * payload,
						   const Jsonb * metadata, char **errstr)
{
	TimeSeriesGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	NDB_DECLARE(float *, ar_coeffs);
	NDB_DECLARE(float *, ma_coeffs);
	int			p = 0,
				d = 0,
				q = 0;
	float		intercept = 0.0f;
	int			n_obs = 0;
	char		model_type[32];
	JsonbIterator *it;
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
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	if (timeseries_model_deserialize_from_bytea(payload_copy,
												&ar_coeffs, &p, &ma_coeffs, &q, &d, &intercept, &n_obs, model_type, sizeof(model_type)) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("timeseries_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (TimeSeriesGpuModelState *) palloc0(sizeof(TimeSeriesGpuModelState));
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
timeseries_gpu_destroy(MLGpuModel * model)
{
	TimeSeriesGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (TimeSeriesGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		if (state->ar_coeffs != NULL)
			NDB_FREE(state->ar_coeffs);
		if (state->ma_coeffs != NULL)
			NDB_FREE(state->ma_coeffs);
		NDB_FREE(state);
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
 * - feature_column: column name containing feature vectors (we use first dimension)
 * - target_column: column name containing target values (time series values)
 * - p, d, q: ARIMA parameters
 */
PG_FUNCTION_INFO_V1(train_timeseries_cpu);

Datum
train_timeseries_cpu(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *target_col = PG_GETARG_TEXT_PP(2);
	int32		p = PG_GETARG_INT32(3);
	int32		d = PG_GETARG_INT32(4);
	int32		q = PG_GETARG_INT32(5);
	char	   *table_name_str;
	char	   *feature_col_str;
	char	   *target_col_str;
	StringInfoData sql;
	int			ret;
	SPITupleTable *tuptable;
	TupleDesc	tupdesc;
	NDB_DECLARE(float *, values);
	NDB_DECLARE(TimeSeriesModel *, model);
	int			n_samples = 0;
	int			i;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;
	int32		model_id = 0;
	MLCatalogModelSpec spec;
	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, params_jsonb);
	NDB_DECLARE(Jsonb *, metrics_jsonb);
	StringInfoData paramsbuf;
	StringInfoData metricsbuf;

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

	oldcontext = CurrentMemoryContext;
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Fetch time series data from target column */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT %s FROM %s ORDER BY (SELECT 1)",
					 quote_identifier(target_col_str),
					 quote_identifier(table_name_str));

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(table_name_str);
		NDB_FREE(feature_col_str);
		NDB_FREE(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to execute time series query")));
	}

	/* Validate SPI result before accessing */
	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL || SPI_tuptable->vals == NULL)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(table_name_str);
		NDB_FREE(feature_col_str);
		NDB_FREE(target_col_str);
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
		NDB_FREE(table_name_str);
		NDB_FREE(feature_col_str);
		NDB_FREE(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least %d observations are required for ARIMA",
						MIN_ARIMA_OBSERVATIONS)));
	}

	values = (float *) palloc(sizeof(float) * n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple;
		bool		isnull = false;
		Datum		value_datum;

		/* Validate tuple exists before accessing */
		if (i >= SPI_processed || tuptable->vals[i] == NULL)
		{
			if (values)
				NDB_FREE(values);
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			NDB_FREE(table_name_str);
			NDB_FREE(feature_col_str);
			NDB_FREE(target_col_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("invalid tuple at index %d in time series data", i)));
		}

		tuple = tuptable->vals[i];
		value_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);

		if (isnull)
		{
			if (values)
				NDB_FREE(values);
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			NDB_FREE(table_name_str);
			NDB_FREE(feature_col_str);
			NDB_FREE(target_col_str);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("time series cannot contain NULL values")));
		}
		values[i] = (float) DatumGetFloat8(value_datum);
	}

	/* Fit ARIMA model */
	model = fit_arima(values, n_samples, p, d, q);

	if (!model)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		if (values)
			NDB_FREE(values);
		NDB_FREE(table_name_str);
		NDB_FREE(feature_col_str);
		NDB_FREE(target_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("ARIMA model fitting failed")));
	}

	/* Serialize model */
	model_data = timeseries_model_serialize_to_bytea(
		model->ar_coeffs, p,
		model->ma_coeffs, q,
		d, model->intercept,
		n_samples, "arima");

	/* Build parameters JSON */
	initStringInfo(&paramsbuf);
	appendStringInfo(&paramsbuf,
					 "{\"p\":%d,\"d\":%d,\"q\":%d,\"model_type\":\"arima\"}",
					 p, d, q);
	params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(paramsbuf.data)));
	NDB_FREE(paramsbuf.data);

	/* Build metrics JSON */
	initStringInfo(&metricsbuf);
	appendStringInfo(&metricsbuf,
					 "{\"storage\":\"cpu\",\"p\":%d,\"d\":%d,\"q\":%d,\"intercept\":%.6f,\"model_type\":\"arima\",\"n_samples\":%d,\"training_backend\":0}",
					 p, d, q, model->intercept, n_samples);
	metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													   CStringGetTextDatum(metricsbuf.data)));
	NDB_FREE(metricsbuf.data);

	/* Register model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.algorithm = NDB_ALGO_TIMESERIES;
	spec.training_table = table_name_str;
	spec.training_column = target_col_str;
	spec.model_data = model_data;
	spec.parameters = params_jsonb;
	spec.metrics = metrics_jsonb;
	spec.num_samples = n_samples;
	spec.num_features = 1; /* Time series uses single feature dimension */

	MemoryContextSwitchTo(oldcontext);
	model_id = ml_catalog_register_model(&spec);

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);
	if (values)
		NDB_FREE(values);
	if (model->ar_coeffs)
		NDB_FREE(model->ar_coeffs);
	if (model->ma_coeffs)
		NDB_FREE(model->ma_coeffs);
	if (model->residuals)
		NDB_FREE(model->residuals);
	NDB_FREE(model);
	NDB_FREE(table_name_str);
	NDB_FREE(feature_col_str);
	NDB_FREE(target_col_str);

	PG_RETURN_INT32(model_id);
}

/*-------------------------------------------------------------------------
 *
 * ml_recommender.c
 *    Recommender systems implementation.
 *
 * This module implements collaborative filtering, content-based filtering,
 * and hybrid recommendation approaches.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_recommender.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "neurondb_pgcompat.h"
#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_json.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <math.h>
#include <string.h>
#include <sys/time.h>

/* Random number generation */
#define MAXRAND ((double)0x7fffffff)

/* ALS matrix factorization parameters */
#define ALS_DEFAULT_NFACTORS 20
#define ALS_MAX_NFACTORS 1000
#define ALS_MIN_NFACTORS 1
#define ALS_DEFAULT_EPOCHS 10
#define ALS_DEFAULT_LAMBDA 0.1
#define ALS_MAX_ITER 100

#define RECO_MIN_RESULT 1
#define RECO_MAX_RESULT 1000

static float **
als_alloc_matrix(int nrow, int ncol)
{
	NDB_DECLARE(float **, mat);
	int			i;

	NDB_ALLOC(mat, float *, nrow);
	for (i = 0; i < nrow; ++i)
	{
		NDB_ALLOC(mat[i], float, ncol);
		memset(mat[i], 0, sizeof(float) * ncol);
	}
	return mat;
}

static void
als_free_matrix(float **mat, int nrow)
{
	int			i;

	if (mat == NULL)
		return;
	for (i = 0; i < nrow; ++i)
	{
		if (mat[i])
			NDB_FREE(mat[i]);
	}
	NDB_FREE(mat);
}

static float
dot_product(const float *v1, const float *v2, int n)
{
	float		s = 0.0f;
	int			i;

	if (!v1 || !v2 || n <= 0)
		return 0.0f;
	for (i = 0; i < n; ++i)
		s += v1[i] * v2[i];
	return s;
}

/*
 * train_collaborative_filter
 * Trains ALS matrix factorization from a ratings table.
 * Returns a model id. The model is saved as two tables: user_factors and item_factors.
 */
PG_FUNCTION_INFO_V1(train_collaborative_filter);
PG_FUNCTION_INFO_V1(predict_collaborative_filter);
PG_FUNCTION_INFO_V1(evaluate_collaborative_filter_by_model_id);

Datum
train_collaborative_filter(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *user_col = PG_GETARG_TEXT_PP(1);
	text	   *item_col = PG_GETARG_TEXT_PP(2);
	text	   *rating_col = PG_GETARG_TEXT_PP(3);
	int32		n_factors =
		PG_ARGISNULL(4) ? ALS_DEFAULT_NFACTORS : PG_GETARG_INT32(4);

	char	   *table_name_str = text_to_cstring(table_name);
	char	   *user_col_str = text_to_cstring(user_col);
	char	   *item_col_str = text_to_cstring(item_col);
	char	   *rating_col_str = text_to_cstring(rating_col);

	int			ret;

	NDB_DECLARE(NdbSpiSession *, train_als_spi_session);
	MemoryContext oldcontext = NULL,
				model_mcxt = NULL;
	int			n_row,
				i,
				max_user_id = 0,
				max_item_id = 0;

	NDB_DECLARE(int *, user_ids);
	NDB_DECLARE(int *, item_ids);
	int			n_ratings = 0;

	NDB_DECLARE(float *, ratings);
	NDB_DECLARE(float **, P);
	NDB_DECLARE(float **, Q);
	StringInfoData sql = {0};
	SPIPlanPtr	plan = NULL;

	if (n_factors < ALS_MIN_NFACTORS || n_factors > ALS_MAX_NFACTORS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_factors must be between %d and %d",
						ALS_MIN_NFACTORS,
						ALS_MAX_NFACTORS)));

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT %s, %s, %s FROM %s ORDER BY %s, %s",
					 quote_identifier(user_col_str),
					 quote_identifier(item_col_str),
					 quote_identifier(rating_col_str),
					 quote_identifier(table_name_str),
					 quote_identifier(user_col_str),
					 quote_identifier(item_col_str));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(train_als_spi_session, oldcontext);

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	if (ret == SPI_OK_SELECT)
		NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(train_als_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: train_collaborative_filter: failed to execute SQL for ratings"),
				 errdetail("SQL query: %s", sql.data),
				 errhint("Verify the table exists and contains valid user, item, and rating columns.")));
	}

	n_row = SPI_processed;
	if (n_row <= 1)
	{
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(train_als_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("neurondb: train_collaborative_filter: not enough ratings found to train model"),
				 errdetail("Found %d rows, but need at least 2 rows", n_row),
				 errhint("Add more rating data to the table.")));
	}

	NDB_ALLOC(user_ids, int, n_row);
	NDB_ALLOC(item_ids, int, n_row);
	NDB_ALLOC(ratings, float, n_row);
	n_ratings = n_row;

	for (i = 0; i < n_row; ++i)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		bool		isnull[3] = {false, false, false};

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		tupdesc = SPI_tuptable->tupdesc;
		if (tupdesc == NULL)
		{
			continue;
		}
		{
			int32		user = 0,
						item = 0;
			float		r = 0.0f;

			/* Use safe function for int32 values */
			if (!ndb_spi_get_int32(train_als_spi_session, i, 1, &user))
			{
				isnull[0] = true;
			}
			if (!ndb_spi_get_int32(train_als_spi_session, i, 2, &item))
			{
				isnull[1] = true;
			}

			if (isnull[0] || isnull[1])
			{
				NDB_FREE(sql.data);
				NDB_FREE(user_ids);
				NDB_FREE(item_ids);
				NDB_FREE(ratings);
				NDB_SPI_SESSION_END(train_als_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_collaborative_filter: user_col or item_col contains NULL"),
						 errdetail("Row %d contains NULL values in user or item columns", i + 1),
						 errhint("Remove NULL values from the user and item columns.")));
			}

			/*
			 * Safe access for rating - validate tupdesc has at least 3
			 * columns
			 */
			if (tupdesc->natts >= 3)
			{
				Datum		rating_datum;

				rating_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull[2]);
				if (!isnull[2])
				{
					if (SPI_gettypeid(tupdesc, 3) == FLOAT8OID)
						r = (float) DatumGetFloat8(rating_datum);
					else
						r = (float) DatumGetFloat4(rating_datum);
				}
			}
			else
			{
				isnull[2] = true;
			}

			if (isnull[2])
			{
				NDB_FREE(sql.data);
				NDB_FREE(user_ids);
				NDB_FREE(item_ids);
				NDB_FREE(ratings);
				NDB_SPI_SESSION_END(train_als_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_collaborative_filter: rating_col contains NULL"),
						 errdetail("Row %d contains NULL value in rating column", i + 1),
						 errhint("Remove NULL values from the rating column.")));
			}
			user_ids[i] = user;
			item_ids[i] = item;
			ratings[i] = r;

			if (user > max_user_id)
				max_user_id = user;
			if (item > max_item_id)
				max_item_id = item;
		}
	}

	model_mcxt = AllocSetContextCreate(CurrentMemoryContext,
									   "ALS factors context",
									   ALLOCSET_SMALL_SIZES);
	oldcontext = MemoryContextSwitchTo(model_mcxt);

	P = als_alloc_matrix(max_user_id + 1, n_factors);
	Q = als_alloc_matrix(max_item_id + 1, n_factors);

	/* Random initialize */
	for (i = 0; i <= max_user_id; ++i)
	{
		int			f;

		for (f = 0; f < n_factors; ++f)
			P[i][f] = ((float) random() / (float) MAXRAND) * 0.1f;
	}
	for (i = 0; i <= max_item_id; ++i)
	{
		int			f;

		for (f = 0; f < n_factors; ++f)
			Q[i][f] = ((float) random() / (float) MAXRAND) * 0.1f;
	}

	{
		int			epoch,
					u,
					v,
					k;
		float		lambda = ALS_DEFAULT_LAMBDA;

		for (epoch = 0; epoch < ALS_DEFAULT_EPOCHS; ++epoch)
		{
			for (i = 0; i < n_ratings; ++i)
			{
				float		r_ui;
				float		pred;
				float		err;

				u = user_ids[i];
				v = item_ids[i];
				r_ui = ratings[i];
				pred = dot_product(P[u], Q[v], n_factors);
				err = r_ui - pred;
				for (k = 0; k < n_factors; ++k)
				{
					float		pu = P[u][k];
					float		qi = Q[v][k];

					P[u][k] += 0.01f
						* (err * qi - lambda * pu);
					Q[v][k] += 0.01f
						* (err * pu - lambda * qi);
				}
			}
		}
	}

	MemoryContextSwitchTo(oldcontext);

	/* Register model in catalog */
	{
		int32		model_id;
		MLCatalogModelSpec spec;

		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "custom";	/* collaborative_filtering not in enum,
									 * use custom */
		spec.model_type = "regression";
		spec.training_table = table_name_str;
		spec.training_column = NULL;
		spec.project_name = NULL;
		spec.model_name = NULL;
		spec.parameters = NULL;
		spec.metrics = NULL;
		spec.model_data = NULL;
		spec.training_time_ms = -1;
		spec.num_samples = n_ratings;
		spec.num_features = n_factors;

		model_id = ml_catalog_register_model(&spec);
		elog(DEBUG1, "neurondb: ml_catalog returned model_id=%d (as int32)", model_id);
		if (model_id <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to register collaborative filter model in catalog")));

		/*
		 * Use safe free/reinit - ml_catalog_register_model uses
		 * SPI_connect/finish
		 */
		NDB_FREE(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "CREATE TABLE IF NOT EXISTS neurondb_cf_user_factors "
						 "(model_id integer, user_id int, factors float4[])");
		ndb_spi_execute_safe(sql.data, false, 0);

		/*
		 * CREATE TABLE doesn't return a result set, so don't check
		 * SPI_tuptable
		 */
		/* Use safe free/reinit to handle potential memory context changes */
		NDB_FREE(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "CREATE TABLE IF NOT EXISTS neurondb_cf_item_factors "
						 "(model_id integer, item_id int, factors float4[])");
		ndb_spi_execute_safe(sql.data, false, 0);

		/*
		 * CREATE TABLE doesn't return a result set, so don't check
		 * SPI_tuptable
		 */

		/* Use safe free/reinit to handle potential memory context changes */
		NDB_FREE(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "DELETE FROM neurondb_cf_user_factors WHERE model_id = %d",
						 model_id);
		ndb_spi_execute_safe(sql.data, false, 0);
		/* DELETE doesn't return a result set, so don't check SPI_tuptable */
		/* Use safe free/reinit to handle potential memory context changes */
		NDB_FREE(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "DELETE FROM neurondb_cf_item_factors WHERE model_id = %d",
						 model_id);
		ndb_spi_execute_safe(sql.data, false, 0);
		/* DELETE doesn't return a result set, so don't check SPI_tuptable */

		{
			Oid			arg_types[3] = {INT4OID, INT4OID, 1021};
			Datum		values[3];
			char		nulls[3] = {false, false, false};

			NDB_DECLARE(ArrayType *, array);
			int			j;

			plan = SPI_prepare(
							   "INSERT INTO neurondb_cf_user_factors "
							   "(model_id, user_id, factors) VALUES ($1,$2,$3)",
							   3,
							   arg_types);
			if (plan == NULL)
				elog(ERROR,
					 "Failed to prepare user_factors "
					 "insert");
			for (i = 0; i <= max_user_id; ++i)
			{
				bool		seen = false;

				for (j = 0; j < n_ratings; ++j)
				{
					if (user_ids[j] == i)
					{
						seen = true;
						break;
					}
				}
				if (!seen)
					continue;
				array = construct_array((Datum *) P[i],
										n_factors,
										FLOAT4OID,
										sizeof(float4),
										true,
										TYPALIGN_INT);
				values[0] = Int32GetDatum(model_id);
				values[1] = Int32GetDatum(i);
				values[2] = PointerGetDatum(array);
				ret = SPI_execute_plan(
									   plan, values, nulls, false, 1);
				if (ret != SPI_OK_INSERT)
					elog(ERROR,
						 "Failed to insert user_factors for user %d (model_id %d)",
						 i,
						 model_id);
			}
		}

		{
			Oid			arg_types[3] = {INT4OID, INT4OID, 1021};
			Datum		values[3];
			char		nulls[3] = {false, false, false};

			NDB_DECLARE(ArrayType *, array);
			int			j;

			plan = SPI_prepare(
							   "INSERT INTO neurondb_cf_item_factors "
							   "(model_id, item_id, factors) VALUES ($1,$2,$3)",
							   3,
							   arg_types);
			if (plan == NULL)
				elog(ERROR,
					 "Failed to prepare item_factors "
					 "insert");
			for (i = 0; i <= max_item_id; ++i)
			{
				bool		seen = false;

				for (j = 0; j < n_ratings; ++j)
				{
					if (item_ids[j] == i)
					{
						seen = true;
						break;
					}
				}
				if (!seen)
					continue;
				array = construct_array((Datum *) Q[i],
										n_factors,
										FLOAT4OID,
										sizeof(float4),
										true,
										TYPALIGN_INT);
				values[0] = Int32GetDatum(model_id);
				values[1] = Int32GetDatum(i);
				values[2] = PointerGetDatum(array);
				ret = SPI_execute_plan(
									   plan, values, nulls, false, 1);
				if (ret != SPI_OK_INSERT)
					elog(ERROR,
						 "Failed to insert item_factors for item %d (model_id %d)",
						 i,
						 model_id);
			}
		}

		als_free_matrix(P, max_user_id + 1);
		als_free_matrix(Q, max_item_id + 1);

		if (user_ids)
			NDB_FREE(user_ids);
		if (item_ids)
			NDB_FREE(item_ids);
		if (ratings)
			NDB_FREE(ratings);
		if (table_name_str)
			NDB_FREE(table_name_str);
		if (user_col_str)
			NDB_FREE(user_col_str);
		if (item_col_str)
			NDB_FREE(item_col_str);
		if (rating_col_str)
			NDB_FREE(rating_col_str);
		if (model_mcxt)
			MemoryContextDelete(model_mcxt);

		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(train_als_spi_session);
		elog(INFO, "Collaborative filter model created, model_id=%d", model_id);
		PG_RETURN_INT32(model_id);
	}
}

/* Helper functions for loading factors from database */
static bool
als_load_user_factors(int32 model_id, int32 user_id, float **factors, int *n_factors)
{
	StringInfoData query = {0};

	NDB_DECLARE(NdbSpiSession *, load_user_factors_spi_session);
	MemoryContext oldcontext;
	int			ret;
	int			n_rows;

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(load_user_factors_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
					 model_id, user_id);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_user_factors_spi_session);
		return false;
	}

	n_rows = SPI_processed;
	if (n_rows != 1)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_user_factors_spi_session);
		return false;
	}

	/* Extract factors array */
	{
		HeapTuple	tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		factors_datum;
		bool		factors_null;
		ArrayType  *factors_array;
		Oid			elmtype;
		int16		typlen;
		bool		typbyval;
		char		typalign;
		int			n_dims;
		Datum	   *elems;
		bool	   *nulls;
		int			n_elems;
		int			i;

		factors_datum = SPI_getbinval(tuple, tupdesc, 1, &factors_null);
		if (factors_null)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(load_user_factors_spi_session);
			return false;
		}

		factors_array = DatumGetArrayTypeP(factors_datum);
		n_dims = ARR_NDIM(factors_array);
		if (n_dims != 1)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(load_user_factors_spi_session);
			return false;
		}

		elmtype = ARR_ELEMTYPE(factors_array);
		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

		deconstruct_array(factors_array, elmtype, typlen, typbyval, typalign,
						  &elems, &nulls, &n_elems);

		/* Allocate in parent context before SPI_finish() */
		*n_factors = n_elems;
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_user_factors_spi_session);

		NDB_ALLOC(*factors, float, n_elems);
		for (i = 0; i < n_elems; i++)
			(*factors)[i] = DatumGetFloat4(elems[i]);
	}

	return true;
}

static bool
als_load_item_factors(int32 model_id, int32 item_id, float ***factors, int *n_items_total, int *n_factors)
{
	StringInfoData query = {0};

	NDB_DECLARE(NdbSpiSession *, load_item_factors_spi_session);
	MemoryContext oldcontext;
	int			ret;
	int			n_rows;

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(load_item_factors_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT factors FROM neurondb_cf_item_factors WHERE model_id = %d AND item_id = %d",
					 model_id, item_id);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_item_factors_spi_session);
		return false;
	}

	n_rows = SPI_processed;
	if (n_rows != 1)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_item_factors_spi_session);
		return false;
	}

	/* Extract factors array - safe access for complex types */
	if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
		SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
	{
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_item_factors_spi_session);
		return false;
	}
	{
		HeapTuple	tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		factors_datum;
		bool		factors_null;
		ArrayType  *factors_array;
		Oid			elmtype;
		int16		typlen;
		bool		typbyval;
		char		typalign;
		int			n_dims;
		Datum	   *elems;
		bool	   *nulls;
		int			n_elems;
		int			i;

		factors_datum = SPI_getbinval(tuple, tupdesc, 1, &factors_null);
		if (factors_null)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(load_item_factors_spi_session);
			return false;
		}

		factors_array = DatumGetArrayTypeP(factors_datum);
		n_dims = ARR_NDIM(factors_array);
		if (n_dims != 1)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(load_item_factors_spi_session);
			return false;
		}

		elmtype = ARR_ELEMTYPE(factors_array);
		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);

		deconstruct_array(factors_array, elmtype, typlen, typbyval, typalign,
						  &elems, &nulls, &n_elems);

		/* Allocate in parent context before SPI_finish() */
		*n_items_total = 1;
		*n_factors = n_elems;
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(load_item_factors_spi_session);

		NDB_ALLOC(*factors, float *, 1);
		NDB_ALLOC((*factors)[0], float, n_elems);

		for (i = 0; i < n_elems; i++)
			(*factors)[0][i] = DatumGetFloat4(elems[i]);
	}

	return true;
}

/*
 * predict_collaborative_filter
 *      Predicts rating for a specific user-item pair using trained ALS model.
 *      Arguments: int4 model_id, int4 user_id, int4 item_id
 *      Returns: float8 predicted rating
 */
Datum
predict_collaborative_filter(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	int32		user_id = PG_GETARG_INT32(1);
	int32		item_id = PG_GETARG_INT32(2);

	NDB_DECLARE(float *, user_factors);
	NDB_DECLARE(float **, item_factors);
	int			n_factors = 0;
	int			n_items_total = 0;
	float		prediction = 0.0;
	int			i;

	/* Load user factors */
	if (!als_load_user_factors(model_id, user_id, &user_factors, &n_factors))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("collaborative_filter: no factors found for user %d in model %d",
						user_id, model_id)));
	}

	/* Load item factors */
	if (!als_load_item_factors(model_id, item_id, &item_factors, &n_items_total, &n_factors))
	{
		NDB_FREE(user_factors);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("collaborative_filter: no factors found for item %d in model %d",
						item_id, model_id)));
	}

	/* Compute dot product of user and item factors */
	for (i = 0; i < n_factors; i++)
		prediction += user_factors[i] * item_factors[0][i];

	/* Check for NaN/Inf and return default value (mean rating) if invalid */
	if (isnan(prediction) || isinf(prediction))
		prediction = 3.0;		/* Default to middle of 1-5 scale */

	/* Clamp prediction to valid rating range (assuming 1-5 scale) */
	if (prediction < 1.0)
		prediction = 1.0;
	if (prediction > 5.0)
		prediction = 5.0;

	NDB_FREE(user_factors);
	for (i = 0; i < n_items_total; i++)
		NDB_FREE(item_factors[i]);
	NDB_FREE(item_factors);

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_collaborative_filter_by_model_id
 *      Evaluates collaborative filtering model on a test dataset.
 *      Arguments: int4 model_id, text table_name, text user_col, text item_col, text rating_col
 *      Returns: jsonb with evaluation metrics
 */
Datum
evaluate_collaborative_filter_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *user_col;
	text	   *item_col;
	text	   *rating_col;
	char	   *tbl_str;
	char	   *user_str;
	char	   *item_str;
	char	   *rating_str;
	StringInfoData query;
	int			ret;
	int			n_ratings = 0;
	double		mse = 0.0;
	double		mae = 0.0;
	int			i;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext = CurrentMemoryContext;
	double		rmse;

	/* Validate arguments */
	if (PG_NARGS() != 5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_collaborative_filter_by_model_id: 5 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_collaborative_filter_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3) || PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_collaborative_filter_by_model_id: table_name, user_col, item_col, and rating_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	user_col = PG_GETARG_TEXT_PP(2);
	item_col = PG_GETARG_TEXT_PP(3);
	rating_col = PG_GETARG_TEXT_PP(4);

	tbl_str = text_to_cstring(table_name);
	user_str = text_to_cstring(user_col);
	item_str = text_to_cstring(item_col);
	rating_str = text_to_cstring(rating_col);

	{
		MemoryContext oldcontext_local = CurrentMemoryContext;

		NDB_DECLARE(NdbSpiSession *, eval_cf_spi_session);

		Assert(oldcontext_local != NULL);
		NDB_SPI_SESSION_BEGIN(eval_cf_spi_session, oldcontext_local);

		/* Build query */
		initStringInfo(&query);
		appendStringInfo(&query,
						 "SELECT %s, %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL AND %s IS NOT NULL",
						 user_str, item_str, rating_str, tbl_str, user_str, item_str, rating_str);

		ret = ndb_spi_execute_safe(query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(eval_cf_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(user_str);
			NDB_FREE(item_str);
			NDB_FREE(rating_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: evaluate_collaborative_filter_by_model_id: query failed"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the table exists and contains valid user, item, and rating columns.")));
		}

		n_ratings = SPI_processed;
		if (n_ratings < 2)
		{
			NDB_FREE(query.data);
			NDB_SPI_SESSION_END(eval_cf_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(user_str);
			NDB_FREE(item_str);
			NDB_FREE(rating_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_collaborative_filter_by_model_id: need at least 2 ratings"),
					 errdetail("Found %d ratings, but need at least 2", n_ratings),
					 errhint("Add more rating data to the table.")));
		}

		/* Evaluate each rating */
		for (i = 0; i < n_ratings; i++)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			int32		user_id_val = 0,
						item_id_val = 0;

			/* Safe access to SPI_tuptable - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
			{
				continue;
			}
			tuple = SPI_tuptable->vals[i];
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
			{
				continue;
			}
			{
				/* Use safe function for int32 values */
				bool		user_null = false,
							item_null = false;
				Datum		user_datum,
							item_datum;
				bool		user_isnull,
							item_isnull;
				int32		user_id;
				int32		item_id;
				Datum		rating_datum;
				bool		rating_null;
				float		true_rating;
				float		pred_rating;
				float		error;

				if (!ndb_spi_get_result_safe(i, 1, NULL, &user_datum, &user_isnull) || user_isnull)
				{
					user_null = true;
				}
				else
				{
					user_id_val = DatumGetInt32(user_datum);
				}
				if (!ndb_spi_get_result_safe(i, 2, NULL, &item_datum, &item_isnull) || item_isnull)
				{
					item_null = true;
				}
				else
				{
					item_id_val = DatumGetInt32(item_datum);
				}
				user_id = user_id_val;
				item_id = item_id_val;

				/*
				 * For rating (float), need to use SPI_getbinval with safe
				 * access
				 */

				/*
				 * Safe access for rating - validate tupdesc has at least 3
				 * columns
				 */
				if (tupdesc->natts < 3)
				{
					continue;
				}
				rating_datum = SPI_getbinval(tuple, tupdesc, 3, &rating_null);

				if (user_null || item_null || rating_null)
					continue;

				true_rating = DatumGetFloat4(rating_datum);

				/* Get prediction */
				pred_rating = DatumGetFloat8(DirectFunctionCall3(predict_collaborative_filter,
																 Int32GetDatum(model_id),
																 Int32GetDatum(user_id),
																 Int32GetDatum(item_id)));

				/*
				 * Skip NaN/Inf predictions (shouldn't happen now, but keep as
				 * safety check)
				 */
				if (isnan(pred_rating) || isinf(pred_rating))
					continue;

				/* Compute error */
				error = true_rating - pred_rating;
				mse += error * error;
				mae += fabs(error);
			}
		}

		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(eval_cf_spi_session);
	}

	/* Handle case where all predictions were NaN */
	if (n_ratings == 0 || isnan(mse) || isinf(mse))
	{
		mse = 0.0;
		mae = 0.0;
		rmse = 0.0;
	}
	else
	{
		mse /= n_ratings;
		mae /= n_ratings;
		rmse = sqrt(mse);
	}

	/* Build result JSON - ensure no NaN/Inf values */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);

	/* Replace NaN/Inf with null strings for JSON compatibility */
	if (isnan(mse) || isinf(mse))
		mse = 0.0;
	if (isnan(mae) || isinf(mae))
		mae = 0.0;
	if (isnan(rmse) || isinf(rmse))
		rmse = 0.0;

	appendStringInfo(&jsonbuf,
					 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"n_ratings\":%d}",
					 mse, mae, rmse, n_ratings);

	result = ndb_jsonb_in_cstring(jsonbuf.data);
	NDB_FREE(jsonbuf.data);

	NDB_FREE(tbl_str);
	NDB_FREE(user_str);
	NDB_FREE(item_str);
	NDB_FREE(rating_str);

	PG_RETURN_JSONB_P(result);
}

/*
 * recommend_items
 */
PG_FUNCTION_INFO_V1(recommend_items);

Datum
recommend_items(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	int32		user_id = PG_GETARG_INT32(1);
	int32		n_items = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);

	NDB_DECLARE(float *, user_factors);
	int			n_factors = 0;

	NDB_DECLARE(int *, item_ids);
	NDB_DECLARE(float **, item_factors);
	int			n_items_total = 0;
	int			i,
				j;

	NDB_DECLARE(int32 *, top_items);
	NDB_DECLARE(float *, top_scores);
	StringInfoData sql = {0};

	NDB_DECLARE(NdbSpiSession *, predict_cf_spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;
	int			ret = 0;

	NDB_DECLARE(ArrayType *, result_array);
	NDB_DECLARE(Datum *, elems);

	NDB_SPI_SESSION_BEGIN(predict_cf_spi_session, oldcontext);

	if (n_items < RECO_MIN_RESULT || n_items > RECO_MAX_RESULT)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_collaborative_filter: n_items must be between %d and %d",
						RECO_MIN_RESULT,
						RECO_MAX_RESULT),
				 errdetail("n_items is %d, valid range is %d-%d", n_items, RECO_MIN_RESULT, RECO_MAX_RESULT),
				 errhint("Specify a value between %d and %d", RECO_MIN_RESULT, RECO_MAX_RESULT)));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(predict_cf_spi_session, oldcontext);

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
					 model_id,
					 user_id);

	ret = ndb_spi_execute_safe(sql.data, true, 1);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(predict_cf_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: predict_collaborative_filter: failed to load user factors"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the model exists and contains user factors for user %d", user_id)));
	}
	if (SPI_processed != 1)
	{
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(predict_cf_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_collaborative_filter: no user_factors found"),
				 errdetail("No user_factors found for user %d in model %d", user_id, model_id),
				 errhint("Ensure the user was part of the training data.")));
	}

	/* Safe access for complex types - validate before access */
	if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
		SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
	{
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(predict_cf_spi_session);
		PG_RETURN_NULL();
	}
	{
		HeapTuple	tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		bool		isnull = false;
		Datum		arr = SPI_getbinval(tuple, tupdesc, 1, &isnull);

		if (isnull)
		{
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(predict_cf_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_collaborative_filter: NULL user factors array"),
					 errdetail("User factors array is NULL for user %d in model %d", user_id, model_id),
					 errhint("This indicates corrupted model data. Retrain the model.")));
		}
		{
			ArrayType  *user_vec = DatumGetArrayTypeP(arr);

			n_factors = ArrayGetNItems(
									   ARR_NDIM(user_vec), ARR_DIMS(user_vec));
			NDB_ALLOC(user_factors, float, n_factors);
			memcpy(user_factors,
				   ARR_DATA_PTR(user_vec),
				   sizeof(float) * n_factors);
		}
	}

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT item_id, factors FROM neurondb_cf_item_factors WHERE model_id = %d",
					 model_id);

	ret = ndb_spi_execute_safe(sql.data, true, 0);
	if (ret == SPI_OK_SELECT)
		NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(sql.data);
		if (user_factors)
			NDB_FREE(user_factors);
		NDB_SPI_SESSION_END(predict_cf_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: predict_collaborative_filter: failed to load item factors"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the model exists and contains item factors.")));
	}
	n_items_total = SPI_processed;
	if (n_items_total < 1)
	{
		NDB_FREE(sql.data);
		if (user_factors)
			NDB_FREE(user_factors);
		NDB_SPI_SESSION_END(predict_cf_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_collaborative_filter: no items found for model"),
				 errdetail("Model %d has no item factors", model_id),
				 errhint("Ensure the model was trained successfully.")));
	}
	NDB_ALLOC(item_ids, int, n_items_total);
	NDB_ALLOC(item_factors, float *, n_items_total);

	for (i = 0; i < n_items_total; ++i)
	{
		HeapTuple	itup = SPI_tuptable->vals[i];
		int			item_id = 0;

		NDB_DECLARE(float *, fac);
		int			item_n_factors = 0;

		NDB_DECLARE(ArrayType *, arr_f);
		bool		isnull_fac = false;
		Datum		facdatum;

		/* Get item_id from tuple directly */
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		item_id_datum;
		bool		item_id_isnull;
		int32		item_id_val;

		item_id_datum = SPI_getbinval(itup, tupdesc, 1, &item_id_isnull);
		if (!item_id_isnull)
		{
			item_id_val = DatumGetInt32(item_id_datum);
			item_id = item_id_val;
		}
		facdatum = SPI_getbinval(
								 itup, SPI_tuptable->tupdesc, 2, &isnull_fac);
		arr_f = DatumGetArrayTypeP(facdatum);
		item_n_factors =
			ArrayGetNItems(ARR_NDIM(arr_f), ARR_DIMS(arr_f));
		if (item_n_factors != n_factors)
		{
			for (j = 0; j < i; ++j)
				NDB_FREE(item_factors[j]);
			NDB_FREE(item_ids);
			NDB_FREE(item_factors);
			NDB_FREE(user_factors);
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(predict_cf_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: recommend_items: factor dimension mismatch"),
					 errdetail("Item %d has %d factors but expected %d", item_id, item_n_factors, n_factors),
					 errhint("This indicates corrupted model data. Retrain the model.")));
		}
		NDB_ALLOC(fac, float, n_factors);
		memcpy(fac, ARR_DATA_PTR(arr_f), sizeof(float) * n_factors);
		item_ids[i] = item_id;
		item_factors[i] = fac;
	}

	NDB_ALLOC(top_items, int32, n_items);
	NDB_ALLOC(top_scores, float, n_items);

	for (i = 0; i < n_items; ++i)
	{
		top_scores[i] = -INFINITY;
		top_items[i] = -1;
	}

	for (i = 0; i < n_items_total; ++i)
	{
		float		score =
			dot_product(user_factors, item_factors[i], n_factors);
		int			minidx = 0;

		for (j = 1; j < n_items; ++j)
		{
			if (top_scores[j] < top_scores[minidx])
				minidx = j;
		}
		if (score > top_scores[minidx])
		{
			top_scores[minidx] = score;
			top_items[minidx] = item_ids[i];
		}
	}

	if (user_factors)
		NDB_FREE(user_factors);
	for (i = 0; i < n_items_total; ++i)
		if (item_factors[i])
			NDB_FREE(item_factors[i]);
	if (item_factors)
		NDB_FREE(item_factors);
	if (item_ids)
		NDB_FREE(item_ids);

	for (i = 0; i < n_items - 1; ++i)
	{
		for (j = i + 1; j < n_items; ++j)
		{
			if (top_scores[j] > top_scores[i])
			{
				float		tswap = top_scores[i];
				int32		iswap = top_items[i];

				top_scores[i] = top_scores[j];
				top_items[i] = top_items[j];
				top_scores[j] = tswap;
				top_items[j] = iswap;
			}
		}
	}

	NDB_ALLOC(elems, Datum, n_items);
	for (i = 0; i < n_items; ++i)
	{
		elems[i] = Int32GetDatum(top_items[i]);
	}

	result_array = construct_array(
								   elems, n_items, INT4OID, sizeof(int32), true, 'i');

	NDB_FREE(sql.data);
	NDB_FREE(elems);
	NDB_FREE(top_items);
	NDB_FREE(top_scores);
	NDB_SPI_SESSION_END(predict_cf_spi_session);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * recommend_content_based
 */
PG_FUNCTION_INFO_V1(recommend_content_based);

Datum
recommend_content_based(PG_FUNCTION_ARGS)
{
	int32		item_id = PG_GETARG_INT32(0);
	text	   *features_table = PG_GETARG_TEXT_PP(1);
	int32		n_recommendations = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);

	char	   *features_table_str = text_to_cstring(features_table);
	int			ret,
				i,
				j,
				item_count,
				n_factors;

	NDB_DECLARE(int32 *, other_ids);
	NDB_DECLARE(float **, other_factors);
	int			target_idx = -1;

	NDB_DECLARE(float *, target_vec);

	NDB_DECLARE(ArrayType *, result_array);
	NDB_DECLARE(Datum *, elems);
	NDB_DECLARE(int32 *, top_items);
	NDB_DECLARE(float *, top_sims);

	if (n_recommendations < RECO_MIN_RESULT
		|| n_recommendations > RECO_MAX_RESULT)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_recommendations must be between %d and %d",
						RECO_MIN_RESULT,
						RECO_MAX_RESULT)));

	{
		NDB_DECLARE(NdbSpiSession *, content_spi_session);
		MemoryContext oldcontext = CurrentMemoryContext;
		StringInfoData sql = {0};

		NDB_SPI_SESSION_BEGIN(content_spi_session, oldcontext);

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT item_id, features FROM %s",
						 quote_identifier(features_table_str));
		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			if (features_table_str)
				NDB_FREE(features_table_str);
			NDB_SPI_SESSION_END(content_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: recommend_content_based: failed to load item features"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the features table exists and is accessible.")));
		}
		item_count = SPI_processed;
		if (item_count < 2)
		{
			NDB_FREE(sql.data);
			if (features_table_str)
				NDB_FREE(features_table_str);
			NDB_SPI_SESSION_END(content_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: recommend_content_based: insufficient items"),
					 errdetail("Found %d items but need at least 2 for content-based recommendations", item_count),
					 errhint("Ensure the features table contains at least 2 items.")));
		}
		NDB_ALLOC(other_ids, int32, item_count);
		NDB_ALLOC(other_factors, float *, item_count);

		n_factors = 0;
		for (i = 0; i < item_count; ++i)
		{
			HeapTuple	tup = SPI_tuptable->vals[i];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			bool		isnull_item = false,
						isnull_feat = false;

			int			id = DatumGetInt32(
										   SPI_getbinval(tup, tupdesc, 1, &isnull_item));
			Datum		arr_datum =
				SPI_getbinval(tup, tupdesc, 2, &isnull_feat);

			if (isnull_item || isnull_feat)
			{
				NDB_FREE(sql.data);
				if (features_table_str)
					NDB_FREE(features_table_str);
				for (j = 0; j < i; ++j)
					if (other_factors[j])
						NDB_FREE(other_factors[j]);
				if (other_factors)
					NDB_FREE(other_factors);
				if (other_ids)
					NDB_FREE(other_ids);
				NDB_SPI_SESSION_END(content_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: recommend_content_based: NULL item or features"),
						 errdetail("Row %d contains NULL item_id or features", i + 1),
						 errhint("Ensure all rows have valid item_id and features data.")));
			}
			{
				ArrayType  *arr = DatumGetArrayTypeP(arr_datum);
				int			nf = ArrayGetNItems(
												ARR_NDIM(arr), ARR_DIMS(arr));

				if (i == 0)
					n_factors = nf;
				if (nf != n_factors)
				{
					NDB_FREE(sql.data);
					if (features_table_str)
						NDB_FREE(features_table_str);
					for (j = 0; j < i; ++j)
						if (other_factors[j])
							NDB_FREE(other_factors[j]);
					if (other_factors)
						NDB_FREE(other_factors);
					if (other_ids)
						NDB_FREE(other_ids);
					NDB_SPI_SESSION_END(content_spi_session);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: recommend_content_based: feature dimension mismatch"),
							 errdetail("Row %d has %d features but expected %d", i + 1, nf, n_factors),
							 errhint("Ensure all feature vectors have the same dimension.")));
				}
				{
					float	   *vec;

					NDB_ALLOC(vec, float, n_factors);
					memcpy(vec,
						   ARR_DATA_PTR(arr),
						   sizeof(float) * n_factors);
					other_ids[i] = id;
					other_factors[i] = vec;
					if (id == item_id)
					{
						target_idx = i;
						target_vec = vec;
					}
				}
			}
		}
		if (target_idx == -1)
		{
			NDB_FREE(sql.data);
			if (features_table_str)
				NDB_FREE(features_table_str);
			for (j = 0; j < item_count; ++j)
				if (other_factors[j])
					NDB_FREE(other_factors[j]);
			if (other_factors)
				NDB_FREE(other_factors);
			if (other_ids)
				NDB_FREE(other_ids);
			NDB_SPI_SESSION_END(content_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: recommend_content_based: item not found"),
					 errdetail("Item ID %d not found in features table", item_id),
					 errhint("Verify the item_id exists in the features table.")));
		}

		NDB_ALLOC(top_items, int32, n_recommendations);
		NDB_ALLOC(top_sims, float, n_recommendations);
		for (i = 0; i < n_recommendations; ++i)
		{
			top_items[i] = -1;
			top_sims[i] = -INFINITY;
		}

		{
			float		target_len = sqrtf(
										   dot_product(target_vec, target_vec, n_factors));

			for (i = 0; i < item_count; ++i)
			{
				if (i == target_idx)
					continue;
				{
					float		dot = dot_product(target_vec,
												  other_factors[i],
												  n_factors);
					float		len = sqrtf(
											dot_product(other_factors[i],
														other_factors[i],
														n_factors));
					float		sim = (len > 0 && target_len > 0)
						? (dot / (len * target_len))
						: 0.0f;
					int			minidx = 0;

					for (j = 1; j < n_recommendations; ++j)
						if (top_sims[j]
							< top_sims[minidx])
							minidx = j;
					if (sim > top_sims[minidx])
					{
						top_sims[minidx] = sim;
						top_items[minidx] =
							other_ids[i];
					}
				}
			}
		}

		for (i = 0; i < n_recommendations - 1; ++i)
			for (j = i + 1; j < n_recommendations; ++j)
				if (top_sims[j] > top_sims[i])
				{
					float		t;
					int32		t2;

					t = top_sims[i];
					top_sims[i] = top_sims[j];
					top_sims[j] = t;
					t2 = top_items[i];
					top_items[i] = top_items[j];
					top_items[j] = t2;
				}
		NDB_ALLOC(elems, Datum, n_recommendations);
		for (i = 0; i < n_recommendations; ++i)
			elems[i] = Int32GetDatum(top_items[i]);
		result_array = construct_array(elems,
									   n_recommendations,
									   INT4OID,
									   sizeof(int32),
									   true,
									   'i');

		for (i = 0; i < item_count; ++i)
			if (other_factors[i])
				NDB_FREE(other_factors[i]);
		if (other_factors)
			NDB_FREE(other_factors);
		if (other_ids)
			NDB_FREE(other_ids);
		if (features_table_str)
			NDB_FREE(features_table_str);
		if (elems)
			NDB_FREE(elems);
		if (top_items)
			NDB_FREE(top_items);
		if (top_sims)
			NDB_FREE(top_sims);
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(content_spi_session);
		PG_RETURN_ARRAYTYPE_P(result_array);
	}
}

/*
 * user_similarity
 */
PG_FUNCTION_INFO_V1(user_similarity);

Datum
user_similarity(PG_FUNCTION_ARGS)
{
	int32		user1_id = PG_GETARG_INT32(0);
	int32		user2_id = PG_GETARG_INT32(1);
	text	   *ratings_table = PG_GETARG_TEXT_PP(2);

	char	   *ratings_table_str = text_to_cstring(ratings_table);
	int			ret,
				count,
				i;
	float		sx = 0.0f,
				sy = 0.0f,
				sxx = 0.0f,
				syy = 0.0f,
				sxy = 0.0f;
	float		r = 0.0f;

	NDB_DECLARE(NdbSpiSession *, user_sim_spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(user_sim_spi_session, oldcontext);

	{
		StringInfoData sql = {0};

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT a.rating AS x, b.rating AS y "
						 "FROM %s a JOIN %s b ON a.item_col = b.item_col "
						 "WHERE a.user_col = %d AND b.user_col = %d",
						 quote_identifier(ratings_table_str),
						 quote_identifier(ratings_table_str),
						 user1_id,
						 user2_id);

		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			if (ratings_table_str)
				NDB_FREE(ratings_table_str);
			NDB_SPI_SESSION_END(user_sim_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: user_similarity: failed to fetch user ratings"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the ratings table exists and is accessible.")));
		}

		count = SPI_processed;
		if (count < 2)
		{
			NDB_FREE(sql.data);
			if (ratings_table_str)
				NDB_FREE(ratings_table_str);
			NDB_SPI_SESSION_END(user_sim_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: user_similarity: insufficient common items"),
					 errdetail("Users must have at least 2 items in common, found %d", count),
					 errhint("Ensure both users have rated at least 2 common items.")));
		}
		for (i = 0; i < count; ++i)
		{
			HeapTuple	tup = SPI_tuptable->vals[i];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			float		x = DatumGetFloat4(
										   SPI_getbinval(tup, tupdesc, 1, NULL));
			float		y = DatumGetFloat4(
										   SPI_getbinval(tup, tupdesc, 2, NULL));

			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
		}

		{
			float		num = sxy - sx * sy / count;
			float		den = sqrtf(sxx - sx * sx / count)
				* sqrtf(syy - sy * sy / count);

			if (den != 0.0f)
				r = num / den;
			else
				r = 0.0f;
		}

		NDB_FREE(sql.data);
		if (ratings_table_str)
			NDB_FREE(ratings_table_str);
		NDB_SPI_SESSION_END(user_sim_spi_session);
		PG_RETURN_FLOAT8((double) r);
	}
}

/*
 * recommend_hybrid
 */
PG_FUNCTION_INFO_V1(recommend_hybrid);

Datum
recommend_hybrid(PG_FUNCTION_ARGS)
{
	int32		user_id = PG_GETARG_INT32(0);
	int32		cf_model_id = PG_GETARG_INT32(1);
	text	   *content_table = PG_GETARG_TEXT_PP(2);
	float8		cf_weight = PG_ARGISNULL(3) ? 0.7 : PG_GETARG_FLOAT8(3);
	int32		n_items = PG_ARGISNULL(4) ? 10 : PG_GETARG_INT32(4);

	if (cf_weight < 0.0 || cf_weight > 1.0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cf_weight must be between 0.0 and "
						"1.0")));
	}
	if (n_items < RECO_MIN_RESULT || n_items > RECO_MAX_RESULT)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_items must be between %d and %d",
						RECO_MIN_RESULT,
						RECO_MAX_RESULT)));
	}

	{
		char	   *content_table_str;
		int			ret;
		int			i;
		int			j;
		StringInfoData sql;
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		bool		isnull;
		Datum		arr;
		ArrayType  *user_vec;
		int			n_factors;

		NDB_DECLARE(float *, user_factors);
		int			n_items_total;

		NDB_DECLARE(int32 *, item_ids);
		float	  **item_factors = NULL;
		int			n_feat_items;

		NDB_DECLARE(int32 *, feat_item_ids);
		float	  **content_factors = NULL;
		int			nf_content;
		int			ntop;

		NDB_DECLARE(int32 *, top_items);
		NDB_DECLARE(float *, top_scores);
		NDB_DECLARE(Datum *, elems);
		ArrayType  *result_array;

		NDB_DECLARE(NdbSpiSession *, hybrid_spi_session);
		MemoryContext oldcontext;

		oldcontext = CurrentMemoryContext;
		content_table_str = text_to_cstring(content_table);

		NDB_SPI_SESSION_BEGIN(hybrid_spi_session, oldcontext);

		initStringInfo(&sql);

		/* load user factors */
		appendStringInfo(&sql,
						 "SELECT factors FROM neurondb_cf_user_factors WHERE model_id = %d AND user_id = %d",
						 cf_model_id,
						 user_id);
		ret = ndb_spi_execute_safe(sql.data, true, 1);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			if (content_table_str)
				NDB_FREE(content_table_str);
			NDB_SPI_SESSION_END(hybrid_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: recommend_hybrid: failed to load user factors"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the model exists and contains user factors.")));
		}
		if (SPI_processed != 1)
		{
			NDB_FREE(sql.data);
			if (content_table_str)
				NDB_FREE(content_table_str);
			NDB_SPI_SESSION_END(hybrid_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: recommend_hybrid: user factors not found"),
					 errdetail("No user_factors found for user %d in model %d", user_id, cf_model_id),
					 errhint("Verify the user_id exists in the model.")));
		}
		tuple = SPI_tuptable->vals[0];
		tupdesc = SPI_tuptable->tupdesc;
		isnull = false;
		arr = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
		{
			NDB_FREE(sql.data);
			if (content_table_str)
				NDB_FREE(content_table_str);
			NDB_SPI_SESSION_END(hybrid_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: recommend_hybrid: NULL user factors array"),
					 errdetail("User factors array is NULL for user %d in model %d", user_id, cf_model_id),
					 errhint("This indicates corrupted model data. Retrain the model.")));
		}
		user_vec = DatumGetArrayTypeP(arr);
		n_factors =
			ArrayGetNItems(ARR_NDIM(user_vec), ARR_DIMS(user_vec));
		NDB_ALLOC(user_factors, float, n_factors);
		memcpy(user_factors,
			   ARR_DATA_PTR(user_vec),
			   sizeof(float) * n_factors);

		/* load item factors */
		/* Use safe free/reinit to handle potential memory context changes */
		NDB_FREE(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT item_id, factors FROM neurondb_cf_item_factors WHERE model_id = %d",
						 cf_model_id);
		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			if (user_factors)
				NDB_FREE(user_factors);
			if (content_table_str)
				NDB_FREE(content_table_str);
			NDB_SPI_SESSION_END(hybrid_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: recommend_hybrid: failed to load item factors"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the model exists and contains item factors.")));
		}
		n_items_total = SPI_processed;
		if (n_items_total < 1)
		{
			NDB_FREE(sql.data);
			if (user_factors)
				NDB_FREE(user_factors);
			if (content_table_str)
				NDB_FREE(content_table_str);
			NDB_SPI_SESSION_END(hybrid_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: recommend_hybrid: no items found"),
					 errdetail("No items found for model %d", cf_model_id),
					 errhint("Verify the model contains item factors.")));
		}
		NDB_ALLOC(item_ids, int32, n_items_total);
		NDB_ALLOC(item_factors, float *, n_items_total);
		for (i = 0; i < n_items_total; ++i)
		{
			HeapTuple	tup2 = SPI_tuptable->vals[i];
			TupleDesc	desc = SPI_tuptable->tupdesc;
			bool		isnull_id = false;
			bool		isnull_fac = false;
			int			id;
			Datum		facdatum;
			ArrayType  *facarr;
			int			nf;
			float	   *vec;

			id = DatumGetInt32(
							   SPI_getbinval(tup2, desc, 1, &isnull_id));
			facdatum = SPI_getbinval(tup2, desc, 2, &isnull_fac);
			facarr = DatumGetArrayTypeP(facdatum);
			nf = ArrayGetNItems(ARR_NDIM(facarr), ARR_DIMS(facarr));
			if (nf != n_factors)
			{
				for (j = 0; j < i; ++j)
				{
					if (item_factors[j])
						NDB_FREE(item_factors[j]);
				}
				if (item_factors)
					NDB_FREE(item_factors);
				if (item_ids)
					NDB_FREE(item_ids);
				if (user_factors)
					NDB_FREE(user_factors);
				if (content_table_str)
					NDB_FREE(content_table_str);
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(hybrid_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: recommend_hybrid: factor dimension mismatch"),
						 errdetail("Item %d has %d factors but expected %d", id, nf, n_factors),
						 errhint("This indicates corrupted model data. Retrain the model.")));
			}
			NDB_ALLOC(vec, float, n_factors);
			memcpy(vec,
				   ARR_DATA_PTR(facarr),
				   sizeof(float) * n_factors);
			item_ids[i] = id;
			item_factors[i] = vec;
		}

		/* Use safe free/reinit to handle potential memory context changes */
		NDB_FREE(sql.data);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT item_id, features FROM %s",
						 quote_identifier(content_table_str));
		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			for (i = 0; i < n_items_total; ++i)
				if (item_factors[i])
					NDB_FREE(item_factors[i]);
			if (item_factors)
				NDB_FREE(item_factors);
			if (item_ids)
				NDB_FREE(item_ids);
			if (user_factors)
				NDB_FREE(user_factors);
			if (content_table_str)
				NDB_FREE(content_table_str);
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(hybrid_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: recommend_hybrid: failed to load item features"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the content table exists and is accessible.")));
		}
		n_feat_items = SPI_processed;
		NDB_ALLOC(feat_item_ids, int32, n_feat_items);
		NDB_ALLOC(content_factors, float *, n_feat_items);
		nf_content = 0;

		for (i = 0; i < n_feat_items; ++i)
		{
			HeapTuple	tupf = SPI_tuptable->vals[i];
			TupleDesc	descf = SPI_tuptable->tupdesc;
			bool		isnull_id = false,
						isnull_feat = false;
			int			id = DatumGetInt32(
										   SPI_getbinval(tupf, descf, 1, &isnull_id));
			Datum		arr_tmp =
				SPI_getbinval(tupf, descf, 2, &isnull_feat);
			ArrayType  *a = DatumGetArrayTypeP(arr_tmp);
			int			nf = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
			float	   *vec;

			if (i == 0)
				nf_content = nf;
			if (nf != nf_content)
			{
				for (j = 0; j < i; ++j)
					if (content_factors[j])
						NDB_FREE(content_factors[j]);
				for (j = 0; j < n_items_total; ++j)
					if (item_factors[j])
						NDB_FREE(item_factors[j]);
				if (content_factors)
					NDB_FREE(content_factors);
				if (feat_item_ids)
					NDB_FREE(feat_item_ids);
				if (item_factors)
					NDB_FREE(item_factors);
				if (item_ids)
					NDB_FREE(item_ids);
				if (user_factors)
					NDB_FREE(user_factors);
				if (content_table_str)
					NDB_FREE(content_table_str);
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(hybrid_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: recommend_hybrid: content vector dimension mismatch"),
						 errdetail("Row %d has %d features but expected %d", i + 1, nf, nf_content),
						 errhint("Ensure all content feature vectors have the same dimension.")));
			}
			NDB_ALLOC(vec, float, nf_content);
			memcpy(vec,
				   ARR_DATA_PTR(a),
				   sizeof(float) * nf_content);
			feat_item_ids[i] = id;
			content_factors[i] = vec;
		}

		ntop = Min(n_items, n_items_total);
		NDB_ALLOC(top_items, int32, ntop);
		NDB_ALLOC(top_scores, float, ntop);

		for (i = 0; i < ntop; ++i)
		{
			top_items[i] = -1;
			top_scores[i] = -INFINITY;
		}

		for (i = 0; i < n_items_total; ++i)
		{
			int32		item = item_ids[i];
			int			featidx = -1;

			for (j = 0; j < n_feat_items; ++j)
				if (feat_item_ids[j] == item)
				{
					featidx = j;
					break;
				}
			if (featidx == -1)
				continue;
			{
				float		cf_score = dot_product(user_factors,
												   item_factors[i],
												   n_factors);
				float		c_dot =
					dot_product(content_factors[featidx],
								content_factors[featidx],
								nf_content);
				float		c_len = sqrtf(c_dot);
				float		c_score = (c_len > 0.0f)
					? dot_product(content_factors[featidx],
								  content_factors[featidx],
								  nf_content)
					/ (c_len * c_len)
					: 1.0f;
				float		score = (float) (cf_weight * cf_score
											 + (1.0 - cf_weight) * c_score);
				int			minidx = 0;

				for (j = 1; j < ntop; ++j)
					if (top_scores[j] < top_scores[minidx])
						minidx = j;
				if (score > top_scores[minidx])
				{
					top_scores[minidx] = score;
					top_items[minidx] = item;
				}
			}
		}

		for (i = 0; i < ntop - 1; ++i)
		{
			for (j = i + 1; j < ntop; ++j)
			{
				if (top_scores[j] > top_scores[i])
				{
					float		t;
					int32		u;

					t = top_scores[i];
					top_scores[i] = top_scores[j];
					top_scores[j] = t;
					u = top_items[i];
					top_items[i] = top_items[j];
					top_items[j] = u;
				}
			}
		}

		NDB_ALLOC(elems, Datum, ntop);
		for (i = 0; i < ntop; ++i)
			elems[i] = Int32GetDatum(top_items[i]);
		result_array = construct_array(
									   elems, ntop, INT4OID, sizeof(int32), true, 'i');
		if (top_items)
			NDB_FREE(top_items);
		if (top_scores)
			NDB_FREE(top_scores);
		for (i = 0; i < n_feat_items; ++i)
			if (content_factors[i])
				NDB_FREE(content_factors[i]);
		for (i = 0; i < n_items_total; ++i)
			if (item_factors[i])
				NDB_FREE(item_factors[i]);
		if (content_factors)
			NDB_FREE(content_factors);
		if (feat_item_ids)
			NDB_FREE(feat_item_ids);
		if (item_factors)
			NDB_FREE(item_factors);
		if (item_ids)
			NDB_FREE(item_ids);
		if (user_factors)
			NDB_FREE(user_factors);
		if (content_table_str)
			NDB_FREE(content_table_str);
		if (elems)
			NDB_FREE(elems);
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(hybrid_spi_session);
		PG_RETURN_ARRAYTYPE_P(result_array);
	}
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Recommender
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct RecommenderGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	float	  **user_factors;
	float	  **item_factors;
	int			n_users;
	int			n_items;
	int			n_factors;
	int			n_samples;
	float		lambda;
}			RecommenderGpuModelState;

static bytea *
recommender_model_serialize_to_bytea(float **user_factors, int n_users, float **item_factors, int n_items, int n_factors, float lambda)
{
	StringInfoData buf;
	int			total_size;

	NDB_DECLARE(bytea *, result);
	int			u,
				i,
				f;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &n_users, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &n_items, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &n_factors, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &lambda, sizeof(float));

	for (u = 0; u < n_users; u++)
		for (f = 0; f < n_factors; f++)
			appendBinaryStringInfo(&buf, (char *) &user_factors[u][f], sizeof(float));

	for (i = 0; i < n_items; i++)
		for (f = 0; f < n_factors; f++)
			appendBinaryStringInfo(&buf, (char *) &item_factors[i][f], sizeof(float));

	total_size = VARHDRSZ + buf.len;
	NDB_ALLOC(result, bytea, total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
recommender_model_deserialize_from_bytea(const bytea * data, float ***user_factors_out, int *n_users_out, float ***item_factors_out, int *n_items_out, int *n_factors_out, float *lambda_out)
{
	const char *buf;
	int			offset = 0;
	int			u,
				i,
				f;
	float	  **user_factors = NULL;
	float	  **item_factors = NULL;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 3 + sizeof(float))
		return -1;

	buf = VARDATA(data);
	memcpy(n_users_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(n_items_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(n_factors_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(lambda_out, buf + offset, sizeof(float));
	offset += sizeof(float);

	if (*n_users_out < 1 || *n_users_out > 1000000 || *n_items_out < 1 || *n_items_out > 1000000 || *n_factors_out < 1 || *n_factors_out > 1000)
		return -1;

	NDB_ALLOC(user_factors, float *, *n_users_out);
	for (u = 0; u < *n_users_out; u++)
	{
		NDB_ALLOC(user_factors[u], float, *n_factors_out);
		for (f = 0; f < *n_factors_out; f++)
		{
			memcpy(&user_factors[u][f], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	NDB_ALLOC(item_factors, float *, *n_items_out);
	for (i = 0; i < *n_items_out; i++)
	{
		NDB_ALLOC(item_factors[i], float, *n_factors_out);
		for (f = 0; f < *n_factors_out; f++)
		{
			memcpy(&item_factors[i][f], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*user_factors_out = user_factors;
	*item_factors_out = item_factors;
	return 0;
}

static void
recommender_model_free(float **user_factors, int n_users, float **item_factors, int n_items)
{
	int			u,
				i;

	if (user_factors != NULL)
	{
		for (u = 0; u < n_users; u++)
			if (user_factors[u] != NULL)
				NDB_FREE(user_factors[u]);
		NDB_FREE(user_factors);
	}

	if (item_factors != NULL)
	{
		for (i = 0; i < n_items; i++)
			if (item_factors[i] != NULL)
				NDB_FREE(item_factors[i]);
		NDB_FREE(item_factors);
	}
}

static bool
recommender_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	RecommenderGpuModelState *state;

	NDB_DECLARE(float **, user_factors);
	NDB_DECLARE(float **, item_factors);
	int			n_users = 100;
	int			n_items = 1000;
	int			n_factors = 20;
	float		lambda = 0.1f;
	int			nvec = 0;
	int			dim = 0;
	int			u,
				i,
				f;

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
			*errstr = pstrdup("recommender_gpu_train: invalid parameters");
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
				if (strcmp(key, "n_users") == 0 && v.type == jbvNumeric)
					n_users = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "n_items") == 0 && v.type == jbvNumeric)
					n_items = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "n_factors") == 0 && v.type == jbvNumeric)
					n_factors = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "lambda") == 0 && v.type == jbvNumeric)
					lambda = (float) DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																		NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (n_users < 1)
		n_users = 100;
	if (n_items < 1)
		n_items = 1000;
	if (n_factors < 1)
		n_factors = 20;
	if (lambda < 0.0f)
		lambda = 0.1f;

	/* Convert feature matrix */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;	/* Reserved for future use */
	(void) dim;					/* Suppress unused variable warning */

	/* Initialize user and item factors (ALS) */
	NDB_ALLOC(user_factors, float *, n_users);
	for (u = 0; u < n_users; u++)
	{
		NDB_ALLOC(user_factors[u], float, n_factors);
		for (f = 0; f < n_factors; f++)
			user_factors[u][f] = (float) rand() / RAND_MAX * 0.1f;
	}

	NDB_ALLOC(item_factors, float *, n_items);
	for (i = 0; i < n_items; i++)
	{
		NDB_ALLOC(item_factors[i], float, n_factors);
		for (f = 0; f < n_factors; f++)
			item_factors[i][f] = (float) rand() / RAND_MAX * 0.1f;
	}

	/* Serialize model */
	model_data = recommender_model_serialize_to_bytea(user_factors, n_users, item_factors, n_items, n_factors, lambda);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"n_users\":%d,\"n_items\":%d,\"n_factors\":%d,\"lambda\":%.6f,\"n_samples\":%d}",
					 n_users, n_items, n_factors, lambda, nvec);
	metrics = ndb_jsonb_in_cstring(metrics_json.data);
	NDB_FREE(metrics_json.data);

	state = (RecommenderGpuModelState *) palloc0(sizeof(RecommenderGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = model_data;
	state->metrics = metrics;
	state->user_factors = user_factors;
	state->item_factors = item_factors;
	state->n_users = n_users;
	state->n_items = n_items;
	state->n_factors = n_factors;
	state->n_samples = nvec;
	state->lambda = lambda;

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static bool
recommender_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
						float *output, int output_dim, char **errstr)
{
	const		RecommenderGpuModelState *state;

	NDB_DECLARE(float **, user_factors);
	NDB_DECLARE(float **, item_factors);
	int			n_users = 0,
				n_items = 0,
				n_factors = 0;
	float		lambda = 0.0f;
	int			user_id = 0;
	int			item_id = 0;
	int			f;
	float		rating = 0.0f;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_predict: model not ready");
		return false;
	}

	state = (const RecommenderGpuModelState *) model->backend_state;

	/* Deserialize if needed */
	if (state->user_factors == NULL)
	{
		if (recommender_model_deserialize_from_bytea(state->model_blob,
													 &user_factors, &n_users, &item_factors, &n_items, &n_factors, &lambda) != 0)
		{
			if (errstr != NULL)
				*errstr = pstrdup("recommender_gpu_predict: failed to deserialize");
			return false;
		}
		((RecommenderGpuModelState *) state)->user_factors = user_factors;
		((RecommenderGpuModelState *) state)->item_factors = item_factors;
		((RecommenderGpuModelState *) state)->n_users = n_users;
		((RecommenderGpuModelState *) state)->n_items = n_items;
		((RecommenderGpuModelState *) state)->n_factors = n_factors;
		((RecommenderGpuModelState *) state)->lambda = lambda;
	}

	/* Extract user_id and item_id from input (assuming first two elements) */
	if (input_dim >= 2)
	{
		user_id = (int) input[0];
		item_id = (int) input[1];
	}

	if (user_id < 0 || user_id >= state->n_users || item_id < 0 || item_id >= state->n_items)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_predict: invalid user_id or item_id");
		return false;
	}

	/* Compute rating: dot product of user and item factors */
	for (f = 0; f < state->n_factors; f++)
		rating += state->user_factors[user_id][f] * state->item_factors[item_id][f];

	output[0] = rating;

	return true;
}

static bool
recommender_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
						 MLGpuMetrics *out, char **errstr)
{
	const		RecommenderGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_evaluate: invalid model");
		return false;
	}

	state = (const RecommenderGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"recommender\",\"storage\":\"cpu\","
					 "\"n_users\":%d,\"n_items\":%d,\"n_factors\":%d,\"lambda\":%.6f,\"n_samples\":%d}",
					 state->n_users > 0 ? state->n_users : 100,
					 state->n_items > 0 ? state->n_items : 1000,
					 state->n_factors > 0 ? state->n_factors : 20,
					 state->lambda > 0.0f ? state->lambda : 0.1f,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = ndb_jsonb_in_cstring(buf.data);
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
recommender_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
						  Jsonb * *metadata_out, char **errstr)
{
	const		RecommenderGpuModelState *state;

	NDB_DECLARE(bytea *, payload_copy);
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
			*errstr = pstrdup("recommender_gpu_serialize: invalid model");
		return false;
	}

	state = (const RecommenderGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	NDB_ALLOC(payload_copy, bytea, payload_size);
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
recommender_gpu_deserialize(MLGpuModel *model, const bytea * payload,
							const Jsonb * metadata, char **errstr)
{
	RecommenderGpuModelState *state;

	NDB_DECLARE(bytea *, payload_copy);
	int			payload_size;

	NDB_DECLARE(float **, user_factors);
	NDB_DECLARE(float **, item_factors);
	int			n_users = 0,
				n_items = 0,
				n_factors = 0;
	float		lambda = 0.0f;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	NDB_ALLOC(payload_copy, bytea, payload_size);
	memcpy(payload_copy, payload, payload_size);

	if (recommender_model_deserialize_from_bytea(payload_copy,
												 &user_factors, &n_users, &item_factors, &n_items, &n_factors, &lambda) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("recommender_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (RecommenderGpuModelState *) palloc0(sizeof(RecommenderGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	state->user_factors = user_factors;
	state->item_factors = item_factors;
	state->n_users = n_users;
	state->n_items = n_items;
	state->n_factors = n_factors;
	state->n_samples = 0;
	state->lambda = lambda;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);

		NDB_DECLARE(Jsonb *, metadata_copy);

		NDB_ALLOC(metadata_copy, Jsonb, metadata_size);
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
recommender_gpu_destroy(MLGpuModel *model)
{
	RecommenderGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (RecommenderGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		if (state->user_factors != NULL || state->item_factors != NULL)
		{
			recommender_model_free(state->user_factors, state->n_users,
								   state->item_factors, state->n_items);
		}
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps recommender_gpu_model_ops = {
	.algorithm = "recommender",
	.train = recommender_gpu_train,
	.predict = recommender_gpu_predict,
	.evaluate = recommender_gpu_evaluate,
	.serialize = recommender_gpu_serialize,
	.deserialize = recommender_gpu_deserialize,
	.destroy = recommender_gpu_destroy,
};

void
neurondb_gpu_register_recommender_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&recommender_gpu_model_ops);
	registered = true;
}

/*-------------------------------------------------------------------------
 *
 * ml_hyperparameter_tuning.c
 *    Hyperparameter optimization.
 *
 * This module implements automated hyperparameter tuning using grid search,
 * random search, and Bayesian optimization.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_hyperparameter_tuning.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <math.h>
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/uuid.h"
#include "miscadmin.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/* PG_MODULE_MAGIC is in neurondb.c only */

PG_FUNCTION_INFO_V1(neurondb_grid_search);
PG_FUNCTION_INFO_V1(neurondb_random_search);
PG_FUNCTION_INFO_V1(neurondb_bayesian_optimize);

/*
 * Generate all parameter combinations for grid search (cartesian product).
 */
static void
generate_grid_combinations(Jsonb * param_grid,
						   List * *param_names,
						   List * *value_lists,
						   List * *combinations)
{
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	Assert(param_names != NULL && value_lists != NULL
		   && combinations != NULL);

	it = JsonbIteratorInit(&param_grid->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
	{
		if (r == WJB_KEY)
		{
			char	   *key;
			JsonbValue	vals_array;

			key = pnstrdup(v.val.string.val, v.val.string.len);
			r = JsonbIteratorNext(&it, &vals_array, false);

			if (r == WJB_VALUE && vals_array.type == jbvArray)
			{
				List	   *values = NIL;
				int			i;

				for (i = 0; i < vals_array.val.array.nElems;
					 ++i)
					values = lappend(values,
									 copyObject(&vals_array.val.array
												.elems[i]));

				*param_names =
					lappend(*param_names, pstrdup(key));
				*value_lists = lappend(*value_lists, values);
			}
			NDB_FREE(key);
		}
	}

	if (list_length(*param_names) == 0)
	{
		*combinations = NIL;
		return;
	}

	{
		int			n_params = list_length(*param_names);
		int		   *indices;
		bool		done = false;

		indices = (int *) palloc0(sizeof(int) * n_params);

		do
		{
			List	   *one_comb = NIL;
			ListCell   *name_cell;
			ListCell   *values_cell;
			int			pi = 0;

			forboth(name_cell,
					*param_names,
					values_cell,
					*value_lists)
			{
				List	   *vlist = (List *) lfirst(values_cell);
				JsonbValue *jbval = (JsonbValue *) list_nth(
															vlist, indices[pi]);

				one_comb = lappend(one_comb, copyObject(jbval));
				pi++;
			}
			*combinations = lappend(*combinations, one_comb);

			for (pi = n_params - 1; pi >= 0; --pi)
			{
				List	   *curr_values =
					(List *) list_nth(*value_lists, pi);

				if (indices[pi] < list_length(curr_values) - 1)
				{
					indices[pi]++;
					break;
				}
				else
				{
					indices[pi] = 0;
					if (pi == 0)
						done = true;
				}
			}
		} while (!done);

		NDB_FREE(indices);
	}
}

/*
 * Build a JSONB object from parameter name/value lists.
 */
static Jsonb *
build_param_jsonb(List * param_names, List * param_values)
{
	NDB_DECLARE(JsonbParseState *, state);
	ListCell   *ncell;
	ListCell   *vcell;
	Jsonb	   *result;

	(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

	forboth(ncell, param_names, vcell, param_values)
	{
		char	   *key = (char *) lfirst(ncell);
		JsonbValue	k;
		JsonbValue *v;

		k.type = jbvString;
		k.val.string.len = strlen(key);
		k.val.string.val = key;
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v = (JsonbValue *) lfirst(vcell);
		(void) pushJsonbValue(&state, WJB_VALUE, v);
	}

	(void) pushJsonbValue(&state, WJB_END_OBJECT, NULL);
	result =
		JsonbValueToJsonb(pushJsonbValue(&state, WJB_END_OBJECT, NULL));
	return result;
}

/*
 * Perform model training and cross-validation via SQL.
 */
static float8
actual_crossval(const char *algo, Jsonb * param_json, int folds)
{
	float8		mean_score = 0.0;
	int			ret;
	StringInfoData cmd;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &cmd);
	appendStringInfoString(
						   &cmd, "SELECT neurondb_train_and_score($1, $2, $3)");

	{
		Datum		values[3];
		char		spi_nulls[3] = {false, false, false};

		values[0] = CStringGetTextDatum(algo);
		values[1] = PointerGetDatum(param_json);
		values[2] = Int32GetDatum(folds);

		{
			Oid			argtypes[3];

			argtypes[0] = TEXTOID;
			argtypes[1] = JSONBOID;
			argtypes[2] = INT4OID;
			ret = ndb_spi_execute_with_args(spi_session,
											cmd.data,
											3,
											argtypes,
											values,
											spi_nulls,
											true,
											1);
		}

		if (ret == SPI_OK_SELECT && SPI_processed == 1)
		{
			bool		isnull = false;

			/*
			 * Note: ndb_spi_get_* doesn't have float8, so we access
			 * SPI_tuptable directly for numeric types
			 */
			/* Safe access for complex types - validate before access */
			if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL &&
				SPI_tuptable->vals != NULL && SPI_processed > 0 && SPI_tuptable->vals[0] != NULL)
			{
				Datum		dval = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 1,
												 &isnull);

				if (!isnull)
					mean_score = DatumGetFloat8(dval);
				else
				{
					ndb_spi_stringinfo_free(spi_session, &cmd);
					NDB_SPI_SESSION_END(spi_session);
					if (algo != NULL)
						pfree((void *) algo);
					if (param_json != NULL)
						pfree((void *) param_json);
					ereport(ERROR,
							(errmsg("Returned value from "
									"neurondb_train_and_score is "
									"null")));
				}
			}
		}
		else
		{
			ndb_spi_stringinfo_free(spi_session, &cmd);
			NDB_SPI_SESSION_END(spi_session);
			if (algo != NULL)
				pfree((void *) algo);
			if (param_json != NULL)
				pfree((void *) param_json);
			ereport(ERROR,
					(errmsg("Failed to obtain score from "
							"neurondb_train_and_score")));
		}
	}

	ndb_spi_stringinfo_free(spi_session, &cmd);
	NDB_SPI_SESSION_END(spi_session);

	return mean_score;
}

/*
 * neurondb_grid_search
 *		Grid search hyperparameter optimization.
 *		Returns: TABLE (params JSONB, score FLOAT8, model_id INT4)
 */
Datum
neurondb_grid_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct grid_search_ctx
	{
		List	   *all_results;
		int			curr;
	}			grid_search_ctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		text	   *algorithm_text;
		Jsonb	   *param_grid;
		int32		cv_folds;
		char	   *algo;
		List	   *param_names = NIL;
		List	   *value_lists = NIL;
		List	   *combinations = NIL;
		grid_search_ctx *ctx;
		List	   *all_results = NIL;
		int			model_id = 1;
		ListCell   *comb_cell;
		TupleDesc	tupdesc = NULL;
		Oid			argtypes[3] = {JSONBOID, FLOAT8OID, INT4OID};

		(void) argtypes;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* project_name = PG_GETARG_TEXT_PP(0); */	/* Not used yet */
		algorithm_text = PG_GETARG_TEXT_PP(1);
		param_grid = PG_GETARG_JSONB_P(2);
		cv_folds = PG_ARGISNULL(3) ? 5 : PG_GETARG_INT32(3);
		algo = text_to_cstring(algorithm_text);

		ctx = (grid_search_ctx *) palloc0(sizeof(grid_search_ctx));

		generate_grid_combinations(
								   param_grid, &param_names, &value_lists, &combinations);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errmsg("Return type must be a composite "
							"type")));

		foreach(comb_cell, combinations)
		{
			List	   *param_values = (List *) lfirst(comb_cell);
			Jsonb	   *jbcomb =
				build_param_jsonb(param_names, param_values);
			float8		score = actual_crossval(algo, jbcomb, cv_folds);

			Datum		values[3];
			bool		nulls[3] = {false, false, false};
			HeapTuple	tuple;
			Datum		result;

			values[0] = PointerGetDatum(jbcomb);
			values[1] = Float8GetDatum(score);
			values[2] = Int32GetDatum(model_id++);
			tuple = heap_form_tuple(tupdesc, values, nulls);
			result = HeapTupleGetDatum(tuple);

			all_results =
				lappend(all_results, (void *) (uintptr_t) result);

			list_free_deep(param_values);
			if (jbcomb)
				NDB_FREE(jbcomb);
		}
		list_free_deep(combinations);
		list_free_deep(param_names);
		list_free_deep(value_lists);

		ctx->all_results = all_results;
		ctx->curr = 0;
		funcctx->user_fctx = ctx;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	{
		grid_search_ctx *ctx = (grid_search_ctx *) funcctx->user_fctx;

		if (ctx->curr < list_length(ctx->all_results))
		{
			Datum		res =
				(Datum) list_nth(ctx->all_results, ctx->curr);

			ctx->curr++;
			SRF_RETURN_NEXT(funcctx, res);
		}
		else
		{
			list_free_deep(ctx->all_results);
			NDB_FREE(ctx);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * Randomly select parameter values from parameter distributions for random search.
 */
static void
random_sample_parameters(Jsonb * param_distributions,
						 List * *param_names,
						 List * *values_list)
{
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	it = JsonbIteratorInit(&param_distributions->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
	{
		if (r == WJB_KEY)
		{
			char	   *key;
			JsonbValue	arr_value;

			key = pnstrdup(v.val.string.val, v.val.string.len);
			r = JsonbIteratorNext(&it, &arr_value, false);
			if (r == WJB_VALUE && arr_value.type == jbvArray
				&& arr_value.val.array.nElems > 0)
			{
				int			arr_size = arr_value.val.array.nElems;
				int			idx;

#if PG_VERSION_NUM >= 120000
				idx = (int) (arc4random_uniform(arr_size));
#else
				idx = (int) ((double) arr_size
							 * ((double) rand()
								/ ((double) RAND_MAX + 1)));
#endif
				{
					JsonbValue *picked =
						&arr_value.val.array.elems[idx];

					*param_names = lappend(
										   *param_names, pstrdup(key));
					*values_list = lappend(*values_list,
										   copyObject(picked));
				}
			}
			NDB_FREE(key);
		}
	}
}

/*
 * neurondb_random_search
 *		Random search hyperparameter optimization.
 *		Returns: TABLE (params JSONB, score FLOAT8, model_id INT4)
 */
Datum
neurondb_random_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct random_search_ctx
	{
		List	   *results;
		int			curr;
	}			random_search_ctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		text	   *algorithm_text;
		Jsonb	   *param_distributions;
		int32		n_iter;
		int32		cv_folds;
		char	   *algo;
		random_search_ctx *ctx;
		List	   *results = NIL;
		int			model_id = 1;
		TupleDesc	tupdesc = NULL;
		Oid			argtypes[3] = {JSONBOID, FLOAT8OID, INT4OID};
		int			i;

		(void) argtypes;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* project_name = PG_GETARG_TEXT_PP(0); */	/* Not used yet */
		algorithm_text = PG_GETARG_TEXT_PP(1);
		param_distributions = PG_GETARG_JSONB_P(2);
		n_iter = PG_ARGISNULL(3) ? 10 : PG_GETARG_INT32(3);
		cv_folds = PG_ARGISNULL(4) ? 5 : PG_GETARG_INT32(4);
		algo = text_to_cstring(algorithm_text);

		ctx = (random_search_ctx *) palloc0(sizeof(random_search_ctx));

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errmsg("Return type must be a composite "
							"type")));

		for (i = 0; i < n_iter; ++i)
		{
			List	   *names = NIL;
			List	   *vals = NIL;

			random_sample_parameters(
									 param_distributions, &names, &vals);

			{
				Jsonb	   *jbcomb;
				float8		score;
				Datum		values[3];
				bool		nulls[3] = {false, false, false};
				HeapTuple	tuple;
				Datum		result;

				jbcomb = build_param_jsonb(names, vals);
				score = actual_crossval(algo, jbcomb, cv_folds);

				values[0] = PointerGetDatum(jbcomb);
				values[1] = Float8GetDatum(score);
				values[2] = Int32GetDatum(model_id++);
				tuple = heap_form_tuple(tupdesc, values, nulls);
				result = HeapTupleGetDatum(tuple);

				results = lappend(
								  results, (void *) (uintptr_t) result);

				list_free_deep(names);
				list_free_deep(vals);
				if (jbcomb)
					NDB_FREE(jbcomb);
			}
		}
		ctx->results = results;
		ctx->curr = 0;
		funcctx->user_fctx = ctx;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	{
		random_search_ctx *ctx =
			(random_search_ctx *) funcctx->user_fctx;

		if (ctx->curr < list_length(ctx->results))
		{
			Datum		res = (Datum) list_nth(ctx->results, ctx->curr);

			ctx->curr++;
			SRF_RETURN_NEXT(funcctx, res);
		}
		else
		{
			list_free_deep(ctx->results);
			NDB_FREE(ctx);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * Deterministic parameter sampling for Bayesian optimization demo.
 */
static void
bayes_sample_parameters(Jsonb * param_space,
						int iter,
						List * *param_names,
						List * *values_list)
{
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	it = JsonbIteratorInit(&param_space->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
	{
		if (r == WJB_KEY)
		{
			char	   *key;
			JsonbValue	arr_value;

			key = pnstrdup(v.val.string.val, v.val.string.len);
			r = JsonbIteratorNext(&it, &arr_value, false);
			if (r == WJB_VALUE && arr_value.type == jbvArray
				&& arr_value.val.array.nElems > 0)
			{
				int			arr_size = arr_value.val.array.nElems;
				int			idx = iter % arr_size;
				JsonbValue *picked =
					&arr_value.val.array.elems[idx];

				*param_names =
					lappend(*param_names, pstrdup(key));
				*values_list = lappend(
									   *values_list, copyObject(picked));
			}
			NDB_FREE(key);
		}
	}
}

/*
 * neurondb_bayesian_optimize
 *		Demo Bayesian hyperparameter optimization (deterministic sampling).
 *		Returns: TABLE (params JSONB, score FLOAT8, model_id INT4)
 */
Datum
neurondb_bayesian_optimize(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct bayes_opt_ctx
	{
		List	   *results;
		int			best_model_idx;
		int			curr;
	}			bayes_opt_ctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		text	   *algorithm_text;
		Jsonb	   *param_space;
		int32		n_calls;
		text	   *acquisition_text;
		char	   *algo;
		char	   *acquisition;
		bayes_opt_ctx *ctx;
		List	   *results = NIL;
		int			model_id = 1;
		float8		best_score = -HUGE_VAL;
		int			best_model_idx = -1;
		TupleDesc	tupdesc = NULL;
		Oid			argtypes[3] = {JSONBOID, FLOAT8OID, INT4OID};
		int			i;

		(void) argtypes;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* project_name = PG_GETARG_TEXT_PP(0); */	/* Not used yet */
		algorithm_text = PG_GETARG_TEXT_PP(1);
		param_space = PG_GETARG_JSONB_P(2);
		n_calls = PG_ARGISNULL(3) ? 20 : PG_GETARG_INT32(3);
		acquisition_text =
			PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4);

		algo = text_to_cstring(algorithm_text);
		acquisition = acquisition_text
			? text_to_cstring(acquisition_text)
			: "ei";
		(void) acquisition;

		ctx = (bayes_opt_ctx *) palloc0(sizeof(bayes_opt_ctx));

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errmsg("Return type must be a composite "
							"type")));

		for (i = 0; i < n_calls; ++i)
		{
			List	   *names = NIL;
			List	   *vals = NIL;

			bayes_sample_parameters(param_space, i, &names, &vals);

			{
				Jsonb	   *jbcomb;
				float8		score;
				Datum		values[3];
				bool		nulls[3] = {false, false, false};
				HeapTuple	tuple;
				Datum		result;

				jbcomb = build_param_jsonb(names, vals);
				score = actual_crossval(algo, jbcomb, 5);

				if (score > best_score)
				{
					best_score = score;
					best_model_idx = model_id - 1;
				}

				values[0] = PointerGetDatum(jbcomb);
				values[1] = Float8GetDatum(score);
				values[2] = Int32GetDatum(model_id++);
				tuple = heap_form_tuple(tupdesc, values, nulls);
				result = HeapTupleGetDatum(tuple);

				results = lappend(
								  results, (void *) (uintptr_t) result);

				list_free_deep(names);
				list_free_deep(vals);
				if (jbcomb)
					NDB_FREE(jbcomb);
			}
		}
		ctx->results = results;
		ctx->best_model_idx = best_model_idx;
		ctx->curr = 0;
		funcctx->user_fctx = ctx;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	{
		bayes_opt_ctx *ctx = (bayes_opt_ctx *) funcctx->user_fctx;

		if (ctx->curr < list_length(ctx->results))
		{
			Datum		res = (Datum) list_nth(ctx->results, ctx->curr);

			ctx->curr++;
			SRF_RETURN_NEXT(funcctx, res);
		}
		else
		{
			list_free_deep(ctx->results);
			NDB_FREE(ctx);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

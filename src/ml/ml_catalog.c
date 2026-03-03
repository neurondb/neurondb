/*-------------------------------------------------------------------------
 *
 * ml_catalog.c
 *    Machine learning model catalog management.
 *
 * This module provides utilities for model registration, retrieval, and
 * versioning in the neurondb.ml_models catalog.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_catalog.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <ctype.h>

#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/elog.h"

#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_json.h"
#include "neurondb_spi.h"
#include "neurondb_constants.h"

/*
 * ml_catalog_default_project
 *		Generate default project name from algorithm and training table.
 *
 * Returns a palloc'd string that must be freed by caller.
 */
static const char *
ml_catalog_default_project(const char *algorithm, const char *training_table)
{
	char *sanitized_table = NULL;
	const char *result;
	int			i,
				j;

	if (algorithm == NULL)
		return pstrdup("ml_default_project");

	/* Check if training_table is NULL, empty, or whitespace-only */
	if (training_table == NULL || strlen(training_table) == 0)
		return psprintf("%s_project", algorithm);

	/* Check if training_table is whitespace-only */
	for (i = 0; training_table[i] != '\0'; i++)
	{
		if (!isspace((unsigned char) training_table[i]))
			break;
	}
	if (training_table[i] == '\0')
		return psprintf("%s_project", algorithm);

	/*
	 * Sanitize table name: replace spaces and other problematic chars with
	 * underscores
	 */
	nalloc(sanitized_table, char, strlen(training_table) + 1);
	j = 0;
	for (i = 0; training_table[i] != '\0'; i++)
	{
		char		c = training_table[i];

		/* Replace spaces and other problematic characters with underscores */
		if (isspace((unsigned char) c) || c == '-' || c == '.' || c == '/')
			sanitized_table[j++] = '_';
		else if (isalnum((unsigned char) c) || c == '_')
			sanitized_table[j++] = c;
		/* Skip other characters */
	}
	sanitized_table[j] = '\0';

	/* If sanitized table is empty after sanitization, use algorithm only */
	if (strlen(sanitized_table) == 0)
	{
		return psprintf("%s_project", algorithm);
	}

	result = psprintf("%s_%s_project", algorithm, sanitized_table);

	/*
	 * sanitized_table will be automatically freed when memory context is
	 * reset
	 */
	return result;
}

/*
 * ml_catalog_register_model
 *		Register a new model in neurondb.ml_models catalog.
 *
 * Validates inputs, creates/updates project entry, acquires versioning lock,
 * calculates next version, and inserts model metadata. Returns the model_id.
 */
int32
ml_catalog_register_model(const MLCatalogModelSpec *spec)
{
	const char *project_name;
	const char *model_type;
	const char *training_table;
	const char *training_column;
	const char *algorithm;
	Jsonb *parameters = NULL;
	Jsonb *metrics = NULL;
	int32		project_id = 0;
	int32		version = 1;
	int32		model_id = 0;
	int32		ret;
	int			error_code = 0;
	StringInfoData sql = {0};
	StringInfoData insert_query = {0};
	StringInfoData select_query = {0};
	StringInfoData insert_sql = {0};
	MemoryContext oldcontext;

	NdbSpiSession *spi_session = NULL;
	char *algorithm_copy = NULL;
	char *training_table_copy = NULL;
	char *training_column_copy = NULL;
	char *project_name_copy = NULL;
	char *params_txt = NULL;
	char *metrics_txt = NULL;
	char *params_quoted = NULL;
	char *metrics_quoted = NULL;
	char *algorithm_quoted = NULL;
	char *table_quoted = NULL;
	char *column_quoted = NULL;
	char *meta_txt = NULL;
	char *saved_project_name = NULL;

	/*
	 * Initialize copy pointers to NULL since we're not doing string copying
	 * anymore
	 */
	project_name_copy = NULL;

	/* Try to safely check if spec is valid by accessing it in a PG_TRY block */
	PG_TRY();
	{
		if (spec == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("invalid spec parameter")),
					 errdetail("The spec parameter is NULL"),
					 errhint("Provide a valid MLCatalogModelSpec structure with required fields.")));
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid spec parameter"),
				 errdetail("The spec parameter appears to be invalid or corrupted"),
				 errhint("The spec pointer may be pointing to invalid memory.")));
	}
	PG_END_TRY();

	Assert(spec != NULL);
	if (spec == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid spec parameter"),
				 errdetail("The spec parameter is NULL"),
				 errhint("Provide a valid MLCatalogModelSpec structure with required fields.")));

	/* Wrap spec->algorithm access in PG_TRY to catch crash */
	PG_TRY();
	{
		/*
		 * Try to access spec->algorithm - this might crash if spec is
		 * corrupted
		 */
		algorithm = spec->algorithm;
	}
	PG_CATCH();
	{
		FlushErrorState();
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid spec parameter"),
				 errdetail("The spec structure appears to be corrupted or pointing to invalid memory"),
				 errhint("The spec pointer may be valid but the structure it points to may be invalid.")));
	}
	PG_END_TRY();
	if (algorithm == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("algorithm must be provided")),
				 errdetail("The algorithm field in spec is NULL"),
				 errhint("Set spec->algorithm to a valid algorithm name (e.g., 'logistic_regression', 'random_forest').")));
	}

	training_table = spec->training_table;
	if (training_table == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("training_table must be provided")),
				 errdetail("The training_table field in spec is NULL"),
				 errhint("Set spec->training_table to the name of the table used for training.")));
	}

	training_column = spec->training_column;

	if (spec->model_type != NULL)
	{
		model_type = spec->model_type;
	}
	else
	{
		model_type = "classification";
	}

	if (spec->project_name != NULL && strlen(spec->project_name) > 0)
	{
		project_name = spec->project_name;
	}
	else
	{
		project_name = ml_catalog_default_project(algorithm, training_table);
	}

	/* Ensure project_name is valid and not empty before proceeding */
	if (project_name == NULL || strlen(project_name) == 0)
	{
		project_name = ml_catalog_default_project(algorithm, training_table);
		if (project_name == NULL || strlen(project_name) == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("project_name is invalid")),
					 errdetail("project_name is NULL or empty after default resolution"),
					 errhint("Ensure project_name is set correctly in the spec or that ml_catalog_default_project returns a valid value.")));
		}
	}

	parameters = spec->parameters;
	metrics = spec->metrics;

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	if (oldcontext == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("CurrentMemoryContext is NULL")),
				 errdetail("Cannot proceed without a valid memory context"),
				 errhint("This is an internal error. Please report this issue.")));
	}

	/*
	 * Strings were already copied in neurondb_train(), skip copying here to
	 * avoid crashes
	 */

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	Assert(spi_session != NULL);
	if (metrics != NULL)
	{
		meta_txt = ndb_jsonb_out_cstring(metrics);
		elog(DEBUG2, "neurondb: ml_catalog_register_model: metrics JSON: %s", meta_txt);
		pfree(meta_txt);
	}

	/* Validate project_name one more time before using it in SQL */
	if (project_name == NULL || strlen(project_name) == 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("project_name is invalid before SQL execution")),
				 errdetail("project_name is NULL or empty: '%s'", project_name ? project_name : "(NULL)"),
				 errhint("This should not happen - project_name should have been validated earlier.")));
	}

	/* Create or update project entry and retrieve project_id */
	ndb_spi_stringinfo_init(spi_session, &insert_query);
	ndb_spi_stringinfo_init(spi_session, &select_query);

	/*
	 * Use INSERT ... RETURNING to get project_id directly, avoiding
	 * transaction visibility issues
	 */
	appendStringInfo(&insert_query,
					 "INSERT INTO " NDB_FQ_ML_PROJECTS " "
					 "(project_name, model_type, description) "
					 "VALUES ('%s', '%s'::" NDB_FQ_TYPE_ML_MODEL ", '%s') "
					 "ON CONFLICT (project_name) DO UPDATE "
					 "SET updated_at = NOW() "
					 "RETURNING " NDB_COL_PROJECT_ID,
					 project_name,
					 model_type,
					 "Auto-created by ml_catalog_register_model");

	ret = ndb_spi_execute(spi_session, insert_query.data, false, 1);

	/* Check for INSERT_RETURNING or UPDATE_RETURNING (both return project_id) */
	if (ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING)
	{
		ereport(WARNING, (errmsg("ml_catalog_register_model: INSERT did not return project_id, ret=%d", ret)));
		/* Fall back to separate SELECT if RETURNING didn't work */
		ndb_spi_stringinfo_free(spi_session, &insert_query);
		ndb_spi_stringinfo_init(spi_session, &select_query);
		appendStringInfo(&select_query,
						 "SELECT " NDB_COL_PROJECT_ID " FROM " NDB_FQ_ML_PROJECTS " WHERE project_name = '%s'",
						 project_name);
		ret = ndb_spi_execute(spi_session, select_query.data, true, 0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			error_code = (ret != SPI_OK_SELECT) ? 2 : 3;
			goto error;
		}
		ndb_spi_stringinfo_free(spi_session, &select_query);
	}
	else
	{
		ndb_spi_stringinfo_free(spi_session, &insert_query);
	}

	if (SPI_processed == 0)
	{
		ereport(WARNING, (errmsg("ml_catalog_register_model: No rows returned for project_name='%s'", project_name)));
		error_code = 3;
		goto error;
	}

	/* Retrieve and validate project_id */
	if (!ndb_spi_get_int32(spi_session, 0, 1, &project_id))
	{
		ereport(WARNING, (errmsg("ml_catalog_register_model: ndb_spi_get_int32 failed for project_name='%s'", project_name)));
		/* Save project_name before error handling might free it */
		if (saved_project_name == NULL)
			saved_project_name = (project_name != NULL && strlen(project_name) > 0) ? pstrdup(project_name) : pstrdup("(unknown)");
		error_code = 3;
		/* Use saved_project_name in error reporting */
		project_name = saved_project_name;
		goto error;
	}

	Assert(project_id > 0);
	if (project_id <= 0)
	{
		error_code = 4;
		goto error;
	}


	/*
	 * Acquire advisory lock to prevent race conditions during version
	 * calculation
	 */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT pg_advisory_xact_lock(%d)", project_id);

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_SELECT)
	{
		error_code = 5;
		goto error;
	}

	/* Calculate next version number */
	ndb_spi_stringinfo_reset(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT COALESCE(MAX(" NDB_COL_VERSION "), 0) + 1 "
					 "FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_PROJECT_ID " = %d",
					 project_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
		version = 1;
	else
	{
		if (!ndb_spi_get_int32(spi_session, 0, 0, &version))
			version = 1;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);

	/* Insert model row with quoted string literals for SQL safety */
	ndb_spi_stringinfo_init(spi_session, &insert_sql);

	/* Quote string literals to prevent SQL injection */
	{
		text *algorithm_text = NULL;
		text *quoted_algorithm = NULL;

		algorithm_text = cstring_to_text(algorithm);
		quoted_algorithm = DatumGetTextP(
										 DirectFunctionCall1(quote_literal,
															 PointerGetDatum(algorithm_text)));
		algorithm_quoted = text_to_cstring(quoted_algorithm);
		pfree(algorithm_text);
		pfree(quoted_algorithm);
	}
	{
		text *table_text = NULL;
		text *quoted_table = NULL;

		table_text = cstring_to_text(training_table);
		quoted_table = DatumGetTextP(
									 DirectFunctionCall1(quote_literal,
														 PointerGetDatum(table_text)));
		table_quoted = text_to_cstring(quoted_table);
		pfree(table_text);
		pfree(quoted_table);
	}
	column_quoted = NULL;
	if (training_column != NULL)
	{
		text *column_text = NULL;
		text *quoted_column = NULL;

		column_text = cstring_to_text(training_column);
		quoted_column = DatumGetTextP(
									  DirectFunctionCall1(quote_literal,
														  PointerGetDatum(column_text)));
		column_quoted = text_to_cstring(quoted_column);
		pfree(column_text);
		pfree(quoted_column);
	}

	params_txt = NULL;
	params_quoted = NULL;
	if (parameters != NULL)
	{
		text *params_text = NULL;
		text *quoted_params = NULL;

		PG_TRY();
		{
			params_txt = ndb_jsonb_out_cstring(parameters);
			if (params_txt == NULL || strlen(params_txt) == 0)
			{
				params_quoted = NULL;
			}
			else
			{
				params_text = cstring_to_text(params_txt);
				quoted_params = DatumGetTextP(
											  DirectFunctionCall1(quote_literal,
																  PointerGetDatum(params_text)));
				params_quoted = text_to_cstring(quoted_params);
				pfree(params_text);
				pfree(quoted_params);
			}
		}
		PG_CATCH();
		{
			elog(WARNING,
				 "neurondb: ml_catalog_register_model: failed to convert parameters JSONB to text, using NULL");
			FlushErrorState();
			params_quoted = NULL;
		}
		PG_END_TRY();
	}

	metrics_txt = NULL;
	metrics_quoted = NULL;
	if (metrics != NULL)
	{
		text *metrics_text = NULL;
		text *quoted_metrics = NULL;

		PG_TRY();
		{
			metrics_txt = ndb_jsonb_out_cstring(metrics);
			if (metrics_txt == NULL || strlen(metrics_txt) == 0)
			{
				metrics_quoted = NULL;
			}
			else
			{
				metrics_text = cstring_to_text(metrics_txt);
				quoted_metrics = DatumGetTextP(
												   DirectFunctionCall1(quote_literal,
																	   PointerGetDatum(metrics_text)));
				metrics_quoted = text_to_cstring(quoted_metrics);
				pfree(metrics_text);
				pfree(quoted_metrics);
			}
		}
		PG_CATCH();
		{
			elog(WARNING,
				 "neurondb: ml_catalog_register_model: failed to convert metrics JSONB to text, using NULL");
			FlushErrorState();
			metrics_quoted = NULL;
		}
		PG_END_TRY();
	}

	{
		char *time_ms_str = NULL;
		char *num_samples_str = NULL;
		char *num_features_str = NULL;

		time_ms_str = (spec->training_time_ms >= 0) ? psprintf("%d", spec->training_time_ms) : NULL;
		num_samples_str = (spec->num_samples > 0) ? psprintf("%d", spec->num_samples) : NULL;
		num_features_str = (spec->num_features > 0) ? psprintf("%d", spec->num_features) : NULL;

		appendStringInfo(&insert_sql,
						 "INSERT INTO " NDB_FQ_ML_MODELS " "
						 "(" NDB_COL_PROJECT_ID ", " NDB_COL_VERSION ", " NDB_COL_ALGORITHM ", " NDB_COL_STATUS ", "
						 NDB_COL_TRAINING_TABLE ", "
						 NDB_COL_TRAINING_COLUMN ", parameters, model_data, " NDB_COL_METRICS ", "
						 "training_time_ms, num_samples, num_features, "
						 "completed_at) "
						 "VALUES (%d, %d, %s::text::" NDB_FQ_TYPE_ML_ALGORITHM ", '" NDB_STATUS_COMPLETED "', "
						 "%s, %s, %s::jsonb, NULL, %s::jsonb, "
						 "%s, %s, %s, NOW()) "
						 "ON CONFLICT (" NDB_COL_PROJECT_ID ", " NDB_COL_VERSION ") DO UPDATE SET "
						 NDB_COL_ALGORITHM " = EXCLUDED." NDB_COL_ALGORITHM ", "
						 NDB_COL_STATUS " = EXCLUDED." NDB_COL_STATUS ", "
						 NDB_COL_TRAINING_TABLE " = EXCLUDED." NDB_COL_TRAINING_TABLE ", "
						 NDB_COL_TRAINING_COLUMN " = EXCLUDED." NDB_COL_TRAINING_COLUMN ", "
						 "parameters = EXCLUDED.parameters, "
						 NDB_COL_METRICS " = EXCLUDED." NDB_COL_METRICS ", "
						 "training_time_ms = EXCLUDED.training_time_ms, "
						 "num_samples = EXCLUDED.num_samples, "
						 "num_features = EXCLUDED.num_features, "
						 "completed_at = EXCLUDED.completed_at "
						 "RETURNING " NDB_COL_MODEL_ID,
						 project_id,
						 version,
						 algorithm_quoted,
						 table_quoted,
						 column_quoted != NULL ? column_quoted : "NULL",
				 params_quoted != NULL && strlen(params_quoted) > 0 ? params_quoted : "NULL",
				 metrics_quoted != NULL && strlen(metrics_quoted) > 0 ? metrics_quoted : "NULL",
			 time_ms_str != NULL ? time_ms_str : "NULL",
			 num_samples_str != NULL ? num_samples_str : "NULL",
			 num_features_str != NULL ? num_features_str : "NULL");

	if (time_ms_str) pfree(time_ms_str);
	if (num_samples_str) pfree(num_samples_str);
	if (num_features_str) pfree(num_features_str);
	}	/* Note: Enum values should be added during extension installation via neurondb--1.0.sql.
	 * We use ::text::enum cast to allow using enum values that may have been added in the same
	 * transaction (though PostgreSQL generally doesn't allow this, the cast is a workaround).
	 * If the enum value doesn't exist, the INSERT will fail with a helpful error message. */

	ret = ndb_spi_execute(spi_session, insert_sql.data, false, 1);

	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) ||
		SPI_processed == 0)
	{
		ereport(WARNING, (errmsg("ml_catalog_register_model: model insert failed or returned 0 rows, ret=%d", ret)));
		error_code = 6;
		goto error;
	}

	if (!ndb_spi_get_int32(spi_session, 0, 1, &model_id))
	{
		ereport(WARNING, (errmsg("ml_catalog_register_model: ndb_spi_get_int32 failed for model_id, algorithm='%s', table='%s'",
								 algorithm ? algorithm : "(NULL)", training_table ? training_table : "(NULL)")));
		error_code = 7;
		goto error;
	}

	Assert(model_id > 0);
	if (model_id <= 0)
	{
		error_code = 8;
		goto error;
	}

	/* Optionally update model_data using parameterized query */
	if (spec->model_data != NULL && model_id > 0)
	{
		Oid			argtypes[2];
		Datum		values[2];
		char		nulls[2];

		ndb_spi_stringinfo_reset(spi_session, &sql);
		appendStringInfo(&sql,
						 "UPDATE " NDB_FQ_ML_MODELS " SET model_data = $1 WHERE " NDB_COL_MODEL_ID " = $2");

		argtypes[0] = BYTEAOID;
		argtypes[1] = INT4OID;
		/* SPI_execute_with_args will copy the bytea Datum automatically */
		values[0] = PointerGetDatum(spec->model_data);
		values[1] = Int32GetDatum(model_id);
		nulls[0] = ' ';
		nulls[1] = ' ';

		ret = ndb_spi_execute_with_args(spi_session, sql.data, 2, argtypes, values, nulls, false, 0);
		if (ret != SPI_OK_UPDATE || SPI_processed != 1)
		{
			elog(WARNING,
				 "neurondb: ml_catalog_register_model: failed to update model_data for model_id %d (ret=%d, processed=%lu)",
				 model_id, ret, (unsigned long) SPI_processed);
		}
		ndb_spi_stringinfo_free(spi_session, &sql);
	}

	ndb_spi_stringinfo_free(spi_session, &insert_sql);
	if (algorithm_quoted) pfree(algorithm_quoted);
	if (table_quoted) pfree(table_quoted);
	if (column_quoted) pfree(column_quoted);
	if (params_txt) pfree(params_txt);
	if (params_quoted) pfree(params_quoted);
	if (metrics_txt) pfree(metrics_txt);
	if (metrics_quoted) pfree(metrics_quoted);

	goto success;

error:
	{
		char *error_project_name = NULL;

		/* Save project_name for error reporting before freeing */
		if (project_name != NULL && strlen(project_name) > 0)
		{
			error_project_name = pstrdup(project_name);
		}
		else if (project_name_copy != NULL && strlen(project_name_copy) > 0)
		{
			error_project_name = pstrdup(project_name_copy);
		}
		else
		{
			error_project_name = pstrdup("(unknown)");
		}

		if (insert_query.data != NULL)
			ndb_spi_stringinfo_free(spi_session, &insert_query);
		if (select_query.data != NULL)
			ndb_spi_stringinfo_free(spi_session, &select_query);
		if (insert_sql.data != NULL)
			ndb_spi_stringinfo_free(spi_session, &insert_sql);
		if (sql.data != NULL)
			ndb_spi_stringinfo_free(spi_session, &sql);
		nfree(algorithm_quoted);
		nfree(table_quoted);
		nfree(column_quoted);
		nfree(params_txt);
		nfree(params_quoted);
		nfree(metrics_txt);
		nfree(metrics_quoted);
		NDB_SPI_SESSION_END(spi_session);
		nfree(project_name_copy);
		nfree(algorithm_copy);
		nfree(training_table_copy);
		nfree(training_column_copy);

		/* Use saved project_name in error messages */
		project_name = error_project_name;
	}

	switch (error_code)
	{
		case 1:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("failed to insert project into catalog")),
					 errdetail("SPI execution returned error code %d. Project name: '%s', Model type: '%s'", ret, project_name, model_type),
					 errhint("Check database permissions and ensure the " NDB_FQ_ML_PROJECTS " table exists and is accessible.")));
			break;
		case 2:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("failed to retrieve project_id from catalog")),
					 errdetail("SPI execution returned code %d (expected %d for SELECT). Project name: '%s'", ret, SPI_OK_SELECT, project_name),
					 errhint("Check database permissions and ensure the " NDB_FQ_ML_PROJECTS " table exists and contains the project.")));
			break;
		case 3:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("failed to retrieve project_id from catalog")),
					 errdetail("No rows returned or invalid data type. Project name: '%s'", project_name),
					 errhint("Verify the project was successfully inserted and the catalog table structure is correct.")));
			break;
		case 4:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("invalid project_id returned from catalog")),
					 errdetail("Project ID is %d (must be > 0). Project name: '%s'", project_id, project_name),
					 errhint("This indicates a database integrity issue. Check the " NDB_FQ_ML_PROJECTS " table.")));
			break;
		case 5:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("failed to acquire advisory lock for project")),
					 errdetail("SPI execution returned code %d (expected %d for SELECT). Project ID: %d, Project name: '%s'", ret, SPI_OK_SELECT, project_id, project_name),
					 errhint("This may indicate a database locking issue. Retry the operation.")));
			break;
		case 6:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("failed to insert/update model in catalog")),
					 errdetail("SPI execution returned code %d (expected %d or %d), processed %lu rows. Project ID: %d, Algorithm: '%s', Table: '%s'", ret, SPI_OK_INSERT_RETURNING, SPI_OK_UPDATE_RETURNING, (unsigned long) SPI_processed, project_id, algorithm, training_table),
					 errhint("Check database permissions, table constraints, and ensure the " NDB_FQ_ML_MODELS " table exists and is accessible.")));
			break;
		case 7:
			{
				const char *err_algorithm = (algorithm != NULL && strlen(algorithm) > 0) ? algorithm : "(NULL or empty)";
				const char *err_table = (training_table != NULL && strlen(training_table) > 0) ? training_table : "(NULL or empty)";

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg(NDB_ERR_MSG("failed to retrieve model_id after insert/update")),
						 errdetail("No rows returned or invalid data type. Project ID: %d, Algorithm: '%s', Table: '%s'", project_id, err_algorithm, err_table),
						 errhint("Verify the model was successfully inserted and the catalog table structure is correct.")));
			}
			break;
		case 8:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("invalid model_id returned from catalog")),
					 errdetail("Model ID is %d (must be > 0). Project ID: %d, Algorithm: '%s', Table: '%s'", model_id, project_id, algorithm, training_table),
					 errhint("This indicates a database integrity issue. Check the " NDB_FQ_ML_MODELS " table.")));
			break;
		default:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("unknown error occurred during model registration")),
					 errdetail("An unexpected error occurred during model registration"),
					 errhint("This is an internal error. Please report this issue.")));
			break;
	}

success:
	NDB_SPI_SESSION_END(spi_session);
	nfree(project_name_copy);
	nfree(algorithm_copy);
	nfree(training_table_copy);
	nfree(training_column_copy);

	return model_id;
}

/*
 * ml_catalog_fetch_model_payload
 *		Fetch model payload (model_data, parameters, metrics) from catalog.
 *
 * Retrieves serialized model data and metadata for the given model_id.
 * All returned data is allocated in caller's memory context.
 * Returns true if model found, false otherwise.
 */
bool
ml_catalog_fetch_model_payload(int32 model_id,
							   bytea * *model_data_out,
							   Jsonb * *parameters_out,
							   Jsonb * *metrics_out)
{
	int			ret;
	StringInfoData sql = {0};
	bool		found = false;
	MemoryContext caller_context;

	NdbSpiSession *spi_session = NULL;

	if (model_data_out != NULL)
		*model_data_out = NULL;
	if (parameters_out != NULL)
		*parameters_out = NULL;
	if (metrics_out != NULL)
		*metrics_out = NULL;

	Assert(model_id > 0);
	if (model_id <= 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("invalid model_id")),
				 errdetail("Model ID must be greater than 0, got %d", model_id),
				 errhint("Provide a valid model_id from the " NDB_FQ_ML_MODELS " table.")));
	}

	caller_context = CurrentMemoryContext;
	Assert(caller_context != NULL);
	if (caller_context == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("CurrentMemoryContext is NULL")),
				 errdetail("Cannot proceed without a valid memory context"),
				 errhint("This is an internal error. Please report this issue.")));
	}

	NDB_SPI_SESSION_BEGIN(spi_session, caller_context);
	Assert(spi_session != NULL);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT model_data, parameters, " NDB_COL_METRICS " "
					 "FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
					 model_id);

	PG_TRY();
	{
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);

		if (ret == SPI_OK_SELECT)
		{
			if (SPI_processed > 0)
			{
				found = true;


				if (model_data_out != NULL)
				{
					bytea *copy = NULL;

					ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: about to get model_data, row=0, col=1")));
					copy = ndb_spi_get_bytea(spi_session, 0, 1, caller_context);
					ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: ndb_spi_get_bytea returned, copy=%p", (void *) copy)));
					if (copy == NULL)
					{
						*model_data_out = NULL;
						ereport(WARNING, (errmsg("ml_catalog_fetch_model_payload: model_data is NULL for model_id %d - model may not have been stored correctly", model_id)));
					}
					else
					{
						elog(DEBUG2, "neurondb: ml_catalog_fetch_model_payload: model_data size=%u", VARSIZE(copy));
						*model_data_out = copy;
						ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: model_data assigned")));
					}
				}

				if (parameters_out != NULL)
				{
					Jsonb *jsonb_val = NULL;

					ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: about to get parameters, row=0, col=2")));
					jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 2, caller_context);
					ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: ndb_spi_get_jsonb returned, jsonb_val=%p", (void *) jsonb_val)));
					if (jsonb_val == NULL)
					{
						*parameters_out = NULL;
					}
					else
					{
						*parameters_out = jsonb_val;
					}
				}

				if (metrics_out != NULL)
				{
					Jsonb *jsonb_val = NULL;

					ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: about to get metrics, row=0, col=3")));
					jsonb_val = ndb_spi_get_jsonb(spi_session, 0, 3, caller_context);
					ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: ndb_spi_get_jsonb returned, jsonb_val=%p", (void *) jsonb_val)));
					if (jsonb_val == NULL)
					{
						*metrics_out = NULL;
					}
					else
					{
						ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: metrics assigned")));
						*metrics_out = jsonb_val;
					}
				}
			}
			else
			{
			}
		}
		else
		{
			elog(WARNING, "neurondb: ml_catalog_fetch_model_payload: SPI execution returned code %d (expected %d) for model_id %d", ret, SPI_OK_SELECT, model_id);
		}

		ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: about to free SQL string")));
		ndb_spi_stringinfo_free(spi_session, &sql);
		ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: SQL string freed")));
	}
	PG_CATCH();
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		PG_RE_THROW();
	}
	PG_END_TRY();

	ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: about to end SPI session, spi_session=%p", (void *) spi_session)));
	NDB_SPI_SESSION_END(spi_session);

	/* Ensure we're in the caller's context before returning */
	if (caller_context != NULL && CurrentMemoryContext != caller_context)
	{
		ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: switching to caller_context before return")));
		MemoryContextSwitchTo(caller_context);
	}

	ereport(DEBUG2, (errmsg("ml_catalog_fetch_model_payload: CurrentMemoryContext=%p, caller_context=%p", (void *) CurrentMemoryContext, (void *) caller_context)));

	/* Store return value to avoid potential stack issues */
	{
		bool		retval = found;

		elog(DEBUG2, "neurondb: ml_catalog_fetch_model_payload: retval=%d, executing return", retval);
		return retval;
	}
}

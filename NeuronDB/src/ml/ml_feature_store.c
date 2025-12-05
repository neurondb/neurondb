/*-------------------------------------------------------------------------
 *
 * ml_feature_store.c
 *    Feature store management.
 *
 * This module provides feature management, versioning, and serving for
 * machine learning workflows.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_feature_store.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/timestamp.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

static char *entity_key_for_store(int32 store_id);
static char *entity_table_for_store(int32 store_id);
static void get_feature_definitions(int32 store_id,
									char ***feat_names,
									char ***feat_types,
									char ***feat_transforms,
									int *n_features);

/*
 * neurondb.create_feature_store()
 */
PG_FUNCTION_INFO_V1(neurondb_create_feature_store);

Datum
neurondb_create_feature_store(PG_FUNCTION_ARGS)
{
	text	   *store_name_text = PG_GETARG_TEXT_PP(0);
	text	   *entity_table_text = PG_GETARG_TEXT_PP(1);
	text	   *entity_key_text = PG_GETARG_TEXT_PP(2);
	text	   *description_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	char	   *store_name = text_to_cstring(store_name_text);
	char	   *entity_table = text_to_cstring(entity_table_text);
	char	   *entity_key = text_to_cstring(entity_key_text);
	char	   *description =
		description_text ? text_to_cstring(description_text) : NULL;

	StringInfoData sql;
	int			ret;
	int32		store_id_val;
	int			store_id;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	elog(DEBUG1,
		 "neurondb.create_feature_store: name='%s', entity_table='%s', entity_key='%s'",
		 store_name,
		 entity_table,
		 entity_key);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Insert feature store definition */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb.feature_stores "
					 "(store_name, entity_table, entity_key, description, "
					 "created_at, updated_at) "
					 "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
					 "RETURNING store_id",
					 quote_literal_cstr(store_name),
					 quote_literal_cstr(entity_table),
					 quote_literal_cstr(entity_key),
					 description ? quote_literal_cstr(description) : "NULL");

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(store_name);
		NDB_FREE(entity_table);
		NDB_FREE(entity_key);
		if (description)
			NDB_FREE(description);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to create feature store")));
	}

	if (!ndb_spi_get_int32(spi_session, 0, 1, &store_id_val))
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(store_name);
		NDB_FREE(entity_table);
		NDB_FREE(entity_key);
		if (description)
			NDB_FREE(description);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to get store_id from result")));
	}
	store_id = store_id_val;

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	NDB_FREE(store_name);
	NDB_FREE(entity_table);
	NDB_FREE(entity_key);
	if (description)
		NDB_FREE(description);

	PG_RETURN_INT32(store_id);
}

/*
 * neurondb.register_feature()
 */
PG_FUNCTION_INFO_V1(neurondb_register_feature);

Datum
neurondb_register_feature(PG_FUNCTION_ARGS)
{
	int32		store_id = PG_GETARG_INT32(0);
	text	   *feature_name_text = PG_GETARG_TEXT_PP(1);
	text	   *feature_type_text = PG_GETARG_TEXT_PP(2);
	text	   *transformation_text =
		PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	char	   *feature_name;
	char	   *feature_type;
	char	   *transformation;
	StringInfoData sql;
	int			ret;
	int32		feature_id_val;
	int			feature_id;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	feature_name = text_to_cstring(feature_name_text);
	feature_type = text_to_cstring(feature_type_text);
	transformation = transformation_text
		? text_to_cstring(transformation_text)
		: NULL;

	elog(DEBUG1,
		 "neurondb.register_feature: store=%d, feature='%s', type=%s, transformation='%s'",
		 store_id,
		 feature_name,
		 feature_type,
		 transformation ? transformation : "none");

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	if (transformation)
	{
		appendStringInfo(&sql,
						 "INSERT INTO neurondb.feature_definitions ("
						 " store_id, feature_name, feature_type, "
						 "transformation, created_at) "
						 "VALUES (%d, %s, %s, %s, CURRENT_TIMESTAMP) "
						 "RETURNING feature_id",
						 store_id,
						 quote_literal_cstr(feature_name),
						 quote_literal_cstr(feature_type),
						 quote_literal_cstr(transformation));
	}
	else
	{
		appendStringInfo(&sql,
						 "INSERT INTO neurondb.feature_definitions ("
						 " store_id, feature_name, feature_type, created_at) "
						 "VALUES (%d, %s, %s, CURRENT_TIMESTAMP) "
						 "RETURNING feature_id",
						 store_id,
						 quote_literal_cstr(feature_name),
						 quote_literal_cstr(feature_type));
	}

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(feature_name);
		NDB_FREE(feature_type);
		if (transformation)
			NDB_FREE(transformation);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to register feature")));
	}

	if (!ndb_spi_get_int32(spi_session, 0, 1, &feature_id_val))
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(feature_name);
		NDB_FREE(feature_type);
		if (transformation)
			NDB_FREE(transformation);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to get feature_id from result")));
	}
	feature_id = feature_id_val;

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	NDB_FREE(feature_name);
	NDB_FREE(feature_type);
	if (transformation)
		NDB_FREE(transformation);

	PG_RETURN_INT32(feature_id);
}

/*
 * neurondb.get_features()
 * Retrieves features for a given list of entity keys and list of feature names.
 *
 * Arguments:
 *    0: int4 store_id
 *    1: text[] entity_ids
 *    2: text[] feature_names (optional, may be NULL for all)
 *
 * Returns:
 *    jsonb: {"entity_id": {"feature1": value1, ...}, ...}
 */
PG_FUNCTION_INFO_V1(neurondb_get_features);

Datum
neurondb_get_features(PG_FUNCTION_ARGS)
{
	int32		store_id = PG_GETARG_INT32(0);
	ArrayType  *entities_array = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType  *features_array =
		PG_ARGISNULL(2) ? NULL : PG_GETARG_ARRAYTYPE_P(2);

	Oid			entity_elemtype,
				feat_elemtype;
	int16		entity_typlen,
				feat_typlen;
	bool		entity_typbyval,
				feat_typbyval;
	char		entity_typalign,
				feat_typalign;
	int			num_entities,
				num_features = 0;
	int			i,
				j,
				ndefs;

	NDB_DECLARE(Datum *, entity_datums);
	NDB_DECLARE(bool *, entity_nulls);
	NDB_DECLARE(Datum *, feat_datums);
	NDB_DECLARE(bool *, feat_nulls);

	NDB_DECLARE(char *, entity_key_col);
	NDB_DECLARE(char *, entity_table);
	char	  **feature_names = NULL;
	char	  **feature_types = NULL;
	char	  **feature_transforms = NULL;
	MemoryContext oldcontext,
				fncontext;
	StringInfoData sql;
	int			spi_ret;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext_spi;

	NDB_DECLARE(JsonbParseState *, ps_top);
	NDB_DECLARE(JsonbValue *, o);
	JsonbValue	ent_key_jbv,
				feature_obj_value;
	Jsonb	   *result_jsonb;

	/* Validate array arguments */
	if (ARR_NDIM(entities_array) != 1 || ARR_HASNULL(entities_array))
		ereport(ERROR,
				(errmsg("entity_ids must be a one-dimensional, not "
						"null array")));
	entity_elemtype = ARR_ELEMTYPE(entities_array);
	get_typlenbyvalalign(entity_elemtype,
						 &entity_typlen,
						 &entity_typbyval,
						 &entity_typalign);
	deconstruct_array(entities_array,
					  entity_elemtype,
					  entity_typlen,
					  entity_typbyval,
					  entity_typalign,
					  &entity_datums,
					  &entity_nulls,
					  &num_entities);

	/* Features */
	if (features_array)
	{
		if (ARR_NDIM(features_array) != 1
			|| ARR_HASNULL(features_array))
			ereport(ERROR,
					(errmsg("feature_names must be a "
							"one-dimensional, not null array")));
		feat_elemtype = ARR_ELEMTYPE(features_array);
		get_typlenbyvalalign(feat_elemtype,
							 &feat_typlen,
							 &feat_typbyval,
							 &feat_typalign);
		deconstruct_array(features_array,
						  feat_elemtype,
						  feat_typlen,
						  feat_typbyval,
						  feat_typalign,
						  &feat_datums,
						  &feat_nulls,
						  &num_features);

		feature_names = (char **) palloc(sizeof(char *) * num_features);
		for (i = 0; i < num_features; ++i)
			feature_names[i] = TextDatumGetCString(feat_datums[i]);
	}

	/* Look up entity table name and entity key for this store */
	entity_key_col = entity_key_for_store(store_id);
	entity_table = entity_table_for_store(store_id);

	/* Find all feature definitions for this store */
	get_feature_definitions(store_id,
							&feature_names,
							&feature_types,
							&feature_transforms,
							&ndefs);

	/* Create a context for our output assembly */
	oldcontext_spi = CurrentMemoryContext;
	fncontext = AllocSetContextCreate(CurrentMemoryContext,
									  "neurondb_get_features_context",
									  ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(fncontext);

	/* Build SQL to select requested features for each entity */
	oldcontext_spi = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	for (i = 0; i < num_entities; ++i)
	{
		char	   *eid = TextDatumGetCString(entity_datums[i]);

		ndb_spi_stringinfo_init(spi_session, &sql);

		appendStringInfo(
						 &sql, "SELECT %s", quote_identifier(entity_key_col));
		for (j = 0; j < ndefs; ++j)
		{
			appendStringInfo(&sql, ", %s", quote_identifier(feature_names[j]));
		}
		appendStringInfo(&sql,
						 " FROM %s WHERE %s = %s",
						 quote_identifier(entity_table),
						 quote_identifier(entity_key_col),
						 quote_literal_cstr(eid));

		/* Execute for this entity */
		spi_ret = ndb_spi_execute(spi_session, sql.data, true, 1);

		if (spi_ret == SPI_OK_SELECT && SPI_processed == 1)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			int			nf;
			JsonbPair  *obj_pairs;
			bool		isnull;

			/* Safe access for complex types - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
			{
				continue;
			}
			tuple = SPI_tuptable->vals[0];
			tupdesc = SPI_tuptable->tupdesc;
			nf = ndefs;
			obj_pairs = (JsonbPair *) palloc(sizeof(JsonbPair) * nf);

			for (j = 0; j < ndefs; ++j)
			{
				Datum		feat_val;

				/* Safe access - validate tupdesc has enough columns */
				if (tupdesc->natts < (j + 2))
				{
					isnull = true;
					continue;
				}
				feat_val = SPI_getbinval(tuple, tupdesc, j + 2, &isnull);

				obj_pairs[j].key.type = jbvString;
				obj_pairs[j].key.val.string.len =
					strlen(feature_names[j]);
				obj_pairs[j].key.val.string.val =
					feature_names[j];
				if (isnull)
				{
					obj_pairs[j].value.type = jbvNull;
				}
				else
				{
					/* Numeric and text only */
					if (strcmp(feature_types[j], "float8")
						== 0)
					{
						obj_pairs[j].value.type =
							jbvNumeric;
						obj_pairs[j]
							.value.val
							.numeric = DatumGetNumeric(
													   DirectFunctionCall1(
																		   float8_numeric,
																		   feat_val));
					}
					else if (strcmp(feature_types[j],
									"int4")
							 == 0)
					{
						obj_pairs[j].value.type =
							jbvNumeric;
						obj_pairs[j]
							.value.val
							.numeric = DatumGetNumeric(
													   DirectFunctionCall1(
																		   int4_numeric,
																		   feat_val));
					}
					else
					{
						char	   *sval =
							TextDatumGetCString(
												feat_val);

						obj_pairs[j].value.type =
							jbvString;
						obj_pairs[j]
							.value.val.string.val =
							sval;
						obj_pairs[j]
							.value.val.string.len =
							strlen(sval);
					}
				}
			}
			/* Now build {"entity_id": {feature1: v1, ...}} */
			ent_key_jbv.type = jbvString;
			ent_key_jbv.val.string.val = eid;
			ent_key_jbv.val.string.len = strlen(eid);

			feature_obj_value.type = jbvObject;
			feature_obj_value.val.object.nPairs = nf;
			feature_obj_value.val.object.pairs = obj_pairs;

			pushJsonbValue(&ps_top, WJB_BEGIN_OBJECT, NULL);
			pushJsonbValue(&ps_top, WJB_KEY, &ent_key_jbv);
			pushJsonbValue(&ps_top, WJB_VALUE, &feature_obj_value);
			o = pushJsonbValue(&ps_top, WJB_END_OBJECT, NULL);
		}
		else
		{
			/* entity row not found: return nulls for all features */
			int			nf = ndefs;
			JsonbPair  *obj_pairs =
				(JsonbPair *) palloc(sizeof(JsonbPair) * nf);

			for (j = 0; j < ndefs; ++j)
			{
				obj_pairs[j].key.type = jbvString;
				obj_pairs[j].key.val.string.len =
					strlen(feature_names[j]);
				obj_pairs[j].key.val.string.val =
					feature_names[j];
				obj_pairs[j].value.type = jbvNull;
			}
			ent_key_jbv.type = jbvString;
			ent_key_jbv.val.string.val = eid;
			ent_key_jbv.val.string.len = strlen(eid);

			feature_obj_value.type = jbvObject;
			feature_obj_value.val.object.nPairs = nf;
			feature_obj_value.val.object.pairs = obj_pairs;

			pushJsonbValue(&ps_top, WJB_BEGIN_OBJECT, NULL);
			pushJsonbValue(&ps_top, WJB_KEY, &ent_key_jbv);
			pushJsonbValue(&ps_top, WJB_VALUE, &feature_obj_value);
			o = pushJsonbValue(&ps_top, WJB_END_OBJECT, NULL);
		}

		NDB_FREE(eid);
		ndb_spi_stringinfo_free(spi_session, &sql);
	}

	NDB_SPI_SESSION_END(spi_session);

	/* Top level: assemble all entity objects into one result object */
	if (!o)
		result_jsonb = DatumGetJsonbP(
									  DirectFunctionCall1(jsonb_in, CStringGetTextDatum("{}")));
	else
		result_jsonb = JsonbValueToJsonb(o);

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(fncontext);

	if (entity_key_col)
		NDB_FREE(entity_key_col);
	if (entity_table)
		NDB_FREE(entity_table);
	if (feature_names)
	{
		for (j = 0; j < ndefs; ++j)
			if (feature_names[j])
				NDB_FREE(feature_names[j]);
		NDB_FREE(feature_names);
	}
	if (feature_types)
	{
		for (j = 0; j < ndefs; ++j)
			if (feature_types[j])
				NDB_FREE(feature_types[j]);
		NDB_FREE(feature_types);
	}
	if (feature_transforms)
	{
		for (j = 0; j < ndefs; ++j)
			if (feature_transforms[j])
				NDB_FREE(feature_transforms[j]);
		NDB_FREE(feature_transforms);
	}
	NDB_FREE(entity_datums);

	PG_RETURN_JSONB_P(result_jsonb);
}

/*
 * neurondb.feature_engineering()
 * For every entity, applies feature transformations as defined in feature_definitions table.
 * Populates features in the entity_table.
 * Arguments:
 *    0 : store_id (int4)
 *    1 : JSONB pipeline (for future, not used)
 *    2 : source_table (text, table to source transformation data from)
 * Returns:
 *    int4: number of rows updated
 */
PG_FUNCTION_INFO_V1(neurondb_feature_engineering);

Datum
neurondb_feature_engineering(PG_FUNCTION_ARGS)
{
	int32		store_id = PG_GETARG_INT32(0);
	text	   *source_table_text;
	char	   *source_table;
	char	   *entity_key;
	char	   *entity_table;
	char	  **feature_names = NULL;
	char	  **feature_types = NULL;
	char	  **feature_transforms = NULL;
	int			ndefs,
				i,
				ret,
				updated = 0;
	StringInfoData sql;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	(void) PG_GETARG_JSONB_P(1);	/* unused for now */
	source_table_text = PG_GETARG_TEXT_PP(2);
	source_table = text_to_cstring(source_table_text);
	entity_key = entity_key_for_store(store_id);
	entity_table = entity_table_for_store(store_id);

	get_feature_definitions(store_id,
							&feature_names,
							&feature_types,
							&feature_transforms,
							&ndefs);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/*
	 * Apply each transformation to each feature. For simplicity, we evaluate
	 * SQL expressions for each transformation, and update the features in the
	 * entity_table via an UPDATE ... FROM ... statement. Transformation is
	 * assumed to be a SQL expression or NULL (in which case, nothing is
	 * done).
	 */
	for (i = 0; i < ndefs; ++i)
	{
		if (feature_transforms[i] == NULL
			|| strlen(feature_transforms[i]) == 0)
			continue;
		/* Build UPDATE SQL */
		ndb_spi_stringinfo_init(spi_session, &sql);
		appendStringInfo(&sql,
						 "UPDATE %s dst SET %s = src.val FROM "
						 "(SELECT %s AS %s, %s AS val FROM %s) src "
						 "WHERE dst.%s = src.%s",
						 quote_identifier(entity_table),
						 quote_identifier(feature_names[i]),
						 quote_identifier(entity_key),
						 quote_identifier(entity_key),
						 feature_transforms[i],
						 quote_identifier(source_table),
						 quote_identifier(entity_key),
						 quote_identifier(entity_key));
		ret = ndb_spi_execute(spi_session, sql.data, false, 0);
		if (ret != SPI_OK_UPDATE)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			NDB_SPI_SESSION_END(spi_session);
			NDB_FREE(source_table);
			if (entity_key)
				NDB_FREE(entity_key);
			if (entity_table)
				NDB_FREE(entity_table);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to update feature '%s'",
							feature_names[i])));
		}
		updated += SPI_processed;
		ndb_spi_stringinfo_free(spi_session, &sql);
	}

	NDB_SPI_SESSION_END(spi_session);

	NDB_FREE(source_table);
	if (entity_key)
		NDB_FREE(entity_key);
	if (entity_table)
		NDB_FREE(entity_table);
	if (feature_names)
	{
		for (i = 0; i < ndefs; ++i)
			if (feature_names[i])
				NDB_FREE(feature_names[i]);
		NDB_FREE(feature_names);
	}
	if (feature_types)
	{
		for (i = 0; i < ndefs; ++i)
			if (feature_types[i])
				NDB_FREE(feature_types[i]);
		NDB_FREE(feature_types);
	}
	if (feature_transforms)
	{
		for (i = 0; i < ndefs; ++i)
			if (feature_transforms[i])
				NDB_FREE(feature_transforms[i]);
		NDB_FREE(feature_transforms);
	}

	PG_RETURN_INT32(updated);
}

/*
 * Helper: get entity key column name for a store
 */
static char *
entity_key_for_store(int32 store_id)
{
	StringInfoData sql;
	int			ret;
	char	   *entity_key;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT entity_key FROM neurondb.feature_stores WHERE store_id = %d",
					 store_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("feature store %d not found",
						store_id)));
	}

	{
		text	   *entity_key_text = ndb_spi_get_text(spi_session, 0, 1, oldcontext);

		entity_key = entity_key_text ? text_to_cstring(entity_key_text) : NULL;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);
	return entity_key;
}

/*
 * Helper: get entity table name for a store
 */
static char *
entity_table_for_store(int32 store_id)
{
	StringInfoData sql;
	int			ret;
	char	   *entity_table;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT entity_table FROM neurondb.feature_stores WHERE store_id = %d",
					 store_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("feature store %d not found",
						store_id)));
	}

	{
		text	   *entity_table_text = ndb_spi_get_text(spi_session, 0, 1, oldcontext);

		entity_table = entity_table_text ? text_to_cstring(entity_table_text) : NULL;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);
	return entity_table;
}

/*
 * Helper: get all features for a store (names, types, optional transforms)
 */
static void
get_feature_definitions(int32 store_id,
						char ***out_feat_names,
						char ***out_feat_types,
						char ***out_feat_transforms,
						int *out_n_features)
{
	StringInfoData sql;
	int			ret,
				i,
				nfeat;
	char	  **fnames = NULL;
	char	  **ftypes = NULL;
	char	  **ftransforms = NULL;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT feature_name, feature_type, transformation "
					 "FROM neurondb.feature_definitions WHERE store_id = %d ORDER BY feature_id",
					 store_id);

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("could not retrieve feature definitions for store %d",
						store_id)));
	}

	nfeat = SPI_processed;
	NDB_ALLOC(fnames, char *, nfeat);
	NDB_ALLOC(ftypes, char *, nfeat);
	NDB_ALLOC(ftransforms, char *, nfeat);

	for (i = 0; i < nfeat; ++i)
	{
		text	   *fname_text = ndb_spi_get_text(spi_session, i, 1, oldcontext);
		text	   *ftype_text = ndb_spi_get_text(spi_session, i, 2, oldcontext);
		text	   *ftransform_text = ndb_spi_get_text(spi_session, i, 3, oldcontext);

		fnames[i] = fname_text ? text_to_cstring(fname_text) : NULL;
		ftypes[i] = ftype_text ? text_to_cstring(ftype_text) : NULL;
		ftransforms[i] = ftransform_text ? text_to_cstring(ftransform_text) : NULL;
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);
	*out_feat_names = fnames;
	*out_feat_types = ftypes;
	*out_feat_transforms = ftransforms;
	*out_n_features = nfeat;
}

/*
 * config.c
 *     Detailed NeuronDB configuration interface implementation
 *
 * Implements SQL and C API for viewing and modifying all NeuronDB
 * configuration variables. Each variable may reference a backend GUC,
 * supports SHOW/SET/RESET, and is accessible via C APIs for internal use.
 *
 * Settings cover index/search/memory/replication. All are surfaced as
 * system catalog views for inspection.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "lib/stringinfo.h"
#include "neurondb_constants.h"

#include <limits.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

/*
 * Internal catalog of NeuronDB configuration options.
 * Each: name, category, description, default value, and GUC type.
 */
typedef enum ConfigType
{
	NEURON_GUC_INT,
	NEURON_GUC_FLOAT,
	NEURON_GUC_BOOL,
	NEURON_GUC_STRING,
	NEURON_GUC_ENUM
}			ConfigType;

typedef struct NeuronDBConfigOpt
{
	const char *name;
	const char *category;
	const char *description;
	const char *default_value;
	ConfigType	type;
	const char *guc_var;
	const char **enum_values;
}			NeuronDBConfigOpt;

static const char *wal_compression_enums[] = {"on", "off", NULL};

static const NeuronDBConfigOpt neuron_config_catalog[] = {
	{"ef_construction",
		"Index",
		"HNSW construction parameter",
		"200",
		NEURON_GUC_INT,
		NDB_GUC_EF_CONSTRUCTION,
	NULL},
	{"ef_search",
		"Search",
		"HNSW search parameter",
		"100",
		NEURON_GUC_INT,
		NDB_GUC_EF_SEARCH,
	NULL},
	{"max_connections",
		"Performance",
		"Maximum concurrent connections",
		"1000",
		NEURON_GUC_INT,
		NDB_GUC_MAX_CONNECTIONS,
	NULL},
	{"buffer_size",
		"Memory",
		"ANN buffer cache size",
		"128MB",
		NEURON_GUC_STRING,
		NDB_GUC_BUFFER_SIZE,
	NULL},
	{"wal_compression",
		"Replication",
		"Enable WAL compression for vectors",
		"on",
		NEURON_GUC_ENUM,
		NDB_GUC_WAL_COMPRESSION,
	wal_compression_enums},
	{"index_parallelism",
		"Performance",
		"Degree of index build parallelism",
		"1",
		NEURON_GUC_INT,
		NDB_GUC_INDEX_PARALLELISM,
	NULL},
	{"vector_dim_limit",
		"Index",
		"Maximum allowed index vector dimension",
		"4096",
		NEURON_GUC_INT,
		NDB_GUC_VECTOR_DIM_LIMIT,
	NULL},
	{"hybrid_threshold",
		"Search",
		"Threshold for hybrid (ANN+keyword) search",
		"0.6",
		NEURON_GUC_FLOAT,
		NDB_GUC_HYBRID_THRESHOLD,
	NULL},
	{"use_gpu",
		"Performance",
		"Enable GPU for vector search",
		"off",
		NEURON_GUC_BOOL,
		NDB_GUC_USE_GPU,
	NULL},
	{NULL, NULL, NULL, NULL, 0, NULL, NULL}
};

static const NeuronDBConfigOpt *
get_config_opt(const char *name)
{
	const		NeuronDBConfigOpt *opt;

	for (opt = neuron_config_catalog; opt && opt->name; opt++)
	{
		if (pg_strcasecmp(opt->name, name) == 0)
			return opt;
	}

	return NULL;
}

static void
set_neurondb_guc(const NeuronDBConfigOpt * opt, const char *value)
{
	switch (opt->type)
	{
		case NEURON_GUC_INT:
			{
				char	   *endp = NULL;
				long		num;

				num = strtol(value, &endp, 10);
				if (!isdigit((unsigned char) value[0]) || endp == value
					|| num < 0 || num > INT_MAX)
					ereport(ERROR,
							(errmsg("invalid integer: \"%s\" for parameter "
									"\"%s\"",
									value,
									opt->name)));
				if (opt->guc_var)
					SetConfigOption(opt->guc_var,
									value,
									PGC_USERSET,
									PGC_S_SESSION);
			}
			break;
		case NEURON_GUC_FLOAT:
			{
				char	   *endp = NULL;
				double		dval;

				dval = strtod(value, &endp);
				if (endp == value || isnan(dval) || isinf(dval))
					ereport(ERROR,
							(errmsg("invalid float: \"%s\" for parameter "
									"\"%s\"",
									value,
									opt->name)));
				if (opt->guc_var)
					SetConfigOption(opt->guc_var,
									value,
									PGC_USERSET,
									PGC_S_SESSION);
			}
			break;
		case NEURON_GUC_BOOL:
			if (pg_strcasecmp(value, "on") == 0
				|| pg_strcasecmp(value, "true") == 0
				|| strcmp(value, "1") == 0)
			{
				SetConfigOption(
								opt->guc_var, "on", PGC_USERSET, PGC_S_SESSION);
			}
			else if (pg_strcasecmp(value, "off") == 0
					 || pg_strcasecmp(value, "false") == 0
					 || strcmp(value, "0") == 0)
			{
				SetConfigOption(opt->guc_var,
								"off",
								PGC_USERSET,
								PGC_S_SESSION);
			}
			else
			{
				ereport(ERROR,
						(errmsg("invalid boolean: \"%s\" for parameter "
								"\"%s\"",
								value,
								opt->name)));
			}
			break;
		case NEURON_GUC_ENUM:
			{
				const char **enum_val;
				bool		found = false;

				for (enum_val = opt->enum_values; enum_val && *enum_val;
					 enum_val++)
				{
					if (pg_strcasecmp(value, *enum_val) == 0)
					{
						found = true;
						break;
					}
				}
				if (!found)
					ereport(ERROR,
							(errmsg("invalid enum value for %s: \"%s\"",
									opt->name,
									value)));
				if (opt->guc_var)
					SetConfigOption(opt->guc_var,
									value,
									PGC_USERSET,
									PGC_S_SESSION);
			}
			break;
		case NEURON_GUC_STRING:
			if (opt->guc_var)
				SetConfigOption(opt->guc_var,
								value,
								PGC_USERSET,
								PGC_S_SESSION);
			break;
		default:
			ereport(ERROR,
					(errmsg("unsupported parameter type for %s",
							opt->name)));
	}
}

/*
 * get_neurondb_guc - Read GUC value as string representation
 */
static char *
get_neurondb_guc(const NeuronDBConfigOpt * opt)
{
	const char *val = NULL;

	if (opt->guc_var)
		val = GetConfigOption(opt->guc_var, false, false);

	if (!val && opt->default_value)
		return pstrdup(opt->default_value);

	return val ? pstrdup(val) : NULL;
}

PG_FUNCTION_INFO_V1(show_vector_config);

Datum
show_vector_config(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx = NULL;
	const		NeuronDBConfigOpt *catalog_row;
	Datum		values[3];
	bool		nulls[3];
	HeapTuple	tuple;
	StringInfoData valbuf;
	char *curr_value = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc	tupdesc;
		MemoryContext oldcontext;
		const		NeuronDBConfigOpt *opt;
		int			count = 0;

		funcctx = SRF_FIRSTCALL_INIT();

		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 1, "category", TEXTOID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 2, "setting", TEXTOID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 3, "description", TEXTOID, -1, 0);

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		/* Count entries in catalog */
		for (opt = neuron_config_catalog; opt && opt->name; opt++)
			count++;

		funcctx->max_calls = count;

		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
		funcctx->user_fctx = (void *) &neuron_config_catalog[0];
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	catalog_row = (const NeuronDBConfigOpt *) funcctx->user_fctx;

	if (!catalog_row || !catalog_row->name)
		SRF_RETURN_DONE(funcctx);

	initStringInfo(&valbuf);
	curr_value = get_neurondb_guc(catalog_row);
	appendStringInfo(&valbuf,
					 "%s=%s",
					 catalog_row->name,
					 curr_value ? curr_value : "(null)");

	values[0] = CStringGetTextDatum(catalog_row->category);
	values[1] = CStringGetTextDatum(valbuf.data);
	values[2] = CStringGetTextDatum(catalog_row->description);
	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

	funcctx->user_fctx = (void *) (catalog_row + 1);

	SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
}

PG_FUNCTION_INFO_V1(set_vector_config);

Datum
set_vector_config(PG_FUNCTION_ARGS)
{
	text	   *config_name;
	text	   *config_value;
	char	   *name_str = NULL;
	char	   *value_str = NULL;
	const		NeuronDBConfigOpt *opt;

	/* Validate argument count */
	if (PG_NARGS() < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: set_vector_config requires at least 2 arguments")));

	/* Validate arguments are not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: set_vector_config: config_name cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: set_vector_config: config_value cannot be NULL")));

	config_name = PG_GETARG_TEXT_PP(0);
	config_value = PG_GETARG_TEXT_PP(1);

	name_str = text_to_cstring(config_name);
	value_str = text_to_cstring(config_value);

	opt = get_config_opt(name_str);
	if (opt == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: unknown configuration "
						"parameter '%s'",
						name_str)));

	set_neurondb_guc(opt, value_str);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(get_vector_config);

Datum
get_vector_config(PG_FUNCTION_ARGS)
{
	text	   *config_name;
	char *name_str = NULL;
	const		NeuronDBConfigOpt *opt;
	char *curr_value = NULL;

	/* Validate argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: get_vector_config requires at least 1 argument")));

	/* Validate argument is not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: get_vector_config: config_name cannot be NULL")));

	config_name = PG_GETARG_TEXT_PP(0);

	name_str = text_to_cstring(config_name);
	opt = get_config_opt(name_str);

	if (opt == NULL)
	{
		ereport(ERROR,
				(errmsg("neurondb: unknown configuration parameter "
						"'%s'",
						name_str)));
		PG_RETURN_NULL();
	}

	curr_value = get_neurondb_guc(opt);

	if (curr_value == NULL)
		PG_RETURN_NULL();

	PG_RETURN_TEXT_P(cstring_to_text(curr_value));
}

PG_FUNCTION_INFO_V1(reset_vector_config);

Datum
reset_vector_config(PG_FUNCTION_ARGS)
{
	const		NeuronDBConfigOpt *opt;

	for (opt = neuron_config_catalog; opt && opt->name; opt++)
	{
		if (opt->guc_var && opt->default_value)
		{
			/* Try to reset the GUC, skip if it doesn't exist */
			PG_TRY();
			{
				SetConfigOption(opt->guc_var,
								opt->default_value,
								PGC_USERSET,
								PGC_S_SESSION);
			}
			PG_CATCH();
			{
				/* Skip unrecognized GUC parameters */
				FlushErrorState();
			}
			PG_END_TRY();
		}
	}

	PG_RETURN_BOOL(true);
}

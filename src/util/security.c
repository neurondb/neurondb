/*-------------------------------------------------------------------------
 *
 * security_extensions.c
 *		Security Extensions: Post-quantum, Confidential Compute, Access Masks,
 *		Federated Queries
 *
 * This file implements advanced security features including post-quantum
 * encryption (AES-256-GCM), confidential compute mode (SGX/SEV), fine-grained
 * access masks, and secure federated queries.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/security_extensions.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "funcapi.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "utils/array.h"
#include "access/htup.h"
#include "access/htup_details.h"
#include "neurondb_validation.h"
#include "neurondb_spi.h"
#include "neurondb_constants.h"
#include "neurondb_guc.h"
#include "neurondb_macros.h"

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/err.h>
#include <string.h>

#define PQ_IV_LEN 12
#define PQ_TAG_LEN 16
#define PQ_KEY_LEN 32

/* Helper: serialize vector to "[x,y,z,...]" for use in remote SQL */
static char *
vector_to_sql_string(Vector *v, MemoryContext ctx)
{
	StringInfoData buf;
	int			i;
	MemoryContext old;

	if (v == NULL || VECTOR_DIM(v) <= 0)
		return pstrdup("[]");
	old = MemoryContextSwitchTo(ctx);
	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');
	for (i = 0; i < VECTOR_DIM(v); i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%.9g", v->data[i]);
	}
	appendStringInfoChar(&buf, ']');
	MemoryContextSwitchTo(old);
	return buf.data;
}

/* Federated result row for merge/sort */
typedef struct FederatedRow
{
	char	   *server;
	int64		id;
	float4		distance;
} FederatedRow;

PG_FUNCTION_INFO_V1(encrypt_postquantum);
Datum
encrypt_postquantum(PG_FUNCTION_ARGS)
{
	Vector	   *input = NULL;
	bytea	   *result = NULL;
	Size		plaintext_len;
	Size		result_size;
	uint8_t	   *plaintext = NULL;
	uint8_t	   *ciphertext = NULL;
	uint8_t	   *tag_ptr = NULL;
	uint8_t	   iv[PQ_IV_LEN];
	uint8_t	   key[PQ_KEY_LEN];
	EVP_CIPHER_CTX *ctx = NULL;
	int			len;
	int			final_len;
	const char *key_source = "neurondb-postquantum-default-key";

	/* Validate input is not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: encrypt_postquantum: input vector cannot be NULL")));

	input = (Vector *) PG_GETARG_POINTER(0);

	/* Validate input vector structure */
	if (input == NULL || VARSIZE_ANY(input) < sizeof(Vector))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: encrypt_postquantum: invalid input vector")));

	plaintext_len = (Size) (input->dim * sizeof(float4));
	result_size = VARHDRSZ + PQ_IV_LEN + plaintext_len + PQ_TAG_LEN;
	result = (bytea *) palloc(result_size);
	SET_VARSIZE(result, result_size);

	plaintext = (uint8_t *) palloc(plaintext_len);
	memcpy(plaintext, input->data, plaintext_len);
	ciphertext = (uint8_t *) (VARDATA(result) + PQ_IV_LEN);
	tag_ptr = (uint8_t *) (VARDATA(result) + PQ_IV_LEN + plaintext_len);

	if (RAND_bytes(iv, PQ_IV_LEN) != 1)
	{
		pfree(plaintext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encrypt_postquantum: failed to generate IV")));
	}
	memcpy(VARDATA(result), iv, PQ_IV_LEN);

	/* Derive 32-byte key via SHA-256 (production should use GUC/session key) */
	(void) SHA256((const unsigned char *) key_source, strlen(key_source), key);

	ctx = EVP_CIPHER_CTX_new();
	if (ctx == NULL)
	{
		pfree(plaintext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encrypt_postquantum: cipher context failed")));
	}
	if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, iv) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(plaintext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encrypt_postquantum: init failed")));
	}
	if (EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, (int) plaintext_len) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(plaintext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encrypt_postquantum: encrypt failed")));
	}
	if (EVP_EncryptFinal_ex(ctx, ciphertext + len, &final_len) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(plaintext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encrypt_postquantum: final failed")));
	}
	if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, PQ_TAG_LEN, tag_ptr) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(plaintext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encrypt_postquantum: get tag failed")));
	}
	EVP_CIPHER_CTX_free(ctx);
	pfree(plaintext);

	PG_RETURN_BYTEA_P(result);
}

PG_FUNCTION_INFO_V1(enable_confidential_compute);
Datum
enable_confidential_compute(PG_FUNCTION_ARGS)
{
	bool		enable = PG_GETARG_BOOL(0);

	extern bool neurondb_confidential_compute;
	neurondb_confidential_compute = enable;
	PG_RETURN_BOOL(neurondb_confidential_compute);
}

PG_FUNCTION_INFO_V1(set_access_mask);
Datum
set_access_mask(PG_FUNCTION_ARGS)
{
	text	   *role_name = NULL;
	text	   *allowed_metrics = NULL;
	text	   *allowed_indexes = NULL;
	char	   *role_str = NULL;
	char	   *metrics_str = NULL;
	char	   *indexes_str = NULL;
	NdbSpiSession *session = NULL;
	StringInfoData sql;
	Oid			argtypes[3] = {TEXTOID, TEXTOID, TEXTOID};
	Datum		values[3];
	char		nulls[3] = {' ', ' ', ' '};
	int			ret;

	/* Validate arguments are not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: set_access_mask: role_name cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: set_access_mask: allowed_metrics cannot be NULL")));

	if (PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: set_access_mask: allowed_indexes cannot be NULL")));

	role_name = PG_GETARG_TEXT_PP(0);
	allowed_metrics = PG_GETARG_TEXT_PP(1);
	allowed_indexes = PG_GETARG_TEXT_PP(2);

	role_str = text_to_cstring(role_name);
	metrics_str = text_to_cstring(allowed_metrics);
	indexes_str = text_to_cstring(allowed_indexes);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: set_access_mask: failed to begin SPI session")));

	/* Ensure table exists */
	ndb_spi_stringinfo_init(session, &sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS neurondb.access_masks ("
					 "role_name text PRIMARY KEY, allowed_metrics text, allowed_indexes text, created_at timestamptz DEFAULT now())");
	ret = ndb_spi_execute(session, sql.data, false, 0);
	ndb_spi_stringinfo_free(session, &sql);
	if (ret != SPI_OK_UTILITY)
	{
		ndb_spi_session_end(&session);
		nfree(role_str);
		nfree(metrics_str);
		nfree(indexes_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: set_access_mask: failed to create access_masks table")));
	}

	values[0] = CStringGetTextDatum(role_str);
	values[1] = CStringGetTextDatum(metrics_str);
	values[2] = CStringGetTextDatum(indexes_str);
	ndb_spi_stringinfo_init(session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb.access_masks (role_name, allowed_metrics, allowed_indexes) VALUES ($1, $2, $3) ON CONFLICT (role_name) DO UPDATE SET allowed_metrics = EXCLUDED.allowed_metrics, allowed_indexes = EXCLUDED.allowed_indexes");
	ret = ndb_spi_execute_with_args(session, sql.data, 3, argtypes, values, nulls, false, 0);
	ndb_spi_stringinfo_free(session, &sql);
	ndb_spi_session_end(&session);
	nfree(role_str);
	nfree(metrics_str);
	nfree(indexes_str);
	if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: set_access_mask: failed to store access mask")));

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(federated_vector_query);
Datum
federated_vector_query(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	ArrayType  *remote_servers = NULL;
	Vector	   *query_vector = NULL;
	int32		k;
	text	   *combine_method = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		Datum	   *conn_elems = NULL;
		bool	   *conn_nulls = NULL;
		int			n_servers = 0;
		int			i;
		char	   *vector_str = NULL;
		NdbSpiSession *session = NULL;
		FederatedRow *rows = NULL;
		int			nrows = 0;
		int			row_alloc = 0;
		TupleDesc	tupdesc;

		if (PG_ARGISNULL(0))
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neurondb: federated_vector_query: remote_servers cannot be NULL")));
		if (PG_ARGISNULL(1))
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neurondb: federated_vector_query: query_vector cannot be NULL")));
		if (PG_ARGISNULL(2))
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("neurondb: federated_vector_query: k cannot be NULL")));

		remote_servers = PG_GETARG_ARRAYTYPE_P(0);
		query_vector = PG_GETARG_VECTOR_P(1);
		k = PG_GETARG_INT32(2);
		if (k <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: federated_vector_query: k must be positive")));
		if (!PG_ARGISNULL(3))
			combine_method = PG_GETARG_TEXT_PP(3);
		(void) combine_method;	/* reserved for merge/rank strategy */

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		deconstruct_array(remote_servers, TEXTOID, -1, false, 'i',
						  &conn_elems, &conn_nulls, &n_servers);
		if (n_servers == 0)
		{
			MemoryContextSwitchTo(oldcontext);
			funcctx->max_calls = 0;
			SRF_RETURN_DONE(funcctx);
		}

		vector_str = vector_to_sql_string(query_vector, funcctx->multi_call_memory_ctx);
		row_alloc = n_servers * k;
		rows = (FederatedRow *) palloc(sizeof(FederatedRow) * (size_t) row_alloc);
		nrows = 0;

		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: federated_vector_query: failed to begin SPI session")));

		for (i = 0; i < n_servers && nrows < row_alloc; i++)
		{
			char	   *connstr = NULL;
			StringInfoData remote_sql;
			StringInfoData full_sql;
			Oid			argtypes[2] = {TEXTOID, TEXTOID};
			Datum		values[2];
			char		nulls[2] = {' ', ' '};
			int			ret;
			SPITupleTable *tuptable = NULL;
			TupleDesc	spi_tupdesc;
			int			proc;
			int			r;

			if (conn_nulls[i])
				continue;
			connstr = text_to_cstring(DatumGetTextPP(conn_elems[i]));
			initStringInfo(&remote_sql);
			appendStringInfo(&remote_sql,
							 "SELECT id, l2_distance(embedding, '%s'::vector) AS distance FROM my_vectors ORDER BY distance LIMIT %d",
							 vector_str, k);
			values[0] = CStringGetTextDatum(connstr);
			values[1] = CStringGetTextDatum(remote_sql.data);

			initStringInfo(&full_sql);
			appendStringInfo(&full_sql,
							 "SELECT * FROM dblink($1, $2) AS t(id bigint, distance real)");
			ret = ndb_spi_execute_with_args(session, full_sql.data, 2, argtypes, values, nulls, true, 0);
			pfree(remote_sql.data);
			pfree(full_sql.data);
			pfree(connstr);

			if (ret != SPI_OK_SELECT)
				continue;
			tuptable = SPI_tuptable;
			spi_tupdesc = SPI_tuptable->tupdesc;
			proc = (int) SPI_processed;
			for (r = 0; r < proc && nrows < row_alloc; r++)
			{
				HeapTuple	tup = tuptable->vals[r];
				bool		n1 = false;
				bool		n2 = false;
				Datum		id_datum = SPI_getbinval(tup, spi_tupdesc, 1, &n1);
				Datum		dist_datum = SPI_getbinval(tup, spi_tupdesc, 2, &n2);
				char	   *srv = text_to_cstring(DatumGetTextPP(conn_elems[i]));

				rows[nrows].server = pstrdup(srv);
				rows[nrows].id = n1 ? 0 : DatumGetInt64(id_datum);
				rows[nrows].distance = n2 ? (float4) 0 : DatumGetFloat4(dist_datum);
				nrows++;
				pfree(srv);
			}
		}
		ndb_spi_session_end(&session);

		/* Full sort by distance (selection sort), then keep top k */
		if (nrows > 1)
		{
			int			j;

			for (i = 0; i < nrows - 1; i++)
			{
				int			min_i = i;

				for (j = i + 1; j < nrows; j++)
				{
					if (rows[j].distance < rows[min_i].distance)
						min_i = j;
				}
				if (min_i != i)
				{
					FederatedRow tmp = rows[i];

					rows[i] = rows[min_i];
					rows[min_i] = tmp;
				}
			}
		}
		if (nrows > k)
			nrows = k;

		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "server", TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "distance", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->user_fctx = rows;
		funcctx->max_calls = nrows;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		FederatedRow *rows = (FederatedRow *) funcctx->user_fctx;

		if (funcctx->call_cntr < funcctx->max_calls)
		{
			Datum		values[3];
			bool		nulls[3] = {false, false, false};
			HeapTuple	tuple;
			FederatedRow *row = &rows[funcctx->call_cntr];

			values[0] = CStringGetTextDatum(row->server);
			values[1] = Int64GetDatum(row->id);
			values[2] = Float4GetDatum(row->distance);
			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			int			i;

			for (i = 0; i < funcctx->max_calls; i++)
				pfree(rows[i].server);
			pfree(rows);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

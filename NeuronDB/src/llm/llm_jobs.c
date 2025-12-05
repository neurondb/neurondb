/*
 * llm_jobs.c
 *
 * Complete, robust, and detailed job helper routines for NeurondB LLM background processing
 * and asynchronous tasks.
 *
 * This file provides detailed C implementations of all helper functions for registering,
 * scheduling, and managing long-running or background jobs related to large language model (LLM)
 * operations within NeurondB. All functions perform careful error checking, full type safety,
 * precise resource cleanup, and handle NULL or edge cases, fully matching the NeurondB SQL schema.
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "access/xact.h"
#include "lib/stringinfo.h"
#include "storage/ipc.h"
#include "utils/timestamp.h"
#include "neurondb_llm.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

/*
 * Submits a new LLM job to the neurondb.llm_jobs table.
 * Returns the id of the new job.
 * Job type is a string (non-empty), payload is a valid JSON string (non-NULL).
 */
int
ndb_llm_job_enqueue(const char *job_type, const char *payload)
{
	int			job_id = 0;
	StringInfoData query;
	Oid			argtypes[2] = {TEXTOID, JSONBOID};
	Datum		values[2];

	NDB_DECLARE(NdbSpiSession *, session);

	/* Validate inputs */
	if (!job_type || strlen(job_type) == 0)
		ereport(ERROR, (errmsg("Job type must not be NULL or empty")));
	if (!payload || strlen(payload) == 0)
		ereport(ERROR,
				(errmsg("Payload JSON must not be NULL or empty")));
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job enqueue")));

	initStringInfo(&query);

	/*
	 * Use parameterization for values. The payload is casted to jsonb since
	 * SQL function takes text input.
	 */
	appendStringInfo(&query,
					 "INSERT INTO neurondb.llm_jobs (operation, "
					 "input_text, status, model_name, created_at) "
					 "VALUES ($1, $2, 'queued', 'default', now()) "
					 "RETURNING job_id");

	values[0] = CStringGetTextDatum(job_type);

	/* Validate that payload is valid JSON */
	{
		/* Try to parse with jsonb_in -- throws error if invalid */
		Oid			typinput;
		Oid			typioparam;
		Datum		jsonb_val;

		getTypeInputInfo(JSONBOID, &typinput, &typioparam);
		jsonb_val = OidFunctionCall3(typinput,
									 CStringGetDatum(payload),
									 ObjectIdGetDatum(InvalidOid),
									 Int32GetDatum(-1));
		values[1] = jsonb_val;
	}

	if (ndb_spi_execute_with_args(session,
								  query.data, 2, argtypes, values, NULL, false, 1)
		== SPI_OK_INSERT_RETURNING
		&& SPI_processed == 1)
	{
		bool		isnull = true;

		job_id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0],
											 SPI_tuptable->tupdesc,
											 1,
											 &isnull));
		if (isnull)
		{
			NDB_FREE(query.data);
			ndb_spi_session_end(&session);
			ereport(ERROR,
					(errmsg("Job ID returned as NULL after "
							"insert")));
		}
	}
	else
	{
		NDB_FREE(query.data);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errmsg("Failed to insert llm job: %s", query.data)));
	}

	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
	return job_id;
}

/*
 * Fetches and locks a single pending 'queued' job for processing by a worker.
 * Sets status to 'processing' and started_at=now().
 * Output: job_id, job_type, payload (all freshly palloc'ed, must be freed by the caller via pfree).
 * Returns true if a job was acquired, false if no pending job was found.
 */
bool
ndb_llm_job_acquire(int *job_id, char **job_type, char **payload)
{
	bool		found = false;
	StringInfoData query;

	NDB_DECLARE(NdbSpiSession *, session);

	/* Must be called within an active transaction */
	if (!IsTransactionState())
		ereport(ERROR,
				(errmsg("ndb_llm_job_acquire must be called within a "
						"transaction")));
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job acquire")));

	/*
	 * Claim a pending job atomically using UPDATE ... FROM+subquery FOR
	 * UPDATE SKIP LOCKED. With RETURNING, we obtain id, job_type, payload
	 * (serialized as text). Only the oldest queued job is claimed (ORDER BY
	 * created_at ASC LIMIT 1).
	 */
	initStringInfo(&query);
	appendStringInfoString(&query,
						   "UPDATE neurondb.llm_jobs "
						   "SET status = 'processing', started_at = now() "
						   "WHERE job_id = ("
						   "SELECT job_id FROM neurondb.llm_jobs "
						   "WHERE status = 'queued' "
						   "ORDER BY created_at ASC "
						   "FOR UPDATE SKIP LOCKED "
						   "LIMIT 1"
						   ") "
						   "RETURNING job_id, operation, input_text");

	if (ndb_spi_execute(session, query.data, false, 1) == SPI_OK_UPDATE_RETURNING
		&& SPI_processed > 0)
	{
		{
			bool		isnull0 = true;
			bool		isnull1 = true;
			bool		isnull2 = true;
			HeapTuple	tuple = SPI_tuptable->vals[0];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			if (job_id)
				*job_id = DatumGetInt64(
										SPI_getbinval(tuple, tupdesc, 1, &isnull0));
			if (job_type)
				*job_type = isnull1
					? NULL
					: pstrdup(TextDatumGetCString(SPI_getbinval(
																tuple, tupdesc, 2, &isnull1)));
			if (payload)
				*payload = isnull2
					? NULL
					: pstrdup(TextDatumGetCString(SPI_getbinval(
																tuple, tupdesc, 3, &isnull2)));
			found = !isnull0;
		}
	}

	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
	return found;
}

/*
 * Updates the status, result, and error for a completed or failed LLM job.
 * 'status' is required ("done", "failed", or any non-NULL status string).
 * 'result' can be NULL/empty and will be stored as NULL if so.
 * 'error' can be NULL/empty and will be stored as NULL if so.
 * Returns true on success, false on no such job.
 */
bool
ndb_llm_job_update(int job_id,
				   const char *status,
				   const char *result,
				   const char *error)
{
	StringInfoData query;
	Oid			argtypes[4] = {INT8OID, TEXTOID, TEXTOID, TEXTOID};
	Datum		values[4];
	bool		ok = false;

	NDB_DECLARE(NdbSpiSession *, session);

	if (job_id <= 0)
		ereport(ERROR,
				(errmsg("Invalid job_id for update: %d", job_id)));
	if (!status || strlen(status) == 0)
		ereport(ERROR,
				(errmsg("Status for job update cannot be NULL or "
						"empty")));
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job update")));

	initStringInfo(&query);
	appendStringInfoString(&query,
						   "UPDATE neurondb.llm_jobs "
						   "SET status = $2, result = $3, error = $4, finished_at = now() "
						   "WHERE id = $1");

	values[0] = Int64GetDatum(job_id);
	values[1] = CStringGetTextDatum(status);

	/*
	 * Treat empty string or NULL as SQL NULL for result or error.
	 */
	values[2] = (result && strlen(result) > 0) ? CStringGetTextDatum(result)
		: (Datum) 0;
	values[3] = (error && strlen(error) > 0) ? CStringGetTextDatum(error)
		: (Datum) 0;
	{
		/* NULLs for result and error */
		char		nulls[4] = {' ', ' ', ' ', ' '};

		if (!result || strlen(result) == 0)
			nulls[2] = 'n';
		if (!error || strlen(error) == 0)
			nulls[3] = 'n';

		if (ndb_spi_execute_with_args(session,
									  query.data, 4, argtypes, values, nulls, false, 0)
			== SPI_OK_UPDATE)
		{
			if (SPI_processed > 0)
				ok = true;
		}
	}
	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
	return ok;
}

/*
 * Delete finished or failed jobs that are older than max_age_days (default 7).
 * Returns number of jobs deleted.
 */
int
ndb_llm_job_prune(int max_age_days)
{
	int			deleted = 0;
	StringInfoData query;
	int			days;

	NDB_DECLARE(NdbSpiSession *, session);

	days = (max_age_days > 0) ? max_age_days : 7;
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job prune")));

	initStringInfo(&query);
	appendStringInfo(&query,
					 "DELETE FROM neurondb.llm_jobs "
					 "WHERE (status = 'done' OR status = 'failed') "
					 "AND completed_at IS NOT NULL "
					 "AND completed_at < now() - interval '%d days'",
					 days);

	if (ndb_spi_execute(session, query.data, false, 0) == SPI_OK_DELETE)
	{
		deleted = SPI_processed;
	}

	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
	return deleted;
}

/*
 * Returns count of jobs in neurondb_llm_jobs with the given status (SQL: WHERE status = $1).
 * Returns 0 if no jobs match.
 */
int
ndb_llm_job_count_status(const char *status)
{
	int			count = 0;
	StringInfoData query;
	Oid			argtypes[1] = {TEXTOID};
	Datum		values[1];

	NDB_DECLARE(NdbSpiSession *, session);

	if (!status || strlen(status) == 0)
		ereport(ERROR,
				(errmsg("Job status cannot be NULL or empty for "
						"count")));
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job count")));

	initStringInfo(&query);
	appendStringInfoString(&query,
						   "SELECT count(*) FROM neurondb.llm_jobs WHERE status "
						   "= $1");
	values[0] = CStringGetTextDatum(status);

	if (ndb_spi_execute_with_args(session,
								  query.data, 1, argtypes, values, NULL, false, 1)
		== SPI_OK_SELECT
		&& SPI_processed == 1)
	{
		bool		isnull = true;

		count = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0],
											SPI_tuptable->tupdesc,
											1,
											&isnull));
		if (isnull)
			count = 0;
	}
	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
	return count;
}

/*
 * Retries failed jobs that have been retried fewer than max_retries times.
 * Resets status to 'queued', resets error/result/started_at/finished_at, increments retry_count.
 * Returns the number of jobs reset for retry.
 */
int
ndb_llm_job_retry_failed(int max_retries)
{
	int			retried = 0;
	StringInfoData query;
	Oid			argtypes[1] = {INT4OID};
	Datum		values[1];

	NDB_DECLARE(NdbSpiSession *, session);

	if (max_retries < 1)
		ereport(ERROR, (errmsg("max_retries must be >= 1")));
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job "
						"retry_failed")));

	initStringInfo(&query);
	appendStringInfoString(&query,
						   "UPDATE neurondb.llm_jobs "
						   "SET status = 'queued', retry_count = COALESCE(retry_count, 0) "
						   "+ 1, "
						   "error = NULL, result = NULL, started_at = NULL, finished_at = "
						   "NULL "
						   "WHERE status = 'failed' "
						   "AND (retry_count IS NULL OR retry_count < $1)");

	values[0] = Int32GetDatum(max_retries);

	if (ndb_spi_execute_with_args(session,
								  query.data, 1, argtypes, values, NULL, false, 0)
		== SPI_OK_UPDATE)
		retried = SPI_processed;

	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
	return retried;
}

/*
 * Delete/truncate all jobs in the table (DANGEROUS: TESTING/ADMIN ONLY!).
 * The function uses TRUNCATE, which is non-transactional, so the effect is immediate.
 * Any error causes ereport(ERROR).
 */
void
ndb_llm_job_clear(void)
{
	StringInfoData query;

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errmsg("Failed to begin SPI session for llm job clear")));

	initStringInfo(&query);
	appendStringInfoString(&query,
						   "TRUNCATE TABLE neurondb.llm_jobs RESTART IDENTITY "
						   "CASCADE");

	if (ndb_spi_execute(session, query.data, false, 0) != SPI_OK_UTILITY)
	{
		NDB_FREE(query.data);
		ndb_spi_session_end(&session);
		ereport(ERROR, (errmsg("Failed to truncate llm jobs table")));
	}

	NDB_FREE(query.data);
	ndb_spi_session_end(&session);
}

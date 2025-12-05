/*-------------------------------------------------------------------------
 *
 * neurondb_spi.h
 *	  Centralized SPI session management for NeurondB
 *
 * Provides a unified interface for all SPI operations:
 * - Connection state tracking (explicit nested SPI support)
 * - Memory context management
 * - Error handling
 * - StringInfoData allocation within the appropriate context
 *
 * IMPORTANT: Error handling pattern
 * ----------------------------------
 * When using this API, you MUST call ndb_spi_session_end() in your
 * PG_CATCH block before rethrowing, if you created a session:
 *
 *   session = ndb_spi_session_begin(ctx, false);
 *   PG_TRY()
 *   {
 *       ... SPI work ...
 *   }
 *   PG_CATCH();
 *   {
 *       ndb_spi_session_end(&session);  // REQUIRED
 *       PG_RE_THROW();
 *   }
 *   PG_END_TRY();
 *   ndb_spi_session_end(&session);
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  include/neurondb_spi.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SPI_H
#define NEURONDB_SPI_H

#include "postgres.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "neurondb_constants.h"

/*
 * NdbSpiSession
 *
 * Opaque handle for SPI session management.
 * Tracks SPI connection and memory context state.
 */
typedef struct NdbSpiSession NdbSpiSession;

/*
 * Begin a new SPI session.
 *
 * If assume_spi_connected is false, always calls SPI_connect().
 * If true, assumes SPI is already connected and does not call SPI_connect().
 *
 * Returns a valid session or raises ERROR.
 */
extern NdbSpiSession *ndb_spi_session_begin(MemoryContext parent_context,
											bool assume_spi_connected);

/*
 * End an SPI session. Only finishes SPI if this session opened it.
 * Session is freed and pointer is set to NULL (safe for NULL arg).
 */
extern void ndb_spi_session_end(NdbSpiSession **session);

/*
 * True if this session controls the SPI connection (i.e., we connected it).
 * If false, the caller is responsible for SPI lifecycle.
 */
extern bool ndb_spi_session_controls_connection(NdbSpiSession *session);

/*
 * Return the memory context for the session (SPI context if active, else parent).
 */
extern MemoryContext ndb_spi_session_get_context(NdbSpiSession *session);

/*
 * Execute a SQL statement via this SPI session.
 * Returns SPI result code, or raises ERROR on failure.
 */
extern int	ndb_spi_execute(NdbSpiSession *session,
							const char *query,
							bool read_only,
							long tcount);

/*
 * Execute parameterized query in this session.
 * Arguments as for SPI_execute_with_args.
 */
extern int	ndb_spi_execute_with_args(NdbSpiSession *session,
									  const char *src,
									  int nargs,
									  Oid * argtypes,
									  Datum * values,
									  const char *nulls,
									  bool read_only,
									  long tcount);

/*
 * Allocate and initialize StringInfoData in SPI session context.
 * Use in place of initStringInfo() for buffers needed in SPI context.
 */
extern void ndb_spi_stringinfo_init(NdbSpiSession *session, StringInfoData * str);

/*
 * Free StringInfoData contents. Always safe (buffer may be SPI- or parent-context).
 */
extern void ndb_spi_stringinfo_free(NdbSpiSession *session, StringInfoData * str);

/*
 * Release and re-init buffer in the appropriate SPI or parent context.
 */
extern void ndb_spi_stringinfo_reset(NdbSpiSession *session, StringInfoData * str);

/*
 * Copy an int32 value from an SPI result tuple at [row,col] to out_value.
 * Returns true on success, false on error.
 */
extern bool ndb_spi_get_int32(NdbSpiSession *session,
							  int row_idx,
							  int col_idx,
							  int32 * out_value);

/*
 * Copy text datum from [row,col] to dest_context.
 * Returns pointer in dest_context, NULL if not available.
 */
extern text * ndb_spi_get_text(NdbSpiSession *session,
							   int row_idx,
							   int col_idx,
							   MemoryContext dest_context);

/*
 * Copy JSONB datum from [row,col] to dest_context.
 * Returns pointer in dest_context, NULL if not available.
 */
extern Jsonb * ndb_spi_get_jsonb(NdbSpiSession *session,
								 int row_idx,
								 int col_idx,
								 MemoryContext dest_context);

/*
 * Copy bytea datum from [row,col] to dest_context.
 * Returns pointer in dest_context, NULL if not available.
 */
extern bytea * ndb_spi_get_bytea(NdbSpiSession *session,
								 int row_idx,
								 int col_idx,
								 MemoryContext dest_context);

/*
 * Macros for begin/end SPI session with error checks.
 *
 * NDB_SPI_SESSION_BEGIN(session, ctx);
 *	 ... use session ...
 * NDB_SPI_SESSION_END(session);
 */

#define NDB_SPI_SESSION_BEGIN(session_var, parent_ctx) \
	do { \
		(session_var) = ndb_spi_session_begin(parent_ctx, false); \
		if ((session_var) == NULL) \
			ereport(ERROR, \
				(errcode(ERRCODE_INTERNAL_ERROR), \
				 errmsg(NDB_ERR_MSG("failed to begin SPI session")))); \
	} while (0)

#define NDB_SPI_SESSION_END(session_var) \
	do { \
		if ((session_var) != NULL) \
		{ \
			NdbSpiSession **_tmp_ptr = &(session_var); \
			ndb_spi_session_end(_tmp_ptr); \
		} \
	} while (0)

#endif							/* NEURONDB_SPI_H */

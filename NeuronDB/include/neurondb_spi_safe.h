/*-------------------------------------------------------------------------
 *
 * neurondb_spi_safe.h
 *    Safe SPI execution wrappers for NeuronDB crash prevention
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_spi_safe.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SPI_SAFE_H
#define NEURONDB_SPI_SAFE_H

#include "postgres.h"
#include "executor/spi.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "access/tupdesc.h"

/* Safe SPI execution */
extern int ndb_spi_execute_safe(const char *query, bool read_only, long tcount);
extern bool ndb_spi_execute_and_validate(const char *query,
										  bool read_only,
										  long tcount,
										  int expected_ret,
										  long min_rows);
extern bool ndb_spi_exec_select_one_row_safe(const char *query,
											   bool read_only,
											   MemoryContext dest_context,
											   TupleDesc *out_tupdesc,
											   Datum **out_datum,
											   bool **out_isnull,
											   int *out_natts);

/* Safe result extraction */
extern bool ndb_spi_get_result_safe(int row_idx,
									 int col_idx,
									 Oid *out_type,
									 Datum *out_datum,
									 bool *out_isnull);
extern Jsonb *ndb_spi_get_jsonb_safe(int row_idx, int col_idx, MemoryContext dest_context);
extern text *ndb_spi_get_text_safe(int row_idx, int col_idx, MemoryContext dest_context);

/* SPI cleanup */
extern void ndb_spi_finish_safe(MemoryContext oldcontext);
extern void ndb_spi_cleanup_safe(MemoryContext oldcontext, MemoryContext callcontext, bool finish_spi);

/* Result iteration */
extern int ndb_spi_iterate_safe(bool (*callback)(int row_idx, HeapTuple tuple, TupleDesc tupdesc, void *userdata),
								 void *userdata);

#endif	/* NEURONDB_SPI_SAFE_H */


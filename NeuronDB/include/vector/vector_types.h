/*-------------------------------------------------------------------------
 *
 * vector_types.h
 *	  Vector type definitions and core type system
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTOR_TYPES_H
#define VECTOR_TYPES_H

#include "postgres.h"
#include "fmgr.h"

/* Vector type I/O functions */
extern Datum vectorp_in(PG_FUNCTION_ARGS);
extern Datum vectorp_out(PG_FUNCTION_ARGS);
extern Datum vectorp_recv(PG_FUNCTION_ARGS);
extern Datum vectorp_send(PG_FUNCTION_ARGS);

/* Vector sparse map type */
extern Datum vecmap_in(PG_FUNCTION_ARGS);
extern Datum vecmap_out(PG_FUNCTION_ARGS);
extern Datum vecmap_recv(PG_FUNCTION_ARGS);
extern Datum vecmap_send(PG_FUNCTION_ARGS);

/* Retrievable text type */
extern Datum rtext_in(PG_FUNCTION_ARGS);
extern Datum rtext_out(PG_FUNCTION_ARGS);
extern Datum rtext_recv(PG_FUNCTION_ARGS);
extern Datum rtext_send(PG_FUNCTION_ARGS);

/* Vector graph type */
extern Datum vgraph_in(PG_FUNCTION_ARGS);
extern Datum vgraph_out(PG_FUNCTION_ARGS);
extern Datum vgraph_recv(PG_FUNCTION_ARGS);
extern Datum vgraph_send(PG_FUNCTION_ARGS);

#endif							/* VECTOR_TYPES_H */

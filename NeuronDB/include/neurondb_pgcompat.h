/*-------------------------------------------------------------------------
 *
 * neurondb_pgcompat.h
 *	  PostgreSQL version compatibility macros
 *
 * Handles API differences between PostgreSQL 16, 17, and 18
 *
 * IDENTIFICATION
 *	  include/neurondb_pgcompat.h
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_PGCOMPAT_H
#define NEURONDB_PGCOMPAT_H

#include "postgres.h"

/*
 * PostgreSQL version detection
 * PG_VERSION_NUM format: MMNNPP (e.g., 160000 for 16.0, 170000 for 17.0)
 */
#if !defined(PG_VERSION_NUM)
#error "PG_VERSION_NUM not defined - PostgreSQL version unknown"
#endif

#define PG_MAJOR_VERSION ((PG_VERSION_NUM / 10000) % 100)

/* Version compatibility macros */
#if PG_VERSION_NUM >= 180000
/* PostgreSQL 18 */
#define NDB_PG_18_PLUS 1
#define NDB_PG_17_PLUS 1
#define NDB_PG_16_PLUS 1
#elif PG_VERSION_NUM >= 170000
/* PostgreSQL 17 */
#define NDB_PG_17_PLUS 1
#define NDB_PG_16_PLUS 1
#undef NDB_PG_18_PLUS
#elif PG_VERSION_NUM >= 160000
/* PostgreSQL 16 */
#define NDB_PG_16_PLUS 1
#undef NDB_PG_17_PLUS
#undef NDB_PG_18_PLUS
#else
#error "PostgreSQL 16+ required. Found version < 16"
#endif

/*
 * Include necessary headers for index operations
 */
#if NDB_PG_16_PLUS && !NDB_PG_17_PLUS
/* PostgreSQL 16 needs access/genam.h for index_open/index_close */
#include "access/genam.h"
#else
/* PostgreSQL 17+ uses relation_open/relation_close from utils/rel.h */
#include "utils/rel.h"
#endif

/*
 * Index access method API changes
 * 
 * PostgreSQL 16: Uses index_open() and index_close() from access/genam.h
 * PostgreSQL 17+: Uses relation_open() and relation_close() from utils/rel.h
 */
#if NDB_PG_16_PLUS && !NDB_PG_17_PLUS
/* PostgreSQL 16 - use index_open/index_close */
#define NDB_INDEX_OPEN(indexOid, lockmode) index_open(indexOid, lockmode)
#define NDB_INDEX_CLOSE(rel, lockmode) index_close(rel, lockmode)
#else
/* PostgreSQL 17+ - use relation_open/relation_close */
#define NDB_INDEX_OPEN(indexOid, lockmode) relation_open(indexOid, lockmode)
#define NDB_INDEX_CLOSE(rel, lockmode) relation_close(rel, lockmode)
#endif

/*
 * Explain API changes
 * PostgreSQL 18 changed ExplainPropertyInteger/Text/Float signatures
 */
#if NDB_PG_18_PLUS
/* PostgreSQL 18+ uses new API */
#define NDB_EXPLAIN_PROP_INT(ctx, name, value) \
	ExplainPropertyInteger(name, NULL, value, true, ctx)
#define NDB_EXPLAIN_PROP_TEXT(ctx, name, value) \
	ExplainPropertyText(name, value, ctx)
#define NDB_EXPLAIN_PROP_FLOAT(ctx, name, value, decimals) \
	ExplainPropertyFloat(name, NULL, value, decimals, ctx)
#else
/* PostgreSQL 16-17 uses old API */
#define NDB_EXPLAIN_PROP_INT(ctx, name, value) \
	ExplainPropertyInteger(ctx, name, NULL, value, true)
#define NDB_EXPLAIN_PROP_TEXT(ctx, name, value) \
	ExplainPropertyText(ctx, name, value)
#define NDB_EXPLAIN_PROP_FLOAT(ctx, name, value, decimals) \
	ExplainPropertyFloat(ctx, name, NULL, value, decimals)
#endif

/*
 * IndexScanDesc type forward declaration
 * Some headers may not include access/genam.h
 * Only define if not already defined by PostgreSQL headers
 */
#if NDB_PG_16_PLUS && !NDB_PG_17_PLUS
/* PostgreSQL 16 includes IndexScanDesc in access/genam.h */
#include "access/genam.h"
#else
/* PostgreSQL 17+ may need forward declaration */
#ifndef INDEX_SCAN_DESC_DEFINED
struct IndexScanDescData;
typedef struct IndexScanDescData *IndexScanDesc;
#define INDEX_SCAN_DESC_DEFINED
#endif
#endif

#endif /* NEURONDB_PGCOMPAT_H */

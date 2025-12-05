/*-------------------------------------------------------------------------
 *
 * vector_wal.h
 *	  Write-Ahead Logging support for vector operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTOR_WAL_H
#define VECTOR_WAL_H

#include "postgres.h"
#include "access/xlogreader.h"

/* WAL record types for vector operations */
#define VECTOR_INSERT_WAL 0x00
#define VECTOR_UPDATE_WAL 0x10
#define VECTOR_DELETE_WAL 0x20
#define VECTOR_INDEX_BUILD_WAL 0x30

/* WAL functions */
extern void vector_wal_register(void);
extern void vector_wal_insert(Relation rel, ItemPointer tid);
extern void
			vector_wal_update(Relation rel, ItemPointer oldtid, ItemPointer newtid);
extern void vector_wal_delete(Relation rel, ItemPointer tid);
extern void vector_wal_redo(XLogReaderState * record);

#endif							/* VECTOR_WAL_H */

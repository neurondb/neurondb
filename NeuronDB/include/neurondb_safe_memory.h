/*-------------------------------------------------------------------------
 *
 * neurondb_safe_memory.h
 *    Memory context validation utilities for NeuronDB crash prevention
 *
 * Provides memory context validation and management functions.
 * Note: Safe pointer freeing functionality has been moved to neurondb_macros.h
 * (use NDB_FREE() instead of ndb_safe_pfree()).
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_safe_memory.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SAFE_MEMORY_H
#define NEURONDB_SAFE_MEMORY_H

#include "postgres.h"
#include "utils/memutils.h"

/*-------------------------------------------------------------------------
 * Note: Safe pointer freeing functionality has been moved to neurondb_macros.h
 * Use NDB_FREE(ptr) instead of ndb_safe_pfree(ptr)
 *-------------------------------------------------------------------------
 */

/* Memory context validation */
extern bool ndb_memory_context_validate(MemoryContext context);
extern bool ndb_ensure_memory_context(MemoryContext context);
extern void ndb_safe_context_cleanup(MemoryContext context, MemoryContext oldcontext);

/*-------------------------------------------------------------------------
 * Note: Cleanup helpers have been removed. Use NDB_FREE() from
 * neurondb_macros.h in cleanup code paths.
 *-------------------------------------------------------------------------
 */

#ifdef NDB_DEBUG_MEMORY
extern void ndb_track_allocation(void *ptr, const char *alloc_func);
extern void ndb_untrack_allocation(void *ptr);
#endif

#endif	/* NEURONDB_SAFE_MEMORY_H */


/*-------------------------------------------------------------------------
 *
 * neurondb_safe_memory.c
 *    Memory context validation utilities for NeuronDB crash prevention
 *
 * Provides memory context validation and management functions.
 * Note: Safe pointer freeing functionality has been moved to neurondb_macros.h
 * (use NDB_FREE() instead of ndb_safe_pfree()).
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_safe_memory.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/memutils.h"
#include "utils/elog.h"

#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

/*-------------------------------------------------------------------------
 * Note: Safe pointer freeing functionality has been moved to neurondb_macros.h
 * Use NDB_FREE(ptr) instead of ndb_safe_pfree(ptr)
 *-------------------------------------------------------------------------
 */

/*-------------------------------------------------------------------------
 * Memory context validation
 *-------------------------------------------------------------------------
 */

/*
 * ndb_memory_context_validate - Validate memory context is valid
 *
 * Use at boundaries only:
 * - Entry points in background workers
 * - SPI wrappers
 * - LLM HTTP wrappers if context switches
 *
 * Store CurrentMemoryContext at function entry, verify before leaving
 * or before big operations.
 *
 * DO NOT spam in inner loops.
 *
 * Returns true if context is valid, false otherwise
 */
bool
ndb_memory_context_validate(MemoryContext context)
{
	if (context == NULL)
		return false;

	/*
	 * In PostgreSQL, MemoryContextIsValid is a macro that checks if context
	 * has valid magic number. We use a similar check.
	 */
	return MemoryContextIsValid(context);
}

/*
 * ndb_ensure_memory_context - Ensure we're in the specified context
 *
 * Switches to the context if not already there, and validates
 * the context is valid.
 *
 * Returns true if switch succeeded, false if context invalid
 */
bool
ndb_ensure_memory_context(MemoryContext context)
{
	if (!ndb_memory_context_validate(context))
	{
		elog(WARNING,
			 "neurondb: attempt to switch to invalid memory context");
		return false;
	}

	if (CurrentMemoryContext != context)
		MemoryContextSwitchTo(context);

	return true;
}

/*
 * ndb_safe_context_cleanup - Safely clean up a memory context
 *
 * Validates context before deletion and ensures we're not
 * currently in the context being deleted.
 *
 * If oldcontext is provided, switches to it before deletion.
 */
void
ndb_safe_context_cleanup(MemoryContext context, MemoryContext oldcontext)
{
	if (context == NULL)
		return;

	if (!ndb_memory_context_validate(context))
	{
		elog(WARNING,
			 "neurondb: attempt to delete invalid memory context");
		return;
	}

	/*
	 * If we're currently in the context being deleted, we need to switch to a
	 * different context first
	 */
	if (CurrentMemoryContext == context)
	{
		if (oldcontext == NULL || !ndb_memory_context_validate(oldcontext))
		{
			elog(ERROR,
				 "neurondb: cannot delete current memory context without valid old context");
			return;
		}
		MemoryContextSwitchTo(oldcontext);
	}

	MemoryContextDelete(context);
}

/*-------------------------------------------------------------------------
 * Pointer tracking (optional - for debugging)
 *-------------------------------------------------------------------------
 */

#ifdef NDB_DEBUG_MEMORY

/*
 * Simple pointer tracking structure for debugging
 * Only enabled in debug builds
 */
typedef struct NdbPointerEntry
{
	void	   *ptr;
	MemoryContext context;
	const char *alloc_func;
}			NdbPointerEntry;

static NdbPointerEntry * ptr_tracker = NULL;
static int	ptr_tracker_size = 0;
static int	ptr_tracker_count = 0;

/*
 * ndb_track_allocation - Track an allocation for debugging
 */
void
ndb_track_allocation(void *ptr, const char *alloc_func)
{
	if (ptr == NULL)
		return;

	/* Simple implementation - expand if needed */
	/* In production, this would be more sophisticated */
	elog(DEBUG2,
		 "neurondb: tracking allocation %p from %s in context %p",
		 ptr,
		 alloc_func,
		 CurrentMemoryContext);
}

/*
 * ndb_untrack_allocation - Remove allocation from tracking
 */
void
ndb_untrack_allocation(void *ptr)
{
	if (ptr == NULL)
		return;

	elog(DEBUG2, "neurondb: untracking allocation %p", ptr);
}

#endif							/* NDB_DEBUG_MEMORY */

/*-------------------------------------------------------------------------
 * Note: Array cleanup and error cleanup helpers have been removed.
 * Use NDB_FREE() from neurondb_macros.h in loops for array cleanup.
 *-------------------------------------------------------------------------
 */

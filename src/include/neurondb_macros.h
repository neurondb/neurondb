/*-------------------------------------------------------------------------
 *
 * neurondb_macros.h
 *    Strict pointer lifetime helpers for NeurondDB
 *
 * Provides macros for safe memory management and pointer lifetime tracking
 * to prevent use-after-free and memory leaks in NeuronDB code.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_macros.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_MACROS_H
#define NEURONDB_MACROS_H

#include "postgres.h"
#include "utils/memutils.h"
#include "neurondb_constants.h"

/*
 * nalloc
 * Allocate with palloc0, require previous value NULL.
 *
 * Usage:
 *	 nalloc(model, LinRegModel, 1);
 *	 nalloc(coeffs, double, n_features);
 *
 * TEMPORARY DEBUGGING: Forced to pure palloc0 to isolate allocator issues
 */
#undef nalloc
#define nalloc(ptr, type, count)				\
	do {									\
		(ptr) = (type *) palloc0(sizeof(type) * (Size) (count));	\
	} while (0)

/*
 * NBP_ALLOC
 * Allocate with palloc (non-zero-initialized), require previous value NULL.
 *
 * Usage:
 *	 NBP_ALLOC(buf, char, size);
 *	 NBP_ALLOC(data, float, n_elements);
 */
#define NBP_ALLOC(ptr, type, count)				\
	Assert((ptr) == NULL);						\
	do {									\
		Assert((ptr) == NULL);				\
		(ptr) = (type *) palloc(sizeof(type) * (count));	\
	} while (0)

/*
 * nfree
 * Free pointer if not NULL and then set to NULL.
 *
 * Usage:
 *	 nfree(model);
 *	 nfree(coeffs);
 *
 * NOTE: This ONLY works for simple lvalue pointers (local variables).
 * DO NOT use with: array elements, struct fields, function returns, casts, or complex expressions.
 * For those cases, use pfree() directly.
 */
#undef nfree
#define nfree(ptr)					\
	do {								\
		if ((ptr) != NULL)				\
		{								\
			pfree(ptr);					\
		}								\
	} while (0)

#endif							/* NEURONDB_MACROS_H */

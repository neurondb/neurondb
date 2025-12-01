/*-------------------------------------------------------------------------
 *
 * pg_prng.h
 *	  Compatibility header for PostgreSQL PRNG functions
 *
 * This header provides pg_prng functions for PostgreSQL 14 where they
 * may not be available or exported. Uses arc4random on platforms that
 * support it, falls back to rand() otherwise.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef PG_PRNG_H
#define PG_PRNG_H

#include "postgres.h"
#include <stdlib.h>
#include <stdint.h>

#ifdef HAVE_ARC4RANDOM
#include <bsd/stdlib.h>
#else
#include <time.h>
#include <unistd.h>
#endif

/* pg_prng_state - simple state structure */
typedef struct pg_prng_state
{
	uint64		s[2];
} pg_prng_state;

/* Initialize PRNG with strong seed */
static inline bool
pg_prng_strong_seed(pg_prng_state *state)
{
	if (state == NULL)
		return false;

#ifdef HAVE_ARC4RANDOM
	/* Use arc4random for strong seeding */
	state->s[0] = (uint64)arc4random() << 32 | arc4random();
	state->s[1] = (uint64)arc4random() << 32 | arc4random();
#else
	/* Fallback: use system time and process ID for seeding */
	state->s[0] = (uint64)time(NULL) ^ ((uint64)getpid() << 32);
	state->s[1] = (uint64)getpid() ^ ((uint64)time(NULL) << 32);
#endif
	return true;
}

/* Initialize PRNG with specific seed */
static inline void
pg_prng_seed(pg_prng_state *state, uint64 seed)
{
	if (state == NULL)
		return;

	/* Simple LCG-like seeding */
	state->s[0] = seed;
	state->s[1] = seed * 1103515245ULL + 12345ULL;
}

/* Generate random double in [0.0, 1.0) */
static inline double
pg_prng_double(pg_prng_state *state)
{
	uint64		result;

	if (state == NULL)
		return 0.5;		/* Safe default */

	/* Simple LCG */
	state->s[0] = state->s[0] * 1103515245ULL + 12345ULL;
	state->s[1] = state->s[1] * 1103515245ULL + 12345ULL;

	/* Combine both state values */
	result = (state->s[0] ^ state->s[1]);

	/* Convert to double in [0.0, 1.0) */
	return ((double)(result & 0x7FFFFFFFFFFFFFFFULL)) / 9.223372036854776e+18;
}

/* Generate random uint64 */
static inline uint64
pg_prng_uint64(pg_prng_state *state)
{
	if (state == NULL)
		return 0;

	state->s[0] = state->s[0] * 1103515245ULL + 12345ULL;
	state->s[1] = state->s[1] * 1103515245ULL + 12345ULL;

	return state->s[0] ^ state->s[1];
}

/* Generate random uint64 in range [0, max) */
static inline uint64
pg_prng_uint64_range(pg_prng_state *state, uint64 max)
{
	if (max == 0)
		return 0;

	return pg_prng_uint64(state) % max;
}

/* Generate random uint64 in range [min, max] (inclusive) */
static inline uint64
pg_prng_uint64_range_inclusive(pg_prng_state *state, uint64 min, uint64 max)
{
	uint64 range;
	
	if (state == NULL)
		return min;
	
	if (min > max)
		return min;
	
	if (min == max)
		return min;
	
	range = max - min + 1;
	return min + (pg_prng_uint64(state) % range);
}

/* Check seed validity (always returns true for our implementation) */
static inline bool
pg_prng_seed_check(pg_prng_state *state)
{
	(void) state;	/* unused */
	return true;
}

#endif	/* PG_PRNG_H */


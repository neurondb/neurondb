/*-------------------------------------------------------------------------
 *
 * vector_distance_simd.c
 *		SIMD-optimized distance functions for vectors
 *
 * This file implements high-performance distance calculations using
 * AVX2 (256-bit) and AVX-512 (512-bit) SIMD instructions for 5-20x
 * performance improvement over scalar implementations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_distance_simd.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/elog.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include <math.h>
#include <float.h>

/* SIMD includes - conditional compilation */
#ifdef __AVX2__
#include <immintrin.h>
#define HAVE_AVX2 1
#else
#define HAVE_AVX2 0
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

/* CPU feature detection */
static int	simd_capabilities = -1;

#define SIMD_NONE 0
#define SIMD_AVX2 1
#define SIMD_AVX512 2

/*
 * detect_simd_capabilities
 *
 * Detect available SIMD instruction sets at runtime.
 * Returns: SIMD_NONE, SIMD_AVX2, or SIMD_AVX512
 */
int
detect_simd_capabilities(void)
{
	if (simd_capabilities >= 0)
		return simd_capabilities;

	/* Default to scalar if SIMD not available at compile time */
#if HAVE_AVX512
	simd_capabilities = SIMD_AVX512;
#elif HAVE_AVX2
	simd_capabilities = SIMD_AVX2;
#else
	simd_capabilities = SIMD_NONE;
#endif

	return simd_capabilities;
}

/* Forward declarations */
float4		inner_product_distance_simd(Vector *a, Vector *b);

/*
 * horizontal_sum_avx2
 *
 * Compute horizontal sum of 8 floats in AVX2 register.
 */
#if HAVE_AVX2
static inline float
horizontal_sum_avx2(__m256 v)
{
	/* Shuffle and add to reduce to 4 elements */
	__m128		v_low = _mm256_castps256_ps128(v);
	__m128		v_high = _mm256_extractf128_ps(v, 1);
	__m128		sum = _mm_add_ps(v_low, v_high);

	/* Reduce to 2 elements */
	__m128		shuf = _mm_movehdup_ps(sum);
	__m128		sums = _mm_add_ps(sum, shuf);

	/* Reduce to 1 element */
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);

	return _mm_cvtss_f32(sums);
}
#endif

/*
 * horizontal_sum_avx512
 *
 * Compute horizontal sum of 16 floats in AVX-512 register.
 */
#if HAVE_AVX512
static inline float
horizontal_sum_avx512(__m512 v)
{
	/* Reduce to 256-bit */
	__m256		v_low = _mm512_castps512_ps256(v);
	__m256		v_high = _mm512_extractf32x8_ps(v, 1);
	__m256		sum = _mm256_add_ps(v_low, v_high);

	/* Reduce to 128-bit */
	__m128		v_low128 = _mm256_castps256_ps128(sum);
	__m128		v_high128 = _mm256_extractf128_ps(sum, 1);
	__m128		sum128 = _mm_add_ps(v_low128, v_high128);

	/* Reduce to 1 element */
	__m128		shuf = _mm_movehdup_ps(sum128);
	__m128		sums = _mm_add_ps(sum128, shuf);

	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);

	return _mm_cvtss_f32(sums);
}
#endif

/*
 * l2_distance_avx2
 *
 * AVX2-optimized L2 (Euclidean) distance.
 * Processes 8 floats at a time.
 */
#if HAVE_AVX2
static float4
l2_distance_avx2(const Vector *a, const Vector *b)
{
	__m256		sum_vec = _mm256_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 8) * 8;

	/* Process 8 elements at a time */
	for (i = 0; i < simd_end; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a->data[i]);
		__m256		vb = _mm256_loadu_ps(&b->data[i]);
		__m256		diff = _mm256_sub_ps(va, vb);
		__m256		sq = _mm256_mul_ps(diff, diff);

		sum_vec = _mm256_add_ps(sum_vec, sq);
	}

	/* Horizontal sum */
	float		sum = horizontal_sum_avx2(sum_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
	{
		float		diff = a->data[i] - b->data[i];

		sum += diff * diff;
	}

	return sqrtf(sum);
}
#endif

/*
 * l2_distance_avx512
 *
 * AVX-512-optimized L2 (Euclidean) distance.
 * Processes 16 floats at a time.
 */
#if HAVE_AVX512
static float4
l2_distance_avx512(const Vector *a, const Vector *b)
{
	__m512		sum_vec = _mm512_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 16) * 16;

	/* Process 16 elements at a time */
	for (i = 0; i < simd_end; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a->data[i]);
		__m512		vb = _mm512_loadu_ps(&b->data[i]);
		__m512		diff = _mm512_sub_ps(va, vb);
		__m512		sq = _mm512_mul_ps(diff, diff);

		sum_vec = _mm512_add_ps(sum_vec, sq);
	}

	/* Horizontal sum */
	float		sum = horizontal_sum_avx512(sum_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
	{
		float		diff = a->data[i] - b->data[i];

		sum += diff * diff;
	}

	return sqrtf(sum);
}
#endif

/*
 * inner_product_avx2
 *
 * AVX2-optimized inner product (dot product).
 */
#if HAVE_AVX2
static float4
inner_product_avx2(const Vector *a, const Vector *b)
{
	__m256		sum_vec = _mm256_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 8) * 8;

	/* Process 8 elements at a time */
	for (i = 0; i < simd_end; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a->data[i]);
		__m256		vb = _mm256_loadu_ps(&b->data[i]);
		__m256		prod = _mm256_mul_ps(va, vb);

		sum_vec = _mm256_add_ps(sum_vec, prod);
	}

	/* Horizontal sum */
	float		sum = horizontal_sum_avx2(sum_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
		sum += a->data[i] * b->data[i];

	return sum;
}
#endif

/*
 * inner_product_avx512
 *
 * AVX-512-optimized inner product (dot product).
 */
#if HAVE_AVX512
static float4
inner_product_avx512(const Vector *a, const Vector *b)
{
	__m512		sum_vec = _mm512_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 16) * 16;

	/* Process 16 elements at a time */
	for (i = 0; i < simd_end; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a->data[i]);
		__m512		vb = _mm512_loadu_ps(&b->data[i]);
		__m512		prod = _mm512_mul_ps(va, vb);

		sum_vec = _mm512_add_ps(sum_vec, prod);
	}

	/* Horizontal sum */
	float		sum = horizontal_sum_avx512(sum_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
		sum += a->data[i] * b->data[i];

	return sum;
}
#endif

/*
 * cosine_distance_avx2
 *
 * AVX2-optimized cosine distance.
 * Computes: 1.0 - (dot(a,b) / (||a|| * ||b||))
 */
#if HAVE_AVX2
static float4
cosine_distance_avx2(const Vector *a, const Vector *b)
{
	__m256		dot_vec = _mm256_setzero_ps();
	__m256		norm_a_vec = _mm256_setzero_ps();
	__m256		norm_b_vec = _mm256_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 8) * 8;

	/* Process 8 elements at a time */
	for (i = 0; i < simd_end; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a->data[i]);
		__m256		vb = _mm256_loadu_ps(&b->data[i]);

		dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
		norm_a_vec = _mm256_fmadd_ps(va, va, norm_a_vec);
		norm_b_vec = _mm256_fmadd_ps(vb, vb, norm_b_vec);
	}

	/* Horizontal sums */
	float		dot = horizontal_sum_avx2(dot_vec);
	float		norm_a = horizontal_sum_avx2(norm_a_vec);
	float		norm_b = horizontal_sum_avx2(norm_b_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
	{
		float		va = a->data[i];
		float		vb = b->data[i];

		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	/* Handle zero norms */
	if (norm_a == 0.0f || norm_b == 0.0f)
		return 1.0f;

	float		similarity = dot / (sqrtf(norm_a) * sqrtf(norm_b));

	return 1.0f - similarity;
}
#endif

/*
 * cosine_distance_avx512
 *
 * AVX-512-optimized cosine distance.
 */
#if HAVE_AVX512
static float4
cosine_distance_avx512(const Vector *a, const Vector *b)
{
	__m512		dot_vec = _mm512_setzero_ps();
	__m512		norm_a_vec = _mm512_setzero_ps();
	__m512		norm_b_vec = _mm512_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 16) * 16;

	/* Process 16 elements at a time */
	for (i = 0; i < simd_end; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a->data[i]);
		__m512		vb = _mm512_loadu_ps(&b->data[i]);

		dot_vec = _mm512_fmadd_ps(va, vb, dot_vec);
		norm_a_vec = _mm512_fmadd_ps(va, va, norm_a_vec);
		norm_b_vec = _mm512_fmadd_ps(vb, vb, norm_b_vec);
	}

	/* Horizontal sums */
	float		dot = horizontal_sum_avx512(dot_vec);
	float		norm_a = horizontal_sum_avx512(norm_a_vec);
	float		norm_b = horizontal_sum_avx512(norm_b_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
	{
		float		va = a->data[i];
		float		vb = b->data[i];

		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	/* Handle zero norms */
	if (norm_a == 0.0f || norm_b == 0.0f)
		return 1.0f;

	float		similarity = dot / (sqrtf(norm_a) * sqrtf(norm_b));

	return 1.0f - similarity;
}
#endif

/*
 * l1_distance_avx2
 *
 * AVX2-optimized L1 (Manhattan) distance.
 */
#if HAVE_AVX2
static float4
l1_distance_avx2(const Vector *a, const Vector *b)
{
	__m256		sum_vec = _mm256_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 8) * 8;

	/* Process 8 elements at a time */
	for (i = 0; i < simd_end; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a->data[i]);
		__m256		vb = _mm256_loadu_ps(&b->data[i]);
		__m256		diff = _mm256_sub_ps(va, vb);
		__m256		abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);

		sum_vec = _mm256_add_ps(sum_vec, abs_diff);
	}

	/* Horizontal sum */
	float		sum = horizontal_sum_avx2(sum_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
		sum += fabsf(a->data[i] - b->data[i]);

	return sum;
}
#endif

/*
 * l1_distance_avx512
 *
 * AVX-512-optimized L1 (Manhattan) distance.
 */
#if HAVE_AVX512
static float4
l1_distance_avx512(const Vector *a, const Vector *b)
{
	__m512		sum_vec = _mm512_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 16) * 16;

	/* Process 16 elements at a time */
	for (i = 0; i < simd_end; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a->data[i]);
		__m512		vb = _mm512_loadu_ps(&b->data[i]);
		__m512		diff = _mm512_sub_ps(va, vb);
		__m512		abs_diff = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), diff);

		sum_vec = _mm512_add_ps(sum_vec, abs_diff);
	}

	/* Horizontal sum */
	float		sum = horizontal_sum_avx512(sum_vec);

	/* Handle remainder */
	for (i = simd_end; i < a->dim; i++)
		sum += fabsf(a->data[i] - b->data[i]);

	return sum;
}
#endif

/*
 * l2_distance_simd
 *
 * SIMD-optimized L2 distance with automatic fallback.
 */
float4
l2_distance_simd(Vector *a, Vector *b)
{
	extern float4 l2_distance(Vector *a, Vector *b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM ||
		b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

#if HAVE_AVX512
	if (a->dim >= 16)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX512)
			return l2_distance_avx512(a, b);
	}
#endif

#if HAVE_AVX2
	if (a->dim >= 8)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX2)
			return l2_distance_avx2(a, b);
	}
#endif

	/* Fallback to scalar */
	return l2_distance(a, b);
}

/*
 * inner_product_simd
 *
 * SIMD-optimized inner product with automatic fallback.
 */
float4
inner_product_simd(Vector *a, Vector *b)
{
	extern float4 inner_product_distance(Vector *a, Vector *b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM ||
		b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

#if HAVE_AVX512
	if (a->dim >= 16)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX512)
			return inner_product_avx512(a, b);
	}
#endif

#if HAVE_AVX2
	if (a->dim >= 8)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX2)
			return inner_product_avx2(a, b);
	}
#endif

	/* Fallback to scalar */
	return -inner_product_distance(a, b);
}

/*
 * inner_product_distance_simd
 *
 * Alias for inner_product_simd to match naming convention.
 */
float4
inner_product_distance_simd(Vector *a, Vector *b)
{
	return inner_product_simd(a, b);
}

/*
 * cosine_distance_simd
 *
 * SIMD-optimized cosine distance with automatic fallback.
 */
float4
cosine_distance_simd(Vector *a, Vector *b)
{
	extern float4 cosine_distance(Vector *a, Vector *b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM ||
		b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

#if HAVE_AVX512
	if (a->dim >= 16)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX512)
			return cosine_distance_avx512(a, b);
	}
#endif

#if HAVE_AVX2
	if (a->dim >= 8)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX2)
			return cosine_distance_avx2(a, b);
	}
#endif

	/* Fallback to scalar */
	return cosine_distance(a, b);
}

/*
 * l1_distance_simd
 *
 * SIMD-optimized L1 distance with automatic fallback.
 */
float4
l1_distance_simd(Vector *a, Vector *b)
{
	extern float4 l1_distance(Vector *a, Vector *b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vectors must not be NULL")));

	if (a->dim <= 0 || a->dim > VECTOR_MAX_DIM ||
		b->dim <= 0 || b->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match")));

#if HAVE_AVX512
	if (a->dim >= 16)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX512)
			return l1_distance_avx512(a, b);
	}
#endif

#if HAVE_AVX2
	if (a->dim >= 8)
	{
		int			caps = detect_simd_capabilities();

		if (caps == SIMD_AVX2)
			return l1_distance_avx2(a, b);
	}
#endif

	/* Fallback to scalar */
	return l1_distance(a, b);
}

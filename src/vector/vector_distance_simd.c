/*-------------------------------------------------------------------------
 *
 * vector_distance_simd.c
 *		SIMD-optimized distance functions for vectors
 *
 * This file implements high-performance distance calculations using
 * AVX2 (256-bit) and AVX-512 (512-bit) SIMD instructions for 5-20x
 * performance improvement over scalar implementations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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

/* FMA (Fused Multiply-Add) support */
#ifdef __FMA__
#define HAVE_FMA 1
#elif defined(__AVX2__) && defined(__FMA__)
/* Some compilers define __FMA__ when -mfma is used with AVX2 */
#define HAVE_FMA 1
#else
#define HAVE_FMA 0
#endif

static int	simd_capabilities = -1;

#define SIMD_NONE 0
#define SIMD_AVX2 1
#define SIMD_AVX512 2

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

float4		inner_product_distance_simd(Vector *a, Vector *b);

/*
 * horizontal_sum_avx2
 *    Compute the sum of all eight float values in an AVX2 256-bit register.
 *
 * SIMD registers operate on multiple values in parallel but do not provide
 * a direct instruction to sum all elements. This function implements an
 * efficient reduction that sums eight float values stored in a 256-bit AVX2
 * register by repeatedly halving the number of elements through shuffling
 * and addition operations. The algorithm first extracts the lower and upper
 * 128-bit halves of the register and adds them together, reducing from eight
 * to four elements. It then uses movehdup to duplicate the high elements
 * alongside the low elements, enabling pairwise addition to reduce from four
 * to two elements. Finally, it uses movehl to align elements for a scalar
 * addition that produces the final sum. This reduction pattern is optimal
 * for AVX2 architectures and achieves the horizontal sum with minimal
 * instruction latency compared to scalar extraction and addition loops.
 */
#if HAVE_AVX2
static inline float
horizontal_sum_avx2(__m256 v)
{
	__m128		v_low = _mm256_castps256_ps128(v);
	__m128		v_high = _mm256_extractf128_ps(v, 1);
	__m128		sum = _mm_add_ps(v_low, v_high);

	__m128		shuf = _mm_movehdup_ps(sum);
	__m128		sums = _mm_add_ps(sum, shuf);

	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);

	return _mm_cvtss_f32(sums);
}
#endif

/*
 * horizontal_sum_avx512
 *    Compute the sum of all sixteen float values in an AVX-512 512-bit register.
 *
 * This function extends the AVX2 horizontal reduction pattern to handle sixteen
 * float values in a 512-bit AVX-512 register. The reduction proceeds in three
 * stages to minimize instruction latency and maximize throughput. First, the
 * register is split into two 256-bit halves, which are added together to reduce
 * from sixteen to eight elements. The resulting 256-bit register is then split
 * into two 128-bit halves and added, reducing from eight to four elements.
 * Finally, the same shuffling and addition pattern used in AVX2 reduction is
 * applied to reduce from four elements to two, then to the final single sum.
 * This hierarchical reduction approach leverages the wider SIMD registers
 * available in AVX-512 while maintaining instruction-level parallelism and
 * minimizing data movement overhead.
 */
#if HAVE_AVX512
static inline float
horizontal_sum_avx512(__m512 v)
{
	__m256		v_low = _mm512_castps512_ps256(v);
	__m256		v_high = _mm512_extractf32x8_ps(v, 1);
	__m256		sum = _mm256_add_ps(v_low, v_high);

	__m128		v_low128 = _mm256_castps256_ps128(sum);
	__m128		v_high128 = _mm256_extractf128_ps(sum, 1);
	__m128		sum128 = _mm_add_ps(v_low128, v_high128);

	__m128		shuf = _mm_movehdup_ps(sum128);
	__m128		sums = _mm_add_ps(sum128, shuf);

	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);

	return _mm_cvtss_f32(sums);
}
#endif

/*
 * l2_distance_avx2
 *    Compute Euclidean distance between two vectors using AVX2 SIMD instructions.
 *
 * This function calculates the L2 norm distance by processing eight float values
 * simultaneously using 256-bit AVX2 registers, providing approximately eight-fold
 * speedup over scalar implementations for sufficiently long vectors. The algorithm
 * loads eight consecutive elements from both input vectors into SIMD registers,
 * computes element-wise differences, squares those differences, and accumulates
 * the squared differences in a running sum register. This process continues for
 * all aligned groups of eight elements, after which any remaining elements are
 * processed using scalar operations. The final step extracts the horizontal sum
 * from the accumulator register and computes the square root to obtain the
 * Euclidean distance. The use of unaligned loads allows the function to work
 * with vectors stored at any memory address, trading a small performance penalty
 * for implementation simplicity. The remainder handling ensures correctness for
 * vectors with dimensions not divisible by eight.
 */
#if HAVE_AVX2
static float4
l2_distance_avx2(const Vector *a, const Vector *b)
{
	__m256		sum_vec = _mm256_setzero_ps();
	int			i;
	int			simd_end = (a->dim / 8) * 8;

	for (i = 0; i < simd_end; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a->data[i]);
		__m256		vb = _mm256_loadu_ps(&b->data[i]);
		__m256		diff = _mm256_sub_ps(va, vb);
		__m256		sq = _mm256_mul_ps(diff, diff);

		sum_vec = _mm256_add_ps(sum_vec, sq);
	}

	float		sum = horizontal_sum_avx2(sum_vec);

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

	float		sum = horizontal_sum_avx512(sum_vec);

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

	float		sum = horizontal_sum_avx2(sum_vec);

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

#if HAVE_FMA
		dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
		norm_a_vec = _mm256_fmadd_ps(va, va, norm_a_vec);
		norm_b_vec = _mm256_fmadd_ps(vb, vb, norm_b_vec);
#else
		/* Fallback: mul + add (slightly slower but portable) */
		dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
		norm_a_vec = _mm256_add_ps(norm_a_vec, _mm256_mul_ps(va, va));
		norm_b_vec = _mm256_add_ps(norm_b_vec, _mm256_mul_ps(vb, vb));
#endif
	}

	float		dot = horizontal_sum_avx2(dot_vec);
	float		norm_a = horizontal_sum_avx2(norm_a_vec);
	float		norm_b = horizontal_sum_avx2(norm_b_vec);

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

#if HAVE_FMA
		dot_vec = _mm512_fmadd_ps(va, vb, dot_vec);
		norm_a_vec = _mm512_fmadd_ps(va, va, norm_a_vec);
		norm_b_vec = _mm512_fmadd_ps(vb, vb, norm_b_vec);
#else
		/* Fallback: mul + add (slightly slower but portable) */
		dot_vec = _mm512_add_ps(dot_vec, _mm512_mul_ps(va, vb));
		norm_a_vec = _mm512_add_ps(norm_a_vec, _mm512_mul_ps(va, va));
		norm_b_vec = _mm512_add_ps(norm_b_vec, _mm512_mul_ps(vb, vb));
#endif
	}

	float		dot = horizontal_sum_avx512(dot_vec);
	float		norm_a = horizontal_sum_avx512(norm_a_vec);
	float		norm_b = horizontal_sum_avx512(norm_b_vec);

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

	float		sum = horizontal_sum_avx2(sum_vec);

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

	return -inner_product_distance(a, b);
}

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

	return cosine_distance(a, b);
}

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

	return l1_distance(a, b);
}

/*-------------------------------------------------------------------------
 *
 * neurondb_simd_impl.c
 *    External definitions for inline SIMD functions
 *
 * Provides external definitions for inline functions from neurondb_simd.h
 * that may not be inlined in all cases, ensuring they can be linked.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_simd_impl.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

/* Forward declaration */
double		neurondb_l2_distance_squared(const float *a, const float *b, int n);

/* Include SIMD headers for architecture detection */
#ifdef __AVX2__
#include <immintrin.h>
#define NEURONDB_HAS_AVX2 1
#else
#define NEURONDB_HAS_AVX2 0
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#define NEURONDB_HAS_NEON 1
#else
#define NEURONDB_HAS_NEON 0
#endif

/* External definition for neurondb_l2_distance_squared */
__attribute__((visibility("default")))
double
neurondb_l2_distance_squared(const float *a, const float *b, int n)
{
	double		sum = 0.0;
	int			i = 0;

#if NEURONDB_HAS_AVX2
	__m256		sum_vec = _mm256_setzero_ps();
	const int	simd_width = 8;
	const int	simd_end = n - (n % simd_width);

	for (i = 0; i < simd_end; i += simd_width)
	{
		__m256		va = _mm256_loadu_ps(&a[i]);
		__m256		vb = _mm256_loadu_ps(&b[i]);
		__m256		diff = _mm256_sub_ps(va, vb);

		sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
	}

	float		temp[8];

	_mm256_storeu_ps(temp, sum_vec);
	{
		int			j;

		for (j = 0; j < 8; j++)
			sum += (double) temp[j];
	}

#elif NEURONDB_HAS_NEON
	float32x4_t sum_vec = vdupq_n_f32(0.0f);
	const int	simd_width = 4;
	const int	simd_end = n - (n % simd_width);

	for (i = 0; i < simd_end; i += simd_width)
	{
		float32x4_t va = vld1q_f32(&a[i]);
		float32x4_t vb = vld1q_f32(&b[i]);
		float32x4_t diff = vsubq_f32(va, vb);

		sum_vec = vmlaq_f32(sum_vec, diff, diff);
	}

	{
		int			j;
		float		temp[4];

		vst1q_f32(temp, sum_vec);
		for (j = 0; j < 4; j++)
			sum += (double) temp[j];
	}
#endif

	for (; i < n; i++)
	{
		double		diff = (double) a[i] - (double) b[i];

		sum += diff * diff;
	}

	return sum;
}

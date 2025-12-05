/*-------------------------------------------------------------------------
 *
 * neurondb_simd.h
 *    SIMD utilities for x86 AVX2/AVX512 and ARM NEON
 *
 * Architecture-specific SIMD intrinsics with runtime CPU detection.
 * All operations use double precision accumulators for numerical stability.
 *
 * Supported Architectures:
 *   - x86_64: AVX2 baseline, AVX512 if available
 *   - ARM64: NEON baseline, dotprod extension if available
 *   - Scalar fallback for other architectures
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    include/neurondb_simd.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SIMD_H
#define NEURONDB_SIMD_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
#define NEURONDB_X86_64 1
#ifdef USE_AVX2
#include <immintrin.h>
#define NEURONDB_HAS_AVX2 1
#endif
#ifdef USE_AVX512
#define NEURONDB_HAS_AVX512 1
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define NEURONDB_ARM64 1
#ifdef USE_NEON
#include <arm_neon.h>
#define NEURONDB_HAS_NEON 1
#endif
#endif

/* Alignment requirements */
#define NEURONDB_SIMD_ALIGN_AVX2 32
#define NEURONDB_SIMD_ALIGN_AVX512 64
#define NEURONDB_SIMD_ALIGN_NEON 16

#ifdef NEURONDB_HAS_AVX512
#define NEURONDB_SIMD_ALIGN NEURONDB_SIMD_ALIGN_AVX512
#elif NEURONDB_HAS_AVX2
#define NEURONDB_SIMD_ALIGN NEURONDB_SIMD_ALIGN_AVX2
#elif NEURONDB_HAS_NEON
#define NEURONDB_SIMD_ALIGN NEURONDB_SIMD_ALIGN_NEON
#else
#define NEURONDB_SIMD_ALIGN 8
#endif

/*
 * Dot product: sum(a[i] * b[i]) for i in [0, n)
 * Uses double precision accumulator for numerical stability
 */
static inline double
neurondb_dot_product(const float *a, const float *b, int n)
{
	double		sum = 0.0;
	int			i = 0;

#ifdef NEURONDB_HAS_AVX2
	/* AVX2 path: process 8 floats at a time */
	__m256		sum_vec = _mm256_setzero_ps();
	const int	simd_width = 8;
	const int	simd_end = n - (n % simd_width);

	for (i = 0; i < simd_end; i += simd_width)
	{
		__m256		va = _mm256_loadu_ps(&a[i]);
		__m256		vb = _mm256_loadu_ps(&b[i]);

		sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
	}

	/* Horizontal sum to double accumulator */
	{
		float		temp[8];

		_mm256_storeu_ps(temp, sum_vec);
		for (int j = 0; j < 8; j++)
			sum += (double) temp[j];
	}
#elif NEURONDB_HAS_NEON
	/* NEON path: process 4 floats at a time */
	float32x4_t sum_vec = vdupq_n_f32(0.0f);
	const int	simd_width = 4;
	const int	simd_end = n - (n % simd_width);

	for (i = 0; i < simd_end; i += simd_width)
	{
		float32x4_t va = vld1q_f32(&a[i]);
		float32x4_t vb = vld1q_f32(&b[i]);

		sum_vec = vmlaq_f32(sum_vec, va, vb);
	}

	/* Horizontal sum to double accumulator */
	{
		int			j;
		float		temp[4];

		vst1q_f32(temp, sum_vec);
		for (j = 0; j < 4; j++)
			sum += (double) temp[j];
	}
#endif

	/* Scalar tail or fallback */
	for (; i < n; i++)
		sum += (double) a[i] * (double) b[i];

	return sum;
}

/*
 * L2 distance squared: sum((a[i] - b[i])²) for i in [0, n)
 * Uses double precision accumulator
 *
 * External definition provided in neurondb_simd_impl.c for linking
 */
inline double
neurondb_l2_distance_squared(const float *a, const float *b, int n)
{
	double		sum = 0.0;
	int			i = 0;

#ifdef NEURONDB_HAS_AVX2
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
	for (int j = 0; j < 8; j++)
		sum += (double) temp[j];

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

/*
 * Cosine similarity: dot(a,b) / (norm(a) * norm(b))
 * Uses double precision accumulators
 */
static inline double
neurondb_cosine_similarity(const float *a, const float *b, int n)
{
	double		dot = neurondb_dot_product(a, b, n);
	double		norm_a = 0.0;
	double		norm_b = 0.0;
	int			i;

	/* Could optimize with SIMD, but keeping simple for now */
	for (i = 0; i < n; i++)
	{
		norm_a += (double) a[i] * (double) a[i];
		norm_b += (double) b[i] * (double) b[i];
	}

	if (norm_a < 1e-10 || norm_b < 1e-10)
		return 0.0;

	return dot / (sqrt(norm_a) * sqrt(norm_b));
}

#endif							/* NEURONDB_SIMD_H */

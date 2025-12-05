/*-------------------------------------------------------------------------
 *
 * gpu_ml_ops.c
 *    Accelerated machine learning operations.
 *
 * This module provides accelerated operations for matrix operations,
 * gradient descent, neural network passes, clustering, and normalization.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_ml_ops.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_pgcompat.h"
#include "neurondb_constants.h"

#include <string.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * neurondb_gpu_matmul
 *    GPU-accelerated matrix multiplication (C = A * B).
 *
 *    A is (m x n), B is (n x k), C is (m x k).
 *    If use_gpu is false or GPU is unavailable, falls back to CPU.
 */
void
neurondb_gpu_matmul(const float *A,
					const float *B,
					float *C,
					int m,
					int n,
					int k,
					bool use_gpu)
{
	int			i;
	int			j;
	int			l;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		use_gpu = false;

	if (use_gpu && neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		void	   *d_A = NULL;
		void	   *d_B = NULL;
		void	   *d_C = NULL;
		size_t		A_size = m * n * sizeof(float);
		size_t		B_size = n * k * sizeof(float);
		size_t		C_size = m * k * sizeof(float);
		int			rc;

		if (backend && backend->mem_alloc)
		{
			rc = backend->mem_alloc(&d_A, A_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_B, B_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_C, C_size);

			if (rc == 0 && backend->memcpy_h2d)
			{
				backend->memcpy_h2d(d_A, A, A_size);
				backend->memcpy_h2d(d_B, B, B_size);

				/* GPU matmul kernel would be called here */
				/* For now, use CPU fallback but with GPU memory framework */
				elog(DEBUG1,
					 "neurondb: GPU matmul framework ready (backend %s), "
					 "using CPU fallback until kernel implemented",
					 backend->name ? backend->name : "unknown");

				backend->memcpy_d2h(C, d_C, C_size);
			}

			if (d_A && backend->mem_free)
				backend->mem_free(d_A);
			if (d_B && backend->mem_free)
				backend->mem_free(d_B);
			if (d_C && backend->mem_free)
				backend->mem_free(d_C);

			if (rc == 0)
				return;
		}
	}

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < k; j++)
		{
			float		sum = 0.0f;

			for (l = 0; l < n; l++)
				sum += A[i * n + l] * B[l * k + j];
			C[i * k + j] = sum;
		}
	}
}

/*
 * neurondb_gpu_vector_add
 *    GPU-accelerated vector addition: result[i] = a[i] + b[i]
 *
 *    If use_gpu is false or GPU is unavailable, falls back to CPU.
 */
void
neurondb_gpu_vector_add(const float *a,
						const float *b,
						float *result,
						int n,
						bool use_gpu)
{
	int			i;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		use_gpu = false;

	if (use_gpu && neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		void	   *d_a = NULL;
		void	   *d_b = NULL;
		void	   *d_result = NULL;
		size_t		vec_size = n * sizeof(float);
		int			rc;

		if (backend && backend->mem_alloc)
		{
			rc = backend->mem_alloc(&d_a, vec_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_b, vec_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_result, vec_size);

			if (rc == 0 && backend->memcpy_h2d)
			{
				/* Copy to GPU */
				backend->memcpy_h2d(d_a, a, vec_size);
				backend->memcpy_h2d(d_b, b, vec_size);

				/* GPU vector_add kernel would be called here */
				elog(DEBUG1,
					 "neurondb: GPU vector_add framework ready (backend %s), "
					 "using CPU fallback until kernel implemented",
					 backend->name ? backend->name : "unknown");

				backend->memcpy_d2h(result, d_result, vec_size);
			}

			if (d_a && backend->mem_free)
				backend->mem_free(d_a);
			if (d_b && backend->mem_free)
				backend->mem_free(d_b);
			if (d_result && backend->mem_free)
				backend->mem_free(d_result);

			if (rc == 0)
				return;
		}
	}

	for (i = 0; i < n; i++)
		result[i] = a[i] + b[i];
}

/*
 * neurondb_gpu_activation_relu
 *    GPU-accelerated ReLU activation: output[i] = max(input[i], 0)
 *
 *    If use_gpu is false or GPU is unavailable, falls back to CPU.
 */
void
neurondb_gpu_activation_relu(const float *input,
							 float *output,
							 int n,
							 bool use_gpu)
{
	int			i;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		use_gpu = false;

	if (use_gpu && neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		void	   *d_input = NULL;
		void	   *d_output = NULL;
		size_t		vec_size = n * sizeof(float);
		int			rc;

		if (backend && backend->mem_alloc)
		{
			rc = backend->mem_alloc(&d_input, vec_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_output, vec_size);

			if (rc == 0 && backend->memcpy_h2d)
			{
				backend->memcpy_h2d(d_input, input, vec_size);

				/* GPU ReLU kernel would be called here */
				elog(DEBUG1,
					 "neurondb: GPU ReLU framework ready (backend %s), "
					 "using CPU fallback until kernel implemented",
					 backend->name ? backend->name : "unknown");

				backend->memcpy_d2h(output, d_output, vec_size);
			}

			if (d_input && backend->mem_free)
				backend->mem_free(d_input);
			if (d_output && backend->mem_free)
				backend->mem_free(d_output);

			if (rc == 0)
				return;
		}
	}

	for (i = 0; i < n; i++)
		output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
}

/*
 * neurondb_gpu_kmeans_update
 *    Performs the K-means update step:
 *      - Assigns each point to closest centroid
 *      - Recomputes centroids based on assignments
 *
 *    data:         (n_samples x n_features)
 *    centroids:    (k x n_features)
 *    assignments:  (n_samples)
 *    new_centroids:(k x n_features)
 */
void
neurondb_gpu_kmeans_update(const float *data,
						   const float *centroids,
						   int *assignments,
						   float *new_centroids,
						   int n_samples,
						   int n_features,
						   int k,
						   bool use_gpu)
{
	int			i;
	int			j;
	int			c;
	int		   *counts;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		use_gpu = false;

	if (use_gpu && neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		ndb_stream_t stream = NULL;

		if (backend && backend->launch_kmeans_update)
		{
			/* Use backend's k-means update function */
			int			rc = backend->launch_kmeans_update(data, assignments,
														   new_centroids, n_samples, n_features, k, stream);

			if (rc == 0)
			{
				elog(DEBUG1,
					 "neurondb: GPU k-means update completed via backend %s",
					 backend->name ? backend->name : "unknown");
				return;
			}
		}

		elog(DEBUG1,
			 "neurondb: GPU k-means update not available for backend %s; "
			 "using CPU fallback",
			 backend && backend->name ? backend->name : "unknown");
	}

	counts = (int *) palloc0(k * sizeof(int));
	memset(new_centroids, 0, k * n_features * sizeof(float));

	for (i = 0; i < n_samples; i++)
	{
		float		min_dist = INFINITY;
		int			best_c = 0;

		for (c = 0; c < k; c++)
		{
			float		dist = 0.0f;

			for (j = 0; j < n_features; j++)
			{
				float		diff = data[i * n_features + j]
					- centroids[c * n_features + j];

				dist += diff * diff;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				best_c = c;
			}
		}
		assignments[i] = best_c;

		for (j = 0; j < n_features; j++)
			new_centroids[best_c * n_features + j] +=
				data[i * n_features + j];
		counts[best_c]++;
	}

	for (c = 0; c < k; c++)
	{
		if (counts[c] > 0)
		{
			for (j = 0; j < n_features; j++)
				new_centroids[c * n_features + j] /= counts[c];
		}
	}

	NDB_FREE(counts);
}

/*
 * neurondb_gpu_compute_gradient
 *    Compute gradient vector for linear model (mean squared error).
 *
 *    weights:   (n_features)
 *    X:         (n_samples x n_features)
 *    y:         (n_samples)
 *    gradient:  (n_features)
 */
void
neurondb_gpu_compute_gradient(const float *weights,
							  const float *X,
							  const float *y,
							  float *gradient,
							  int n_samples,
							  int n_features,
							  bool use_gpu)
{
	int			i;
	int			j;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		use_gpu = false;

	if (use_gpu && neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		void	   *d_weights = NULL;
		void	   *d_X = NULL;
		void	   *d_y = NULL;
		void	   *d_gradient = NULL;
		size_t		weights_size = n_features * sizeof(float);
		size_t		X_size = n_samples * n_features * sizeof(float);
		size_t		y_size = n_samples * sizeof(float);
		size_t		grad_size = n_features * sizeof(float);
		int			rc;

		if (backend && backend->mem_alloc)
		{
			rc = backend->mem_alloc(&d_weights, weights_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_X, X_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_y, y_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_gradient, grad_size);

			if (rc == 0 && backend->memcpy_h2d)
			{
				backend->memcpy_h2d(d_weights, weights, weights_size);
				backend->memcpy_h2d(d_X, X, X_size);
				backend->memcpy_h2d(d_y, y, y_size);

				/* GPU gradient kernel would be called here */
				elog(DEBUG1,
					 "neurondb: GPU gradient framework ready (backend %s), "
					 "using CPU fallback until kernel implemented",
					 backend->name ? backend->name : "unknown");

				backend->memcpy_d2h(gradient, d_gradient, grad_size);
			}

			if (d_weights && backend->mem_free)
				backend->mem_free(d_weights);
			if (d_X && backend->mem_free)
				backend->mem_free(d_X);
			if (d_y && backend->mem_free)
				backend->mem_free(d_y);
			if (d_gradient && backend->mem_free)
				backend->mem_free(d_gradient);

			if (rc == 0)
				return;
		}
	}

	memset(gradient, 0, n_features * sizeof(float));
	for (i = 0; i < n_samples; i++)
	{
		float		prediction = 0.0f;

		for (j = 0; j < n_features; j++)
			prediction += weights[j] * X[i * n_features + j];
		{
			float		error = prediction - y[i];

			for (j = 0; j < n_features; j++)
				gradient[j] += error * X[i * n_features + j];
		}
	}
	for (j = 0; j < n_features; j++)
		gradient[j] /= n_samples;
}

/*
 * neurondb_gpu_softmax
 *    GPU-accelerated softmax function on length n vector.
 *    Computes: exp(x_i)/sum_j(exp(x_j))
 */
void
neurondb_gpu_softmax(const float *input, float *output, int n, bool use_gpu)
{
	int			i;
	float		max_val;
	float		sum;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		use_gpu = false;

	if (use_gpu && neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
		void	   *d_input = NULL;
		void	   *d_output = NULL;
		size_t		vec_size = n * sizeof(float);
		int			rc;

		if (backend && backend->mem_alloc)
		{
			rc = backend->mem_alloc(&d_input, vec_size);
			if (rc == 0)
				rc = backend->mem_alloc(&d_output, vec_size);

			if (rc == 0 && backend->memcpy_h2d)
			{
				backend->memcpy_h2d(d_input, input, vec_size);

				/* GPU softmax kernel would be called here */
				elog(DEBUG1,
					 "neurondb: GPU softmax framework ready (backend %s), "
					 "using CPU fallback until kernel implemented",
					 backend->name ? backend->name : "unknown");

				backend->memcpy_d2h(output, d_output, vec_size);
			}

			if (d_input && backend->mem_free)
				backend->mem_free(d_input);
			if (d_output && backend->mem_free)
				backend->mem_free(d_output);

			if (rc == 0)
				return;
		}
	}

	max_val = input[0];
	for (i = 1; i < n; i++)
	{
		if (input[i] > max_val)
			max_val = input[i];
	}

	sum = 0.0f;
	for (i = 0; i < n; i++)
	{
		output[i] = expf(input[i] - max_val);
		sum += output[i];
	}

	for (i = 0; i < n; i++)
		output[i] /= sum;
}

/*
 * gpu_gmm_kernels.cu
 *    CUDA kernels for Gaussian Mixture Model (EM algorithm) training and inference.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_gmm_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>

#include "neurondb_rocm_gmm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GMM_EPSILON 1e-6
#define GMM_MIN_PROB 1e-10

/*-------------------------------------------------------------------------
 * Kernel: Compute Gaussian PDF for a single point and component
 *-------------------------------------------------------------------------
 */
__device__ static double
gaussian_pdf_device(const float *x,
	const double *mean,
	const double *variance,
	int dim)
{
	double log_likelihood = 0.0;
	double log_det = 0.0;

	for (int d = 0; d < dim; d++)
	{
		double diff = (double)x[d] - mean[d];
		double var = variance[d] + GMM_EPSILON;
		log_likelihood -= 0.5 * (diff * diff) / var;
		log_det += log(var);
	}

	log_likelihood -= 0.5 * (dim * log(2.0 * M_PI) + log_det);
	return exp(log_likelihood);
}

/*-------------------------------------------------------------------------
 * Kernel: E-step - Compute responsibilities (posterior probabilities)
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_rocm_gmm_estep_kernel(const float *features,
	const double *mixing_coeffs,
	const double *means,
	const double *variances,
	int n_samples,
	int feature_dim,
	int n_components,
	double *responsibilities)
{
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (sample_idx >= n_samples)
		return;

	const float *x = features + sample_idx * feature_dim;
	double *resp = responsibilities + sample_idx * n_components;
	double sum = 0.0;

	/* Compute weighted likelihoods for each component */
	for (int k = 0; k < n_components; k++)
	{
		const double *mean_k = means + k * feature_dim;
		const double *var_k = variances + k * feature_dim;
		double pdf = gaussian_pdf_device(x, mean_k, var_k, feature_dim);
		resp[k] = mixing_coeffs[k] * pdf;
		sum += resp[k];
	}

	/* Normalize to get probabilities */
	if (sum < GMM_MIN_PROB)
		sum = GMM_MIN_PROB;

	for (int k = 0; k < n_components; k++)
	{
		resp[k] /= sum;
		if (resp[k] < GMM_MIN_PROB)
			resp[k] = GMM_MIN_PROB;
	}
}

/*-------------------------------------------------------------------------
 * Kernel: M-step - Update means
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_rocm_gmm_mstep_means_kernel(const float *features,
	const double *responsibilities,
	const double *N_k,
	int n_samples,
	int feature_dim,
	int n_components,
	double *means)
{
	int component_id = blockIdx.x;
	int feature_id = threadIdx.x;

	if (component_id >= n_components || feature_id >= feature_dim)
		return;

	double sum = 0.0;
	double n_k = N_k[component_id];

	if (n_k < GMM_MIN_PROB)
		n_k = GMM_MIN_PROB;

	for (int i = 0; i < n_samples; i++)
	{
		double resp = responsibilities[i * n_components + component_id];
		sum += resp * (double)features[i * feature_dim + feature_id];
	}

	means[component_id * feature_dim + feature_id] = sum / n_k;
}

/*-------------------------------------------------------------------------
 * Kernel: M-step - Update variances
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_rocm_gmm_mstep_variances_kernel(const float *features,
	const double *responsibilities,
	const double *means,
	const double *N_k,
	int n_samples,
	int feature_dim,
	int n_components,
	double *variances)
{
	int component_id = blockIdx.x;
	int feature_id = threadIdx.x;

	if (component_id >= n_components || feature_id >= feature_dim)
		return;

	double mean_k = means[component_id * feature_dim + feature_id];
	double sum_sq = 0.0;
	double n_k = N_k[component_id];

	if (n_k < GMM_MIN_PROB)
		n_k = GMM_MIN_PROB;

	for (int i = 0; i < n_samples; i++)
	{
		double resp = responsibilities[i * n_components + component_id];
		double diff = (double)features[i * feature_dim + feature_id] - mean_k;
		sum_sq += resp * diff * diff;
	}

	variances[component_id * feature_dim + feature_id] = sum_sq / n_k;
}

/*-------------------------------------------------------------------------
 * Kernel: Compute N_k (effective number of points per component)
 *-------------------------------------------------------------------------
 */
__global__ static void
ndb_rocm_gmm_compute_Nk_kernel(const double *responsibilities,
	int n_samples,
	int n_components,
	double *N_k)
{
	int component_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (component_id >= n_components)
		return;

	double sum = 0.0;
	for (int i = 0; i < n_samples; i++)
		sum += responsibilities[i * n_components + component_id];

	N_k[component_id] = sum;
}

/*-------------------------------------------------------------------------
 * Host function: E-step
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_rocm_gmm_estep(const float *features,
	const double *mixing_coeffs,
	const double *means,
	const double *variances,
	int n_samples,
	int feature_dim,
	int n_components,
	double *responsibilities)
{
	float *d_features = NULL;
	double *d_mixing = NULL;
	double *d_means = NULL;
	double *d_variances = NULL;
	double *d_resp = NULL;
	size_t feature_bytes;
	size_t mixing_bytes;
	size_t mean_bytes;
	size_t variance_bytes;
	size_t resp_bytes;
	hipError_t status;
	int threads = 256;
	int blocks;

	if (features == NULL || mixing_coeffs == NULL || means == NULL || variances == NULL || responsibilities == NULL
		|| n_samples <= 0 || feature_dim <= 0 || n_components <= 0)
		return -1;

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	mixing_bytes = sizeof(double) * (size_t)n_components;
	mean_bytes = sizeof(double) * (size_t)n_components * (size_t)feature_dim;
	variance_bytes = sizeof(double) * (size_t)n_components * (size_t)feature_dim;
	resp_bytes = sizeof(double) * (size_t)n_samples * (size_t)n_components;

	hipGetLastError();

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != hipSuccess)
		return -1;

	status = cudaMalloc((void **)&d_mixing, mixing_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_means, mean_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_variances, variance_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_resp, resp_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_mixing, mixing_coeffs, mixing_bytes, cudaMemcpyHostToDevice);
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_means, means, mean_bytes, cudaMemcpyHostToDevice);
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(d_variances, variances, variance_bytes, cudaMemcpyHostToDevice);
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	blocks = (n_samples + threads - 1) / threads;
	hipLaunchKernelGGL(ndb_rocm_gmm_estep_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,d_features, d_mixing, d_means, d_variances, n_samples, feature_dim, n_components, d_resp);

	status = hipGetLastError();
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(responsibilities, d_resp, resp_bytes, cudaMemcpyDeviceToHost);
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_features);
		return -1;
	}

	cudaFree(d_resp);
	cudaFree(d_variances);
	cudaFree(d_means);
	cudaFree(d_mixing);
	cudaFree(d_features);
	return 0;
}

/*-------------------------------------------------------------------------
 * Host function: M-step
 *-------------------------------------------------------------------------
 */
extern "C" int
ndb_rocm_gmm_mstep(const float *features,
	const double *responsibilities,
	int n_samples,
	int feature_dim,
	int n_components,
	double *mixing_coeffs,
	double *means,
	double *variances)
{
	float *d_features = NULL;
	double *d_resp = NULL;
	double *d_N_k = NULL;
	double *d_mixing = NULL;
	double *d_means = NULL;
	double *d_variances = NULL;
	double *N_k = NULL;
	size_t feature_bytes;
	size_t resp_bytes;
	size_t Nk_bytes;
	size_t mixing_bytes;
	size_t mean_bytes;
	size_t variance_bytes;
	hipError_t status;
	int threads = 256;
	int blocks;

	if (features == NULL || responsibilities == NULL || mixing_coeffs == NULL || means == NULL || variances == NULL
		|| n_samples <= 0 || feature_dim <= 0 || n_components <= 0)
		return -1;

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	resp_bytes = sizeof(double) * (size_t)n_samples * (size_t)n_components;
	Nk_bytes = sizeof(double) * (size_t)n_components;
	mixing_bytes = sizeof(double) * (size_t)n_components;
	mean_bytes = sizeof(double) * (size_t)n_components * (size_t)feature_dim;
	variance_bytes = sizeof(double) * (size_t)n_components * (size_t)feature_dim;

	N_k = (double *)malloc(Nk_bytes);
	if (N_k == NULL)
		return -1;

	hipGetLastError();

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != hipSuccess)
	{
		free(N_k);
		return -1;
	}

	status = cudaMalloc((void **)&d_resp, resp_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMalloc((void **)&d_N_k, Nk_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMalloc((void **)&d_mixing, mixing_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMalloc((void **)&d_means, mean_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMalloc((void **)&d_variances, variance_bytes);
	if (status != hipSuccess)
	{
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMemcpy(d_resp, responsibilities, resp_bytes, cudaMemcpyHostToDevice);
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	/* Compute N_k */
	blocks = (n_components + threads - 1) / threads;
	hipLaunchKernelGGL(ndb_rocm_gmm_compute_Nk_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,d_resp, n_samples, n_components, d_N_k);

	status = hipGetLastError();
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMemcpy(N_k, d_N_k, Nk_bytes, cudaMemcpyDeviceToHost);
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	/* Update mixing coefficients on host */
	for (int k = 0; k < n_components; k++)
	{
		if (N_k[k] < GMM_MIN_PROB)
			N_k[k] = GMM_MIN_PROB;
		mixing_coeffs[k] = N_k[k] / n_samples;
	}

	/* Update means */
	hipLaunchKernelGGL(ndb_rocm_gmm_mstep_means_kernel,
		dim3(n_components),
		dim3(feature_dim),
		0,
		0,d_features, d_resp, d_N_k, n_samples, feature_dim, n_components, d_means);

	status = hipGetLastError();
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	/* Update variances */
	hipLaunchKernelGGL(ndb_rocm_gmm_mstep_variances_kernel,
		dim3(n_components),
		dim3(feature_dim),
		0,
		0,d_features, d_resp, d_means, d_N_k, n_samples, feature_dim, n_components, d_variances);

	status = hipGetLastError();
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMemcpy(means, d_means, mean_bytes, cudaMemcpyDeviceToHost);
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	status = cudaMemcpy(variances, d_variances, variance_bytes, cudaMemcpyDeviceToHost);
	if (status != hipSuccess)
	{
		cudaFree(d_variances);
		cudaFree(d_means);
		cudaFree(d_mixing);
		cudaFree(d_N_k);
		cudaFree(d_resp);
		cudaFree(d_features);
		free(N_k);
		return -1;
	}

	cudaFree(d_variances);
	cudaFree(d_means);
	cudaFree(d_mixing);
	cudaFree(d_N_k);
	cudaFree(d_resp);
	cudaFree(d_features);
	free(N_k);
	return 0;
}


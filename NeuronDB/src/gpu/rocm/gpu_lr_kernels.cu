/*-------------------------------------------------------------------------
 *
 * gpu_lr_kernels.cu
 *    HIP kernels for Logistic Regression training and inference on AMD GPUs
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_lr_kernels.cu
 *
 *-------------------------------------------------------------------------
 */

#ifdef NDB_GPU_HIP

#include <hip/hip_runtime.h>
#include <math.h>
#include "neurondb_cuda_lr.h"

__global__ static void
ndb_rocm_lr_forward_pass_kernel(const float *features,
	const float *weights,
	float bias,
	int n_samples,
	int feature_dim,
	double *outputs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples)
		return;

	const float *row = features + idx * feature_dim;
	double z = (double)bias;

	for (int j = 0; j < feature_dim; j++)
		z += (double)weights[j] * (double)row[j];

	outputs[idx] = z;
}

__global__ void
ndb_rocm_lr_sigmoid_kernel(const double *inputs, int n, double *outputs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	double z = inputs[idx];

	if (z > 500.0)
		outputs[idx] = 1.0;
	else if (z < -500.0)
		outputs[idx] = 0.0;
	else
		outputs[idx] = 1.0 / (1.0 + exp(-z));
}

/* Portable atomic add for double */
__device__ static double
ndb_atomicAdd_double(double *addr, double val)
{
	unsigned long long *addr_as_ull = (unsigned long long *)addr;
	unsigned long long old = *addr_as_ull;
	unsigned long long assumed;

	do
	{
		assumed = old;
		old = atomicCAS(addr_as_ull,
			assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);

	return __longlong_as_double(old);
}

__global__ static void
ndb_rocm_lr_compute_gradients_kernel(const float *features,
	const double *labels,
	const double *predictions,
	int n_samples,
	int feature_dim,
	double *grad_weights,
	double *grad_bias)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples)
		return;

	double error = predictions[idx] - labels[idx];
	const float *row = features + idx * feature_dim;

	ndb_atomicAdd_double(grad_bias, error);

	for (int j = 0; j < feature_dim; j++)
		ndb_atomicAdd_double(&grad_weights[j], error * (double)row[j]);
}

__global__ void
compute_errors_kernel(const double *predictions,
	const double *labels,
	float *errors,
	int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	errors[idx] = (float)(predictions[idx] - labels[idx]);
}

__global__ void
reduce_errors_bias_kernel(const float *errors,
	int n,
	double *out_sum)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double val = 0.0;

	if (idx < n)
		val = (double)errors[idx];

	sdata[tid] = val;
	__syncthreads();

	/* Reduction in shared memory */
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

	/* Write result for this block to global memory */
	if (tid == 0)
		ndb_atomicAdd_double(out_sum, sdata[0]);
}

extern "C" int
ndb_rocm_lr_reduce_errors_bias_gpu(const float *d_errors,
	int n,
	double *d_error_sum)
{
	int threads = 256;
	int blocks;
	size_t shared_mem;

	if (d_errors == NULL || d_error_sum == NULL || n <= 0)
		return (int)hipErrorInvalidValue;

	hipGetLastError();

	blocks = (n + threads - 1) / threads;
	if (blocks > 65535)
		blocks = 65535;

	shared_mem = sizeof(double) * threads;
	hipLaunchKernelGGL(reduce_errors_bias_kernel,
		dim3(blocks),
		dim3(threads),
		shared_mem,
		0,
		d_errors, n, d_error_sum);

	return (int)hipGetLastError();
}

__global__ void
update_weights_kernel(float *weights,
	const float *grad_weights,
	float learning_rate,
	float lambda,
	int feature_dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= feature_dim)
		return;

	float grad = grad_weights[idx];
	float w = weights[idx];
	float new_w = w - learning_rate * (grad + lambda * w);

	if (new_w > 1000.0f)
		new_w = 1000.0f;
	else if (new_w < -1000.0f)
		new_w = -1000.0f;

	weights[idx] = new_w;
}

__global__ void
update_bias_kernel(double *d_bias, double bias_update)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		ndb_atomicAdd_double(d_bias, bias_update);
	}
}

extern "C" int
ndb_rocm_lr_update_weights_gpu(float *d_weights,
	const float *d_grad_weights,
	float learning_rate,
	float lambda,
	int feature_dim,
	double *d_bias,
	double grad_bias)
{
	int threads = 256;
	int blocks;
	hipError_t status;

	if (d_weights == NULL || d_grad_weights == NULL || feature_dim <= 0)
		return (int)hipErrorInvalidValue;

	hipGetLastError();

	blocks = (feature_dim + threads - 1) / threads;
	hipLaunchKernelGGL(update_weights_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_weights, d_grad_weights, learning_rate, lambda, feature_dim);

	status = hipGetLastError();
	if (status != hipSuccess)
		return (int)status;

	if (d_bias != NULL)
	{
		double bias_update = -learning_rate * grad_bias;
		hipLaunchKernelGGL(update_bias_kernel,
			dim3(1),
			dim3(1),
			0,
			0,
			d_bias, bias_update);
		status = hipGetLastError();
		if (status != hipSuccess)
			return (int)status;
	}

	return (int)hipSuccess;
}

__global__ void
convert_z_add_bias_kernel(const float *d_z_float,
	double bias,
	int n,
	double *d_z_double)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	double z_val = (double)d_z_float[idx] + bias;
	if (z_val > 500.0)
		d_z_double[idx] = 500.0;
	else if (z_val < -500.0)
		d_z_double[idx] = -500.0;
	else
		d_z_double[idx] = z_val;
}

extern "C" int
ndb_rocm_lr_convert_z_add_bias_gpu(const float *d_z_float,
	double bias,
	int n,
	double *d_z_double)
{
	int threads = 256;
	int blocks;

	if (d_z_float == NULL || d_z_double == NULL || n <= 0)
		return (int)hipErrorInvalidValue;

	hipGetLastError();

	blocks = (n + threads - 1) / threads;
	hipLaunchKernelGGL(convert_z_add_bias_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_z_float, bias, n, d_z_double);

	return (int)hipGetLastError();
}

extern "C" int
ndb_rocm_lr_compute_errors_gpu(const double *d_predictions,
	const double *d_labels,
	float *d_errors,
	int n)
{
	int threads = 256;
	int blocks;

	if (d_predictions == NULL || d_labels == NULL || d_errors == NULL || n <= 0)
		return (int)hipErrorInvalidValue;

	hipGetLastError();

	blocks = (n + threads - 1) / threads;
	hipLaunchKernelGGL(compute_errors_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_predictions, d_labels, d_errors, n);

	return (int)hipGetLastError();
}

extern "C" int
ndb_rocm_lr_forward_pass_gpu(const float *d_features,
	const float *d_weights,
	float bias,
	int n_samples,
	int feature_dim,
	double *d_outputs)
{
	hipError_t status;
	int threads = 256;
	int blocks;

	if (d_features == NULL || d_weights == NULL || d_outputs == NULL
		|| n_samples <= 0 || feature_dim <= 0)
		return -1;

	hipGetLastError();

	blocks = (n_samples + threads - 1) / threads;

	hipLaunchKernelGGL(ndb_rocm_lr_forward_pass_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_features, d_weights, bias, n_samples, feature_dim, d_outputs);

	status = hipGetLastError();
	if (status != hipSuccess)
		return -1;

	return 0;
}

extern "C" int
ndb_rocm_lr_sigmoid_gpu(const double *d_in, int n, double *d_out)
{
	int threads = 256;
	int blocks;

	if (d_in == NULL || d_out == NULL || n <= 0)
		return (int)hipErrorInvalidValue;

	hipGetLastError();

	blocks = (n + threads - 1) / threads;
	hipLaunchKernelGGL(ndb_rocm_lr_sigmoid_kernel,
		dim3(blocks),
		dim3(threads),
		0,
		0,
		d_in, n, d_out);

	return (int)hipGetLastError();
}

__global__ void
ndb_rocm_lr_eval_kernel(const float *features,
	const double *labels,
	const double *weights,
	double bias,
	double threshold,
	int n_samples,
	int feature_dim,
	long long *tp_out,
	long long *tn_out,
	long long *fp_out,
	long long *fn_out,
	double *log_loss_out,
	long long *count_out)
{
	extern __shared__ double w_shared[];
	int tid = threadIdx.x;

	/* Load weights into shared memory */
	for (int i = tid; i < feature_dim; i += blockDim.x)
		w_shared[i] = weights[i];
	__syncthreads();

	long long local_tp = 0;
	long long local_tn = 0;
	long long local_fp = 0;
	long long local_fn = 0;
	double local_log_loss = 0.0;
	long long local_count = 0;

	int stride = blockDim.x * gridDim.x;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (; sample_idx < n_samples; sample_idx += stride)
	{
		double z = bias;
		const float *feat_row = features + (sample_idx * feature_dim);

		for (int i = 0; i < feature_dim; i++)
		{
			z += w_shared[i] * (double)feat_row[i];
		}

		double probability;
		if (z > 500.0)
			probability = 1.0;
		else if (z < -500.0)
			probability = 0.0;
		else
			probability = 1.0 / (1.0 + exp(-z));

		double y_true = labels[sample_idx];
		int y_true_class = (y_true > 0.5) ? 1 : 0;
		int y_pred_class = (probability >= threshold) ? 1 : 0;

		if (y_true_class == 1)
		{
			if (y_pred_class == 1)
				local_tp++;
			else
				local_fn++;
		}
		else
		{
			if (y_pred_class == 1)
				local_fp++;
			else
				local_tn++;
		}

		probability = fmax(1e-15, fmin(1.0 - 1e-15, probability));
		if (y_true > 0.5)
			local_log_loss -= log(probability);
		else
			local_log_loss -= log(1.0 - probability);

		local_count++;
	}

	if (local_tp > 0)
		atomicAdd((unsigned long long *)tp_out, (unsigned long long)local_tp);
	if (local_tn > 0)
		atomicAdd((unsigned long long *)tn_out, (unsigned long long)local_tn);
	if (local_fp > 0)
		atomicAdd((unsigned long long *)fp_out, (unsigned long long)local_fp);
	if (local_fn > 0)
		atomicAdd((unsigned long long *)fn_out, (unsigned long long)local_fn);
	if (local_log_loss != 0.0)
		ndb_atomicAdd_double(log_loss_out, local_log_loss);
	if (local_count > 0)
		atomicAdd((unsigned long long *)count_out, (unsigned long long)local_count);
}

extern "C" hipError_t
launch_lr_eval_kernel_hip(const float *features,
	const double *labels,
	const double *weights,
	double bias,
	double threshold,
	int n_samples,
	int feature_dim,
	long long *tp_out,
	long long *tn_out,
	long long *fp_out,
	long long *fn_out,
	double *log_loss_out,
	long long *count_out)
{
	if (n_samples <= 0)
		return hipErrorInvalidValue;

	if (features == NULL || labels == NULL || weights == NULL ||
		tp_out == NULL || tn_out == NULL || fp_out == NULL || fn_out == NULL ||
		log_loss_out == NULL || count_out == NULL)
		return hipErrorInvalidValue;

	hipGetLastError();

	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;

	if (blocks > 65535)
		blocks = 65535;

	size_t shared_mem_size = sizeof(double) * (size_t)feature_dim;

	hipLaunchKernelGGL(ndb_rocm_lr_eval_kernel,
		dim3(blocks),
		dim3(threads_per_block),
		shared_mem_size,
		0,
		features,
		labels,
		weights,
		bias,
		threshold,
		n_samples,
		feature_dim,
		tp_out,
		tn_out,
		fp_out,
		fn_out,
		log_loss_out,
		count_out);

	return hipGetLastError();
}

#endif /* NDB_GPU_HIP */


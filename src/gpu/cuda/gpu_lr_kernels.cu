/*
 * gpu_lr_kernels.cu
 *    CUDA kernels for Logistic Regression training and inference.
 */

#include <cuda_runtime.h>
#include <math.h>

#include "neurondb_cuda_lr.h"

__global__ static void
ndb_cuda_lr_forward_pass_kernel(const float *features,
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
ndb_cuda_lr_sigmoid_kernel(const double *inputs, int n, double *outputs)
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

/*
 * ndb_atomicAdd_double
 *    Thread-safe atomic addition for double-precision floating point values.
 *
 * This function provides portable atomic addition for double values on GPU
 * architectures that do not natively support atomic operations on doubles.
 * CUDA introduced native atomicAdd for doubles in the Pascal architecture,
 * but older GPUs require this emulation using compare-and-swap loops. The
 * function uses the atomicCAS primitive on unsigned long long integers to
 * implement atomic double addition. It converts the double value to its
 * integer representation, performs an atomic compare-and-swap operation
 * in a loop until successful, and converts back to double. This pattern
 * ensures thread safety when multiple GPU threads simultaneously update
 * the same memory location during gradient accumulation in parallel
 * machine learning operations. The loop continues until the compare-and-swap
 * succeeds, guaranteeing that all updates are correctly applied without
 * race conditions.
 */
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

/*
 * ndb_cuda_lr_compute_gradients_kernel
 *    GPU kernel to compute gradients for logistic regression training.
 *
 * This kernel computes gradients for logistic regression weight updates
 * by processing multiple training samples in parallel across GPU threads.
 * Each thread handles one training sample, computing the prediction error
 * as the difference between predicted and actual labels, then accumulating
 * this error into global gradient arrays using atomic operations. The
 * gradient for the bias term is simply the sum of errors across all samples,
 * while the gradient for each weight is the sum of errors multiplied by the
 * corresponding feature value. Atomic operations are necessary because
 * multiple threads may update the same gradient element simultaneously when
 * processing different samples. The function uses portable atomic double
 * addition to ensure correctness on all GPU architectures, enabling
 * efficient parallel gradient computation that scales with the number of
 * available GPU cores.
 */
__global__ static void
ndb_cuda_lr_compute_gradients_kernel(const float *features,
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

/* Device kernel to compute errors = predictions - labels (as float) */
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

/* Device kernel to reduce errors for grad_bias computation */
/* Uses shared memory reduction pattern */
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

/* Host wrapper for error reduction kernel */
extern "C" int
ndb_cuda_lr_reduce_errors_bias_gpu(const float *d_errors,
	int n,
	double *d_error_sum)
{
	int threads = 256;
	int blocks;
	size_t shared_mem;

	if (d_errors == NULL || d_error_sum == NULL || n <= 0)
		return (int)cudaErrorInvalidValue;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (n + threads - 1) / threads;
	/* Cap blocks to maximum grid dimension */
	if (blocks > 65535)
		blocks = 65535;

	shared_mem = sizeof(double) * threads;
	reduce_errors_bias_kernel<<<blocks, threads, shared_mem>>>(
		d_errors, n, d_error_sum);

	return (int)cudaGetLastError();
}

/* Device kernel to update weights: w = w - lr * grad + lambda * w (L2 regularization) */
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
	
	/* Update: w = w - lr * grad + lambda * w */
	/* Simplified: w = w * (1 - lambda) - lr * grad */
	/* For L2 regularization: w = w - lr * (grad + lambda * w) */
	float new_w = w - learning_rate * (grad + lambda * w);

	/* Clamp to prevent overflow/underflow */
	if (new_w > 1000.0f)
		new_w = 1000.0f;
	else if (new_w < -1000.0f)
		new_w = -1000.0f;

	weights[idx] = new_w;
}

/* Device kernel to update bias */
__global__ void
update_bias_kernel(double *d_bias, double bias_update)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		ndb_atomicAdd_double(d_bias, bias_update);
	}
}

/* Host wrapper for weight update kernel */
extern "C" int
ndb_cuda_lr_update_weights_gpu(float *d_weights,
	const float *d_grad_weights,
	float learning_rate,
	float lambda,
	int feature_dim,
	double *d_bias,
	double grad_bias)
{
	int threads = 256;
	int blocks;
	cudaError_t status;

	if (d_weights == NULL || d_grad_weights == NULL || feature_dim <= 0)
		return (int)cudaErrorInvalidValue;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (feature_dim + threads - 1) / threads;
	update_weights_kernel<<<blocks, threads>>>(
		d_weights, d_grad_weights, learning_rate, lambda, feature_dim);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		return (int)status;

	/* Update bias on device if provided */
	if (d_bias != NULL)
	{
		/* Use device kernel to update bias atomically */
		double bias_update = -learning_rate * grad_bias;
		update_bias_kernel<<<1, 1>>>(d_bias, bias_update);
		status = cudaGetLastError();
		if (status != cudaSuccess)
			return (int)status;
	}

	return (int)cudaSuccess;
}

/* GPU kernel to convert float z to double z and add bias on device */
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
	/* Defensive: Clamp to reasonable range if non-finite */
	if (z_val > 500.0)
		d_z_double[idx] = 500.0;
	else if (z_val < -500.0)
		d_z_double[idx] = -500.0;
	else
		d_z_double[idx] = z_val;
}

/* Host wrapper for convert_z_add_bias_kernel */
extern "C" int
ndb_cuda_lr_convert_z_add_bias_gpu(const float *d_z_float,
	double bias,
	int n,
	double *d_z_double)
{
	int threads = 256;
	int blocks;

	if (d_z_float == NULL || d_z_double == NULL || n <= 0)
		return (int)cudaErrorInvalidValue;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (n + threads - 1) / threads;
	convert_z_add_bias_kernel<<<blocks, threads>>>(d_z_float, bias, n, d_z_double);

	return (int)cudaGetLastError();
}

/* Device-only wrapper for error computation kernel */
extern "C" int
ndb_cuda_lr_compute_errors_gpu(const double *d_predictions,
	const double *d_labels,
	float *d_errors,
	int n)
{
	int threads = 256;
	int blocks;

	if (d_predictions == NULL || d_labels == NULL || d_errors == NULL || n <= 0)
		return (int)cudaErrorInvalidValue;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (n + threads - 1) / threads;
	compute_errors_kernel<<<blocks, threads>>>(d_predictions, d_labels, d_errors, n);

	return (int)cudaGetLastError();
}

extern "C" int
ndb_cuda_lr_forward_pass(const float *features,
	const double *weights,
	double bias,
	int n_samples,
	int feature_dim,
	double *outputs)
{
	float *d_features = NULL;
	float *d_weights = NULL;
	double *d_outputs = NULL;
	size_t feature_bytes;
	size_t weight_bytes;
	size_t output_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (features == NULL || weights == NULL || outputs == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		/* Use elog via PostgreSQL's error reporting - but this is CUDA code, so we can't */
		/* Just return -1 and let caller report */
		return -1;
	}

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	weight_bytes = sizeof(float) * (size_t)feature_dim;
	output_bytes = sizeof(double) * (size_t)n_samples;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != cudaSuccess)
	{
		return -1;
	}

	status = cudaMalloc((void **)&d_weights, weight_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_features);
		return -1;
	}

	status = cudaMalloc((void **)&d_outputs, output_bytes);
	if (status != cudaSuccess)
	{
		cudaFree(d_weights);
		cudaFree(d_features);
		return -1;
	}

	status = cudaMemcpy(
		d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		goto cleanup_outputs;
	}

	{
		float *h_weights = (float *)malloc(weight_bytes);
		int i;

		for (i = 0; i < feature_dim; i++)
			h_weights[i] = (float)weights[i];
		status = cudaMemcpy(d_weights,
			h_weights,
			weight_bytes,
			cudaMemcpyHostToDevice);
		free(h_weights);
		if (status != cudaSuccess)
			goto cleanup_outputs;
	}

	blocks = (n_samples + threads - 1) / threads;
	ndb_cuda_lr_forward_pass_kernel<<<blocks, threads>>>(d_features,
		d_weights,
		(float)bias,
		n_samples,
		feature_dim,
		d_outputs);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		goto cleanup_outputs;
	}

	status = cudaMemcpy(
		outputs, d_outputs, output_bytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		goto cleanup_outputs;
	}

cleanup_outputs:
	if (d_outputs != NULL)
		cudaFree(d_outputs);
	if (d_weights != NULL)
		cudaFree(d_weights);
	if (d_features != NULL)
		cudaFree(d_features);

	if (status == cudaSuccess)
		return 0;
	else
		return -1;
}

/* Host-side wrapper for launching forward pass kernel with pre-allocated GPU memory */
extern "C" int
ndb_cuda_lr_forward_pass_gpu(const float *d_features,
	const float *d_weights,
	float bias,
	int n_samples,
	int feature_dim,
	double *d_outputs)
{
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (d_features == NULL || d_weights == NULL || d_outputs == NULL
		|| n_samples <= 0 || feature_dim <= 0)
		return -1;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (n_samples + threads - 1) / threads;

	ndb_cuda_lr_forward_pass_kernel<<<blocks, threads>>>(
		d_features, d_weights, bias, n_samples, feature_dim, d_outputs);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		return -1;

	return 0;
}

/* Device-only wrapper for sigmoid kernel (takes device pointers) */
extern "C" int
ndb_cuda_lr_sigmoid_gpu(const double *d_in, int n, double *d_out)
{
	int threads = 256;
	int blocks;

	if (d_in == NULL || d_out == NULL || n <= 0)
		return (int)cudaErrorInvalidValue;

	/* Clear any previous CUDA errors */
	cudaGetLastError();

	blocks = (n + threads - 1) / threads;
	ndb_cuda_lr_sigmoid_kernel<<<blocks, threads>>>(d_in, n, d_out);

	return (int)cudaGetLastError();
}

/*
 * CUDA kernel for batch evaluation: computes predictions with sigmoid and accumulates metrics
 * Each thread processes multiple samples, accumulates in registers, then does atomic adds
 * 
 * Inputs:
 *   features: [n_samples, feature_dim] - feature matrix (row-major, float)
 *   labels: [n_samples] - true labels (double, 0.0 or 1.0)
 *   weights: [feature_dim] - model weights (double)
 *   bias: scalar bias term (double)
 *   threshold: classification threshold (double, typically 0.5)
 * 
 * Outputs (atomic accumulators):
 *   tp_out: true positives (long long)
 *   tn_out: true negatives (long long)
 *   fp_out: false positives (long long)
 *   fn_out: false negatives (long long)
 *   log_loss_out: sum of log loss (double)
 *   count_out: number of samples processed (long long)
 */
__global__ void
ndb_cuda_lr_eval_kernel(const float *features,
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
	/* Shared memory for weights (loaded once per block) */
	extern __shared__ double w_shared[];
	int tid = threadIdx.x;
	
	/* Load weights into shared memory cooperatively */
	for (int i = tid; i < feature_dim; i += blockDim.x)
		w_shared[i] = weights[i];
	__syncthreads();
	
	/* Per-thread local accumulators in registers (fastest) */
	long long local_tp = 0;
	long long local_tn = 0;
	long long local_fp = 0;
	long long local_fn = 0;
	double local_log_loss = 0.0;
	long long local_count = 0;
	
	/* Grid-stride loop: each thread processes every stride-th sample */
	int stride = blockDim.x * gridDim.x;
	int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	/* Process samples assigned to this thread */
	for (; sample_idx < n_samples; sample_idx += stride)
	{
		/* Compute z = bias + sum(weights[i] * features[i]) */
		double z = bias;
		const float *feat_row = features + (sample_idx * feature_dim);
		
		/* Dot product: weights * features (using shared memory) */
		for (int i = 0; i < feature_dim; i++)
		{
			z += w_shared[i] * (double)feat_row[i];
		}
		
		/* Apply sigmoid */
		double probability;
		if (z > 500.0)
			probability = 1.0;
		else if (z < -500.0)
			probability = 0.0;
		else
			probability = 1.0 / (1.0 + exp(-z));
		
		/* Get true label */
		double y_true = labels[sample_idx];
		int y_true_class = (y_true > 0.5) ? 1 : 0;
		
		/* Apply threshold to get predicted class */
		int y_pred_class = (probability >= threshold) ? 1 : 0;
		
		/* Update confusion matrix */
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
		
		/* Compute log loss: -[y*log(p) + (1-y)*log(1-p)] */
		/* Clamp probability to avoid log(0) */
		probability = fmax(1e-15, fmin(1.0 - 1e-15, probability));
		if (y_true > 0.5)
			local_log_loss -= log(probability);
		else
			local_log_loss -= log(1.0 - probability);
		
		local_count++;
	}
	
	/* Single atomic add per thread per metric */
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

/*
 * Host launch wrapper for lr_eval_kernel
 * 
 * Returns cudaSuccess on success, or cudaError_* on failure
 */
extern "C" cudaError_t
launch_lr_eval_kernel(const float *features,
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
	/* Parameter validation */
	if (n_samples <= 0)
	{
		return cudaErrorInvalidValue;
	}
	
	if (features == NULL || labels == NULL || weights == NULL ||
		tp_out == NULL || tn_out == NULL || fp_out == NULL || fn_out == NULL ||
		log_loss_out == NULL || count_out == NULL)
	{
		return cudaErrorInvalidValue;
	}
	
	/* Clear any previous CUDA errors */
	cudaGetLastError();
	
	/* Configure kernel launch */
	int threads_per_block = 256;
	int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
	
	/* Cap blocks to maximum grid dimension */
	if (blocks > 65535)
		blocks = 65535;
	
	/* Allocate shared memory for weights (one double per feature) */
	size_t shared_mem_size = sizeof(double) * (size_t)feature_dim;
	
	/* Launch kernel with shared memory */
	ndb_cuda_lr_eval_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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
	
	return cudaGetLastError();
}

extern "C" int
ndb_cuda_lr_sigmoid(const double *inputs, int n, double *outputs)
{
	double *d_inputs = NULL;
	double *d_outputs = NULL;
	size_t input_bytes;
	size_t output_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (inputs == NULL || outputs == NULL || n <= 0)
		return -1;

	input_bytes = sizeof(double) * (size_t)n;
	output_bytes = sizeof(double) * (size_t)n;

	status = cudaMalloc((void **)&d_inputs, input_bytes);
	if (status != cudaSuccess)
		return -1;
	status = cudaMalloc((void **)&d_outputs, output_bytes);
	if (status != cudaSuccess)
		goto cleanup_inputs;

	status = cudaMemcpy(
		d_inputs, inputs, input_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_outputs;

	blocks = (n + threads - 1) / threads;
	ndb_cuda_lr_sigmoid_kernel<<<blocks, threads>>>(d_inputs, n, d_outputs);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto cleanup_outputs;

	status = cudaMemcpy(
		outputs, d_outputs, output_bytes, cudaMemcpyDeviceToHost);

cleanup_outputs:
	cudaFree(d_outputs);
cleanup_inputs:
	cudaFree(d_inputs);

	return (status == cudaSuccess) ? 0 : -1;
}

extern "C" int
ndb_cuda_lr_compute_gradients(const float *features,
	const double *labels,
	const double *predictions,
	int n_samples,
	int feature_dim,
	double *grad_weights,
	double *grad_bias)
{
	float *d_features = NULL;
	double *d_labels = NULL;
	double *d_predictions = NULL;
	double *d_grad_weights = NULL;
	double *d_grad_bias = NULL;
	size_t feature_bytes;
	size_t label_bytes;
	size_t pred_bytes;
	size_t grad_weight_bytes;
	size_t grad_bias_bytes;
	cudaError_t status;
	int threads = 256;
	int blocks;

	if (features == NULL || labels == NULL || predictions == NULL
		|| grad_weights == NULL || grad_bias == NULL || n_samples <= 0
		|| feature_dim <= 0)
		return -1;

	feature_bytes = sizeof(float) * (size_t)n_samples * (size_t)feature_dim;
	label_bytes = sizeof(double) * (size_t)n_samples;
	pred_bytes = sizeof(double) * (size_t)n_samples;
	grad_weight_bytes = sizeof(double) * (size_t)feature_dim;
	grad_bias_bytes = sizeof(double);

	memset(grad_weights, 0, grad_weight_bytes);
	*grad_bias = 0.0;

	status = cudaMalloc((void **)&d_features, feature_bytes);
	if (status != cudaSuccess)
		return -1;
	status = cudaMalloc((void **)&d_labels, label_bytes);
	if (status != cudaSuccess)
		goto cleanup_features;
	status = cudaMalloc((void **)&d_predictions, pred_bytes);
	if (status != cudaSuccess)
		goto cleanup_labels;
	status = cudaMalloc((void **)&d_grad_weights, grad_weight_bytes);
	if (status != cudaSuccess)
		goto cleanup_predictions;
	status = cudaMalloc((void **)&d_grad_bias, grad_bias_bytes);
	if (status != cudaSuccess)
		goto cleanup_grad_weights;

	status = cudaMemcpy(
		d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemcpy(
		d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemcpy(
		d_predictions, predictions, pred_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;

	status = cudaMemset(d_grad_weights, 0, grad_weight_bytes);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemset(d_grad_bias, 0, grad_bias_bytes);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;

	blocks = (n_samples + threads - 1) / threads;
	ndb_cuda_lr_compute_gradients_kernel<<<blocks, threads>>>(d_features,
		d_labels,
		d_predictions,
		n_samples,
		feature_dim,
		d_grad_weights,
		d_grad_bias);

	status = cudaGetLastError();
	if (status != cudaSuccess)
		goto cleanup_grad_bias;

	status = cudaMemcpy(grad_weights,
		d_grad_weights,
		grad_weight_bytes,
		cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto cleanup_grad_bias;
	status = cudaMemcpy(grad_bias,
		d_grad_bias,
		grad_bias_bytes,
		cudaMemcpyDeviceToHost);

cleanup_grad_bias:
	cudaFree(d_grad_bias);
cleanup_grad_weights:
	cudaFree(d_grad_weights);
cleanup_predictions:
	cudaFree(d_predictions);
cleanup_labels:
	cudaFree(d_labels);
cleanup_features:
	cudaFree(d_features);

	return (status == cudaSuccess) ? 0 : -1;
}

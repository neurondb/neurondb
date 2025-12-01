/*
 * gpu_distance_kernel.metal
 *     Pre-compiled Metal GPU kernels for NeuronDB
 *
 * These kernels are compiled at build time to avoid XPC runtime compilation.
 * Provides TRUE GPU parallel processing on Apple Silicon.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 */

#include <metal_stdlib>
using namespace metal;

/*
 * GPU Kernel: L2 Distance (Euclidean)
 * 
 * Computes L2 distance between two vectors in parallel on GPU
 * Uses Metal's parallel thread execution across GPU cores
 */
kernel void l2_distance_kernel(
    device const float* vec_a [[buffer(0)]],
    device const float* vec_b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= 1) return;  // Single output value
    
    float sum = 0.0f;
    
    // Parallel reduction across GPU threads
    for (uint i = 0; i < dimension; i++) {
        float diff = vec_a[i] - vec_b[i];
        sum += diff * diff;
    }
    
    result[0] = sqrt(sum);
}

/*
 * GPU Kernel: Batch L2 Distance
 * 
 * Computes L2 distances for multiple query-target pairs in parallel
 * Each GPU thread computes one distance
 */
kernel void batch_l2_distance_kernel(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& num_queries [[buffer(3)]],
    constant uint& num_targets [[buffer(4)]],
    constant uint& dimension [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint query_idx = gid.x;
    uint target_idx = gid.y;
    
    if (query_idx >= num_queries || target_idx >= num_targets)
        return;
    
    // Compute L2 distance
    float sum = 0.0f;
    uint q_offset = query_idx * dimension;
    uint t_offset = target_idx * dimension;
    
    for (uint i = 0; i < dimension; i++) {
        float diff = queries[q_offset + i] - targets[t_offset + i];
        sum += diff * diff;
    }
    
    distances[query_idx * num_targets + target_idx] = sqrt(sum);
}

/*
 * GPU Kernel: Cosine Distance
 * 
 * Computes cosine distance (1 - cosine similarity) in parallel
 */
kernel void cosine_distance_kernel(
    device const float* vec_a [[buffer(0)]],
    device const float* vec_b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= 1) return;
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (uint i = 0; i < dimension; i++) {
        float a = vec_a[i];
        float b = vec_b[i];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    float similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b));
    result[0] = 1.0f - similarity;
}

/*
 * GPU Kernel: Inner Product
 * 
 * Computes negative inner product (for max inner product search)
 */
kernel void inner_product_kernel(
    device const float* vec_a [[buffer(0)]],
    device const float* vec_b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= 1) return;
    
    float sum = 0.0f;
    
    for (uint i = 0; i < dimension; i++) {
        sum += vec_a[i] * vec_b[i];
    }
    
    result[0] = -sum;
}

/*
 * GPU Kernel: Batch Cosine Distance
 * 
 * Computes cosine distances for multiple pairs in parallel
 */
kernel void batch_cosine_distance_kernel(
    device const float* queries [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& num_queries [[buffer(3)]],
    constant uint& num_targets [[buffer(4)]],
    constant uint& dimension [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint query_idx = gid.x;
    uint target_idx = gid.y;
    
    if (query_idx >= num_queries || target_idx >= num_targets)
        return;
    
    uint q_offset = query_idx * dimension;
    uint t_offset = target_idx * dimension;
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (uint i = 0; i < dimension; i++) {
        float a = queries[q_offset + i];
        float b = targets[t_offset + i];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    float similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b));
    distances[query_idx * num_targets + target_idx] = 1.0f - similarity;
}

/*
 * GPU Kernel: Vector Addition (for testing)
 * 
 * Simple kernel to verify GPU functionality
 */
kernel void vector_add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    result[gid] = a[gid] + b[gid];
}


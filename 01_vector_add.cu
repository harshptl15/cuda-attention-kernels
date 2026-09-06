// 01_vector_add.cu
//
// Element-wise vector addition. Baseline kernel covering the fundamentals:
// thread indexing, host/device memory management, and correctness checking.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Always check CUDA calls. Silent failures are the #1 source of confusing bugs.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// The kernel: each thread computes exactly one output element.
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20; // ~1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());     // catches launch-time errors
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime errors

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Correctness check against the CPU
    double maxErr = 0.0;
    for (int i = 0; i < N; ++i) {
        double expected = (double)h_a[i] + (double)h_b[i];
        maxErr = fmax(maxErr, fabs(expected - h_c[i]));
    }
    printf("N = %d\n", N);
    printf("Grid: %d blocks x %d threads\n", blocksPerGrid, threadsPerBlock);
    printf("Max error vs CPU: %e\n", maxErr);
    printf(maxErr < 1e-5 ? "PASSED\n" : "FAILED\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);

    return 0;
}

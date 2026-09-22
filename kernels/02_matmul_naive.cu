// 02_matmul_naive.cu
//
// Naive matrix multiply: no shared memory, no tiling. Every thread reads its
// full row/column straight from global memory. This is the baseline that
// 03_matmul_tiled.cu improves on.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// --- Timing harness -------------------------------------------------------
// A T4 idles at P8 with its SM clock parked at 300 MHz and needs on the order
// of a second of sustained load to reach boost. A fixed launch-count warmup is
// nowhere near enough, and the error is large: this project's attention kernel
// measured 5.996 ms when it ran straight after an nvcc compile (GPU cold) and
// 2.469 ms run back-to-back with the other benchmarks (GPU hot) -- a 2.4x
// swing from the identical binary on the identical input. cuBLAS on the same
// machine ranged over 3072-6160 GFLOP/s for the same reason.
//
// So: warm up by wall-clock time rather than by launch count, and report the
// median of several timed blocks plus the observed spread, so a run that is
// still unstable says so in its own output instead of looking authoritative.
#define WARMUP_MS 1500.0
#define ITERS     20
#define REPEATS   5

template <typename LaunchFn>
static float benchmarkMs(LaunchFn launch, float* spreadPct) {
    auto t0 = std::chrono::steady_clock::now();
    do {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
    } while (std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count() < WARMUP_MS);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms / ITERS);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    float median = samples[REPEATS / 2];
    if (spreadPct) *spreadPct = 100.0f * (samples.back() - samples.front()) / median;
    return median;
}

// C = A * B, all square N x N, row-major.
// One thread computes one output element C[row][col].
__global__ void matmulNaive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmulCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

int main() {
    const int N = 512; // kept modest so the CPU reference check doesn't take forever
    const size_t bytes = (size_t)N * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    float msSpread = 0.0f;
    float ms = benchmarkMs([&]{
        matmulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }, &msSpread);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU reference (this is the slow part — O(N^3) on one core)
    matmulCPU(h_A, h_B, h_C_ref, N);

    double maxErr = 0.0;
    for (int i = 0; i < N * N; ++i) {
        maxErr = fmax(maxErr, fabs((double)h_C[i] - (double)h_C_ref[i]));
    }

    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;

    printf("N = %d x %d\n", N, N);
    printf("Naive matmul time: %.3f ms  [median of %d, spread %.1f%%]\n",
           ms, REPEATS, msSpread);
    printf("Approx throughput: %.2f GFLOP/s\n", gflops);
    printf("Max error vs CPU: %e\n", maxErr);
    printf(maxErr < 1e-2 ? "PASSED\n" : "FAILED\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_C_ref);

    return 0;
}

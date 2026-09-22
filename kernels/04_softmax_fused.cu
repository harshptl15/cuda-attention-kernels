// 04_softmax_fused.cu
//
// Row-wise softmax using the online/streaming algorithm: a single pass that
// tracks a running max and running sum, rescaling as the max updates, plus
// warp-shuffle reductions instead of shared-memory reductions.
//
// The same running-max/running-sum rescaling is reused in the fused attention
// kernel (05), there with an added weighted-sum-over-V accumulator.

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

#define WARP_SIZE 32

// One warp handles one row. Each thread in the warp strides across the row,
// maintaining its own (local_max, local_sum) via the online-softmax update
// rule, then the warp combines its 32 partial results with __shfl_down_sync
// instead of a shared-memory reduction (no __syncthreads needed within a warp).
__global__ void softmaxFusedWarp(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int rows, int cols) {
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= rows) return;

    const float* rowPtr = input + (size_t)row * cols;
    float* outPtr = output + (size_t)row * cols;

    // Pass 1: each lane streams through its slice of the row, maintaining a
    // running max (m) and running sum (l) of exp(x - m), rescaling l whenever
    // m updates. This is the "online softmax" trick — no separate max-pass needed.
    float m = -INFINITY;
    float l = 0.0f;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        float x = rowPtr[c];
        float m_new = fmaxf(m, x);
        l = l * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    // Combine all 32 lanes' (m, l) pairs into one row-wide (m, l) via warp shuffle.
    // Classic butterfly reduction: each step halves the number of "active" values.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_m = __shfl_down_sync(0xffffffff, m, offset);
        float other_l = __shfl_down_sync(0xffffffff, l, offset);
        float new_m = fmaxf(m, other_m);
        l = l * expf(m - new_m) + other_l * expf(other_m - new_m);
        m = new_m;
    }
    // After the reduction, lane 0 holds the row-wide (m, l). Broadcast it back.
    m = __shfl_sync(0xffffffff, m, 0);
    l = __shfl_sync(0xffffffff, l, 0);

    // Pass 2: write out exp(x - m) / l for this lane's slice of the row.
    for (int c = lane; c < cols; c += WARP_SIZE) {
        outPtr[c] = expf(rowPtr[c] - m) / l;
    }
}

void softmaxCPU(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* in = input + (size_t)r * cols;
        float* out = output + (size_t)r * cols;
        float maxVal = -INFINITY;
        for (int c = 0; c < cols; ++c) maxVal = fmaxf(maxVal, in[c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) { out[c] = expf(in[c] - maxVal); sum += out[c]; }
        for (int c = 0; c < cols; ++c) out[c] /= sum;
    }
}

int main() {
    const int ROWS = 4096;
    const int COLS = 1024;
    const size_t bytes = (size_t)ROWS * COLS * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < ROWS * COLS; ++i)
        h_in[i] = (static_cast<float>(rand()) / RAND_MAX) * 20.0f - 10.0f; // range [-10, 10]

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 4 warps per block -> 128 threads per block, each block handles 4 rows.
    int warpsPerBlock = 4;
    int threadsPerBlock = warpsPerBlock * WARP_SIZE;
    int blocks = (ROWS + warpsPerBlock - 1) / warpsPerBlock;

    float msSpread = 0.0f;
    float ms = benchmarkMs([&]{
        softmaxFusedWarp<<<blocks, threadsPerBlock>>>(d_in, d_out, ROWS, COLS);
    }, &msSpread);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    softmaxCPU(h_in, h_ref, ROWS, COLS);

    double maxErr = 0.0;
    for (int i = 0; i < ROWS * COLS; ++i)
        maxErr = fmax(maxErr, fabs((double)h_out[i] - (double)h_ref[i]));

    double gbMoved = 2.0 * bytes / 1e9; // read once, write once
    double gbps = gbMoved / (ms / 1000.0);

    printf("Rows x Cols = %d x %d\n", ROWS, COLS);
    printf("Fused softmax time: %.3f ms  [median of %d, spread %.1f%%]\n",
           ms, REPEATS, msSpread);
    printf("Approx bandwidth: %.2f GB/s\n", gbps);
    printf("Max error vs CPU: %e\n", maxErr);
    printf(maxErr < 1e-4 ? "PASSED\n" : "FAILED\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);

    return 0;
}

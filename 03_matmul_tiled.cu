// 03_matmul_tiled.cu
//
// Tiled matrix multiply using shared memory, benchmarked directly against the
// naive version in the same run.
//
// Instead of every thread re-reading full rows/columns from slow global memory,
// each block cooperatively loads a TILE x TILE chunk of A and B into shared
// memory once, and every thread in the block reuses it. This cuts global memory
// traffic by roughly a factor of TILE.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define TILE 16

__global__ void matmulNaive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matmulTiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Cooperative load: each thread loads exactly one element of each tile.
        // Bounds-checked so N doesn't need to be a multiple of TILE.
        As[ty][tx] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        __syncthreads(); // wait until the whole tile is loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads(); // wait until everyone's done reading before the next load overwrites it
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_naive = (float*)malloc(bytes);
    float* h_C_tiled = (float*)malloc(bytes);

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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // --- naive ---
    cudaEvent_t s1, e1;
    CUDA_CHECK(cudaEventCreate(&s1)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(s1));
    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float msNaive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msNaive, s1, e1));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- tiled ---
    cudaEvent_t s2, e2;
    CUDA_CHECK(cudaEventCreate(&s2)); CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventRecord(s2));
    matmulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float msTiled = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTiled, s2, e2));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    // Tiled should match naive (both should match a CPU reference too, but
    // at N=1024 the CPU reference is slow — checking tiled-vs-naive agreement
    // is a fast, still-meaningful correctness check).
    double maxDiff = 0.0;
    for (int i = 0; i < N * N; ++i)
        maxDiff = fmax(maxDiff, fabs((double)h_C_naive[i] - (double)h_C_tiled[i]));

    double gflopsNaive = (2.0 * N * N * N) / (msNaive / 1000.0) / 1e9;
    double gflopsTiled = (2.0 * N * N * N) / (msTiled / 1000.0) / 1e9;

    printf("N = %d x %d, TILE = %d\n", N, N, TILE);
    printf("Naive : %.3f ms  (%.2f GFLOP/s)\n", msNaive, gflopsNaive);
    printf("Tiled : %.3f ms  (%.2f GFLOP/s)\n", msTiled, gflopsTiled);
    printf("Speedup: %.2fx\n", msNaive / msTiled);
    printf("Max diff naive vs tiled: %e\n", maxDiff);
    printf(maxDiff < 1e-2 ? "PASSED (naive and tiled agree)\n" : "FAILED (results diverge)\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);

    return 0;
}

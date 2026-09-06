// 05_flash_attention_simplified.cu
//
// Fuses QK^T, softmax, and the weighted sum over V into a single kernel that
// never materializes the full [N x N] attention matrix in global memory.
//
// This is a simplified, single-head, single-block-per-query version of the
// FlashAttention algorithm: it keeps the core idea (tile over keys/values,
// track a running max/sum, rescale the output accumulator as new tiles arrive)
// without the multi-query-tile-per-block parallelism of the full paper.
//
// Layout: Q, K, V are all [N, D] row-major, single head, batch size 1. One
// block handles one query row. Keys/values are streamed in tiles of TILE_N so
// the full N x N score matrix is never stored anywhere.

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

#define D 64        // head dimension
#define TILE_N 32   // keys/values processed per tile (also blockDim.x)

__global__ void flashAttentionSimplified(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O,
                                          int N, float scale) {
    int qi = blockIdx.x;   // this block owns query row qi
    int tid = threadIdx.x; // 0..TILE_N-1, also "which key in the tile" this thread loads/scores

    __shared__ float Qs[D];
    __shared__ float Ks[TILE_N][D];
    __shared__ float Vs[TILE_N][D];
    __shared__ float p[TILE_N];        // exp(score - running_max) for the current tile
    __shared__ float acc[D];           // running output accumulator (unnormalized)
    __shared__ float reduceBuf[TILE_N];
    __shared__ float m_prev, l_prev;   // running max / running sum (the "online softmax" state)

    // Load this block's query row and zero the accumulator.
    for (int c = tid; c < D; c += blockDim.x) {
        Qs[c] = Q[(size_t)qi * D + c];
        acc[c] = 0.0f;
    }
    if (tid == 0) { m_prev = -INFINITY; l_prev = 0.0f; }
    __syncthreads();

    int numTiles = (N + TILE_N - 1) / TILE_N;
    for (int t = 0; t < numTiles; ++t) {
        int keyIdx = t * TILE_N + tid;

        // Cooperative load of one K/V tile into shared memory (each thread
        // loads the full D-length row for "its" key — simple, not the most
        // bandwidth-optimal load pattern, but easy to reason about and correct).
        if (keyIdx < N) {
            for (int c = 0; c < D; ++c) {
                Ks[tid][c] = K[(size_t)keyIdx * D + c];
                Vs[tid][c] = V[(size_t)keyIdx * D + c];
            }
        } else {
            for (int c = 0; c < D; ++c) { Ks[tid][c] = 0.0f; Vs[tid][c] = 0.0f; }
        }
        __syncthreads();

        // Score this thread's key against the query row.
        float score = -INFINITY;
        if (keyIdx < N) {
            float dot = 0.0f;
            #pragma unroll
            for (int c = 0; c < D; ++c) dot += Qs[c] * Ks[tid][c];
            score = dot * scale;
        }

        // Block-wide max reduction over this tile's scores.
        reduceBuf[tid] = score;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduceBuf[tid] = fmaxf(reduceBuf[tid], reduceBuf[tid + stride]);
            __syncthreads();
        }
        float tileMax = reduceBuf[0];
        __syncthreads();

        // Online softmax update: fold this tile into the running max/sum.
        float m_new = fmaxf(m_prev, tileMax);
        float correction = expf(m_prev - m_new); // rescales everything accumulated so far

        float pi = (keyIdx < N) ? expf(score - m_new) : 0.0f;
        p[tid] = pi;
        __syncthreads();

        reduceBuf[tid] = pi;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduceBuf[tid] += reduceBuf[tid + stride];
            __syncthreads();
        }
        float tileSum = reduceBuf[0];
        __syncthreads();

        float l_new = l_prev * correction + tileSum;

        // Rescale the existing accumulator and fold in this tile's weighted V.
        for (int c = tid; c < D; c += blockDim.x) {
            float weighted = 0.0f;
            for (int j = 0; j < TILE_N; ++j) weighted += p[j] * Vs[j][c];
            acc[c] = acc[c] * correction + weighted;
        }
        __syncthreads();

        if (tid == 0) { m_prev = m_new; l_prev = l_new; }
        __syncthreads();
    }

    // Final normalization: divide the accumulator by the running sum.
    for (int c = tid; c < D; c += blockDim.x) {
        O[(size_t)qi * D + c] = acc[c] / l_prev;
    }
}

// Reference implementation: the textbook (non-fused) version — materializes
// the full [N x N] score matrix, just like a naive PyTorch implementation would.
void attentionCPU(const float* Q, const float* K, const float* V, float* O,
                   int N, float scale) {
    float* scores = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        float maxVal = -INFINITY;
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int c = 0; c < D; ++c) dot += Q[i * D + c] * K[j * D + c];
            scores[j] = dot * scale;
            maxVal = fmaxf(maxVal, scores[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) { scores[j] = expf(scores[j] - maxVal); sum += scores[j]; }
        for (int c = 0; c < D; ++c) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) acc += scores[j] * V[j * D + c];
            O[i * D + c] = acc / sum;
        }
    }
    free(scores);
}

int main() {
    const int N = 512; // sequence length — modest so the CPU reference finishes in reasonable time
    const float scale = 1.0f / sqrtf((float)D);
    const size_t bytes = (size_t)N * D * sizeof(float);

    float* h_Q = (float*)malloc(bytes);
    float* h_K = (float*)malloc(bytes);
    float* h_V = (float*)malloc(bytes);
    float* h_O = (float*)malloc(bytes);
    float* h_O_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * D; ++i) {
        h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    dim3 grid(N);
    dim3 block(TILE_N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    flashAttentionSimplified<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost));

    attentionCPU(h_Q, h_K, h_V, h_O_ref, N, scale);

    double maxErr = 0.0;
    for (int i = 0; i < N * D; ++i)
        maxErr = fmax(maxErr, fabs((double)h_O[i] - (double)h_O_ref[i]));

    printf("N (seq len) = %d, D (head dim) = %d, TILE_N = %d\n", N, D, TILE_N);
    printf("Fused attention kernel time: %.3f ms\n", ms);
    printf("Max error vs unfused CPU reference: %e\n", maxErr);
    printf(maxErr < 1e-3 ? "PASSED\n" : "FAILED\n");
    printf("Peak shared memory per block: ~%.1f KB (well under the 48KB default limit)\n",
           (2.0 * TILE_N * D * sizeof(float) + D * sizeof(float) * 2 + TILE_N * sizeof(float) * 2) / 1024.0);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_ref);

    return 0;
}

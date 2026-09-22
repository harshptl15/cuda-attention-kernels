# CUDA Kernels for Attention Inference

I wrote these to learn CUDA properly — memory hierarchy, shared memory, warp
primitives, kernel fusion — by implementing each idea and measuring it instead
of reading about it. They build from a vector-add up to a fused,
flash-attention-style kernel, each benchmarked against PyTorch/cuBLAS.

I don't beat cuBLAS anywhere. But "they're faster" is useless without saying by
how much and why, so most of this README is me working out where my time goes.

```
kernels/
  01_vector_add.cu                  # thread indexing, memory management
  02_matmul_naive.cu                # baseline: no shared memory
  03_matmul_tiled.cu                # shared-memory tiling
  04_softmax_fused.cu               # online softmax, warp-shuffle reduction
  05_flash_attention_simplified.cu  # fused QK^T + softmax + weighted-V
notebooks/cuda_kernels_colab.ipynb  # runs everything on a free Colab T4
```

I don't own a GPU, so this was built on Colab. Open the notebook, set the
runtime to a GPU, run top to bottom — about five minutes. Locally, `make run`
builds and runs each kernel (edit `ARCH` in the Makefile if you're not on a T4).

## Results

Colab **Tesla T4**, fp32. Everything passes a correctness check against a CPU or
PyTorch reference — the attention kernel matches an unfused CPU version to
1.1e-07. Each number is the median of 5 timed blocks of 20 launches.

| Kernel | Mine | PyTorch / cuBLAS | % of T4 peak |
|---|---|---|---|
| Matmul, naive (1024×1024) | 4.860 ms (442 GFLOP/s) | — | 5.5% |
| Matmul, tiled (1024×1024) | 3.040 ms (706 GFLOP/s) | 0.350 ms (cuBLAS) | 8.7% vs 75.8% |
| Softmax (4096×1024) | 0.240 ms (140 GB/s) | 0.168 ms (torch) | 43.7% vs 62.2% of bandwidth |
| Attention (N=512, D=64) | 1.243 ms | 0.078 ms (unfused), 0.177 ms (SDPA) | see below |

**Tiled vs. naive matmul: 1.60× speedup** from shared-memory tiling alone.

![benchmark comparison](results/benchmark_comparison.png)

## Two things I got wrong

**Timing.** My first harness timed a single cold launch. A T4 idles at 300 MHz
and needs about a second of load to reach boost, so five warmup launches (~15 ms)
does nothing. The same attention binary measured **5.996 ms** right after an
`nvcc` compile and **2.469 ms** run back-to-back with the other benchmarks — a
2.4× swing from clock state alone, while I was reporting a 1.59× speedup. cuBLAS
ranged over 3072–6160 GFLOP/s for the same reason. The kernels now warm up by
wall-clock time and report a median plus spread. The tell: softmax never moved,
because it's the only memory-bound kernel here and memory clocks don't swing.

**A bank conflict, in the wrong place.** I'd claimed the conflict was in the
accumulation loop, `Vs[j][c]`. Wrong — conflicts are about what 32 lanes touch at
the same instant, not what one thread touches over time, and there the lanes
share `j` and differ in `c`, so they hit 32 distinct banks. The real conflict is
the load above it: `Ks[tid][c] = K[keyIdx * D + c]`, where `(tid*64 + c) % 32`
is `c % 32` for every lane — all 32 on one bank. Padding to `[TILE_N][D + 1]`
makes it `(tid + c) % 32`. One character, and attention went 2.469 → 1.243 ms.

## Why cuBLAS and PyTorch are still faster

**Matmul.** Each of my threads computes one output element, so the inner loop
does two shared-memory loads per FMA — that caps it well below peak before
latency even matters. The fix is register blocking: one thread computes a 4×4 or
8×8 patch so one set of loads feeds sixteen or sixty-four FMAs. cuBLAS does that
plus vectorized loads and double-buffering. I also assumed wrongly at first that
cuBLAS was using Tensor Cores — Turing's don't do fp32 (TF32 arrived with
Ampere), so it's running on the same CUDA cores mine is.

**Softmax.** My closest result. It's memory-bound — 16 MB in, 16 MB out, almost
no arithmetic — so percent of peak bandwidth is the honest metric, not the ratio
to torch.

**Attention.** At N=512, D=64, single head, the problem is only 67 MFLOP — small
enough that torch's *unfused* path (0.078 ms) beats its own fused SDPA
(0.177 ms), because dispatch costs more than the math. So the real baseline is
the unfused one, and I lose by 16×. The algorithm is right; the remaining cost is
memory access:

1. **Uncoalesced loads** — each thread reads a whole 64-float K/V row alone, so
   adjacent threads are 64 floats apart and every load is its own transaction.
   This is where most of the time goes.
2. **One warp per block** — 32 threads against 16.6 KB of shared memory is ~3
   resident blocks and under 10% occupancy, so there's nothing to hide latency.

## What's next

Register blocking, vectorized loads, coalescing the K/V loads, and
re-benchmarking attention at a realistic shape — at N=512 the score matrix is
1 MB and fits in L2, so the materialization my kernel avoids isn't costing the
unfused version anything yet. Tensor Cores are the biggest win on paper (65 vs
8.1 TFLOP/s) but change the numerics, so they'd need their own correctness story.

MIT licensed. Requires a CUDA toolkit or a Colab GPU runtime, plus PyTorch for
the comparisons.

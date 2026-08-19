# Py2Tensor benchmark evidence — 2026-08-19

This directory contains a reproducibility artifact for commit
`e1f66c9f17d6c4038473bdcde021a1a8fb76ca53` of `salihcankurnaz/py2tensor`.

## What this artifact supports

- **Correctness in the tested scope:** all **10/10** comparisons passed against the
  hand-written PyTorch reference with `atol=1e-3` and `rtol=1e-3`.
- The recorded maximum absolute error is **0.0** for all ten comparisons.
- For the five tested 10-million-element CUDA workloads, the `tensorize` path produced
  median throughput between **97.39%**
  and **100.04%**
  of the hand-written PyTorch reference.
- The `tensorize(compile=True)` path produced median throughput between
  **97.66%**
  and **99.98%**
  of the same reference.

## Median execution results

| Workload | Hand PyTorch median | `tensorize` median | `tensorize` vs hand | `compile=True` median | `compile=True` vs hand |
|---|---:|---:|---:|---:|---:|
| Polynomial x^2+2x+1 | 1.623040 ms | 1.644016 ms | 98.72% | 1.644544 ms | 98.69% |
| Gaussian PDF exp(-x^2/2)/sqrt(2pi) | 1.644528 ms | 1.662464 ms | 98.92% | 1.662976 ms | 98.89% |
| Branching if/else | 1.876496 ms | 1.917808 ms | 97.85% | 1.900544 ms | 98.73% |
| Sigmoid 1/(1+exp(-x)) | 1.815552 ms | 1.864160 ms | 97.39% | 1.859072 ms | 97.66% |
| Newton sqrt (10 iterations) | 14.890496 ms | 14.884864 ms | 100.04% | 14.893056 ms | 99.98% |

Each method/workload cell above is based on **50 CUDA-event timing rounds** after
**5 warmup calls**. The input tensor contains 10,000,000 elements and is already
resident on the GPU before timing.

## Workloads

1. Polynomial `x^2 + 2x + 1`
2. Gaussian PDF
3. Branching `if/else`
4. Sigmoid
5. Newton square root with 10 iterations

## Timing scope

The benchmark compares three CUDA execution paths on the same GPU-resident input:

- `@tensorize`
- `@tensorize(compile=True)`
- hand-written PyTorch

The measurements do **not** include CPU scalar execution, host-to-device transfer,
data loading, or an end-to-end application pipeline. CPU/GPU speedup is therefore
not reported by this artifact.

## `compile=True` interpretation

The audited repository implementation attempts `torch.compile(gpu_fn)` for
`compile=True`, but falls back to the uncompiled generated GPU function if that
attempt raises an exception. V30 did not independently record the selected internal
backend. For that reason, this artifact labels the measurement
**`tensorize(compile=True)` path** rather than claiming a confirmed TorchInductor
compiled kernel.

## Reproducibility

- Raw timings: [`gpu_raw_rounds.csv`](gpu_raw_rounds.csv)
- Aggregate statistics: [`gpu_summary.csv`](gpu_summary.csv)
- Correctness checks: [`correctness.csv`](correctness.csv)
- Environment: [`environment.json`](environment.json)
- Source/import hashes: [`source_hashes.json`](source_hashes.json)
- Methodology: [`methodology.json`](methodology.json)
- Machine-readable scoped results: [`CLAIM_SAFE_RESULTS.json`](CLAIM_SAFE_RESULTS.json)

The raw table contains **750 measurements**: 5 workloads × 3 methods × 50 rounds.

## Explicit limitations

This artifact does **not** establish:

- arbitrary Python function coverage;
- universal exactness;
- universal performance or speedup;
- CPU-versus-GPU speedup;
- confirmed use of a particular `torch.compile` backend; or
- performance on hardware or workloads other than those recorded here.

The results are intentionally scoped to the five functions, tolerance, hardware,
software environment, tensor size, and timing methodology documented in this artifact.

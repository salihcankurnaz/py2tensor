# Py2Tensor v2.0

> **Research status:** experimental Python-to-tensor lowering toolkit. The implementation supports a defined subset of Python patterns; it does not compile arbitrary Python programs.

**Write supported Python patterns and lower them to GPU-oriented tensor operations without writing custom CUDA kernels.**

```python
from tensorize_all import tensorize_all

@tensorize_all
def insurance(age, bmi, smoker, claims):
    rates = {0: 200, 1: 400, 2: 600, 3: 1000}
    if age > 60: factor = 3.0
    else:
        if age > 40: factor = 2.0
        else: factor = 1.0
    if smoker > 0.5: factor = factor * 2
    else: factor = factor
    return rates[claims] * factor

# GPU-oriented vectorized execution for supported inputs
quotes = insurance(ages, bmis, smokers, claims)
```

## What It Does

Lowers a **supported subset** of Python functions to PyTorch/tensor operations. Unsupported syntax or semantics should be treated as out of scope unless covered by the current tests.

`if/else` -> `torch.where` | `for` -> unroll | `dict` -> tensor lookup | `math.sin` -> `torch.sin` | `try/except` -> safe execution

## Backends and benchmarking

The repository contains several experimental execution paths, including `@tensorize`,
compiled/fused execution, Triton-oriented experiments, pure tensor execution, automatic
selection, and the broader `@tensorize_all` path.

Performance is workload- and hardware-dependent. Reproducible benchmark evidence is
committed under [`benchmarks/publication-evidence/2026-08-19-v30/`](benchmarks/publication-evidence/2026-08-19-v30/).
That artifact records the exact source revision, environment, methodology, correctness
checks, raw round timings, and machine-readable summaries for five benchmark workloads.
It should be treated as configuration-specific evidence rather than a universal speedup claim.

For reproducible performance work, start with:

- `final_benchmark.py` — compares Python, NumPy, tensorized execution, compiled execution,
  and hand-written PyTorch on a fixed benchmark harness.
- `shap_benchmark.py` — SHAP-related benchmark experiments.

When reporting results, record the commit, GPU, CUDA/PyTorch versions, tensor shapes,
warm-up policy, round count, and the raw output.

## Supported Python Patterns

**Arithmetic**: `+` `-` `*` `/` `**` `%`  
**Math**: `sin cos tan exp log sqrt tanh atan2 pi e`  
**Control**: `if/else` (nested, multi-var, multi-statement + return)  
**Loops**: `for range(N)` (unrolled), `while` (auto-bounded)  
**Data**: `dict` literals, `list` literals, `min(a,b)` `max(a,b)`  
**Advanced**: `+=` `-=` `*=`, ternary, tuple return, `abs`  
**Error**: `try/except` (auto-stripped, safe execution)  
**Types**: float32, float16, numpy, pandas

## Explored Lowering Patterns

| Pattern | Experimental lowering approach |
|---|---|
| String comparison | char -> int tensor |
| Dictionary lookup | embedding tensor |
| Dynamic list | mask + filter |
| Hash table | modular arithmetic |
| State machine | transition matrix |
| Try/except | condition check |
| File I/O | streaming pipeline |

## Tested Real-World Functions

- Insurance premium (4 factors, nested if)
- Tax calculator (6 progressive brackets)
- Projectile with air resistance (50 iterations)
- FICO credit scoring (5 inputs, weighted rules)
- Black-Scholes option pricing (log, sqrt, exp)
- Damped spring simulation (30 iteration ODE)
- Fraud detection rule engine (8 rules)
- Trading signals (RSI, Bollinger, momentum)
- Monte Carlo option pricing (1M paths x 12 steps)
- Mandelbrot set (2048x2048, 32 iterations)

These patterns were explored during development. Verify the current implementation and tests before relying on a specific pattern in another environment.

## Install

```bash
git clone https://github.com/salihcankurnaz/py2tensor.git
cd py2tensor
pip install -e .
```

## License

MIT. See [LICENSE](LICENSE).

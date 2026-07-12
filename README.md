# Py2Tensor

An experimental source-to-source compiler that converts a documented subset of ordinary Python functions into batched tensor operations.

Py2Tensor explores whether rule-heavy numerical functions can be expressed once in familiar Python syntax and executed efficiently through PyTorch or optional GPU-oriented backends.

## Example

```python
from tensorize_all import tensorize_all

@tensorize_all
def insurance(age, bmi, smoker, claims):
    rates = {0: 200, 1: 400, 2: 600, 3: 1000}

    if age > 60:
        factor = 3.0
    elif age > 40:
        factor = 2.0
    else:
        factor = 1.0

    if smoker > 0.5:
        factor = factor * 2

    return rates[claims] * factor
```

The transformed function can operate on tensor batches rather than processing records one at a time in Python.

## Supported subset

Current transformations cover selected forms of:

- Arithmetic and comparison expressions
- Nested `if`/`else` control flow
- Fixed-range loops and bounded loop lowering
- Dictionary and list literals used as lookup tables
- Common mathematical functions
- Tuple returns and augmented assignment
- NumPy, pandas, and tensor inputs in supported paths

Support depends on the selected backend and the exact Python construct. Py2Tensor does **not** compile arbitrary Python programs, unrestricted dynamic behavior, general file I/O, or every exception-handling pattern.

## Backends

The repository contains several experimental execution paths, including PyTorch-based lowering and optional compiled or GPU-oriented backends. Their supported features differ, so tests should be run for the backend used by an application.

## Benchmarks

The repository includes local benchmarks for numerical rules, decision trees, pricing functions, simulations, and trading indicators.

Benchmark results are hardware-, input-, dtype-, and backend-dependent. Treat the included figures as project measurements rather than universal performance guarantees, and reproduce them on the target environment before relying on them.

## Tested use cases

Examples in the test and benchmark suites include:

- Insurance and tax rules
- Fraud-detection rules
- Decision-tree-style functions
- Black–Scholes calculations
- Iterative numerical simulations
- Trading indicators
- Monte Carlo workloads

## Installation

```bash
git clone https://github.com/salihcankurnaz/py2tensor.git
cd py2tensor
pip install -e .
```

Run the repository tests and benchmarks to confirm which constructs are supported in your environment.

## Project status

Py2Tensor is a research and engineering prototype. The most important future work is to formalize the supported language subset, add stronger semantic-equivalence tests, report reproducible benchmark environments, and improve diagnostics for unsupported syntax.

## License

MIT

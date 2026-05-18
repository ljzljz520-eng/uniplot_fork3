# Plan: Full Numba Parallelism + Optional Dependency

## Background

`_render_batch_of_lines` was rewritten from vectorized NumPy into explicit loops and decorated
with `@njit(cache=True, fastmath=True, parallel=True)`, achieving a 29x total speedup
(3,500ms → 119ms at 1M points). See `NUMBA_OPTIMIZATION_RESULTS.md`.

However, `parallel=True` is currently a no-op: the loops use `range`, not `numba.prange`.
The reason is a sequential dependency — a shared `idx` counter threads through the generation
loop, so each iteration must know where the previous one finished before it can write.

Additionally, Numba is now a hard dependency, which breaks installs on Alpine/musl Docker
images, PyPy, and environments where Numba wheels are unavailable.

This plan addresses both issues.

---

## Part 1: Make Numba an Optional Dependency

### Goal
`pip install uniplot` works everywhere. `pip install uniplot[fast]` adds Numba.

### Changes

**`pyproject.toml`**
- Remove `numba` from `dependencies`
- Add under `[project.optional-dependencies]`:
  ```toml
  [project.optional-dependencies]
  fast = ["numba >=0.57.0"]
  ```

**`uniplot/pixel_matrix.py`**
- Replace the top-level `from numba import njit` with a try/except that defines two complete
  implementations at import time:

```python
try:
    from numba import njit, prange

    @njit(cache=True, fastmath=True, parallel=True)
    def _render_batch_of_lines(...):
        # parallelised loop-based implementation (see Part 2)

except ImportError:
    def _render_batch_of_lines(...):
        # restored vectorized NumPy implementation (recovered from git history)
```

No runtime flag, no dispatch overhead — the right function is simply bound at import time.

**`README.md`**
- Add install note in the installation section, e.g.:
  > For best performance with large datasets, install the optional Numba dependency:
  > `pip install uniplot[fast]`

---

## Part 2: Actually Exploit `parallel=True` with `prange`

### The Problem

The current pixel generation loop cannot use `prange` because of the shared `idx` counter:

```python
idx = 0
for i in range(len(x0)):   # <-- range, not prange
    ...
    for step in range(n):
        x_all[idx] = ...
        idx += 1            # each iteration depends on previous idx
```

### The Solution: Prefix Sum

Decouple index computation from pixel generation so each line can write independently.

**Step 1 — Compute per-line pixel counts (parallelisable reduction):**
```python
counts = np.empty(len(x0), dtype=np.int64)
for i in prange(len(x0)):
    if steep[i]:
        counts[i] = max(1, int(np.round(abs(y1[i] - y0[i]))) + 1)
    else:
        counts[i] = max(1, int(np.round(abs(x1[i] - x0[i]))) + 1)
```

**Step 2 — Prefix sum to get per-line start offsets:**
```python
offsets = np.empty(len(x0) + 1, dtype=np.int64)
offsets[0] = 0
for i in range(len(counts)):
    offsets[i + 1] = offsets[i] + counts[i]
total_pixels = offsets[-1]
```
Note: the prefix sum itself must stay sequential (Numba does not provide a parallel
`cumsum`), but it is O(n) over *lines*, not pixels — cheap.

**Step 3 — Parallel pixel generation (each line writes to its own slice):**
```python
x_all = np.empty(total_pixels, dtype=np.float64)
y_all = np.empty(total_pixels, dtype=np.float64)

for i in prange(len(x0)):          # <-- prange: no shared state
    start = offsets[i]
    # ... steep/shallow line logic writes to x_all[start:start+counts[i]]
```

**Step 4 — Parallel pixel write (benign race condition):**
```python
for i in prange(total_pixels):
    xi = int(np.round(x_all[i]))
    yi = height - 1 - int(np.round(y_all[i]))
    if 0 <= xi < width and 0 <= yi < height:
        pixels[yi, xi] = layer
```
Two threads writing the same pixel is not a correctness problem here: `layer` is a
constant, so the last writer always produces the correct value. No atomics needed.

### Expected Additional Speedup

The 29x gain already achieved was from eliminating memory allocation overhead (not from
parallelism). The parallelism in `prange` adds on top of that:
- Outer line loop parallelises across CPU cores
- Inner step loop remains sequential per line (correct, as steps within a line are dependent)
- On an 8-core machine: theoretical up to ~8x; realistic 3–5x accounting for overhead

At 1M points: 119ms → estimated 30–60ms.

---

## Implementation Order

1. **Optional dependency first** (lower risk, independent of Part 2)
   - `pyproject.toml` change
   - try/except in `pixel_matrix.py` with NumPy fallback restored
   - README update
   - Run full test suite

2. **Parallelism second** (only touches the Numba branch)
   - Rewrite the `@njit` function body with prefix sum + `prange`
   - Benchmark against current `range`-based version to confirm gain
   - Run full test suite

## Test Plan

- `uv run pytest` — all 134 tests must stay green after each phase
- `uv run python3 scripts/scaling_benchmark.py` — log-log plot should show improvement
  over current `oo` curve after Part 2
- Manual smoke test: `pip install uniplot` (no extras) should work and render correctly

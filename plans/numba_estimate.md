# Numba Performance Estimate for pixel_matrix.py

## Current Performance (after Phase 1 algorithmic optimization)
- **1M points with lines:** 350ms
- **Target:** <200ms
- **Gap:** Need 1.75x speedup

## Profiling Data
From `profile_lines.py`:
```
0.382 total
└─ 0.343 _render_batch_of_lines (90% of total)
   ├─ 0.289 [self] - array operations (84% of function)
   ├─ 0.028 np.repeat
   └─ 0.024 np.clip
```

## Current Code Characteristics

**Operations that create temporary arrays (overhead):**
1. Broadcasting: `steps = steps[None, :] * np.ones((len(n), 1))` - creates large 2D array
2. Boolean masking: `x_vals[mask_steps]` - creates masked copy
3. Indexing: `x0s[:, None]` - creates broadcasted views
4. Clipping: `np.clip()` - creates new arrays
5. Concatenation: `np.concatenate(all_x)` - allocates and copies

For 500K lines × 34 pixels average:
- Broadcast creates: 500K × 101 = 50M floats = 400MB per array × 4 arrays = 1.6GB
- Then masks down to: 17M actual pixels (66% waste)

**What Numba loop-based rewrite would eliminate:**
- All temporary 2D arrays (1.6GB → 0GB)
- Broadcasting overhead
- Boolean masking overhead
- Multiple array copies
- Replace with: simple loop over lines, generating pixels directly

## Speedup Estimate

### Conservative (Pessimistic): 2-3x speedup
**Rationale:** NumPy is already well-optimized, Numba mainly removes allocation overhead

- **Current:** 350ms
- **With Numba:** 117-175ms
- **Result:** May narrowly miss or barely hit 200ms target ⚠️

### Moderate (Realistic): 3-5x speedup
**Rationale:** Replacing vectorization with loops in this specific case removes significant overhead

Evidence:
- [Numba can be 5× faster and more memory efficient](https://pythonspeed.com/articles/slow-numba/) when avoiding temporary arrays
- Our code has 66% memory waste from broadcasting
- Direct pixel generation loop eliminates all temp arrays

- **Current:** 350ms
- **With Numba:** 70-117ms
- **Result:** Solidly under 200ms target ✓

### Optimistic: 5-10x speedup
**Rationale:** Add parallelization across lines with `parallel=True`

- **Current:** 350ms
- **With Numba + parallel:** 35-70ms
- **Result:** Well under target ✓

## Algorithm Comparison

### Current (NumPy vectorized)
```python
# For all lines at once:
n = np.maximum(np.round(x1s - x0s).astype(int) + 1, 1)  # 500K values
steps = np.arange(n.max())  # Array of 101 values
steps = steps[None, :] * np.ones((len(n), 1))  # 500K × 101 = 50M floats!
mask_steps = steps < n[:, None]  # Another 50M bools!
x_vals = np.round(x0s)[:, None] + steps  # 50M floats
# ... more operations on 50M element arrays
x_vals = x_vals[mask_steps]  # Finally mask down to 17M
```

**Memory:** 50M × 8 bytes × 4 arrays = 1.6GB
**CPU:** Vectorized operations on large arrays
**Cache:** Poor - arrays don't fit in L3 cache (16MB typical)

### Numba loop-based
```python
@njit(cache=True, fastmath=True, parallel=True)
def _render_batch_of_lines(...):
    # Pre-allocate output based on actual pixel count
    total_pixels = sum(max(1, round(abs(x1s[i] - x0s[i])) + 1) for i in range(len(x0s)))
    x_all = np.empty(total_pixels, dtype=np.float64)
    y_all = np.empty(total_pixels, dtype=np.float64)

    idx = 0
    for i in prange(len(x0s)):  # Parallel over lines
        n_steps = max(1, round(abs(x1s[i] - x0s[i])) + 1)

        for step in range(n_steps):  # Inner loop per line
            t = step / max(1, n_steps - 1)
            x_all[idx] = x0s[i] + t * (x1s[i] - x0s[i])
            y_all[idx] = y0s[i] + t * (y1s[i] - y0s[i])
            idx += 1

    # Convert to pixel coordinates and write
    for i in range(total_pixels):
        xi = round(x_all[i])
        yi = round(y_all[i])
        if 0 <= xi < width and 0 <= yi < height:
            pixels[yi, xi] = layer
```

**Memory:** 17M × 8 bytes × 2 arrays = 272MB (6× less!)
**CPU:** Compiled C-speed loops
**Cache:** Better - sequential access, smaller working set

## Implementation Effort

**Files to modify:**
- `pixel_matrix.py::_render_batch_of_lines()` - complete rewrite (~100 lines)
- `pixel_matrix.py::_render_batch_of_dots()` - similar rewrite (~30 lines)

**Complexities to handle:**
1. Replace boolean masking → explicit if checks in loops
2. Replace broadcasting → nested loops
3. Replace fancy indexing → direct index computation
4. Handle steep vs shallow lines in loop logic
5. Manage parallel reduction for pixel writing (race conditions!)

**Effort estimate:**
- **Writing code:** 4-6 hours (straightforward translation)
- **Debugging:** 4-8 hours (edge cases, parallel race conditions, off-by-one errors)
- **Testing:** 2-4 hours (verify against 23 unit tests)
- **Total:** 1-2 days

## Risk Assessment

**Low risk:**
- Algorithm is well-understood (Bresenham line drawing)
- 23 comprehensive unit tests catch regressions
- Can keep old code as fallback initially

**Medium risk:**
- Parallel pixel writing needs atomics or separate buffers
- Numba compilation errors can be cryptic
- Different edge case behavior than NumPy

## Recommendation

**Expected outcome:** 3-5x speedup (realistic estimate)
- 350ms → 70-117ms
- **Comfortably meets <200ms target** ✓

**Effort:** 1-2 days implementation + testing

**Verdict:** **Worth attempting** given:
1. High probability of meeting target (3-5x from research benchmarks)
2. Manageable 1-2 day effort
3. Low risk with good test coverage
4. Even pessimistic 2x gets us to 175ms (close to target)

**Alternative if this fails:** Fall back to Cython (1 week effort, similar speedup)

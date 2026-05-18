import numpy as np
from numpy.typing import NDArray
from typing import Optional, Final


BATCH_SIZE: Final = 10_000


def render(
    xs: NDArray,
    ys: NDArray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    lines: bool = False,
    pixels: Optional[NDArray] = None,
    layer: int = 1,
    batch_size: int = BATCH_SIZE,
):
    if pixels is None:
        pixels = np.zeros((height, width), dtype=np.int32)

    # Always render points
    for start in range(0, len(xs), batch_size):
        end = min(start + batch_size, len(xs))
        pixels = _render_batch_of_dots(
            xs=xs[start:end],
            ys=ys[start:end],
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            width=width,
            height=height,
            pixels=pixels,
            layer=layer,
        )

    # Optionally render lines
    if lines and len(xs) >= 2:
        valid = (
            ~np.isnan(xs[:-1])
            & ~np.isnan(xs[1:])
            & ~np.isnan(ys[:-1])
            & ~np.isnan(ys[1:])
        )
        xs0 = xs[:-1][valid]
        xs1 = xs[1:][valid]
        ys0 = ys[:-1][valid]
        ys1 = ys[1:][valid]

        for start in range(0, len(xs0), batch_size):
            end = min(start + batch_size, len(xs0))
            x_pairs = np.stack([xs0[start:end], xs1[start:end]], axis=1).reshape(-1)
            y_pairs = np.stack([ys0[start:end], ys1[start:end]], axis=1).reshape(-1)

            pixels = _render_batch_of_lines(
                xs=x_pairs,
                ys=y_pairs,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                width=width,
                height=height,
                pixels=pixels,
                layer=layer,
            )

    return pixels


def merge_on_top(
    low_layer: NDArray, high_layer: NDArray, width: int, height: int
) -> NDArray:
    """
    Put a pixel matrix on top of another, with an optional single solid line of
    "shadow", including diagonal fields.

    If activated, this shadow will ensure that later 2x2 squares exclusively
    belong to one particular line.

    TODO I stopped using this but still there is the unused shadow stuff,
    I would delete it as well as the tests
    """
    merged_layer = np.copy(low_layer)

    not_zero_high_layer = high_layer != 0
    merged_layer[not_zero_high_layer] = high_layer[not_zero_high_layer]

    return merged_layer


###########
# private #
###########


def _render_batch_of_dots(
    xs: NDArray,
    ys: NDArray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    pixels: Optional[NDArray] = None,
    layer: int = 1,
) -> NDArray:
    if pixels is None:
        pixels = np.zeros((height, width), dtype=np.int32)

    if len(xs) == 0:
        return pixels

    valid = ~np.isnan(xs) & ~np.isnan(ys)

    xs_pix = (width - 1) * (xs[valid] - x_min) / (x_max - x_min)
    ys_pix = (height - 1) * (ys[valid] - y_min) / (y_max - y_min)

    xi = np.round(xs_pix).astype(int)
    yi = np.round(ys_pix).astype(int)
    yi = height - 1 - yi  # flip Y for image coordinates
    valid = (
        (~np.isnan(xi))
        & (xi >= 0)
        & (xi < width)
        & (~np.isnan(yi))
        & (yi >= 0)
        & (yi < height)
    )
    pixels[yi[valid], xi[valid]] = layer
    return pixels


try:
    from numba import njit, prange  # type: ignore

    @njit(cache=True, fastmath=True, parallel=True)  # type: ignore
    def _render_batch_of_lines(
        xs: NDArray,
        ys: NDArray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: int,
        height: int,
        pixels: Optional[NDArray] = None,
        layer: int = 1,
    ) -> NDArray:
        if pixels is None:
            pixels = np.zeros((height, width), dtype=np.int32)

        if len(xs) == 0:
            return pixels

        xs_pix = (width - 1) * (xs - x_min) / (x_max - x_min)
        ys_pix = (height - 1) * (ys - y_min) / (y_max - y_min)

        x0, x1 = xs_pix[::2], xs_pix[1::2]
        y0, y1 = ys_pix[::2], ys_pix[1::2]

        valid = ~np.isnan(x0) & ~np.isnan(x1) & ~np.isnan(y0) & ~np.isnan(y1)
        x0, x1 = x0[valid], x1[valid]
        y0, y1 = y0[valid], y1[valid]

        if len(x0) == 0:
            return pixels

        steep = np.abs(y1 - y0) > np.abs(x1 - x0)
        n_lines = len(x0)

        # Step 1: per-line pixel counts (parallel scatter, no shared state)
        counts = np.empty(n_lines, dtype=np.int64)
        for i in prange(n_lines):
            if steep[i]:
                counts[i] = max(1, int(np.round(abs(y1[i] - y0[i]))) + 1)
            else:
                counts[i] = max(1, int(np.round(abs(x1[i] - x0[i]))) + 1)

        # Step 2: prefix sum → per-line start offsets (sequential over lines, not pixels)
        offsets = np.empty(n_lines + 1, dtype=np.int64)
        offsets[0] = 0
        for i in range(n_lines):
            offsets[i + 1] = offsets[i] + counts[i]
        total_pixels = offsets[n_lines]

        # Step 3: allocate output arrays
        x_all = np.empty(total_pixels, dtype=np.float64)
        y_all = np.empty(total_pixels, dtype=np.float64)

        # Step 4: parallel pixel generation — each line writes to its own slice
        for i in prange(n_lines):
            x_start, x_end = x0[i], x1[i]
            y_start, y_end = y0[i], y1[i]
            start = offsets[i]

            if steep[i]:
                if y_start > y_end:
                    y_start, y_end = y_end, y_start
                    x_start, x_end = x_end, x_start

                n = counts[i]
                y_base = np.round(y_start)

                for step in range(n):
                    y_val = y_base + step
                    safe_dy = y_end - y_start
                    if abs(safe_dy) < 1e-10:
                        safe_dy = 1.0
                    t = (y_val - y_start) / safe_dy
                    x_val = x_start + t * (x_end - x_start)

                    y_val = max(min(y_val, max(y_start, y_end)), min(y_start, y_end))
                    x_val = max(min(x_val, max(x_start, x_end)), min(x_start, x_end))

                    x_all[start + step] = x_val
                    y_all[start + step] = y_val
            else:
                if x_start > x_end:
                    x_start, x_end = x_end, x_start
                    y_start, y_end = y_end, y_start

                n = counts[i]
                x_base = np.round(x_start)

                for step in range(n):
                    x_val = x_base + step
                    safe_dx = x_end - x_start
                    if abs(safe_dx) < 1e-10:
                        safe_dx = 1.0
                    t = (x_val - x_start) / safe_dx
                    y_val = y_start + t * (y_end - y_start)

                    x_val = max(min(x_val, max(x_start, x_end)), min(x_start, x_end))
                    y_val = max(min(y_val, max(y_start, y_end)), min(y_start, y_end))

                    x_all[start + step] = x_val
                    y_all[start + step] = y_val

        # Step 5: parallel pixel write — benign race: constant value, last writer wins
        for i in prange(total_pixels):
            xi = int(np.round(x_all[i]))
            yi = height - 1 - int(np.round(y_all[i]))
            if 0 <= xi < width and 0 <= yi < height:
                pixels[yi, xi] = layer

        return pixels

except ImportError:

    def _render_batch_of_lines(  # type: ignore[misc]
        xs: NDArray,
        ys: NDArray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: int,
        height: int,
        pixels: Optional[NDArray] = None,
        layer: int = 1,
    ) -> NDArray:
        if pixels is None:
            pixels = np.zeros((height, width), dtype=np.int32)

        if len(xs) == 0:
            return pixels

        xs_pix = (width - 1) * (xs - x_min) / (x_max - x_min)
        ys_pix = (height - 1) * (ys - y_min) / (y_max - y_min)

        x0, x1 = xs_pix[::2], xs_pix[1::2]
        y0, y1 = ys_pix[::2], ys_pix[1::2]

        valid = ~np.isnan(x0) & ~np.isnan(x1) & ~np.isnan(y0) & ~np.isnan(y1)
        x0, x1 = x0[valid], x1[valid]
        y0, y1 = y0[valid], y1[valid]

        if len(x0) == 0:
            return pixels

        steep = np.abs(y1 - y0) > np.abs(x1 - x0)

        all_x, all_y = [], []

        # Shallow lines
        mask = ~steep
        if np.any(mask):
            x0s, x1s = x0[mask], x1[mask]
            y0s, y1s = y0[mask], y1[mask]

            swap = x0s > x1s
            x0s[swap], x1s[swap] = x1s[swap], x0s[swap]
            y0s[swap], y1s[swap] = y1s[swap], y0s[swap]

            n = np.maximum(np.round(x1s - x0s).astype(int) + 1, 1)
            steps = np.arange(n.max())
            steps = steps[None, :] * np.ones((len(n), 1))
            mask_steps = steps < n[:, None]

            x_vals = np.round(x0s)[:, None] + steps
            safe_dx = x1s - x0s
            safe_dx[safe_dx == 0] = 1
            t = (x_vals - x0s[:, None]) / safe_dx[:, None]
            y_vals = y0s[:, None] + t * (y1s - y0s)[:, None]

            x_vals = np.clip(
                x_vals,
                np.minimum(x0s[:, None], x1s[:, None]),
                np.maximum(x0s[:, None], x1s[:, None]),
            )
            y_vals = np.clip(
                y_vals,
                np.minimum(y0s[:, None], y1s[:, None]),
                np.maximum(y0s[:, None], y1s[:, None]),
            )

            all_x.append(x_vals[mask_steps])
            all_y.append(y_vals[mask_steps])

        # Steep lines
        mask = steep
        if np.any(mask):
            x0s, x1s = x0[mask], x1[mask]
            y0s, y1s = y0[mask], y1[mask]

            swap = y0s > y1s
            x0s[swap], x1s[swap] = x1s[swap], x0s[swap]
            y0s[swap], y1s[swap] = y1s[swap], y0s[swap]

            n = np.maximum(np.round(y1s - y0s).astype(int) + 1, 1)
            steps = np.arange(n.max())
            steps = steps[None, :] * np.ones((len(n), 1))
            mask_steps = steps < n[:, None]

            y_vals = np.round(y0s)[:, None] + steps
            safe_dy = y1s - y0s
            safe_dy[safe_dy == 0] = 1
            t = (y_vals - y0s[:, None]) / safe_dy[:, None]
            x_vals = x0s[:, None] + t * (x1s - x0s)[:, None]

            y_vals = np.clip(
                y_vals,
                np.minimum(y0s[:, None], y1s[:, None]),
                np.maximum(y0s[:, None], y1s[:, None]),
            )
            x_vals = np.clip(
                x_vals,
                np.minimum(x0s[:, None], x1s[:, None]),
                np.maximum(x0s[:, None], x1s[:, None]),
            )

            all_x.append(x_vals[mask_steps])
            all_y.append(y_vals[mask_steps])

        if not all_x:
            return pixels

        x_all = np.round(np.concatenate(all_x)).astype(int)
        y_all = np.round(np.concatenate(all_y)).astype(int)
        y_all = height - 1 - y_all

        valid = (x_all >= 0) & (x_all < width) & (y_all >= 0) & (y_all < height)
        pixels[y_all[valid], x_all[valid]] = layer

        return pixels

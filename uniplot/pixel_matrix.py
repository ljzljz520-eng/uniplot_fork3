import numpy as np
from numpy.typing import NDArray
from typing import Optional, Final
from numba import njit  # type: ignore


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


@njit(cache=True, fastmath=True, parallel=True)
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

    dx = x1 - x0
    dy = y1 - y0
    steep = np.abs(dy) > np.abs(dx)

    # Calculate total pixels needed (exact allocation)
    total_pixels = 0
    for i in range(len(x0)):
        if steep[i]:
            n = max(1, int(np.round(abs(y1[i] - y0[i]))) + 1)
        else:
            n = max(1, int(np.round(abs(x1[i] - x0[i]))) + 1)
        total_pixels += n

    # Allocate exact size arrays
    x_all = np.empty(total_pixels, dtype=np.float64)
    y_all = np.empty(total_pixels, dtype=np.float64)

    # Generate all pixels using nested loops
    idx = 0
    for i in range(len(x0)):
        x_start, x_end = x0[i], x1[i]
        y_start, y_end = y0[i], y1[i]

        if steep[i]:
            # Steep line: iterate along y-axis
            # Ensure y_start < y_end for consistent direction
            if y_start > y_end:
                y_start, y_end = y_end, y_start
                x_start, x_end = x_end, x_start

            n = max(1, int(np.round(y_end - y_start)) + 1)
            y_base = np.round(y_start)

            for step in range(n):
                y_val = y_base + step
                # Calculate t parameter for interpolation
                safe_dy = y_end - y_start
                if abs(safe_dy) < 1e-10:
                    safe_dy = 1.0
                t = (y_val - y_start) / safe_dy
                x_val = x_start + t * (x_end - x_start)

                # Clipping
                y_val = max(min(y_val, max(y_start, y_end)), min(y_start, y_end))
                x_val = max(min(x_val, max(x_start, x_end)), min(x_start, x_end))

                x_all[idx] = x_val
                y_all[idx] = y_val
                idx += 1
        else:
            # Shallow line: iterate along x-axis
            # Ensure x_start < x_end for consistent direction
            if x_start > x_end:
                x_start, x_end = x_end, x_start
                y_start, y_end = y_end, y_start

            n = max(1, int(np.round(x_end - x_start)) + 1)
            x_base = np.round(x_start)

            for step in range(n):
                x_val = x_base + step
                # Calculate t parameter for interpolation
                safe_dx = x_end - x_start
                if abs(safe_dx) < 1e-10:
                    safe_dx = 1.0
                t = (x_val - x_start) / safe_dx
                y_val = y_start + t * (y_end - y_start)

                # Clipping
                x_val = max(min(x_val, max(x_start, x_end)), min(x_start, x_end))
                y_val = max(min(y_val, max(y_start, y_end)), min(y_start, y_end))

                x_all[idx] = x_val
                y_all[idx] = y_val
                idx += 1

    # Round and convert to integer pixel coordinates
    x_all = np.round(x_all).astype(np.int64)  # type: ignore
    y_all = np.round(y_all).astype(np.int64)  # type: ignore
    y_all = height - 1 - y_all  # Flip Y for image coordinates

    # Write pixels with bounds checking
    for i in range(len(x_all)):
        if 0 <= x_all[i] < width and 0 <= y_all[i] < height:
            pixels[y_all[i], x_all[i]] = layer

    return pixels

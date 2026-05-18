"""
Tests for the pure NumPy fallback path of _render_batch_of_lines.

When run as part of the full suite (numba installed), a module-scoped fixture
patches numba away and reloads pixel_matrix, then restores it afterward so
subsequent test modules are unaffected.

In CI the no-numba job runs without numba installed, so no patching is needed
and the fallback is used automatically.
"""

import sys
import importlib

import numpy as np
import pytest

import uniplot.pixel_matrix as _pm

_ABSENT = object()  # sentinel: numba was not in sys.modules at all


@pytest.fixture(autouse=True, scope="module")
def _numpy_only():
    saved = sys.modules.get("numba", _ABSENT)
    sys.modules["numba"] = None  # type: ignore[assignment]
    importlib.reload(_pm)
    yield
    if saved is _ABSENT:
        sys.modules.pop("numba", None)
    else:
        sys.modules["numba"] = saved  # type: ignore[assignment]
    importlib.reload(_pm)


def render(*args, **kwargs):
    # Call through the module reference so we always hit the reloaded version.
    return _pm.render(*args, **kwargs)


# ------------------------------------------------------------------ #
# Line rendering tests (mirror of test_pixel_matrix.py line tests)   #
# ------------------------------------------------------------------ #


def test_diagonal_line():
    pixels = render(
        xs=np.array([1, 2]),
        ys=np.array([1, 2]),
        x_min=1,
        y_min=1,
        x_max=2.1,
        y_max=2.1,
        width=5,
        height=5,
        lines=True,
    )
    desired = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(pixels, desired)


def test_horizontal_line():
    pixels = render(
        xs=np.array([1, 2]),
        ys=np.array([1, 1]),
        x_min=1,
        y_min=1,
        x_max=2.1,
        y_max=2.1,
        width=5,
        height=3,
        lines=True,
    )
    desired = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(pixels, desired)


def test_vertical_line():
    pixels = render(
        xs=np.array([1, 1]),
        ys=np.array([1, 2]),
        x_min=1,
        y_min=1,
        x_max=2.1,
        y_max=2.1,
        width=2,
        height=4,
        lines=True,
    )
    desired = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    np.testing.assert_array_equal(pixels, desired)


def test_forward_line_with_steep_upward_slope():
    pixels = render(
        xs=np.array([1, 20]),
        ys=np.array([1, 200]),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=2,
        height=4,
        lines=True,
    )
    desired = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    np.testing.assert_array_equal(pixels, desired)


def test_forward_line_with_shallow_upward_slope():
    pixels = render(
        xs=np.array([1, 20]),
        ys=np.array([1, 200]),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=4,
        height=2,
        lines=True,
    )
    desired = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
    np.testing.assert_array_equal(pixels, desired)


def test_forward_line_with_steep_downward_slope():
    pixels = render(
        xs=np.array([1, 20]),
        ys=np.array([200, 1]),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=2,
        height=4,
        lines=True,
    )
    desired = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(pixels, desired)


def test_forward_line_with_shallow_downward_slope():
    pixels = render(
        xs=np.array([1, 20]),
        ys=np.array([200, 1]),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=4,
        height=2,
        lines=True,
    )
    desired = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    np.testing.assert_array_equal(pixels, desired)


def test_backward_line_with_steep_upward_slope():
    pixels = render(
        xs=np.flip(np.array([1, 20])),
        ys=np.flip(np.array([1, 200])),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=2,
        height=4,
        lines=True,
    )
    desired = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    np.testing.assert_array_equal(pixels, desired)


def test_backward_line_with_shallow_upward_slope():
    pixels = render(
        xs=np.flip(np.array([1, 20])),
        ys=np.flip(np.array([1, 200])),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=4,
        height=2,
        lines=True,
    )
    desired = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
    np.testing.assert_array_equal(pixels, desired)


def test_backward_line_with_steep_downward_slope():
    pixels = render(
        xs=np.flip(np.array([1, 20])),
        ys=np.flip(np.array([200, 1])),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=2,
        height=4,
        lines=True,
    )
    desired = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(pixels, desired)


def test_backward_line_with_shallow_downward_slope():
    pixels = render(
        xs=np.flip(np.array([1, 20])),
        ys=np.flip(np.array([200, 1])),
        x_min=1,
        y_min=1,
        x_max=20.1,
        y_max=200.1,
        width=4,
        height=2,
        lines=True,
    )
    desired = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    np.testing.assert_array_equal(pixels, desired)


def test_draw_triangular_line():
    pixels = render(
        xs=np.array([1, 3, 2, 1]),
        ys=np.array([1, 1, 2, 1]),
        x_min=1,
        y_min=1,
        x_max=3.01,
        y_max=2.01,
        width=5,
        height=3,
        lines=True,
    )
    desired = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(pixels, desired)


def test_no_mysterious_extra_vertical_lines():
    pixels = render(
        xs=np.array([1, 1]),
        ys=np.array([0, 1]),
        x_min=3,
        y_min=0,
        x_max=6,
        y_max=1.1,
        width=60,
        height=17,
        lines=True,
    )
    np.testing.assert_array_equal(pixels, np.zeros((17, 60), dtype=int))


def test_that_vertical_lines_partially_out_of_view_are_fully_drawn():
    pixels = render(
        xs=np.array([1, 1]),
        ys=np.array([0, 1]),
        x_min=0,
        y_min=0,
        x_max=2,
        y_max=0.8,
        width=60,
        height=17,
        lines=True,
    )
    for row_index in range(17):
        assert np.sum(pixels[row_index]) == 1


def test_should_plot_no_lines_where_nan_values_are():
    pixels = render(
        xs=np.array([1.0, 2.0, 3.0]),
        ys=np.array([1.0, np.nan, 3.0]),
        x_min=0.9,
        y_min=0.9,
        x_max=3.1,
        y_max=3.1,
        width=3,
        height=3,
        lines=True,
    )
    desired = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(pixels, desired)

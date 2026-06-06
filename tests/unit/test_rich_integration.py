"""Tests for the Rich renderable integration on `plot_gen`."""

import builtins
import re

import pytest

from uniplot import plot_gen

# All tests here need rich; skip the whole module if it is not installed.
rich = pytest.importorskip("rich")
from rich.console import Console  # noqa: E402
from rich.columns import Columns  # noqa: E402
from rich.measure import Measurement  # noqa: E402
from rich.panel import Panel  # noqa: E402

ANSI_ESCAPE = re.compile(r"\033\[[\d;]+m")
YS = [1, 2, 4, 3, 5, 2]


def _render(renderable, width=80, no_color=False):
    console = Console(record=True, width=width, no_color=no_color, legacy_windows=False)
    console.print(renderable)
    return console.export_text()


def test_console_print_renders_plot():
    output = _render(plot_gen(ys=YS))
    # The plot box should be present.
    assert "┌" in output and "┘" in output


def test_no_raw_ansi_escapes_leak():
    # A colored plot printed through a no-color console must not contain raw
    # ANSI escape sequences; Rich should own all styling.
    output = _render(plot_gen(ys=[YS, [v + 1 for v in YS]], color=True), no_color=True)
    assert "\033[" not in output
    assert ANSI_ESCAPE.search(output) is None


def test_fits_inside_panel_without_overflow():
    width = 60
    output = _render(Panel(plot_gen(ys=YS)), width=width)
    for line in output.splitlines():
        assert len(line) <= width


def test_fits_inside_columns():
    output = _render(Columns([plot_gen(ys=YS), plot_gen(ys=YS)]), width=120)
    assert "┌" in output
    for line in output.splitlines():
        assert len(line) <= 120


def test_rich_measure_is_sane():
    console = Console(width=80)
    measurement = plot_gen(ys=YS).__rich_measure__(console, console.options)
    assert isinstance(measurement, Measurement)
    assert 0 < measurement.minimum <= measurement.maximum
    assert measurement.maximum <= console.options.max_width


def test_render_to_string_is_idempotent():
    plot = plot_gen(ys=YS)
    original_width = plot.options.width
    original_cap = plot.options.line_length_hard_cap

    first = plot._render_to_string()
    plot._render_to_string(max_width=40)
    second = plot._render_to_string()

    # Options are unchanged after rendering at different widths.
    assert plot.options.width == original_width
    assert plot.options.line_length_hard_cap == original_cap
    # Rendering without a width constraint is stable.
    assert first == second


def test_render_to_string_respects_max_width():
    plot = plot_gen(ys=YS)
    constrained = plot._render_to_string(max_width=40)
    for line in constrained.split("\n"):
        assert len(ANSI_ESCAPE.sub("", line)) <= 40


def test_helpful_error_when_rich_missing(monkeypatch):
    plot = plot_gen(ys=YS)
    console = Console(width=80)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rich.text" or name.startswith("rich.text"):
            raise ImportError("No module named 'rich.text'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install uniplot\[rich\]"):
        list(plot.__rich_console__(console, console.options))

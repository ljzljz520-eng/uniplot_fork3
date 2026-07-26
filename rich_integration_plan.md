# Rich Integration for Uniplot — Engineering Plan

## Goal

Make a uniplot plot a **first-class Rich renderable** so it can be dropped into
any Rich application, panel, layout, column set, or dashboard:

```python
from rich.console import Console
from uniplot import plot_gen

console = Console()
console.print(plot_gen(ys=[1, 2, 4, 3]))
```

…while preserving uniplot's core promise of being **lightweight** (no new
mandatory dependencies).

---

## Decisions (locked in)

| Topic | Decision |
|-------|----------|
| **Dependency** | `rich` is an **optional extra** (`pip install uniplot[rich]`). It is *lazily imported* inside the Rich hooks. uniplot's core deps stay `numpy` + `readchar`. |
| **Public API** | Implement the Rich protocol on the **existing `plot_gen` object** — no new public `Plot` class. `console.print(plot_gen(ys=...))` just works. |
| **Width** | **Auto-fit** to the width Rich allocates, driven through uniplot's existing `line_length_hard_cap` (total-line cap). `height` stays the vertical control. |
| **README** | Add a Rich section; also raise the profile of the numpy/pandas/polars ecosystem and the agent/Claude use case. Exact copy proposed at the end for review. |

---

## Background: how sizing already works

`uniplot/sections.py` already honors `line_length_hard_cap`:

```python
if options.line_length_hard_cap is not None:
    options.reset_width()  # idempotent: resets to initial width first
    max_y_label_length = max(len(ls) for ls in y_axis_labels)
    if 2 + options.width + 1 + max_y_label_length > options.line_length_hard_cap:
        options.width = options.line_length_hard_cap - (2 + 1 + max_y_label_length)
```

So the **right lever for Rich** is `line_length_hard_cap`, not `width`:
the cap bounds the *whole* line (`2` borders `+ width + 1` space `+ y-label`),
and because `reset_width()` runs first, re-rendering the same object at
different caps is safe and repeatable. We never fight the region `width`
directly.

## Current state / problems to fix

The WIP introduced three issues that this plan removes:

1. `uniplot/uniplot.py:3` — `from rich.text import Text` is an **unconditional
   top-level import**, but `rich` is only in the `dev` group. Any end user
   without rich installed would get an `ImportError` on `import uniplot`. → Must
   be lazy.
2. `plot_gen.__rich__` (lines 124–128) prints debug output and returns a `Text`
   from a method annotated `-> str`; it also calls `update(return_string=True)`
   (wrong — `return_string` is a constructor flag, not an `update` kwarg). →
   Replace with proper hooks.
3. `examples/10-rich_integration.py` calls `plot_gen(ys)` **positionally**,
   which binds `ys` to the first parameter `return_string`. → Fix to
   `plot_gen(ys=ys)`.

---

## Design

### 1. A non-printing render path

`plot_gen.update()` either prints or returns a string based on the `Final`
`return_string` flag set at construction — unsuitable for a renderable created
by a normal user (`return_string=False` ⇒ it prints as a side effect).

Add a private helper that renders to a string **without printing or erasing**,
optionally constrained to a max total width, and **without permanently mutating
user options**:

```python
def _render_to_string(self, max_width: Optional[int] = None) -> str:
    # Combine any user-set cap with the Rich-allocated width
    saved_cap = self.options.line_length_hard_cap
    try:
        if max_width is not None:
            self.options.line_length_hard_cap = (
                max_width if saved_cap is None else min(saved_cap, max_width)
            )
        header = sections.generate_header(self.options)
        x_lab, y_lab, matrix = sections.generate_body_raw_elements(
            self.series, self.options
        )
        body = sections.generate_body(x_lab, y_lab, matrix, self.options)
        return "\n".join(header + body)
    finally:
        self.options.line_length_hard_cap = saved_cap
        self.options.reset_width()  # undo width mutation done by the cap logic
```

`plot_to_string()` / `histogram_to_string()` can optionally be refactored to use
this later, but that is **not required** for this change (keep the diff focused).

### 2. Rich hooks on `plot_gen`

Lazy import keeps rich optional; helpful error if the extra is missing.

```python
def __rich_console__(self, console, options):
    try:
        from rich.text import Text
    except ImportError as e:
        raise ImportError(
            "Rich integration requires the 'rich' package. "
            "Install it with:  pip install uniplot[rich]"
        ) from e
    plot_string = self._render_to_string(max_width=options.max_width)
    # from_ansi parses uniplot's ANSI color codes into native Rich styling,
    # so no raw escape sequences leak into Rich output.
    yield Text.from_ansi(plot_string)


def __rich_measure__(self, console, options):
    from rich.measure import Measurement

    plot_string = self._render_to_string(max_width=options.max_width)
    widths = [_visible_len(line) for line in plot_string.split("\n")]
    natural = max(widths) if widths else 0
    return Measurement(min(natural, options.max_width), natural)
```

`_visible_len` strips ANSI before measuring, reusing the existing
`COLOR_CODE_REGEX` from `uniplot/colors.py` (no new regex). `__rich_measure__`
is what lets `Columns([...])` and tables size plots correctly.

Remove the old `__rich__` method and the top-level `from rich.text import Text`
import.

### 3. Color handling (Level 2)

uniplot emits raw ANSI for both named colors (`\033[34m`) and truecolor RGB
(`\033[38;2;r;g;bm`). `Text.from_ansi` parses both into native Rich style spans,
so colors render correctly inside containers, respect `console.no_color`, and
never appear as literal escape codes. No conversion code of our own needed.

### 4. Packaging

`pyproject.toml`:

```toml
[project.optional-dependencies]
fast = ["numba >=0.57.0"]
rich = ["rich >=13.0"]          # NEW — broad, stable floor (from_ansi + Measurement)
```

`rich` stays in the `dev` group (already pinned `>=14.3.4`) so tests run.

---

## Files to change

| File | Change |
|------|--------|
| `uniplot/uniplot.py` | Remove top-level `rich` import; remove broken `__rich__`; add `_render_to_string`, `__rich_console__`, `__rich_measure__`, and a small `_visible_len` helper (or import the regex). |
| `pyproject.toml` | Add `rich` optional-dependencies extra. |
| `examples/10-rich_integration.py` | Fix to `plot_gen(ys=ys)`; expand into a real demo (plain, in a `Panel`, colored, and a `Columns` layout). |
| `tests/unit/test_rich_integration.py` | **NEW** — see test plan. |
| `README.md` | New "Rich integration" section + ecosystem/agent emphasis (copy proposed for review). |

No changes to the core plotting pipeline — sizing reuses existing
`line_length_hard_cap` behavior.

---

## Test plan

New `tests/unit/test_rich_integration.py`, using `pytest.importorskip("rich")`:

1. **Renders via Console** — `Console(record=True)`; `console.print(plot_gen(ys=...))`;
   assert the exported text contains plot box characters and the data renders.
2. **No escape leakage** — colored plot (`color=True`) printed through a
   no-color console contains **no raw `\033[` sequences**.
3. **Fits in a Panel** — wrap in `Panel`, render at a fixed console width, assert
   **no line exceeds the console width** (overflow guard / Level 3).
4. **Columns layout** — `Columns([plot_gen(...), plot_gen(...)])` renders without
   error and within width.
5. **`__rich_measure__`** — returns a `Measurement` with `0 < minimum <= maximum`
   and `maximum <= max_width`.
6. **Idempotency** — calling `_render_to_string()` twice (and once with a
   `max_width`) leaves `options.width` / `line_length_hard_cap` unchanged.
7. **Lazy-import safety** — `import uniplot` and a normal `plot_to_string` path
   work; simulate missing rich (monkeypatch `builtins.__import__`) and assert the
   `__rich_console__` hook raises the helpful `pip install uniplot[rich]` error.

**Full gate (must be green before handing back):**

```
uv run pytest
uv run ruff check .
uv run mypy uniplot
```

---

## Out of scope (future follow-ups)

Rich-specific widgets, Textual integration, interactive Rich controls, new
plotting functionality.

Live-updating dashboards are **promoted to Phase 2** below (design + effort
assessment only — not yet scheduled for implementation).

---

## Step order

1. `pyproject.toml` extra + `uv sync` (rich already in dev).
2. Refactor `uniplot.py`: lazy import, `_render_to_string`, hooks, remove `__rich__`.
3. Write tests; iterate to green.
4. Update the example into a real multi-pattern demo; run it.
5. Run full gate (pytest + ruff + mypy).
6. Propose README copy for your review, then apply.

> **Status:** Phase 1 (Levels 1–3 above) is implemented and shipped. The section
> below is a *design + effort assessment* for a possible Phase 2; no code yet.

---

# Phase 2: Live updates (design & effort assessment)

## Target experience

```python
from rich.live import Live
from uniplot import plot_gen

plot = plot_gen(ys=initial_data)
with Live(plot, refresh_per_second=4) as live:
    while True:
        new_data = consume()  # may arrive thousands of times per second
        plot.set_data(ys=new_data)  # cheap, no printing
        # No explicit re-render needed — Live's own thread redraws at 4 fps.
```

## What already works today (verified)

* The **render side is done.** Mutating a plot's state and re-rendering through
  `__rich_console__` already reflects the new data — confirmed empirically.
* A `plot_gen(return_string=True)` already has a **non-printing `update()`**, so
  a basic Live loop functions *today* without new code.

So "does this already exist via `return_string`?" — **mechanically yes.** The
gaps are (a) the `return_string=True` flag misrepresents intent for this use
case (you ignore the returned string), and (b) **thread safety** and **rate
decoupling**, which are the parts actually worth designing.

## The interesting problem: decouple high-rate ingest from low-rate render

Goal (your words): *consume at a high rate, but update the plot only a few times
a second.*

**Key insight — we should NOT build a render thread inside uniplot.** Rich
`Live` already runs a background refresh thread at `refresh_per_second`. That is
exactly the throttled render loop we want. So the division of labor is:

| Role | Who | Rate |
|------|-----|------|
| **Producer** — push incoming data into the plot's state | user thread / main loop | high (unbounded) |
| **Renderer** — call `__rich_console__` and redraw | Rich `Live`'s own thread | fixed (e.g. 4 fps) |

Because the renderer always reads "the current state," **many producer updates
between two ticks naturally coalesce into a single render** — no queue, no
backlog, no custom timing code. This is the elegant part: the decoupling falls
out of Live's existing design once our state is thread-safe.

## Thread safety — the one real complexity

Two threads now touch shared state: the producer mutating `self.series` /
`self.options`, and Live's thread rendering. Today's `_render_to_string` makes
this unsafe because it **temporarily mutates `self.options`**
(`line_length_hard_cap` + `width`) and restores them in a `finally`. A
concurrent producer update — or two overlapping renders — can observe torn
state.

Two ways to fix it, in increasing cleanliness:

### Option A — Lock (simple, recommended first step)

Add a `threading.Lock` to `plot_gen`; acquire it around (a) state mutation in
`set_data`/`update` and (b) the whole of `_render_to_string`. Renders are
milliseconds at 4 fps, so contention is negligible. ~10 lines. The default
`Live(auto_refresh=True)` then "just works."

### Option B — Make rendering read-only (cleaner; also fixes a latent wart)

Refactor `_render_to_string` so it never mutates shared state: compute the
width-capped options on a **copy** (`dataclasses.replace(self.options, ...)`)
and render against that. Rendering becomes pure/read-only, so the only writer is
the producer's reference swap (`self.series = ...`), which is atomic under the
GIL — **little or no locking needed**, and the ugly try/finally restore
disappears. This is my recommended end state; it improves Phase 1 code quality
independent of Live.

(Realistically: do **B**, and keep a small lock for the series/options swap to be
safe against multiple producer threads.)

## De-simplifying streaming: a bounded window

The current streaming example grows its data list forever, so memory and
per-render cost climb without bound, and auto-ranging eventually flattens recent
detail. For sustained high-rate ingest we'd want a **rolling window** — keep the
last *N* points (or last *T* seconds). Options:

* A `max_points: Optional[int]` option that drops oldest points on update, **or**
* A small `append(y, x=None)` method backed by a `collections.deque(maxlen=N)`
  so high-rate ingest is O(1) and memory is bounded.

This is the piece that most directly addresses "the streaming use case always
felt very simplified." It's an independent, optional enhancement.

## Proposed public API (for a future round)

* `plot.set_data(ys=..., xs=..., **kwargs)` — replace data, no printing,
  thread-safe. (Clearer than overloading `return_string=True`; `update()` and
  existing streaming docs stay untouched.)
* *(optional)* `plot.append(y, x=None)` + `max_points=N` — bounded rolling
  stream.
* No new dependency: `Live` ships inside `rich`, already the optional extra.

## Effort & added complexity

| Item | Effort | Notes |
|------|--------|-------|
| Reuse `return_string` + document Live | **trivial** | Works now; just docs + example. |
| `set_data()` non-printing method | **low** | Thin wrapper over existing state-rebuild logic. |
| Make `_render_to_string` non-mutating (Option B) | **low** | ~10 lines via `dataclasses.replace`; also cleans up Phase 1. |
| Thread-safety lock | **low** | One `Lock`; guard swap + render. |
| Bounded window (`max_points` / `append`) | **low–moderate** | The main scope expander; touches `MultiSeries` ingestion. |
| Example `examples/11-rich_live.py` + tests + docs | **moderate** | Concurrency tests are fiddly — prefer deterministic single-thread tests + one bounded stress test. |

**Overall: small-to-moderate**, no new dependencies. The render path is already
proven. The genuine new surface area is (1) the threading contract — which
methods are safe to call from which thread, clearly documented — and (2) the
optional bounded-window feature.

## Risks / things to decide later

* **Encouraging threads in user code.** We must document precisely what is
  thread-safe. A clean recipe (producer mutates via `set_data`; Live's thread
  renders) keeps users on the safe path.
* **`Live` needs a real TTY.** In non-interactive/redirected output it degrades
  to plain prints; tests must account for this (force/record console).
* **Backwards compatibility.** Adding `set_data` is purely additive. Changing
  `_render_to_string` internals is invisible to users. The existing streaming
  `update()` behavior is preserved.

## Recommendation

If/when we build Phase 2: do **Option B** (read-only render) + a small lock +
`set_data()` + an `examples/11-rich_live.py` driven by `Live(refresh_per_second=...)`.
Treat the bounded rolling window (`max_points`/`append`) as a closely-related but
separately-scoped follow-up, since it's the item with the most design surface.

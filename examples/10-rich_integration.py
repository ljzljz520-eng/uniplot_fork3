"""
Rich integration demo.

A uniplot plot is a first-class Rich renderable: pass a `plot_gen` object
straight to `console.print`, or embed it in any Rich container such as
`Panel`, `Group` or `Columns`. Colors and width-fitting work automatically.

Requires the optional `rich` dependency:  pip install uniplot[rich]
"""

import math

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from uniplot import plot_gen

console = Console()

xs = [i / 5 for i in range(60)]
sine = [math.sin(x) for x in xs]
cosine = [math.cos(x) for x in xs]

# 1. Print a plot directly.
console.rule("[bold]console.print(plot)")
console.print(plot_gen(xs=xs, ys=sine, lines=True, title="sin(x)"))

# 2. Embed a plot in a Panel — it auto-fits the panel's width.
console.rule("[bold]Inside a Panel")
console.print(
    Panel(
        plot_gen(xs=xs, ys=sine, lines=True, title="sin(x)"),
        title="Wrapped in Rich",
    )
)

# 3. Colors are preserved (converted to native Rich styling).
console.rule("[bold]Colored multi-series")
console.print(
    plot_gen(
        xs=[xs, xs],
        ys=[sine, cosine],
        lines=True,
        color=True,
        legend_labels=["sin", "cos"],
        title="sin(x) and cos(x)",
    )
)

# 4. Side-by-side plots in a Columns layout.
console.rule("[bold]Columns layout")
console.print(
    Columns(
        [
            Panel(plot_gen(xs=xs, ys=sine, lines=True, title="sin"), title="left"),
            Panel(plot_gen(xs=xs, ys=cosine, lines=True, title="cos"), title="right"),
        ]
    )
)

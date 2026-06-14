"""
Rich Live integration demo.

A uniplot plot can be driven by `rich.live.Live` for a smoothly updating
terminal display. The pattern decouples data ingestion from rendering:

  * the producer calls `plt.set_data(...)` as fast as data arrives (no printing),
  * Rich's own background thread re-renders the plot at `refresh_per_second`.

Many `set_data` calls between two refresh ticks simply coalesce into one render,
so a fast producer does not flood the terminal. `set_data` is thread-safe.

Requires the optional `rich` dependency:  pip install uniplot[rich]
"""

import math
import time

from rich.live import Live

from uniplot import plot_gen

# A pinned y range keeps the axis steady while data scrolls past.
plt = plot_gen(ys=[0.0], lines=True, y_min=-1.2, y_max=1.2, title="Live sine")

window = 100  # number of points kept in view (a simple rolling window)
ys = []

with Live(plt, refresh_per_second=10) as live:
    for i in range(400):
        ys.append(math.sin(i / 10))
        # High-rate state update (~200 Hz here); Rich renders at 10 fps.
        plt.set_data(ys=ys[-window:])
        time.sleep(0.005)

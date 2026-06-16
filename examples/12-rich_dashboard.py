"""
Rich dashboard demo: a live system monitor built from uniplot plots.

Several uniplot plots are stacked in a Rich layout and updated in real time with
`psutil` system metrics. Each plot is a Rich renderable driven by `set_data()`;
Rich's `Live` owns the screen and redraws at a fixed rate, so the sampling loop
never floods the terminal. The plots fill the width Rich gives them, so the
dashboard scales to the size of the terminal. Press Ctrl-C to quit.

Note that uniplot draws its own frame and title, so no Rich `Panel` is used --
that would double the borders. Each plot's live value is shown in its title.

By default the dashboard runs for 5 seconds and then exits (handy for CI and
demos); pass `--duration 0` to run until Ctrl-C instead.

Requires the optional `rich` dependency plus `psutil`:
    pip install uniplot[rich] psutil
"""

import argparse
from collections import deque
import datetime
import time

import psutil
from rich.console import Group
from rich.live import Live

from uniplot import plot_gen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--duration",
    type=float,
    default=5.0,
    help="seconds to run before exiting; use 0 (or negative) to run until Ctrl-C",
)
args = parser.parse_args()

WINDOW = 100  # number of samples kept in view
INTERVAL = 0.5  # seconds between samples

# One plot per metric. CPU and memory are percentages, so pin their y-axis to
# 0..100 (pinned bounds are preserved across updates); network throughput
# auto-ranges, with only the lower bound pinned to zero. Braille gives the
# highest resolution and a sleek look.
style = dict(
    lines=True, height=8, character_set="braille", x_gridlines=[], y_gridlines=[]
)
cpu_plot = plot_gen(y_min=0, y_max=100, y_unit="%", color=["cyan"], **style)
mem_plot = plot_gen(y_min=0, y_max=100, y_unit="%", color=["magenta"], **style)
net_plot = plot_gen(
    y_min=0, y_unit=" kB/s", color=True, legend_labels=["down", "up"], **style
)

# The plots are live renderables, so the layout is built once: Live re-renders
# it on every refresh and picks up the latest data (and titles) automatically.
dashboard = Group(cpu_plot, mem_plot, net_plot)

time_hist: deque = deque(maxlen=WINDOW)  # wall-clock timestamps for the x-axis
cpu_hist: deque = deque(maxlen=WINDOW)
mem_hist: deque = deque(maxlen=WINDOW)
down_hist: deque = deque(maxlen=WINDOW)
up_hist: deque = deque(maxlen=WINDOW)

# Prime the CPU meter (the first reading is meaningless without a baseline) and
# capture initial network counters.
psutil.cpu_percent(interval=None)
prev = psutil.net_io_counters()

deadline = time.monotonic() + args.duration if args.duration > 0 else None

with Live(dashboard, refresh_per_second=4):
    while deadline is None or time.monotonic() < deadline:
        time.sleep(INTERVAL)

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        now = psutil.net_io_counters()
        down = (now.bytes_recv - prev.bytes_recv) / 1024 / INTERVAL
        up = (now.bytes_sent - prev.bytes_sent) / 1024 / INTERVAL
        prev = now

        time_hist.append(datetime.datetime.now())
        cpu_hist.append(cpu)
        mem_hist.append(mem)
        down_hist.append(down)
        up_hist.append(up)

        # Cheap, non-printing state updates; Live redraws on its own schedule.
        # Passing timestamps as `xs` gives a wall-clock x-axis; the current
        # value goes in each plot's own title.
        times = list(time_hist)
        cpu_plot.set_data(xs=times, ys=list(cpu_hist), title=f"CPU  {cpu:.0f}%")
        mem_plot.set_data(xs=times, ys=list(mem_hist), title=f"Memory  {mem:.0f}%")
        net_plot.set_data(
            xs=[times, times],
            ys=[list(down_hist), list(up_hist)],
            title=f"Network  down {down:.0f} / up {up:.0f} kB/s",
        )

# Add a shortcut such that users of the library can write `from uniplot import plot`
# instead of `from uniplot.uniplot import plot`.
from uniplot.layer_assembly import (
    BandLayer,
    EventInterval,
    MarkerLayer,
    PlotLayer,
    TextAnnotation,
    ThresholdBand,
    ViewTransform,
    ZOrder,
)
from uniplot.uniplot import (
    histogram,
    histogram_to_string,
    plot,
    plot_gen,
    plot_to_string,
)

__all__ = [
    "BandLayer",
    "EventInterval",
    "MarkerLayer",
    "PlotLayer",
    "TextAnnotation",
    "ThresholdBand",
    "ViewTransform",
    "ZOrder",
    "histogram",
    "histogram_to_string",
    "plot",
    "plot_gen",
    "plot_to_string",
]

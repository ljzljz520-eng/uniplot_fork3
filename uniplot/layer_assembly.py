from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from uniplot import layer_factory
from uniplot.colors import Color
from uniplot.layer_factory import (
    BAND_FILL_CHARACTER,
    MARKER_CHARACTER,
    ViewTransform,
)
from uniplot.options import Options

__all__ = [
    "BandLayer",
    "EventInterval",
    "HorizontalGridlineLayer",
    "MarkerLayer",
    "PlotLayer",
    "PointsLayer",
    "TextAnnotation",
    "ThresholdBand",
    "VerticalGridlineLayer",
    "ViewTransform",
    "ZOrder",
    "assemble_scatter_plot",
    "merge_layers",
]


class ZOrder(IntEnum):
    """
    Stacking order of layers, drawn from low to high. Higher layers paint over
    lower ones. Custom layers may use these levels or any integer in between.
    """

    BAND = 20
    GRID = 30
    DATA = 50
    MARKER = 60
    LABEL = 80
    ANNOTATION = 90


class PlotLayer:
    """
    Contract for everything drawn on the graph surface.

    A layer renders into a character matrix of shape `(height, width)` against
    the shared `ViewTransform` (data coordinates), and declares its `z_order`
    (stacking priority). The assembler merges all layers in z-order, so
    gridlines, data, bands, markers and annotations can be mixed freely while
    sharing one coordinate system.

    Custom layers only need to subclass this and implement `render`.
    """

    z_order: int = ZOrder.DATA

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        raise NotImplementedError


###################################
# Built-in layers: grid and data #
###################################


class HorizontalGridlineLayer(PlotLayer):
    z_order = ZOrder.GRID

    def __init__(self, y: float, index: int = 0) -> None:
        self.y = y
        self.index = index

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        return layer_factory.render_horizontal_gridline(
            y=self.y, options=options, index=self.index
        )


class VerticalGridlineLayer(PlotLayer):
    z_order = ZOrder.GRID

    def __init__(self, x: float, index: int = 0) -> None:
        self.x = x
        self.index = index

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        return layer_factory.render_vertical_gridline(
            x=self.x, options=options, index=self.index
        )


class PointsLayer(PlotLayer):
    z_order = ZOrder.DATA

    def __init__(self, xs: list[NDArray], ys: list[NDArray]) -> None:
        self.xs = xs
        self.ys = ys

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        return layer_factory.render_points(xs=self.xs, ys=self.ys, options=options)


#######################################
# Built-in layers: overlays in data  #
# coordinates (bands/markers/text)   #
#######################################


class BandLayer(PlotLayer):
    """
    Filled rectangle in data coordinates. An unset `x_low`/`x_high` (or
    `y_low`/`y_high`) pair means the band spans the full axis extent. An
    optional `label` is drawn at the top-left corner of the band.
    """

    def __init__(
        self,
        x_low: float | None = None,
        x_high: float | None = None,
        y_low: float | None = None,
        y_high: float | None = None,
        *,
        fill_character: str = BAND_FILL_CHARACTER,
        color: Color | None = None,
        label: str | None = None,
        label_color: Color | None = None,
        z_order: int = int(ZOrder.BAND),
    ) -> None:
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high
        self.fill_character = fill_character
        self.color = color
        self.label = label
        self.label_color = label_color
        self.z_order = int(z_order)

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        return layer_factory.render_band(
            view,
            x_low=self.x_low,
            x_high=self.x_high,
            y_low=self.y_low,
            y_high=self.y_high,
            fill_character=self.fill_character,
            color=self.color,
            label=self.label,
            label_color=self.label_color,
        )


class ThresholdBand(BandLayer):
    """
    Horizontal (default, `axis="y"`) or vertical (`axis="x"`) band marking a
    threshold range, e.g. a "normal operating" region between two values.
    """

    def __init__(
        self,
        low: float,
        high: float,
        *,
        axis: str = "y",
        fill_character: str = BAND_FILL_CHARACTER,
        color: Color | None = None,
        z_order: int = int(ZOrder.BAND),
    ) -> None:
        if axis == "y":
            super().__init__(
                y_low=low,
                y_high=high,
                fill_character=fill_character,
                color=color,
                z_order=z_order,
            )
        elif axis == "x":
            super().__init__(
                x_low=low,
                x_high=high,
                fill_character=fill_character,
                color=color,
                z_order=z_order,
            )
        else:
            raise ValueError(f"Invalid axis '{axis}', expected 'x' or 'y'.")


class EventInterval(BandLayer):
    """
    Vertical (default, `axis="x"`) or horizontal (`axis="y"`) band marking an
    event spanning a time/value interval, with an optional `label`.
    """

    def __init__(
        self,
        start: float,
        end: float,
        *,
        axis: str = "x",
        label: str | None = None,
        fill_character: str = BAND_FILL_CHARACTER,
        color: Color | None = None,
        label_color: Color | None = None,
        z_order: int = int(ZOrder.BAND),
    ) -> None:
        if axis == "x":
            super().__init__(
                x_low=start,
                x_high=end,
                fill_character=fill_character,
                color=color,
                label=label,
                label_color=label_color,
                z_order=z_order,
            )
        elif axis == "y":
            super().__init__(
                y_low=start,
                y_high=end,
                fill_character=fill_character,
                color=color,
                label=label,
                label_color=label_color,
                z_order=z_order,
            )
        else:
            raise ValueError(f"Invalid axis '{axis}', expected 'x' or 'y'.")


class MarkerLayer(PlotLayer):
    """
    Emphasis markers at data positions. Accepts either scalar coordinates or
    array-like sequences of coordinates.
    """

    def __init__(
        self,
        x: float | NDArray,
        y: float | NDArray,
        *,
        character: str = MARKER_CHARACTER,
        color: Color | None = None,
        z_order: int = int(ZOrder.MARKER),
    ) -> None:
        self.x = x
        self.y = y
        self.character = character
        self.color = color
        self.z_order = int(z_order)

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        return layer_factory.render_markers(
            view,
            xs=np.asarray(self.x, dtype=float),
            ys=np.asarray(self.y, dtype=float),
            character=self.character,
            color=self.color,
        )


class TextAnnotation(PlotLayer):
    """
    Text label anchored at a data position. `anchor` is `"start"`, `"middle"`
    or `"end"` (horizontal placement relative to `x`).
    """

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        *,
        color: Color | None = None,
        anchor: str = "start",
        z_order: int = int(ZOrder.ANNOTATION),
    ) -> None:
        if anchor not in ("start", "middle", "end"):
            raise ValueError(
                f"Invalid anchor '{anchor}', expected 'start', 'middle' or 'end'."
            )
        self.x = x
        self.y = y
        self.text = str(text)
        self.color = color
        self.anchor = anchor
        self.z_order = int(z_order)

    def render(self, view: ViewTransform, options: Options) -> NDArray:
        return layer_factory.render_text_annotation(
            view,
            x=self.x,
            y=self.y,
            text=self.text,
            color=self.color,
            anchor=self.anchor,
        )


#############
# Assembly #
#############


def assemble_scatter_plot(
    xs: list[NDArray], ys: list[NDArray], options: Options
) -> NDArray:
    """
    Assemble the graph surface for a scatter plot.

    The default layers (gridlines and data points) are composed with any
    user-supplied `options.layers`, then merged in z-order. All layers share
    the same data-coordinate view, so overlays stay aligned when panning or
    zooming.
    """
    all_layers = default_layers(xs=xs, ys=ys, options=options)
    all_layers.extend(getattr(options, "layers", None) or [])
    return merge_layers(all_layers, options=options)


def default_layers(
    xs: list[NDArray], ys: list[NDArray], options: Options
) -> list[PlotLayer]:
    """
    The layers drawn on every plot: horizontal gridlines, vertical gridlines,
    then the data points, in stable insertion order within their z-order.
    """
    layers: list[PlotLayer] = [
        HorizontalGridlineLayer(y=y, index=i) for i, y in enumerate(options.y_gridlines)
    ]
    layers += [
        VerticalGridlineLayer(x=x, index=i) for i, x in enumerate(options.x_gridlines)
    ]
    layers.append(PointsLayer(xs=xs, ys=ys))
    return layers


def merge_layers(layers: list[PlotLayer], options: Options) -> NDArray:
    """
    Render every layer against the shared view and paint them onto one matrix,
    sorted by ascending `z_order`. Ties keep insertion order, so a layer added
    later with the same z-order paints over earlier ones.
    """
    view = ViewTransform.from_options(options)
    merged_layer = layer_factory.blank_character_matrix(
        width=options.width, height=options.height
    )

    # Stable sort by z-order; the insertion index breaks ties.
    ordered_layers = sorted(
        enumerate(layers),
        key=lambda pair: (int(getattr(pair[1], "z_order", 0)), pair[0]),
    )

    for _, layer in ordered_layers:
        character_layer = layer.render(view, options)
        # Just checking
        assert character_layer.shape == (options.height, options.width), (
            f"{character_layer.shape} != {(options.height, options.width)}"
        )
        to_replace_mask = character_layer != ""
        merged_layer[to_replace_mask] = character_layer[to_replace_mask]

    return merged_layer

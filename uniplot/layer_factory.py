import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uniplot import character_sets, pixel_matrix
from uniplot.colors import COLOR_RESET_CODE, Color
from uniplot.conversions import convert_matrix_to_rows_of_submatrices
from uniplot.discretizer import discretize
from uniplot.options import CharacterSet, Options

Y_GRIDLINE_CHARACTERS = ["▔", "─", "▁"]

# Default appearance of overlay layers
BAND_FILL_CHARACTER = "░"
MARKER_CHARACTER = "◆"


@dataclass(frozen=True)
class ViewTransform:
    """
    Shared mapping from data coordinates to character-matrix cells.

    Every layer renders against the same `ViewTransform`, so gridlines, data
    points, bands, markers and annotations all share one coordinate system:
    columns count left to right, rows count top to bottom, and both are clipped
    to the current view window.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int
    height: int

    @classmethod
    def from_options(cls, options: Options) -> "ViewTransform":
        return cls(
            x_min=float(options.x_min),
            x_max=float(options.x_max),
            y_min=float(options.y_min),
            y_max=float(options.y_max),
            width=options.width,
            height=options.height,
        )

    def x_is_visible(self, x: float) -> bool:
        x = float(x)
        return not math.isnan(x) and self.x_min <= x <= self.x_max

    def y_is_visible(self, y: float) -> bool:
        y = float(y)
        return not math.isnan(y) and self.y_min <= y <= self.y_max

    def col(self, x: float) -> int:
        """
        Column of the cell containing `x`, clipped to the matrix width.
        """
        col_index = discretize(
            x=x, x_min=self.x_min, x_max=self.x_max, steps=self.width
        )
        return min(max(col_index, 0), self.width - 1)

    def row(self, y: float) -> int:
        """
        Row of the cell containing `y`, clipped to the matrix height. Rows are
        counted from the top, mirroring the character-matrix indexing.
        """
        row_index = (
            self.height
            - 1
            - discretize(x=y, x_min=self.y_min, x_max=self.y_max, steps=self.height)
        )
        return min(max(row_index, 0), self.height - 1)

    def col_span(
        self, x_low: float | None, x_high: float | None
    ) -> tuple[int, int] | None:
        """
        Half-open column range `[start, end)` covered by the data interval
        `[x_low, x_high]`. `None` bounds mean "full axis extent". Returns
        `None` if the interval is outside of the view.
        """
        if x_low is None or x_high is None:
            return (0, self.width)
        return _cell_span(
            float(x_low), float(x_high), self.x_min, self.x_max, self.width
        )

    def row_span(
        self, y_low: float | None, y_high: float | None
    ) -> tuple[int, int] | None:
        """
        Half-open *screen-row* range `[top, bottom)` covered by the data
        interval `[y_low, y_high]`. `None` bounds mean "full axis extent".
        Returns `None` if the interval is outside of the view.
        """
        if y_low is None or y_high is None:
            return (0, self.height)
        bottom_span = _cell_span(
            float(y_low), float(y_high), self.y_min, self.y_max, self.height
        )
        if bottom_span is None:
            return None
        # Convert bottom-counted bin indices into top-counted screen rows
        bottom_start, bottom_end = bottom_span
        return (self.height - bottom_end, self.height - bottom_start)


def blank_character_matrix(width: int, height: int) -> NDArray:
    """
    Initialize an empty character matrix as a NumPy array.
    """
    return _init_character_matrix(width, height, value=" ")


def render_horizontal_gridline(y: float, options: Options, index: int = 0) -> NDArray:
    """
    Render the pixel matrix that only consists of a line where the `y` value is.

    Because a character is higher than wide, this is rendered with "super-resolution"
    Unicode characters.
    """
    pixels = _init_character_matrix(width=options.width, height=options.height)
    if y < options.y_min or y >= options.y_max:
        return pixels

    if options.character_set == CharacterSet.ASCII:
        y_index = (
            options.height
            - 1
            - discretize(
                x=y, x_min=options.y_min, x_max=options.y_max, steps=options.height
            )
        )
        character = "─"
        if options.y_gridlines_color and len(options.y_gridlines_color) > 0:
            color = options.y_gridlines_color[index % len(options.y_gridlines_color)]
            character = color.colorize(character)
        pixels[y_index, :] = character
    else:
        y_index_superresolution = (
            3 * options.height
            - 1
            - discretize(
                x=y, x_min=options.y_min, x_max=options.y_max, steps=3 * options.height
            )
        )
        y_index = int(y_index_superresolution / 3)
        character = Y_GRIDLINE_CHARACTERS[y_index_superresolution % 3]
        if options.y_gridlines_color and len(options.y_gridlines_color) > 0:
            color = options.y_gridlines_color[index % len(options.y_gridlines_color)]
            character = color.colorize(character)
        pixels[y_index, :] = character

    return pixels


def render_vertical_gridline(x: float, options: Options, index: int = 0) -> NDArray:
    """
    Render the pixel matrix that only consists of a line where the `x` value is.
    """
    pixels = _init_character_matrix(width=options.width, height=options.height)
    if float(x) < float(options.x_min) or float(x) >= float(options.x_max):
        return pixels

    x_index = discretize(
        x=x, x_min=options.x_min, x_max=options.x_max, steps=options.width
    )

    character = "│"
    if options.x_gridlines_color and len(options.x_gridlines_color) > 0:
        color = options.x_gridlines_color[index % len(options.x_gridlines_color)]
        character = color.colorize(character)
    pixels[:, x_index] = character

    return pixels


def render_points(xs: list[NDArray], ys: list[NDArray], options: Options) -> NDArray:
    # Setup: determine submatrix size, encoder, and character list
    scale_w, scale_h, encoder, char_list = _set_up_submatrix_shape_and_encoders(options)
    full_width = scale_w * options.width
    full_height = scale_h * options.height

    # Render input points into a full pixel matrix
    px_matrix = np.zeros((full_height, full_width), dtype=np.int32)
    for series_index, (x, y, lines) in enumerate(zip(xs, ys, options.lines)):
        px_matrix = pixel_matrix.render(
            xs=x,
            ys=y,
            x_min=options.x_min,
            x_max=options.x_max,
            y_min=options.y_min,
            y_max=options.y_max,
            width=full_width,
            height=full_height,
            lines=lines,
            pixels=px_matrix,
            layer=series_index + 1,
        )  # type: ignore

    # Initialize output character matrix
    char_matrix = _init_character_matrix(width=options.width, height=options.height)

    # Break down pixel matrix into submatrices per character cell
    submatrices = convert_matrix_to_rows_of_submatrices(
        px_matrix,
        width_submatrix=scale_w,
        height_submatrix=scale_h,
    )

    color_matrix = submatrices.max(axis=2) - 1  # For optional color use

    if options.character_set != CharacterSet.ASCII:
        max_vals = submatrices.max(axis=2, keepdims=True)
        submatrices = ((submatrices == max_vals) & (max_vals > 0)).astype(int)

    # Encode submatrices into integer values representing character shapes
    int_matrix = (submatrices * encoder).sum(axis=2)
    mask_nonzero = int_matrix != 0

    # Character decoding and assignment
    decoder = np.array(char_list)
    decoder[..., 0] = ""  # Blank character for zero entries
    char_matrix[mask_nonzero] = decoder[int_matrix[mask_nonzero]]

    # Apply color if enabled
    if options.color:
        color_enable_strings = [c.enable_str() for c in options.color]

        decoder_colored = np.array(
            [
                np.char.add(np.char.add(c, char_list), COLOR_RESET_CODE)
                for c in color_enable_strings
            ]
        )
        indices = (
            color_matrix[mask_nonzero] % len(color_enable_strings),
            int_matrix[mask_nonzero],
        )
        decoder_colored[..., 0] = ""
        char_matrix[mask_nonzero] = decoder_colored[indices]

    return char_matrix


def print_raw_pixel_matrix(pixels: NDArray, verbose: bool = False) -> None:
    """
    Just print the pixels.

    Used for testing and debugging.
    """
    join_char = "," if verbose else ""
    for row in pixels:
        print("DEBUG: (" + join_char.join(list(row)) + ")")


def render_band(
    view: ViewTransform,
    *,
    x_low: float | None = None,
    x_high: float | None = None,
    y_low: float | None = None,
    y_high: float | None = None,
    fill_character: str = BAND_FILL_CHARACTER,
    color: Color | None = None,
    label: str | None = None,
    label_color: Color | None = None,
) -> NDArray:
    """
    Render a filled band (threshold region, event interval, etc.) defined in
    data coordinates. Each pair of bounds is optional: an unset pair means the
    band spans the full axis extent. An optional `label` is drawn at the top
    left corner of the band.
    """
    pixels = _init_character_matrix(width=view.width, height=view.height)

    col_range = view.col_span(x_low, x_high)
    row_range = view.row_span(y_low, y_high)
    if col_range is None or row_range is None:
        return pixels
    col_start, col_end = col_range
    row_start, row_end = row_range
    if col_end <= col_start or row_end <= row_start:
        return pixels

    fill = color.colorize(fill_character) if color is not None else fill_character
    pixels[row_start:row_end, col_start:col_end] = fill

    if label:
        label_color = label_color if label_color is not None else color
        _draw_text_in_place(
            pixels,
            text=label,
            row=row_start,
            col_start=col_start,
            col_end=col_end,
            color=label_color,
        )

    return pixels


def render_markers(
    view: ViewTransform,
    xs: NDArray,
    ys: NDArray,
    *,
    character: str = MARKER_CHARACTER,
    color: Color | None = None,
) -> NDArray:
    """
    Render one marker character per data point, at cell granularity. Points
    outside of the view are skipped.
    """
    pixels = _init_character_matrix(width=view.width, height=view.height)
    marker = color.colorize(character) if color is not None else character

    xs = np.atleast_1d(np.asarray(xs, dtype=float)).ravel()
    ys = np.atleast_1d(np.asarray(ys, dtype=float)).ravel()
    for x, y in zip(xs, ys):
        if not view.x_is_visible(x) or not view.y_is_visible(y):
            continue
        pixels[view.row(y), view.col(x)] = marker

    return pixels


def render_text_annotation(
    view: ViewTransform,
    x: float,
    y: float,
    text: str,
    *,
    color: Color | None = None,
    anchor: str = "start",
) -> NDArray:
    """
    Render a text label at a data position. `anchor` controls the horizontal
    placement relative to the anchor point: `"start"` (text starts at `x`),
    `"middle"` (centered on `x`) or `"end"` (text ends at `x`). The label is
    clipped to the plot width; if the anchor point is outside of the view,
    nothing is drawn.
    """
    pixels = _init_character_matrix(width=view.width, height=view.height)
    if not view.x_is_visible(x) or not view.y_is_visible(y) or text == "":
        return pixels

    anchor_col = view.col(x)
    if anchor == "middle":
        start_col = anchor_col - len(text) // 2
    elif anchor == "end":
        start_col = anchor_col - len(text) + 1
    else:
        start_col = anchor_col

    _draw_text_in_place(
        pixels,
        text=text,
        row=view.row(y),
        col_start=start_col,
        col_end=start_col + len(text),
        color=color,
    )
    return pixels


###########
# private #
###########


def _cell_span(
    low: float, high: float, minimum: float, maximum: float, cells: int
) -> tuple[int, int] | None:
    """
    Half-open cell range `[start, end)` covering the data interval
    `[low, high]` along one axis, or `None` if it is fully outside of the
    view. Degenerate zero-width intervals cover the single cell they fall in.
    """
    if math.isnan(low) or math.isnan(high) or maximum <= minimum:
        return None
    if low > high:
        low, high = high, low
    if high < minimum or low > maximum:
        return None

    scale = cells / (maximum - minimum)
    start = math.floor((low - minimum) * scale)
    end = math.ceil((high - minimum) * scale)
    start = max(0, min(cells, start))
    end = max(0, min(cells, end))

    if end <= start:
        # Zero-width interval inside of (or on the boundary of) one cell
        single_cell = int((max(minimum, min(maximum, low)) - minimum) * scale)
        single_cell = max(0, min(cells - 1, single_cell))
        return (single_cell, single_cell + 1)

    return (start, end)


def _draw_text_in_place(
    pixels: NDArray,
    *,
    text: str,
    row: int,
    col_start: int,
    col_end: int,
    color: Color | None,
) -> None:
    """
    Write `text` into one row, keeping it inside both the matrix and the
    given column window (used to keep band labels inside their band).
    """
    height, width = pixels.shape
    if row < 0 or row >= height:
        return
    for offset, character in enumerate(text):
        col = col_start + offset
        if col < 0 or col >= width or col < col_start or col >= col_end:
            continue
        pixels[row, col] = color.colorize(character) if color is not None else character


def _init_character_matrix(width: int, height: int, value: str = "") -> NDArray:
    return np.full((height, width), fill_value=value, dtype="<U25")


def _set_up_submatrix_shape_and_encoders(
    options: Options,
) -> tuple[int, int, NDArray, list[str]]:
    if options.character_set == CharacterSet.ASCII:
        return (1, 1, np.array([1], ndmin=3), [" "] + options.force_ascii_characters)
    if options.character_set == CharacterSet.BRAILLE:
        return (
            2,
            4,
            np.array([1, 2, 4, 8, 16, 32, 64, 128], ndmin=3),
            character_sets.BRAILLE_CHARACTER_SET,
        )
    return (2, 2, np.array([1, 2, 4, 8], ndmin=3), character_sets.UNICODE_CHARACTER_SET)

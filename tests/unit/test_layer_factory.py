import numpy as np

from uniplot import plot_to_string
from uniplot.colors import Color
from uniplot.layer_assembly import (
    EventInterval,
    MarkerLayer,
    PlotLayer,
    TextAnnotation,
    ThresholdBand,
    ZOrder,
    assemble_scatter_plot,
    merge_layers,
)
from uniplot.layer_factory import (
    ViewTransform,
    render_band,
    render_markers,
    render_points,
    render_text_annotation,
)
from uniplot.options import CharacterSet, Options


def test_ascii_characters_without_color():
    xs = [np.array([1, 2, 3])]
    ys = [np.array([2, 1, 4])]
    opts = Options(
        character_set=CharacterSet.ASCII, x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0
    )
    matrix = render_points(xs, ys, opts)
    assert "+" in matrix


def test_ascii_characters_with_terminal_color():
    xs = [np.array([1, 2, 3])]
    ys = [np.array([2, 1, 4])]
    opts = Options(
        character_set=CharacterSet.ASCII,
        x_min=0.0,
        x_max=5.0,
        y_min=0.0,
        y_max=5.0,
        color=[Color.from_terminal("red")],
    )
    matrix = render_points(xs, ys, opts)
    assert _wrap_in_red("+") in matrix


def test_ascii_characters_with_rgb_color():
    xs = [np.array([1, 2, 3])]
    ys = [np.array([2, 1, 4])]
    opts = Options(
        character_set=CharacterSet.ASCII,
        x_min=0.0,
        x_max=5.0,
        y_min=0.0,
        y_max=5.0,
        height=3,
        width=3,
        color=[Color.from_rgb(146, 255, 12)],
    )
    matrix = render_points(xs, ys, opts)
    assert _wrap_in_lime_green("+") in matrix


def test_block_characters_without_color():
    xs = [np.array([0, 5])]
    ys = [np.array([0, 5])]
    opts = Options(x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0)
    matrix = render_points(xs, ys, opts)
    assert "▖" in matrix


def test_block_characters_with_color():
    xs = [np.array([0, 5])]
    ys = [np.array([0, 5])]
    opts = Options(
        x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0, color=[Color.from_terminal("red")]
    )
    matrix = render_points(xs, ys, opts)
    assert _wrap_in_red("▖") in matrix


def test_block_characters_where_the_top_layer_should_hide_the_lower_layer():
    pixel_width = 5.0 / 10
    second_pixel_coord = 1.5 * pixel_width
    xs = [np.array([0, second_pixel_coord]), np.array([0, 5])]
    ys = [np.array([second_pixel_coord, 0]), np.array([0, 5])]
    opts = Options(
        x_min=0.0,
        x_max=5.0,
        y_min=0.0,
        y_max=5.0,
        width=6,
        height=6,
        lines=[False, False],
        color=False,
    )
    matrix = render_points(xs, ys, opts)
    # Make sure that the result is the same as in `test_block_characters_without_color`,
    # because the second series overwrites the first one.
    assert "▖" in matrix


def test_braille_characters_without_color():
    xs = [np.array([0, 5])]
    ys = [np.array([0, 5])]
    opts = Options(
        x_min=0.0,
        x_max=5.0,
        y_min=0.0,
        y_max=5.0,
        width=3,
        height=3,
        character_set=CharacterSet.BRAILLE,
    )
    matrix = render_points(xs, ys, opts)
    assert "⡀" in matrix


def test_braille_characters_with_color():
    xs = [np.array([0, 5])]
    ys = [np.array([0, 5])]
    opts = Options(
        x_min=0.0,
        x_max=5.0,
        y_min=0.0,
        y_max=5.0,
        width=3,
        height=3,
        character_set=CharacterSet.BRAILLE,
        color=[Color.from_terminal("red")],
    )
    matrix = render_points(xs, ys, opts)
    print(matrix)
    assert _wrap_in_red("⡀") in matrix


#########################################
# Testing: shared coordinate transform #
#########################################


def _test_view() -> ViewTransform:
    return ViewTransform(
        x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, width=10, height=10
    )


def test_view_transform_col_and_row():
    view = _test_view()
    assert view.col(0.0) == 0
    assert view.col(5.0) == 5
    # Values on the far edge are clipped into the last cell
    assert view.col(10.0) == 9
    assert view.row(10.0) == 0
    assert view.row(0.0) == 9


def test_view_transform_spans():
    view = _test_view()
    assert view.col_span(2.0, 4.0) == (2, 4)
    assert view.col_span(None, None) == (0, 10)
    # A zero-width interval still covers the single cell it falls in
    assert view.col_span(5.0, 5.0) == (5, 6)
    # Fully outside of the view
    assert view.col_span(11.0, 12.0) is None
    # Screen rows are counted from the top
    assert view.row_span(2.0, 4.0) == (6, 8)
    assert view.row_span(None, None) == (0, 10)


##########################################
# Testing: band/marker/text primitives #
##########################################


def test_render_vertical_band():
    view = _test_view()
    matrix = render_band(view, x_low=2.0, x_high=4.0)
    assert (matrix[:, 2:4] == "░").all()
    assert (matrix[:, 0:2] == "").all()
    assert (matrix[:, 4:] == "").all()


def test_render_horizontal_band_with_label():
    view = _test_view()
    matrix = render_band(view, y_low=2.0, y_high=4.0, label="E")
    # Label sits at the top-left corner of the band and replaces the fill there
    assert matrix[6, 0] == "E"
    assert matrix[6, 1] == "░"
    assert (matrix[7, :] == "░").all()
    assert (matrix[0:6, :] == "").all()


def test_render_band_outside_of_view_is_blank():
    view = _test_view()
    matrix = render_band(view, x_low=20.0, x_high=30.0, y_low=20.0, y_high=30.0)
    assert (matrix == "").all()


def test_render_markers():
    view = _test_view()
    matrix = render_markers(view, xs=[5.0, 50.0], ys=[8.0, 8.0])
    assert matrix[1, 5] == "◆"
    # The out-of-view marker is skipped
    assert "◆" in matrix
    assert (matrix == "◆").sum() == 1


def test_render_markers_with_custom_character_and_color():
    view = _test_view()
    matrix = render_markers(
        view, xs=5.0, ys=5.0, character="x", color=Color.from_terminal("red")
    )
    assert matrix[4, 5] == _wrap_in_red("x")


def test_render_text_annotation_anchors():
    view = _test_view()
    start_matrix = render_text_annotation(view, 1.0, 9.0, "hi", anchor="start")
    assert start_matrix[0, 1] == "h"
    assert start_matrix[0, 2] == "i"

    middle_matrix = render_text_annotation(view, 5.0, 5.0, "AB", anchor="middle")
    assert middle_matrix[4, 4] == "A"
    assert middle_matrix[4, 5] == "B"

    end_matrix = render_text_annotation(view, 5.0, 5.0, "AB", anchor="end")
    assert end_matrix[4, 4] == "A"
    assert end_matrix[4, 5] == "B"


def test_render_text_annotation_is_clipped_and_skips_invisible_anchor():
    view = _test_view()
    clipped = render_text_annotation(view, 9.0, 5.0, "XYZ")
    assert clipped[4, 9] == "X"
    assert (clipped == "Y").sum() == 0

    invisible = render_text_annotation(view, 50.0, 50.0, "XYZ")
    assert (invisible == "").all()


###################################
# Testing: layer z-order contract #
###################################


class _SingleCharLayer(PlotLayer):
    def __init__(self, char: str, z_order: int) -> None:
        self.char = char
        self.z_order = z_order

    def render(self, view: ViewTransform, options: Options) -> np.ndarray:
        matrix = np.full((options.height, options.width), fill_value="", dtype="<U25")
        matrix[0, 0] = self.char
        return matrix


def test_merge_layers_paints_higher_z_order_on_top():
    options = Options(width=4, height=4, x_gridlines=[], y_gridlines=[])
    matrix = merge_layers(
        [_SingleCharLayer("LOW", ZOrder.BAND), _SingleCharLayer("HIGH", ZOrder.DATA)],
        options=options,
    )
    assert matrix[0, 0] == "HIGH"


def test_merge_layers_same_z_order_keeps_insertion_order():
    options = Options(width=4, height=4, x_gridlines=[], y_gridlines=[])
    matrix = merge_layers(
        [
            _SingleCharLayer("FIRST", ZOrder.DATA),
            _SingleCharLayer("SECOND", ZOrder.DATA),
        ],
        options=options,
    )
    assert matrix[0, 0] == "SECOND"


def test_data_points_paint_over_bands_and_markers_over_points():
    band_options = Options(
        character_set=CharacterSet.ASCII,
        x_min=0.0,
        x_max=4.0,
        y_min=0.0,
        y_max=4.0,
        width=4,
        height=4,
        x_gridlines=[],
        y_gridlines=[],
        layers=[ThresholdBand(0.0, 4.0)],
    )
    band_matrix = assemble_scatter_plot(
        xs=[np.array([1.0])], ys=[np.array([1.0])], options=band_options
    )
    assert band_matrix[2, 1] == "+"
    assert "░" in band_matrix

    marker_options = Options(
        character_set=CharacterSet.ASCII,
        x_min=0.0,
        x_max=4.0,
        y_min=0.0,
        y_max=4.0,
        width=4,
        height=4,
        x_gridlines=[],
        y_gridlines=[],
        layers=[MarkerLayer(1.0, 1.0, character="*")],
    )
    marker_matrix = assemble_scatter_plot(
        xs=[np.array([1.0])], ys=[np.array([1.0])], options=marker_options
    )
    assert marker_matrix[2, 1] == "*"


def test_gridlines_paint_over_bands():
    options = Options(
        character_set=CharacterSet.ASCII,
        x_min=0.0,
        x_max=4.0,
        y_min=0.0,
        y_max=4.0,
        width=4,
        height=4,
        x_gridlines=[],
        y_gridlines=[0.0],
        layers=[ThresholdBand(0.0, 4.0)],
    )
    matrix = assemble_scatter_plot(
        xs=[np.array([])], ys=[np.array([])], options=options
    )
    # The horizontal gridline at y=0 is on the bottom row and stays visible
    assert matrix[3, 0] == "─"


def test_overlay_layers_end_to_end():
    output = plot_to_string(
        ys=[1, 2, 3],
        xs=[1, 2, 3],
        x_min=0.0,
        x_max=4.0,
        y_min=0.0,
        y_max=4.0,
        width=20,
        height=10,
        x_gridlines=[],
        layers=[
            ThresholdBand(1.0, 2.0),
            EventInterval(2.5, 3.5, label="E"),
            MarkerLayer(2.0, 3.0),
            TextAnnotation(0.2, 3.5, "peak"),
        ],
    )
    assert "░" in output
    assert "◆" in output
    assert "peak" in output
    assert "E" in output


###########
# private #
###########


def _wrap_in_red(char: str) -> str:
    return "\033[31m" + str(char) + "\033[0m"


def _wrap_in_lime_green(char: str) -> str:
    return "\033[38;2;146;255;12m" + str(char) + "\033[0m"

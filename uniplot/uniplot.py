from typing import List, Dict, Optional, Final, Any, TYPE_CHECKING
from readchar import readkey, key

from uniplot.multi_series import MultiSeries
from uniplot.options import Options
from uniplot.param_initializer import validate_and_transform_options
import uniplot.colors as colors
import uniplot.sections as sections
import uniplot.plot_elements as elements

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.measure import Measurement


def plot(ys: Any, xs: Optional[Any] = None, **kwargs) -> None:
    """
    2D plot on the terminal.

    Parameters:

    - `ys` are the y coordinates of the points to plot. This parameter is
      mandatory and can either be a list or a list of lists, or the equivalent
      NumPy array.
    - `xs` are the x coordinates of the points to plot. This parameter is
      optional and can either be a `None` or of the same shape as `ys`.
    - Any additional keyword arguments are passed to the
      `uniplot.options.Options` class.
    """
    plt = plot_gen(xs=xs, ys=ys, **kwargs)

    # Main loop for interactive mode. Will only be executed once when not in
    # interactive mode.
    first_iteration: bool = True
    while first_iteration or plt.options.interactive:
        plt.update()

        if plt.options.interactive:
            plt.print_subscript("Move h/j/k/l, zoom u/n, or r to reset. q to quit.")
            key_pressed = readkey()

            # Here we support 3 ways to move: Vim-style, arrow keys and FPS-style
            if key_pressed in ["h", key.LEFT, "a"]:
                plt.options.shift_view_left()
            elif key_pressed in ["l", key.RIGHT, "d"]:
                plt.options.shift_view_right()
            elif key_pressed in ["j", key.DOWN, "s"]:
                plt.options.shift_view_down()
            elif key_pressed in ["k", key.UP, "w"]:
                plt.options.shift_view_up()
            elif key_pressed in ["u", "]"]:
                plt.options.zoom_in()
            elif key_pressed in ["n", "["]:
                plt.options.zoom_out()
            elif key_pressed == "r":
                plt.options.reset_view()
            elif key_pressed in ["q", "Q", key.ESC]:
                break

        first_iteration = False


class plot_gen:
    def __init__(self, return_string=False, **kwargs) -> None:
        self.default_arguments: Final[Dict] = kwargs
        self.last_nr_of_lines: int = 0
        self.return_string: Final[bool] = return_string
        self.series: MultiSeries = MultiSeries([])
        self.options: Options = Options()
        if "ys" in kwargs:
            self.series = MultiSeries(xs=kwargs.get("xs"), ys=kwargs.get("ys", []))
            if "xs" in kwargs:
                del kwargs["xs"]
            del kwargs["ys"]
            self.options = validate_and_transform_options(
                series=self.series, kwargs=kwargs
            )

    def update(self, **kwargs) -> Optional[str]:
        header_buffer: List[str] = []
        body_buffer: List[str] = []

        full_kwargs = {**self.default_arguments, **kwargs}

        if "xs" in kwargs or "ys" in kwargs:
            self.series = MultiSeries(
                xs=full_kwargs.get("xs"), ys=full_kwargs.get("ys")
            )
        if len(kwargs.keys()) > 0 or self.options is None:
            # New options provided, so regenerate `self.options`
            # NOTE This overwrites the view window if not supplied explicitely
            if "xs" in full_kwargs:
                del full_kwargs["xs"]
            del full_kwargs["ys"]
            self.options = validate_and_transform_options(
                series=self.series, kwargs=full_kwargs
            )

        header_buffer = sections.generate_header(self.options)

        # Generate and collect plot content
        body_buffer = []
        (
            x_axis_labels,
            y_axis_labels,
            pixel_character_matrix,
        ) = sections.generate_body_raw_elements(self.series, self.options)
        body_buffer += sections.generate_body(
            x_axis_labels, y_axis_labels, pixel_character_matrix, self.options
        )

        # Delete plot before we re-draw
        if not self.return_string:
            elements.erase_previous_lines(self.last_nr_of_lines)

        # Output plot
        output = "\n".join(header_buffer + body_buffer)
        self.last_nr_of_lines = elements.count_lines(output)
        if self.return_string:
            return output
        print(output)
        return None

    def print_subscript(self, text: str) -> None:
        self.last_nr_of_lines += elements.count_lines(text)
        print(text)

    def _render_to_string(self, max_width: Optional[int] = None) -> str:
        """
        Render the current plot to a string without printing or erasing, and
        without permanently mutating the options.

        If `max_width` is given, the total line length is constrained to it via
        the existing `line_length_hard_cap` mechanism (combined with any cap the
        user already set). This is what makes the plot fit the space Rich
        allocates.
        """
        saved_cap = self.options.line_length_hard_cap
        try:
            if max_width is not None:
                self.options.line_length_hard_cap = (
                    max_width if saved_cap is None else min(saved_cap, max_width)
                )
            header_buffer = sections.generate_header(self.options)
            (
                x_axis_labels,
                y_axis_labels,
                pixel_character_matrix,
            ) = sections.generate_body_raw_elements(self.series, self.options)
            body_buffer = sections.generate_body(
                x_axis_labels, y_axis_labels, pixel_character_matrix, self.options
            )
            return "\n".join(header_buffer + body_buffer)
        finally:
            # Restore the options to their pre-render state. The cap logic in
            # `sections` mutates `width`, so reset both.
            self.options.line_length_hard_cap = saved_cap
            self.options.reset_width()

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        """
        Rich renderable protocol. Allows `console.print(plot_gen(ys=...))` and
        embedding plots in Rich containers such as `Panel`, `Group`, `Columns`.

        Requires the optional `rich` dependency: `pip install uniplot[rich]`.
        """
        try:
            from rich.text import Text
        except ImportError as e:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "Rich integration requires the 'rich' package. "
                "Install it with:  pip install uniplot[rich]"
            ) from e

        plot_string = self._render_to_string(max_width=options.max_width)
        # `from_ansi` parses uniplot's ANSI color codes into native Rich styling,
        # so colors are preserved and no raw escape sequences leak into output.
        yield Text.from_ansi(plot_string)

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        """
        Report the plot's width to Rich so layouts (e.g. `Columns`, tables) can
        size it correctly.
        """
        from rich.measure import Measurement

        plot_string = self._render_to_string(max_width=options.max_width)
        widths = [
            len(colors.COLOR_CODE_REGEX.sub("", line))
            for line in plot_string.split("\n")
        ]
        natural = max(widths) if widths else 0
        return Measurement(min(natural, options.max_width), natural)


def plot_to_string(ys: Any, xs: Optional[Any] = None, **kwargs) -> str:
    """
    Same as `plot`, but the return type is string. Ignores the `interactive`
    option.

    Can be used to integrate uniplot in other applications, or if the output is
    desired to be not stdout.
    """
    plt = plot_gen(return_string=True)
    return str(plt.update(xs=xs, ys=ys, **kwargs))


#####################################
# Experimental features, see Readme #
#####################################


def histogram(
    xs: Any,
    bins: int = 20,
    bins_min: Optional[float] = None,
    bins_max: Optional[float] = None,
    **kwargs,
) -> None:
    """
    Plot a histogram to the terminal.

    Parameters:

    - `xs` are the values of the points to plot. This parameter is mandatory
      and can either be a list or a list of lists, or the equivalent NumPy
      array.
    - Any additional keyword arguments are passed to the
      `uniplot.options.Options` class.
    """
    # HACK Use the `MultiSeries` constructor to cast values to uniform format
    multi_series = MultiSeries(ys=xs)
    xs_histo, ys_histo = elements.prepare_histogram(
        multi_series, bins, bins_min, bins_max
    )

    # Histograms usually make sense only with lines
    kwargs["lines"] = True
    plot(xs=xs_histo, ys=ys_histo, **kwargs)


def histogram_to_string(
    xs: Any,
    bins: int = 20,
    bins_min: Optional[float] = None,
    bins_max: Optional[float] = None,
    **kwargs,
) -> str:
    """
    Same as `histogram`, but the return type is string. Ignores the `interactive`
    option.

    Can be used to integrate uniplot in other applications, or if the output is
    desired to be not stdout.
    """
    # HACK Use the `MultiSeries` constructor to cast values to uniform format
    multi_series = MultiSeries(ys=xs)
    xs_histo, ys_histo = elements.prepare_histogram(
        multi_series, bins, bins_min, bins_max
    )

    plt = plot_gen(return_string=True)
    # Histograms usually make sense only with lines
    kwargs["lines"] = True
    return str(plt.update(xs=xs_histo, ys=ys_histo, **kwargs))

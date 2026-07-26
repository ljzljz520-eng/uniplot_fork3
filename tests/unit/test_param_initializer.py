import pytest

from uniplot.multi_series import MultiSeries
from uniplot.param_initializer import validate_and_transform_options


def test_passing_simple_list():
    series = MultiSeries(ys=[1, 2, 3])
    options = validate_and_transform_options(series=series)
    assert not options.interactive


def test_lines_option_with_simple_list():
    series = MultiSeries(ys=[1, 2, 3])
    options = validate_and_transform_options(series=series, kwargs={"lines": True})
    assert options.lines == [True]


def test_lines_option_with_multiple_lists():
    series = MultiSeries(ys=[[1, 2, 3], [100, 1000, 10000]])
    options = validate_and_transform_options(
        series=series, kwargs={"lines": [False, True]}
    )
    assert options.lines == [False, True]


def test_invalid_lines_option_with_multiple_lists():
    series = MultiSeries(ys=[[1, 2, 3], [100, 1000, 10000]])
    with pytest.raises(ValueError):
        validate_and_transform_options(series=series, kwargs={"lines": [False]})


def test_unit_scaling_option_is_normalized():
    series = MultiSeries(ys=[1, 2, 3])
    options = validate_and_transform_options(
        series=series, kwargs={"x_unit_scaling": "SI "}
    )
    assert options.x_unit_scaling == "si"


def test_invalid_unit_scaling_option_raises():
    series = MultiSeries(ys=[1, 2, 3])
    with pytest.raises(ValueError):
        validate_and_transform_options(
            series=series, kwargs={"x_unit_scaling": "currency"}
        )

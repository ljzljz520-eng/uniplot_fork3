import numpy as np

from uniplot.axis_labels.label_set import LabelSet, SI_PREFIXES


def test_redering_to_string():
    labels = np.array([0, 1, 2])
    ls = LabelSet(
        labels, x_min=-0.5, x_max=2.5, available_space=30, vertical_direction=False
    )
    render = ls.render()
    assert len(render) == 1
    assert len(render[0]) >= 15
    assert " 0 " in render[0]
    assert ls.compute_if_render_does_overlap() is False


def test_redering_detects_overlap():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    ls = LabelSet(
        labels, x_min=-0.5, x_max=2.5, available_space=3, vertical_direction=False
    )
    assert ls.compute_if_render_does_overlap() is True


def test_redering_to_string_with_unit():
    labels = np.array([0, 1, 2])
    ls = LabelSet(
        labels,
        x_min=-0.5,
        x_max=2.5,
        available_space=30,
        unit=" apples",
        vertical_direction=False,
    )
    render = ls.render()
    assert len(render) == 1
    assert len(render[0]) >= 15
    assert " 0 apples" in render[0]
    assert ls.compute_if_render_does_overlap() is False


def test_si_units_scale_up_to_kilo():
    labels = np.array([0.0, 2000.0, 4000.0])
    ls = LabelSet(
        labels,
        x_min=-100.0,
        x_max=4100.0,
        available_space=40,
        unit="m",
        unit_as_si=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "2km" in render
    assert "4km" in render


def test_si_units_scale_down_to_milli():
    labels = np.array([0.0, 0.002, 0.004])
    ls = LabelSet(
        labels,
        x_min=-0.0001,
        x_max=0.0041,
        available_space=40,
        unit="m",
        unit_as_si=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "2mm" in render
    assert "4mm" in render


def test_si_units_stay_in_base_below_kilo():
    labels = np.array([0.0, 200.0, 400.0])
    ls = LabelSet(
        labels,
        x_min=-10.0,
        x_max=410.0,
        available_space=40,
        unit="m",
        unit_as_si=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "200m" in render
    assert "400m" in render
    assert "k" not in render


def test_si_units_disabled_by_default():
    labels = np.array([0.0, 2000.0, 4000.0])
    ls = LabelSet(
        labels,
        x_min=-100.0,
        x_max=4100.0,
        available_space=40,
        unit="m",
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "2,000m" in render
    assert "k" not in render


def test_si_prefix_applied_even_with_blank_unit():
    # A blank unit still gets a prefix, so 200,000 renders as "200k".
    labels = np.array([0.0, 100_000.0, 200_000.0])
    ls = LabelSet(
        labels,
        x_min=-1000.0,
        x_max=210_000.0,
        available_space=40,
        unit="",
        unit_as_si=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "100k" in render
    assert "200k" in render


def test_si_prefix_inserted_before_first_non_whitespace_of_unit():
    # The prefix goes right before the unit's first non-whitespace character,
    # so a leading space is preserved: " m" -> " km", never "k m".
    ls = LabelSet(
        np.array([0.0, 2000.0, 4000.0]),
        x_min=-100.0,
        x_max=4100.0,
        available_space=40,
        unit=" m",
        unit_as_si=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "2 km" in render
    assert "4 km" in render
    assert "k m" not in render


def test_apply_si_prefix_placement():
    ls = LabelSet(np.array([1.0]))
    assert ls._apply_si_prefix("k", "") == "k"
    assert ls._apply_si_prefix("k", "m") == "km"
    assert ls._apply_si_prefix("k", " m") == " km"
    assert ls._apply_si_prefix("k", "  apples") == "  kapples"
    # An empty prefix leaves the unit untouched, whitespace included.
    assert ls._apply_si_prefix("", " m") == " m"


def _si_label_strings(labels, unit="g", log=False):
    """Helper: the composed SI label strings for a set of labels.

    On a log axis the labels are stored as exponents, so the given values are
    converted to their base-10 logarithm first.
    """
    a = np.array(labels, dtype=float)
    if log:
        a = np.log10(a)
    lo, hi = float(a.min()), float(a.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ls = LabelSet(
        a,
        x_min=lo - pad,
        x_max=hi + pad,
        available_space=60,
        unit=unit,
        unit_as_si=True,
        log=log,
        vertical_direction=False,
    )
    return ls._compute_label_strings()


def test_si_small_integers_stay_in_base_unit():
    assert _si_label_strings([0, 1, 2, 5, 10, 20]) == [
        "0g",
        "1g",
        "2g",
        "5g",
        "10g",
        "20g",
    ]


def test_si_stays_in_base_unit_up_to_and_including_1000():
    # Anchoring to the smallest nonzero label keeps these in grams rather than
    # tipping to fractional kilograms (0.25kg etc.).
    assert _si_label_strings([0, 250, 500, 750, 1000]) == [
        "0g",
        "250g",
        "500g",
        "750g",
        "1,000g",
    ]


def test_si_boundary_around_1000_does_not_tip_eagerly():
    assert _si_label_strings([999, 1000, 1001]) == ["999g", "1,000g", "1,001g"]


def test_si_scales_up_when_all_labels_are_large():
    assert _si_label_strings([0, 1000, 2000, 3000, 4000, 5000]) == [
        "0kg",
        "1kg",
        "2kg",
        "3kg",
        "4kg",
        "5kg",
    ]


def test_si_tiny_linear_values_scale_down():
    assert _si_label_strings([0, 0.001, 0.002, 0.003]) == [
        "0mg",
        "1mg",
        "2mg",
        "3mg",
    ]


def test_si_log_axis_spans_many_decades_with_per_label_prefix():
    labels = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    assert _si_label_strings(labels, log=True) == [
        "1ng",
        "10ng",
        "100ng",
        "1µg",
        "10µg",
        "100µg",
        "1mg",
    ]


def test_si_uses_extreme_prefixes_at_the_edges_of_the_known_range():
    # Yotta (10^24) and yocto (10^-24) are the largest/smallest known prefixes.
    yotta = LabelSet(
        np.array([1e24]), unit="m", unit_as_si=True, vertical_direction=False
    )
    assert yotta._si_divisor_and_prefix() == (1e24, "Y")
    yocto = LabelSet(
        np.array([1e-24]), unit="m", unit_as_si=True, vertical_direction=False
    )
    assert yocto._si_divisor_and_prefix() == (1e-24, "y")


def test_si_linear_falls_back_to_plain_above_known_range():
    # Beyond yotta (10^24) there is no known prefix, so no scaling is applied.
    ls = LabelSet(
        np.array([1e27, 2e27]),
        x_min=0.9e27,
        x_max=2.1e27,
        unit="m",
        unit_as_si=True,
        vertical_direction=False,
    )
    assert ls._si_divisor_and_prefix() == (1.0, "")


def test_si_linear_falls_back_to_plain_below_known_range():
    # Below yocto (10^-24) there is no known prefix, so no scaling is applied.
    ls = LabelSet(
        np.array([1e-25, 2e-25]),
        x_min=0.9e-25,
        x_max=2.1e-25,
        unit="m",
        unit_as_si=True,
        vertical_direction=False,
    )
    assert ls._si_divisor_and_prefix() == (1.0, "")


def test_si_log_per_label_falls_back_to_plain_outside_known_range():
    # Use "s" as the unit since (unlike "m") it is not also an SI prefix letter,
    # so we can safely check that no prefix leaked into the output.
    ls = LabelSet(np.array([1.0]), unit="s", unit_as_si=True)
    # In range: prefixes are used.
    assert ls._format_value_si(1e24) == "1Ys"
    assert ls._format_value_si(1e-24) == "1ys"
    # Out of range: plain value plus the bare unit, no invented prefix.
    assert ls._format_value_si(1e27).endswith("s")
    assert not any(p and p in ls._format_value_si(1e27) for p in SI_PREFIXES.values())
    assert ls._format_value_si(1e-27).endswith("s")


def test_log_axis_renders_actual_values_not_exponents():
    # Labels on a log axis are stored as exponents.
    labels = np.array([-1.0, 0.0, 1.0, 2.0])
    ls = LabelSet(
        labels,
        x_min=-1.1,
        x_max=2.1,
        available_space=60,
        log=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "^" not in render
    assert "0.1" in render
    assert "100" in render


def test_log_axis_fractional_exponents_become_real_values():
    # 10^0.3 ≈ 2, 10^0.6 ≈ 4. Previously these rendered as "10^0.3" etc.
    labels = np.array([0.0, 0.3, 0.6])
    ls = LabelSet(
        labels,
        x_min=-0.05,
        x_max=0.65,
        available_space=60,
        log=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "^" not in render
    assert "2" in render
    assert "3.98" in render


def test_log_axis_uses_per_label_si_prefix():
    # Exponents -1, 1, 3 => 0.1 m, 10 m, 1000 m => each gets its own prefix.
    labels = np.array([-1.0, 1.0, 3.0])
    ls = LabelSet(
        labels,
        x_min=-1.1,
        x_max=3.1,
        available_space=60,
        unit="m",
        unit_as_si=True,
        log=True,
        vertical_direction=False,
    )
    render = ls.render()[0]
    assert "100mm" in render
    assert "10m" in render
    assert "1km" in render


def test_floating_point_wrapping_issue():
    """
    This is covering issue #15. It tests that the right number of digits are
    displayed even is only a part of the labels have digits on both sides of
    the decimal point.
    """
    ls = LabelSet(
        labels=np.array([-1.4, -0.9, -0.4]),
        x_min=-1.846526999372103,
        x_max=-0.1651564799721942,
        available_space=17,
        vertical_direction=True,
    )
    render = ls.render()
    assert "-1.4" in render
    assert "-0.4" in render


def test_floating_point_rounding_issue():
    """
    This is covering an issue observed in testing, where labels 0.99 and
    1.00 were displayed as "0" and "1". The underlying reason was that
    0.99 at zero digits would incorrectyly render to "0". This test ensures
    that we have fixed the issue.
    """
    ls = LabelSet(
        labels=np.array([0.99, 1]),
        x_min=0.985,
        x_max=1.05,
        available_space=17,
        vertical_direction=True,
    )
    render = ls.render()
    assert "0.99" in render
    assert "1.00" in render


def test_string_representation_of_full_integers():
    """
    Make sure we add the right number of digits.

    Inspired by the Rust version.
    """
    ls = LabelSet(labels=np.array([1.0, 2.0, 3.0]))
    represent = ls._find_shortest_string_representation()
    assert represent[0] == "1"
    assert represent[1] == "2"
    assert represent[2] == "3"


def test_string_representation_of_half_integers():
    """
    Make sure we add the right number of digits even when they are only
    needed for a subset of the labels.

    Inspired by the Rust version.
    """
    ls = LabelSet(labels=np.array([1.0, 1.5, 2.0]))
    represent = ls._find_shortest_string_representation()
    assert represent[0] == "1.0"
    assert represent[1] == "1.5"
    assert represent[2] == "2.0"


def test_string_representation_of_large_integers():
    """
    Make sure we add the right number of digits.

    Inspired by the Rust version.
    """
    ls = LabelSet(labels=np.array([0.0, 100.0, 1_000_000.0]))
    represent = ls._find_shortest_string_representation()
    assert represent[0] == "0"
    assert represent[1] == "100"
    assert represent[2] == "1,000,000"


def test_string_representation_of_large_negative_integers():
    """
    Make sure we add the right number of digits.

    Inspired by the Rust version.
    """
    ls = LabelSet(
        labels=np.array([0.0, -100.0, -10_000.0]),
        x_min=1.0,
        x_max=2.0,
        available_space=17,
    )
    represent = ls._find_shortest_string_representation()
    assert represent[0] == "0"
    assert represent[1] == "-100"
    assert represent[2] == "-10,000"

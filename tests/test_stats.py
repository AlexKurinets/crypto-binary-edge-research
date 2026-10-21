import numpy as np
import pytest

from edgelab.stats import benjamini_hochberg, brier, cluster_bootstrap, min_detectable_edge, outcomes_needed, reliability


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])


def test_repeating_a_trade_inside_its_hour_does_not_narrow_the_interval():
    rng = np.random.default_rng(4)
    v = rng.choice([0.5, -0.5], 300) + 0.02
    clusters = np.arange(300)
    once = cluster_bootstrap(v, np.ones(300), clusters, 1000, 9)
    five = cluster_bootstrap(np.repeat(v, 5), np.ones(1500), np.repeat(clusters, 5), 1000, 9)
    naive = cluster_bootstrap(np.repeat(v, 5), np.ones(1500), np.arange(1500), 1000, 9)
    assert five.mean == pytest.approx(once.mean)
    assert (five.ci_lo, five.ci_hi) == pytest.approx((once.ci_lo, once.ci_hi))
    assert naive.ci_hi - naive.ci_lo < 0.6 * (once.ci_hi - once.ci_lo)


def test_fair_coin_interval_covers_zero():
    rng = np.random.default_rng(5)
    v = rng.choice([0.5, -0.5], 2000)
    est = cluster_bootstrap(v, np.ones(2000), np.arange(2000), 1000, 1)
    assert est.ci_lo < 0 < est.ci_hi and est.p_value > 0.05


def test_power_for_a_month_of_fifteen_minute_windows():
    assert float(min_detectable_edge(30 * 96)) == pytest.approx(0.0232, abs=2e-4)
    assert outcomes_needed(0.02) == 3865


def test_brier_and_reliability():
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    rel = reliability([0.05, 0.15, 0.95], [0, 1, 1])
    assert list(rel["bin"]) == [0, 1, 9]

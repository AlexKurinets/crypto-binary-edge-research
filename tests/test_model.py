import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from edgelab.model import HarModel, effective_seconds, fit_har, fit_recalibration, prob_at_or_above


def test_averaging_window_counts_a_third():
    assert effective_seconds(60) == pytest.approx(20)
    assert effective_seconds(300) == pytest.approx(260)
    with pytest.raises(ValueError):
        effective_seconds(30)


def test_at_the_money_is_one_half_and_strikes_are_monotone():
    strikes = np.array([63_500, 64_000, 64_500])
    p = prob_at_or_above(64_000, strikes, 1_200, 1e-7, dof=5)
    assert p[1] == pytest.approx(0.5)
    assert p[0] > p[1] > p[2]


def test_averaging_makes_far_strikes_less_likely():
    averaged = prob_at_or_above(64_000, 64_100, 120, 2e-7, dof=5, window_s=60)
    point = prob_at_or_above(64_000, 64_100, 120, 2e-7, dof=5, window_s=0)
    assert averaged < point


def test_student_t_is_scaled_to_the_forecast_variance():
    rng = np.random.default_rng(0)
    dof, var = 5, 4e-6
    draws = rng.standard_t(dof, 400_000) * np.sqrt(var * (dof - 2) / dof)
    assert draws.var() == pytest.approx(var, rel=0.03)
    strike = 64_000 * np.exp(0.003)
    analytic = prob_at_or_above(64_000, strike, 60 + 3 * 60, var / (180 + 20) * 60, dof)
    assert analytic == pytest.approx((draws >= 0.003).mean(), abs=0.003)


def test_recalibration_recovers_a_known_distortion():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 60_000)
    y = rng.random(60_000) < expit(0.3 + 0.8 * logit(p))
    r = fit_recalibration(p, y)
    assert r.a == pytest.approx(0.3, abs=0.05)
    assert r.b == pytest.approx(0.8, abs=0.05)


def test_har_recovers_linear_coefficients_and_floors_predictions():
    rng = np.random.default_rng(2)
    rows = pd.DataFrame(rng.uniform(1e-8, 1e-6, size=(500, 3)), columns=["rv15", "rv60", "rv1440"])
    rows["fwd_rv60"] = 1e-9 + 0.2 * rows["rv15"] + 0.3 * rows["rv60"] + 0.4 * rows["rv1440"]
    har = fit_har(rows)
    assert (har.b15, har.b60, har.b1440) == pytest.approx((0.2, 0.3, 0.4), abs=1e-6)
    assert HarModel(-1.0, 0, 0, 0).predict(0, 0, 4e-7) == pytest.approx(1e-7)

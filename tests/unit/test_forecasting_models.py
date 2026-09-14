"""
Fase 10 - Pruebas T-08, T-09, T-10, T-11 (RF-04, RF-13, RF-14, RF-15).

Casos cubiertos de la matriz oficial:
- MODEL-A-01: caminata aleatoria sin deriva, benchmark analítico.
- MODEL-B-01: lognormal con deriva, benchmark analítico.
- PATH-B-01: trayectoria analítica completa de 1 a H (Modelo B).
- HORIZON-01: regla de suficiencia T>=max(2m,5H)+H+9 (H=27 la viola con
  T=539, m=252; H=1,5,17 la cumplen).
- WF-01: validación walk-forward con AAPL real del fixture, T=539, m=252,
  H=20, últimos 10 orígenes expansivos (510 a 519).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics import log_returns
from src.forecasting import (
    ForecastModel,
    fit_lognormal_drift,
    fit_random_walk,
    forecast_path,
    is_walkforward_sufficient,
    max_valid_horizon,
    terminal_distribution,
    walk_forward_validate,
)
from tests.fixtures_path import FIXTURE_20_PRICES

ATOL = 1e-8
RTOL = 1e-10


def test_model_a_terminal_distribution():
    """MODEL-A-01: P0=100, sigma=0.0158113883008419, H=4 (caminata aleatoria)."""
    model = ForecastModel(name="A", mean_per_period=0.0, std_per_period=0.0158113883008419, n_train=999)
    dist = terminal_distribution(model, p0=100.0, horizon=4)
    assert np.isclose(dist.m_h, 0.0, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.v_h, 0.001, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.v_h ** 0.5, 0.031622776601684, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.median_price, 100.0, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.mean_price, 100.050012502084, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.q05_price, 94.931478005803, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.q95_price, 105.339137344820, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.prob_above_entry, 0.5, atol=ATOL, rtol=RTOL)


def test_model_b_terminal_distribution():
    """MODEL-B-01: P0=100, mu=0.01, sigma=0.02, H=3 (lognormal con deriva)."""
    model = ForecastModel(name="B", mean_per_period=0.01, std_per_period=0.02, n_train=999)
    dist = terminal_distribution(model, p0=100.0, horizon=3)
    assert np.isclose(dist.m_h, 0.03, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.v_h, 0.0012, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.median_price, 103.045453395352, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.mean_price, 103.107299219281, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.q05_price, 97.338129194272, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.q95_price, 109.087420863215, atol=ATOL, rtol=RTOL)
    assert np.isclose(dist.prob_above_entry, 0.806761884614384, atol=ATOL, rtol=RTOL)


def test_forecast_path_model_b_path01():
    """PATH-B-01: trayectoria analítica h=1,2,3 del Modelo B."""
    model = ForecastModel(name="B", mean_per_period=0.01, std_per_period=0.02, n_train=999)
    steps = forecast_path(model, p0=100.0, horizon=3)
    expected = {
        1: dict(median=101.005016708417, mean=101.025219731994, q05=97.736307609620, q95=104.383045050327),
        2: dict(median=102.020134002676, mean=102.060950218976, q05=97.382517216574, q95=106.878606544712),
        3: dict(median=103.045453395352, mean=103.107299219281, q05=97.338129194272, q95=109.087420863215),
    }
    assert len(steps) == 3
    for s in steps:
        exp = expected[s.step]
        assert np.isclose(s.median_price, exp["median"], atol=ATOL, rtol=RTOL)
        assert np.isclose(s.mean_price, exp["mean"], atol=ATOL, rtol=RTOL)
        assert np.isclose(s.q05_price, exp["q05"], atol=ATOL, rtol=RTOL)
        assert np.isclose(s.q95_price, exp["q95"], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("horizon", [1, 5, 17])
def test_horizon_valid_within_limit_horizon01(horizon):
    """HORIZON-01: T=539, m=252 -> H=1,5,17 cumplen T>=max(2m,5H)+H+9."""
    assert is_walkforward_sufficient(n_returns=539, periods_per_year=252, horizon=horizon)


def test_horizon_27_violates_sufficiency_rule_horizon01():
    """HORIZON-01: H=27 viola T>=max(2m,5H)+H+9 con T=539, m=252."""
    assert not is_walkforward_sufficient(n_returns=539, periods_per_year=252, horizon=27)


@pytest.mark.parametrize("horizon", [0, -1])
def test_horizon_invalid_values_rejected(horizon):
    """T-10: H cero o negativo nunca es suficiente (se rechaza)."""
    assert not is_walkforward_sufficient(n_returns=539, periods_per_year=252, horizon=horizon)


def test_max_valid_horizon_horizon01():
    """HORIZON-01: con T=539, m=252, el límite justificado es H_max=26 (H=27 ya lo excede)."""
    assert max_valid_horizon(n_returns=539, periods_per_year=252) == 26


@pytest.mark.skipif(not FIXTURE_20_PRICES.exists(), reason="Falta tests/fixtures/Fixture_20_activos_sintetico_TBS_EIA.csv")
def test_walk_forward_aapl_wf01():
    """
    WF-01: AAPL real del fixture, T=539, m=252, H=20, n_min=504, orígenes
    510..519 (10 orígenes expansivos consecutivos), Modelo A y Modelo B.
    """
    raw = pd.read_csv(FIXTURE_20_PRICES, parse_dates=["date"])
    aapl_prices = raw[raw["ticker"] == "AAPL"].sort_values("date").set_index("date")["adjusted_close"]
    returns = log_returns(aapl_prices)
    assert len(returns) == 539

    wf_a = walk_forward_validate(returns, fit_random_walk, "A", horizon=20, periods_per_year=252)
    wf_b = walk_forward_validate(returns, fit_lognormal_drift, "B", horizon=20, periods_per_year=252)

    assert wf_a.ok and wf_b.ok
    assert [o.origin_index for o in wf_a.origins] == list(range(510, 520))

    assert np.isclose(wf_a.mae, 0.047838991594412, atol=ATOL, rtol=RTOL)
    assert np.isclose(wf_a.rmse, 0.049889389325085, atol=ATOL, rtol=RTOL)
    assert np.isclose(wf_a.coverage_90, 0.200000000000, atol=ATOL, rtol=RTOL)
    assert wf_a.directional_accuracy is None  # modelo A: mediana logarítmica cero -> N/A

    assert np.isclose(wf_b.mae, 0.050673668010446, atol=ATOL, rtol=RTOL)
    assert np.isclose(wf_b.rmse, 0.052630163459918, atol=ATOL, rtol=RTOL)
    assert np.isclose(wf_b.coverage_90, 0.100000000000, atol=ATOL, rtol=RTOL)
    assert np.isclose(wf_b.directional_accuracy, 0.000000000000, atol=ATOL, rtol=RTOL)
    assert wf_b.n_directional_defined == 10
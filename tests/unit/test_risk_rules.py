"""
Fase 10 - Pruebas T-02, T-13, T-14, T-15 (RF-13, RF-16, RF-17, RF-18).

Casos cubiertos de la matriz oficial:
- CONST-01: precios constantes -> mu=sigma=VaR=0, todos los cuantiles=P0,
  sin división por cero, y no señal porque SL=E=TP.
- VAR99-A-01 / VAR99-B-01: VaR paramétrico al 99% de confianza (Modelos A y B).
- COST-B-01: niveles y probabilidades con costos de compra/venta (Modelo B).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics import log_returns
from src.forecasting import ForecastModel, fit_random_walk
from src.risk_rules import RiskParameters, compute_var, evaluate_signal, terminal_probabilities

ATOL = 1e-8
RTOL = 1e-10


def test_constant_prices_no_signal_const01():
    """CONST-01: P=100 repetido, Modelo A, H=4, pL=.05, pU=.95, cb=cs=0."""
    import pandas as pd

    const_prices = pd.Series([100.0] * 10)
    returns = log_returns(const_prices)
    model = fit_random_walk(returns)

    assert model.mean_per_period == 0.0
    assert model.std_per_period == 0.0

    var_result = compute_var(model, p0=100.0, horizon=4, confidence=0.95, capital=10000.0)
    assert var_result.var_fraction == 0.0
    assert var_result.var_dollar == 0.0

    signal = evaluate_signal(
        model, p0=100.0, horizon=4,
        params=RiskParameters(p_l=0.05, p_u=0.95, br_min=1.0, c_b=0.0, c_s=0.0),
    )
    assert signal.sl_h == 100.0
    assert signal.tp_h == 100.0
    assert signal.ok is False  # SL_H = E = TP_H -> no señal, sin dividir por cero
    assert len(signal.reasons) > 0

    probs = terminal_probabilities(model, p0=100.0, horizon=4, p_be=signal.p_be)
    assert np.isclose(probs.prob_win + probs.prob_lose + probs.prob_neutral, 1.0, atol=ATOL, rtol=RTOL)
    assert probs.prob_neutral == 1.0  # rama determinista, precio terminal = P_BE = 100


def test_var_99_model_a_var99a01():
    """VAR99-A-01: MODEL-A-01 con c=.99, C=10000."""
    model = ForecastModel(name="A", mean_per_period=0.0, std_per_period=0.0158113883008419, n_train=999)
    var_result = compute_var(model, p0=100.0, horizon=4, confidence=0.99, capital=10000.0)
    assert np.isclose(var_result.var_fraction, 0.070924784140090, atol=ATOL, rtol=RTOL)
    assert np.isclose(var_result.var_dollar, 709.247841400895, atol=ATOL, rtol=RTOL)


def test_var_99_model_b_var99b01():
    """VAR99-B-01: MODEL-B-01 con c=.99, C=10000."""
    model = ForecastModel(name="B", mean_per_period=0.01, std_per_period=0.02, n_train=999)
    var_result = compute_var(model, p0=100.0, horizon=3, confidence=0.99, capital=10000.0)
    assert np.isclose(var_result.var_fraction, 0.049328834922328, atol=ATOL, rtol=RTOL)
    assert np.isclose(var_result.var_dollar, 493.288349223279, atol=ATOL, rtol=RTOL)


def test_costs_and_levels_model_b_costb01():
    """COST-B-01: MODEL-B-01 con cb=cs=.001, pL=.05, pU=.95."""
    model = ForecastModel(name="B", mean_per_period=0.01, std_per_period=0.02, n_train=999)
    signal = evaluate_signal(
        model, p0=100.0, horizon=3,
        params=RiskParameters(p_l=0.05, p_u=0.95, br_min=1.0, c_b=0.001, c_s=0.001),
    )
    assert np.isclose(signal.p_be, 100.200200200200, atol=ATOL, rtol=RTOL)
    assert np.isclose(signal.u_neto, 8.878333442352, atol=ATOL, rtol=RTOL)
    assert np.isclose(signal.d_neto, 2.859208934922, atol=ATOL, rtol=RTOL)
    assert np.isclose(signal.br_neto, 3.105171270946, atol=ATOL, rtol=RTOL)

    probs = terminal_probabilities(model, p0=100.0, horizon=3, p_be=signal.p_be)
    assert np.isclose(probs.prob_win, 0.790538273989, atol=ATOL, rtol=RTOL)
    assert np.isclose(probs.prob_lose, 0.209461726011, atol=ATOL, rtol=RTOL)
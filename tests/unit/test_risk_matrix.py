# tests/unit/test_risk_matrix.py
"""
Escalamiento v2 - Sección 4 (riesgo conjunto) y Sección 5 (Markowitz).

Casos cubiertos, con el caso de control oficial de dos activos
(``dos_activos.json``) de la guía v2:

- P2-6: el log-retorno de la cartera NO es la suma ponderada de logs
  individuales (ejemplo +10 %/-10 % con pesos iguales).
- P2-7: la covarianza manual (vía numpy) coincide con ``sample_moments``
  (que usa ``pandas.cov(ddof=1)``).
- P2-8: una varianza cero produce correlación indefinida (NaN), no un
  valor forzado.
- P2-9: PSD, simetría, autovalores y condicionamiento se revisan
  explícitamente.
- P2-13: la GMV de dos activos coincide con el control analítico
  ``Sigma^-1 1 / (1'Sigma^-1 1)``.
- P2-16: máximo rendimiento esperado, tangente y óptimo personal son tres
  carteras distintas (no se confunden entre sí).
- P2-17: con cotas largo-solo (sin apalancamiento), los pesos siguen
  sumando 1 y están dentro de [0,1].
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.risk_matrix import (
    check_covariance_matrix,
    correlation_from_covariance,
    efficient_frontier,
    gmv_weights,
    gmv_weights_closed_form,
    max_expected_return_weights,
    personal_optimal_weights,
    personal_optimal_weights_closed_form,
    portfolio_log_return_vs_weighted_log_sum,
    portfolio_variance,
    risk_contributions,
    sample_moments,
    tangency_weights,
    tangency_weights_closed_form,
)
from tests.fixtures_path import DOS_ACTIVOS

ATOL = 1e-8
RTOL = 1e-8

pytestmark = pytest.mark.skipif(
    not DOS_ACTIVOS.exists(),
    reason="Falta tests/fixtures/v2/dos_activos.json (caso de control oficial v2).",
)


@pytest.fixture(scope="module")
def caso_dos_activos() -> dict:
    with open(DOS_ACTIVOS, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture()
def mu_sigma(caso_dos_activos: dict) -> tuple[pd.Series, pd.DataFrame]:
    assets = caso_dos_activos["assets"]
    mu = pd.Series(caso_dos_activos["mu"], index=assets)
    sigma = pd.DataFrame(caso_dos_activos["Sigma"], index=assets, columns=assets)
    return mu, sigma


def test_portfolio_log_return_differs_from_weighted_log_sum_p2_6():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    simple_returns_row = pd.Series({"A": 0.10, "B": -0.10})

    log_port, weighted_log_sum = portfolio_log_return_vs_weighted_log_sum(
        weights, simple_returns_row
    )

    # Retorno simple de la cartera = 0 => log(1+0) = 0.
    assert log_port == pytest.approx(0.0, abs=ATOL)
    # Promedio de logs individuales es negativo: 0.5*ln(1.10)+0.5*ln(0.90) < 0.
    assert weighted_log_sum < -1e-4
    assert log_port != pytest.approx(weighted_log_sum, abs=1e-4)


def test_sample_moments_matches_manual_covariance_p2_7():
    rng = np.random.default_rng(20261018)
    data = pd.DataFrame(rng.normal(size=(50, 3)), columns=["A", "B", "C"])

    mu, sigma = sample_moments(data)

    manual_mu = data.mean(axis=0)
    manual_sigma = pd.DataFrame(
        np.cov(data.values, rowvar=False, ddof=1), index=data.columns, columns=data.columns
    )
    pd.testing.assert_series_equal(mu, manual_mu, check_exact=False, atol=ATOL, rtol=RTOL)
    pd.testing.assert_frame_equal(sigma, manual_sigma, check_exact=False, atol=ATOL, rtol=RTOL)


def test_zero_variance_asset_gives_undefined_correlation_p2_8():
    sigma = pd.DataFrame(
        {"A": [0.0, 0.0], "B": [0.0, 0.04]}, index=["A", "B"]
    )

    corr = correlation_from_covariance(sigma)

    assert np.isnan(corr.loc["A", "B"])
    assert np.isnan(corr.loc["B", "A"])
    assert corr.loc["B", "B"] == pytest.approx(1.0, abs=ATOL)


def test_covariance_diagnostics_psd_and_condition_number_p2_9(mu_sigma):
    _, sigma = mu_sigma

    diag = check_covariance_matrix(sigma)

    assert diag["is_symmetric"] is True
    assert diag["is_psd"] is True
    assert diag["is_singular"] is False
    assert diag["condition_number"] > 1.0


def test_gmv_matches_analytical_control_p2_13(mu_sigma):
    _, sigma = mu_sigma

    w_closed = gmv_weights_closed_form(sigma)
    w_bounded = gmv_weights(sigma, bounds=[(0.0, 1.0), (0.0, 1.0)])

    # Control manual: Sigma^-1 1 / (1'Sigma^-1 1) con Sigma=[[0.01,0.004],[0.004,0.04]]
    expected = pd.Series({"A": 0.857142857, "B": 0.142857143})
    pd.testing.assert_series_equal(
        w_closed, expected[w_closed.index], check_exact=False, atol=ATOL, rtol=RTOL
    )
    assert w_closed.sum() == pytest.approx(1.0, abs=ATOL)
    # Las cotas largo-solo no estaban activas: debe coincidir con el cerrado.
    pd.testing.assert_series_equal(
        w_closed, w_bounded[w_closed.index], check_exact=False, atol=1e-6, rtol=1e-6
    )


def test_four_portfolios_are_distinct_p2_16(mu_sigma):
    mu, sigma = mu_sigma
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    rf = 0.03
    gamma = 5.0

    w_gmv = gmv_weights(sigma, bounds=bounds)
    w_tan = tangency_weights(mu, sigma, rf, bounds=bounds)
    w_perso = personal_optimal_weights(mu, sigma, gamma, bounds=bounds)
    w_max = max_expected_return_weights(mu, sigma, bounds=bounds)

    for w in (w_gmv, w_tan, w_perso, w_max):
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-8).all() and (w <= 1.0 + 1e-8).all()

    # Máximo rendimiento esperado: todo en el activo B (mayor mu) por las cotas [0,1].
    assert w_max["B"] == pytest.approx(1.0, abs=1e-6)
    assert w_max["A"] == pytest.approx(0.0, abs=1e-6)

    # Las tres carteras de riesgo/retorno intermedio deben ser distintas entre sí.
    assert not np.allclose(w_gmv.values, w_tan.values, atol=1e-3)
    assert not np.allclose(w_tan.values, w_perso.values, atol=1e-3)
    assert not np.allclose(w_gmv.values, w_max.values, atol=1e-3)


def test_tangency_closed_form_matches_bounded_when_not_binding(mu_sigma):
    mu, sigma = mu_sigma
    rf = 0.03

    w_closed = tangency_weights_closed_form(mu, sigma, rf)
    w_bounded = tangency_weights(mu, sigma, rf, bounds=[(0.0, 1.0), (0.0, 1.0)])

    assert w_closed.sum() == pytest.approx(1.0, abs=ATOL)
    pd.testing.assert_series_equal(
        w_closed, w_bounded[w_closed.index], check_exact=False, atol=1e-4, rtol=1e-4
    )


def test_personal_optimal_closed_form_matches_bounded(mu_sigma):
    mu, sigma = mu_sigma
    gamma = 5.0

    w_closed = personal_optimal_weights_closed_form(mu, sigma, gamma)
    w_bounded = personal_optimal_weights(mu, sigma, gamma, bounds=[(0.0, 1.0), (0.0, 1.0)])

    pd.testing.assert_series_equal(
        w_closed, w_bounded[w_closed.index], check_exact=False, atol=1e-4, rtol=1e-4
    )


def test_portfolio_variance_and_risk_contributions_reconcile_p2_11(mu_sigma):
    _, sigma = mu_sigma
    weights = pd.Series({"A": 0.6, "B": 0.4})

    var_p = portfolio_variance(weights, sigma)
    rc, sigma_p = risk_contributions(weights, sigma)

    assert var_p == pytest.approx(sigma_p ** 2, abs=ATOL)
    assert rc.sum() == pytest.approx(sigma_p, abs=ATOL)


def test_risk_contributions_zero_risk_branch_is_explicit():
    sigma = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=["A", "B"])
    weights = pd.Series({"A": 0.5, "B": 0.5})

    rc, sigma_p = risk_contributions(weights, sigma)

    assert sigma_p == pytest.approx(0.0, abs=ATOL)
    assert (rc == 0.0).all()


def test_efficient_frontier_marks_both_branches(mu_sigma):
    mu, sigma = mu_sigma
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    frontier = efficient_frontier(mu, sigma, bounds=bounds, n_points=15)

    assert frontier["is_efficient"].any()
    # El riesgo de la rama eficiente debe ser no decreciente en el retorno objetivo.
    eff = frontier[frontier["is_efficient"]].sort_values("target_return")
    assert (eff["risk"].diff().dropna() >= -1e-9).all()
"""
Fase 10 - Pruebas T-01 y T-07 (RF-06, RF-07, RF-10, RF-11).

Casos cubiertos de la matriz oficial (Anexo_matriz_casos_prueba...):
- LOG-01: rendimientos logarítmicos con una serie sintética de 3 precios.
- STAT-01: media, varianza muestral, desviación y anualización descriptiva.
- FREQ-01: factores de anualización 252 (diaria), 52 (semanal) y 12 (mensual).
- RESAMPLE-01: remuestrear ANTES de calcular rendimientos, con AAPL del
  fixture oficial de 20 activos (semanal W-FRI y mensual ME).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics import annualize_mean, annualize_vol, descriptive_stats, log_returns
from tests.fixtures_path import FIXTURE_20_PRICES

ATOL = 1e-8
RTOL = 1e-10


def test_log_returns_known_values_log01():
    """LOG-01: prices=100|110|99 -> g = 0.095310179804325 | -0.105360515657826."""
    prices = pd.Series([100.0, 110.0, 99.0])
    g = log_returns(prices)
    assert np.isclose(g.iloc[0], 0.095310179804325, atol=ATOL, rtol=RTOL)
    assert np.isclose(g.iloc[1], -0.105360515657826, atol=ATOL, rtol=RTOL)


def test_descriptive_stats_known_values_stat01():
    """STAT-01: g=-0.02|-0.01|0|0.01|0.02, m=252."""
    g = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    stats = descriptive_stats(g, periods_per_year=252)
    assert np.isclose(stats.mean, 0.0, atol=ATOL, rtol=RTOL)
    assert np.isclose(stats.variance, 0.000250000000000, atol=ATOL, rtol=RTOL)
    assert np.isclose(stats.std, 0.015811388300842, atol=ATOL, rtol=RTOL)
    assert np.isclose(stats.std_annualized, 0.250998007960223, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "periods_per_year, expected_annual_sd",
    [(52, 0.114017542509914), (12, 0.054772255750517)],
)
def test_annualization_factors_freq01(periods_per_year, expected_annual_sd):
    """FREQ-01: mismos rendimientos, factores semanal (m=52) y mensual (m=12)."""
    g = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    mean = float(g.mean())
    sd = float(g.std(ddof=1))
    assert np.isclose(annualize_mean(mean, periods_per_year), 0.0, atol=ATOL, rtol=RTOL)
    assert np.isclose(annualize_vol(sd, periods_per_year), expected_annual_sd, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not FIXTURE_20_PRICES.exists(), reason="Falta tests/fixtures/Fixture_20_activos_sintetico_TBS_EIA.csv")
def test_resample_before_returns_aapl_resample01():
    """
    RESAMPLE-01: remuestrear ANTES de calcular rendimientos (RF-07). Nunca se
    agregan rendimientos diarios ya calculados.
    """
    raw = pd.read_csv(FIXTURE_20_PRICES, parse_dates=["date"])
    aapl = raw[raw["ticker"] == "AAPL"].sort_values("date").set_index("date")["adjusted_close"]

    weekly_prices = aapl.resample("W-FRI").last().dropna()
    weekly_returns = log_returns(weekly_prices)
    assert len(weekly_prices) == 109
    assert len(weekly_returns) == 108
    assert np.isclose(
        annualize_mean(float(weekly_returns.mean()), 52), 0.024364449534971, atol=ATOL, rtol=RTOL
    )
    assert np.isclose(
        annualize_vol(float(weekly_returns.std(ddof=1)), 52), 0.137253313867469, atol=ATOL, rtol=RTOL
    )

    monthly_prices = aapl.resample("ME").last().dropna()
    monthly_returns = log_returns(monthly_prices)
    assert len(monthly_prices) == 26
    assert len(monthly_returns) == 25
    assert np.isclose(
        annualize_mean(float(monthly_returns.mean()), 12), 0.005722470978255, atol=ATOL, rtol=RTOL
    )
    assert np.isclose(
        annualize_vol(float(monthly_returns.std(ddof=1)), 12), 0.098672726885537, atol=ATOL, rtol=RTOL
    )
# tests/unit/test_universe.py
"""
Escalamiento v2 - Puente M2->M3/M4: ensambla el universo real (posiblemente
mixto en moneda) y construye la matriz de retornos simples que alimenta a
``src.risk_matrix``.

- UNIV-01: con moneda mixta y las series FX dadas, los dos activos quedan
  en la misma ventana común de fechas y en la misma moneda base — nada se
  excluye por incompatibilidad de moneda.
- UNIV-02: sin pasar ``fx_rates_quote_per_usd`` (``None``), el
  comportamiento es idéntico al de v1: el activo en otra moneda se
  excluye, sin romper lo que ya estaba verificado.
- UNIV-03: la matriz de retornos resultante conecta sin fricción con
  ``src.risk_matrix.sample_moments`` (covarianza/correlación).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import AssetData
from src.risk_matrix import sample_moments
from src.universe import build_return_matrix


def _asset(currency: str, closes: "list[float]") -> AssetData:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    prices = pd.DataFrame({"adjusted_close": closes}, index=index)
    return AssetData(ticker="X", ok=True, prices=prices, currency=currency, source="Yahoo Finance (yfinance)")


def test_build_return_matrix_mixed_currency_merges_via_fx_univ01():
    index = pd.date_range("2026-01-01", periods=6, freq="B")
    fx_cop_per_usd = pd.Series([4000.0, 4005.0, 3995.0, 4010.0, 4020.0, 4015.0], index=index)

    assets = {
        "AAPL": _asset("USD", [100.0, 101.0, 102.0, 101.5, 103.0, 104.0]),
        "ECOPETROL": _asset("COP", [40000.0, 40200.0, 39800.0, 40500.0, 40900.0, 40700.0]),
    }

    matrix, result = build_return_matrix(
        assets,
        frequency="Diaria",
        periods_per_year=252,
        base_currency="USD",
        fx_rates_quote_per_usd={"COP": fx_cop_per_usd},
    )

    assert result.ok is True
    assert len(result.excluded) == 0
    assert set(matrix.columns) == {"AAPL", "ECOPETROL"}
    assert len(matrix) == 5  # 6 precios -> 5 retornos


def test_build_return_matrix_without_fx_falls_back_to_v1_blocking_univ02():
    assets = {
        "AAPL": _asset("USD", [100.0, 101.0, 102.0]),
        "ECOPETROL": _asset("COP", [40000.0, 40200.0, 39800.0]),
    }

    matrix, result = build_return_matrix(
        assets, frequency="Diaria", periods_per_year=252, base_currency="USD",
        fx_rates_quote_per_usd=None,
    )

    assert result.ok is True
    excluded_tickers = {e.ticker for e in result.excluded}
    assert "ECOPETROL" in excluded_tickers
    assert "AAPL" in matrix.columns
    assert "ECOPETROL" not in matrix.columns


def test_return_matrix_feeds_sample_moments_univ03():
    index = pd.date_range("2026-01-01", periods=10, freq="B")
    assets = {
        "AAPL": _asset("USD", [100 + i * 0.5 for i in range(10)]),
        "MSFT": _asset("USD", [200 + (i % 3) for i in range(10)]),
    }

    matrix, result = build_return_matrix(
        assets, frequency="Diaria", periods_per_year=252, base_currency="USD"
    )

    assert result.ok is True
    mu, sigma = sample_moments(matrix)
    assert set(mu.index) == {"AAPL", "MSFT"}
    assert sigma.shape == (2, 2)
    assert sigma.loc["AAPL", "MSFT"] == pytest.approx(sigma.loc["MSFT", "AAPL"])
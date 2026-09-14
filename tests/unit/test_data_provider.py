"""
Fase 10 - Prueba T-03 (RF-05, RF-08).

PROVIDER-01: con mocks (sin depender de internet ni de yfinance real) se
prueban respuesta válida, vacía, timeout, ticker inexistente/vacío y precio
no positivo. Ningún caso debe lanzar una excepción sin controlar, y cada
fallo debe aislarse sin detener el resto del lote (RF-08, T-06).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import fetch_asset_data, fetch_many


def _fake_history(closes, index=None):
    if index is None:
        index = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=index)


def _make_mock_ticker(history_return=None, history_side_effect=None, currency="USD", timezone="America/New_York"):
    mock_ticker = MagicMock()
    if history_side_effect is not None:
        mock_ticker.history.side_effect = history_side_effect
    else:
        mock_ticker.history.return_value = history_return
    fast_info = MagicMock()
    fast_info.currency = currency
    fast_info.timezone = timezone
    mock_ticker.fast_info = fast_info
    return mock_ticker


def test_provider_valid_response():
    """PROVIDER-01: respuesta válida se parsea con moneda y zona horaria."""
    mock_ticker = _make_mock_ticker(history_return=_fake_history([100.0, 101.0, 102.5]))
    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        result = fetch_asset_data("AAPL", "2026-01-01", "2026-01-10")
    assert result.ok is True
    assert result.error is None
    assert len(result.prices) == 3
    assert result.currency == "USD"
    assert result.timezone == "America/New_York"


def test_provider_empty_response():
    """PROVIDER-01: respuesta vacía (DataFrame vacío) se aísla como error, sin excepción."""
    mock_ticker = _make_mock_ticker(history_return=pd.DataFrame())
    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        result = fetch_asset_data("EIA_INVALID_2026", "2026-01-01", "2026-01-10")
    assert result.ok is False
    assert "vacía" in result.error.lower() or "vacio" in result.error.lower()


def test_provider_timeout():
    """PROVIDER-01: timeout/excepción del proveedor se captura y se reporta, sin lanzar excepción."""
    mock_ticker = _make_mock_ticker(history_side_effect=TimeoutError("simulated timeout"))
    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        result = fetch_asset_data("SLOWTICK", "2026-01-01", "2026-01-10")
    assert result.ok is False
    assert "error al descargar" in result.error.lower()


def test_provider_nonpositive_price_dropped_but_valid_remain():
    """PROVIDER-01: precios no positivos se descartan (RF-06); si quedan válidos, ok=True."""
    mock_ticker = _make_mock_ticker(history_return=_fake_history([100.0, -5.0, 0.0, 103.0]))
    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        result = fetch_asset_data("MIXED", "2026-01-01", "2026-01-10")
    assert result.ok is True
    assert len(result.prices) == 2  # solo 100.0 y 103.0 sobreviven
    assert result.dropped_rows == 2


def test_provider_all_nonpositive_prices_insufficient_sample():
    """PROVIDER-01: si TODOS los precios son no positivos, no hay muestra válida (ok=False)."""
    mock_ticker = _make_mock_ticker(history_return=_fake_history([0.0, -1.0, -2.0]))
    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        result = fetch_asset_data("ALLBAD", "2026-01-01", "2026-01-10")
    assert result.ok is False
    assert "insuficiente" in result.error.lower()


@pytest.mark.parametrize("empty_ticker", ["", "   ", None])
def test_provider_ticker_missing_or_empty(empty_ticker):
    """PROVIDER-01: ticker vacío o solo espacios se rechaza sin llamar al proveedor."""
    result = fetch_asset_data(empty_ticker, "2026-01-01", "2026-01-10")
    assert result.ok is False
    assert "vacío" in result.error.lower() or "vacio" in result.error.lower()


def test_fetch_many_isolates_failures_without_stopping_batch():
    """
    RF-08/T-06: un ticker con timeout y otro vacío no detienen el resto del
    lote; cada activo válido conserva su propio resultado.
    """
    def ticker_factory(ticker_symbol):
        if ticker_symbol == "GOOD1":
            return _make_mock_ticker(history_return=_fake_history([100.0, 101.0]))
        if ticker_symbol == "GOOD2":
            return _make_mock_ticker(history_return=_fake_history([50.0, 51.0]))
        if ticker_symbol == "TIMEOUTTICK":
            return _make_mock_ticker(history_side_effect=TimeoutError("timeout"))
        return _make_mock_ticker(history_return=pd.DataFrame())  # EMPTYTICK

    with patch("src.data.yf.Ticker", side_effect=ticker_factory):
        results = fetch_many(
            ["GOOD1", "TIMEOUTTICK", "GOOD2", "EMPTYTICK"],
            "2026-01-01", "2026-01-10", frequency="Diaria",
        )

    assert results["GOOD1"].ok is True
    assert results["GOOD2"].ok is True
    assert results["TIMEOUTTICK"].ok is False
    assert results["EMPTYTICK"].ok is False
    assert sum(1 for r in results.values() if r.ok) == 2
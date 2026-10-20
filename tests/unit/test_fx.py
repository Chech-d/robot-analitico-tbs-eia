# tests/unit/test_fx.py
"""
Escalamiento v2 - Sección 3.2 (conversión de moneda para universo mixto).

- FX-01: ``fetch_fx_series`` se prueba con mock (sin red), igual que
  ``src.data`` en ``test_data_provider.py``.
- FX-02: ``P_COP = P_USD * FX`` y su inversa ``P_USD = P_COP / FX``
  (convención de la guía, sección 3.2) hacen ida y vuelta sin pérdida.
- FX-03: ``harmonize_universe_currency`` convierte un activo en COP a la
  base USD y dos activos ya en la base quedan intactos (sin reconvertir).
- FX-04: si falta la serie FX que un activo necesita, ese activo queda
  marcado ``ok=False`` con un motivo explícito, no se descarta en silencio
  ni queda con una etiqueta de moneda incorrecta.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import AssetData
from src.fx import (
    convert_prices_from_usd,
    convert_prices_to_usd,
    fetch_fx_series,
    harmonize_universe_currency,
)

ATOL = 1e-8


def test_fetch_fx_series_mocked():
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    fake_history = pd.DataFrame({"Close": [4000.0, 4010.0, 4010.0, 3995.0]}, index=index)
    mock_ticker = type("MockTicker", (), {"history": lambda self, **kw: fake_history})()

    with patch("src.fx.yf.Ticker", return_value=mock_ticker):
        series = fetch_fx_series("COP", "2026-01-01", "2026-01-10")

    assert len(series) == 4
    assert series.name == "USDCOP"
    assert series.is_monotonic_increasing is False or series.index.is_monotonic_increasing


def test_fetch_fx_series_unknown_currency_raises():
    with pytest.raises(ValueError):
        fetch_fx_series("EUR", "2026-01-01", "2026-01-10")


def test_convert_prices_round_trip_p2_fx():
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    prices_usd = pd.Series([100.0, 101.0, 99.5, 102.0, 103.0], index=index, name="adjusted_close")
    fx_cop_per_usd = pd.Series([4000.0, 4010.0, 3990.0, 4005.0, 4020.0], index=index)

    prices_cop = convert_prices_from_usd(prices_usd, fx_cop_per_usd)
    # Chequeo manual puntual: P_COP = P_USD * FX (sección 3.2 de la guía).
    assert prices_cop.iloc[0] == pytest.approx(100.0 * 4000.0, rel=1e-9)
    assert prices_cop.iloc[-1] == pytest.approx(103.0 * 4020.0, rel=1e-9)

    back_to_usd = convert_prices_to_usd(prices_cop, fx_cop_per_usd)
    pd.testing.assert_series_equal(
        back_to_usd.rename(None), prices_usd.rename(None), check_exact=False, atol=ATOL, rtol=1e-9
    )


def test_convert_prices_no_lookahead_drops_dates_before_first_fx():
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    prices_usd = pd.Series([100.0, 101.0, 99.5, 102.0, 103.0], index=index, name="adjusted_close")
    # La serie FX solo empieza desde el tercer día: los dos primeros deben descartarse.
    fx_cop_per_usd = pd.Series([3990.0, 4005.0, 4020.0], index=index[2:])

    prices_cop = convert_prices_from_usd(prices_usd, fx_cop_per_usd)

    assert len(prices_cop) == 3
    assert prices_cop.index.min() == index[2]


def _asset(currency: str, closes: "list[float]", ok: bool = True) -> AssetData:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    prices = pd.DataFrame({"adjusted_close": closes}, index=index) if ok else None
    return AssetData(ticker="X", ok=ok, prices=prices, currency=currency if ok else None)


def test_harmonize_universe_currency_converts_non_base_p2_fx():
    index = pd.date_range("2026-01-01", periods=3, freq="B")
    fx_cop_per_usd = pd.Series([4000.0, 4010.0, 4020.0], index=index)

    assets = {
        "AAPL": _asset("USD", [100.0, 101.0, 102.0]),
        "ECOPETROL": _asset("COP", [40000.0, 40100.0, 40200.0]),
    }

    harmonized = harmonize_universe_currency(assets, "USD", {"COP": fx_cop_per_usd})

    assert harmonized["AAPL"].currency == "USD"
    assert harmonized["AAPL"] is assets["AAPL"]  # no se toca si ya está en la base
    assert harmonized["ECOPETROL"].ok is True
    assert harmonized["ECOPETROL"].currency == "USD"
    # 40000 COP / 4000 COP-por-USD = 10 USD
    assert harmonized["ECOPETROL"].prices["adjusted_close"].iloc[0] == pytest.approx(10.0, rel=1e-9)


def test_harmonize_missing_fx_series_marks_asset_failed_p2_fx():
    assets = {"ECOPETROL": _asset("COP", [40000.0, 40100.0])}

    harmonized = harmonize_universe_currency(assets, "USD", {})  # falta la serie COP

    assert harmonized["ECOPETROL"].ok is False
    assert "fx" in harmonized["ECOPETROL"].error.lower()
    assert harmonized["ECOPETROL"].prices is None


def test_harmonize_leaves_already_failed_assets_untouched():
    assets = {"BAD": _asset("COP", [], ok=False)}

    harmonized = harmonize_universe_currency(assets, "USD", {})

    assert harmonized["BAD"].ok is False
    assert harmonized["BAD"] is assets["BAD"]
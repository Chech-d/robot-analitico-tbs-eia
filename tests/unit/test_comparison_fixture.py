"""
Fase 10 - Pruebas T-05, T-06, T-16, T-17 (RF-03, RF-08, RF-19, RF-20, RF-21).

Casos cubiertos de la matriz oficial, usando el fixture obligatorio de 20
activos sintéticos y su archivo de resultados esperados:
- MAP-20-01 / ASSET-01 a ASSET-20: los 20 tickers válidos producen
  exactamente las coordenadas históricas (media y volatilidad anualizadas),
  RVR, cuantiles, drawdown y clasificación de dominancia del archivo oficial
  de resultados esperados.
- FAIL-20P1-01 / ASSET-X: un ticker inválido adicional (EIA_INVALID_2026) no
  elimina los 20 resultados válidos y se reporta como error aislado (RF-08).
- MAP-H-01: las coordenadas del mapa histórico no dependen del horizonte de
  pronóstico H (build_comparison no recibe H como parámetro).
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.comparison import build_comparison, non_dominated_tickers, preselect_max_rvr
from src.data import AssetData
from tests.fixtures_path import EXPECTED_RESULTS_20, FIXTURE_20_PRICES

ATOL = 1e-8
RTOL = 1e-10

pytestmark = pytest.mark.skipif(
    not (FIXTURE_20_PRICES.exists() and EXPECTED_RESULTS_20.exists()),
    reason="Faltan los CSV oficiales en tests/fixtures/ (fixture de 20 activos y resultados esperados).",
)


def _load_fixture_assets() -> "dict[str, AssetData]":
    raw = pd.read_csv(FIXTURE_20_PRICES, parse_dates=["date"])
    assets = {}
    for ticker, group in raw.groupby("ticker"):
        group = group.sort_values("date")
        prices = group.set_index("date")[["adjusted_close"]]
        assets[ticker] = AssetData(
            ticker=ticker, ok=True, prices=prices,
            currency=group["currency"].iloc[0], source=group["source"].iloc[0],
        )
    return assets


@pytest.fixture(scope="module")
def expected_results() -> pd.DataFrame:
    return pd.read_csv(EXPECTED_RESULTS_20).set_index("ticker")


@pytest.fixture(scope="module")
def comparison_result():
    assets = _load_fixture_assets()
    result = build_comparison(assets, frequency="Diaria", periods_per_year=252)
    assert result.ok
    return result


def test_20_valid_assets_processed_map2001(comparison_result):
    """MAP-20-01: los 20 tickers válidos aparecen en el resultado."""
    assert len(comparison_result.summaries) == 20
    assert comparison_result.base_currency == "USD"
    assert len(comparison_result.excluded) == 0


@pytest.mark.parametrize(
    "ticker",
    [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "PG",
        "KO", "PEP", "WMT", "HD", "COST", "UNH", "V", "MA", "CAT", "MCD",
    ],
)
def test_asset_matches_expected_results(comparison_result, expected_results, ticker):
    """ASSET-01 a ASSET-20: cada activo coincide numéricamente con el archivo oficial."""
    summary = next(s for s in comparison_result.summaries if s.ticker == ticker)
    expected = expected_results.loc[ticker]

    assert summary.n == int(expected["return_observations"])
    assert np.isclose(summary.last_price, expected["last_adjusted_close"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.mean_period, expected["mean_log_period"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.std_period, expected["sample_sd_period"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.mean_annual, expected["mean_log_annual_252"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.vol_annual, expected["volatility_annual_252"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.rvr, expected["individual_rvr"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.q05, expected["q05_log_type7"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.q95, expected["q95_log_type7"], atol=ATOL, rtol=RTOL)
    assert np.isclose(summary.max_dd, expected["max_drawdown"], atol=ATOL, rtol=RTOL)


def test_non_dominated_classification_matches_expected(comparison_result, expected_results):
    """MAP-20-01: el conjunto no dominado coincide exactamente con el archivo oficial."""
    non_dominated = non_dominated_tickers(comparison_result.summaries)
    expected_non_dominated = set(expected_results.index[expected_results["non_dominated"]])
    assert non_dominated == expected_non_dominated


def test_max_rvr_selection_matches_expected_pg(comparison_result, expected_results):
    """MAP-20-01: PG es el único activo con selected_max_rvr=true (mayor RVR individual)."""
    selected = set(preselect_max_rvr(comparison_result.summaries))
    expected_selected = set(expected_results.index[expected_results["selected_max_rvr"]])
    assert selected == expected_selected == {"PG"}


def test_invalid_ticker_does_not_remove_20_valid_fail20p1_01():
    """FAIL-20P1-01 / ASSET-X: EIA_INVALID_2026 no elimina los 20 activos válidos."""
    assets = _load_fixture_assets()
    assets["EIA_INVALID_2026"] = AssetData(ticker="EIA_INVALID_2026", ok=False, error="Ticker inexistente.")

    result = build_comparison(assets, frequency="Diaria", periods_per_year=252)

    assert result.ok
    assert len(result.summaries) == 20
    assert len(result.excluded) == 1
    assert result.excluded[0].ticker == "EIA_INVALID_2026"
    assert result.excluded[0].reason  # motivo explícito, no vacío


def test_map_coordinates_do_not_depend_on_horizon_maph01():
    """
    MAP-H-01: el mapa histórico no cambia con H. build_comparison ni
    AssetSummary reciben o exponen un horizonte de pronóstico: las
    coordenadas (media y volatilidad anualizadas) son puramente históricas.
    """
    params = inspect.signature(build_comparison).parameters
    assert "horizon" not in params and "h" not in params

    from src.comparison import AssetSummary
    field_names = {f for f in AssetSummary.__dataclass_fields__}
    assert not any("horizon" in f.lower() for f in field_names)
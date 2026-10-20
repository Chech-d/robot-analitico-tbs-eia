# src/universe.py
"""Ensambla el universo real de activos (posiblemente mixto en moneda) para
las matrices de riesgo y Markowitz (M2 -> M3/M4 de la guía v2).

Conecta ``src.data`` (descarga), ``src.fx`` (armonización de moneda) y
``src.comparison`` (ventana común de fechas) con ``src.risk_matrix``
(covarianza y carteras). No duplica lógica: reutiliza
``build_comparison`` tal cual para obtener la intersección común de
fechas, ya con todos los activos en la misma moneda — así la tabla
comparativa de v1 (RF-19/RF-20) sigue funcionando igual, ahora sobre un
universo que puede venir en varias monedas de origen.
"""
from __future__ import annotations

import pandas as pd

from src.analytics import log_returns
from src.comparison import ComparisonResult, build_comparison
from src.data import AssetData
from src.fx import harmonize_universe_currency
from src.risk_matrix import simple_from_log_returns


def build_return_matrix(
    assets: "dict[str, AssetData]",
    frequency: str,
    periods_per_year: int,
    base_currency: str,
    fx_rates_quote_per_usd: "dict[str, pd.Series] | None" = None,
) -> "tuple[pd.DataFrame, ComparisonResult]":
    """Devuelve ``(retornos_simples, ComparisonResult)`` listos para
    ``src.risk_matrix.sample_moments``.

    Si se pasa ``fx_rates_quote_per_usd``, los activos en una moneda
    distinta a ``base_currency`` se convierten primero (``src.fx``); si no
    se pasa (``None``), el comportamiento es idéntico al de v1: los activos
    en otra moneda quedan excluidos por ``build_comparison`` sin
    conversión, para no romper el comportamiento ya verificado cuando no se
    pide explícitamente moneda mixta.

    El DataFrame resultante tiene una columna por ticker con retornos
    SIMPLES por periodo (no logarítmicos), ya en la ventana común de
    fechas — exactamente lo que pide ``sample_moments``.
    """
    harmonized = (
        harmonize_universe_currency(assets, base_currency, fx_rates_quote_per_usd)
        if fx_rates_quote_per_usd
        else assets
    )

    result = build_comparison(harmonized, frequency, periods_per_year, base_currency=base_currency)
    if not result.ok or not result.common_prices:
        return pd.DataFrame(), result

    returns = {}
    for ticker, prices in result.common_prices.items():
        log_rets = log_returns(prices)
        returns[ticker] = simple_from_log_returns(log_rets)
    matrix = pd.DataFrame(returns)
    return matrix, result
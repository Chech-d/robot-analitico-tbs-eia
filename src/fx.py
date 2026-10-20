# src/fx.py
"""Conversión de moneda para universos mixtos (escalamiento v2, sección 3.2).

v1 (``src.comparison.build_comparison``) BLOQUEA activos en una moneda
distinta a la base (RF-20) en vez de convertirlos — ese comportamiento se
conserva intacto, sin tocarlo, para no romper ninguna de sus pruebas. Este
módulo es una capa opcional de PREPROCESAMIENTO: convierte los precios de
un ``AssetData`` a la moneda base ANTES de pasarlos a ``build_comparison``,
para que el universo pueda incluir activos en varias monedas (ej. acciones
en USD y ETFs/activos en COP) sin modificar la lógica ya verificada de v1.

Convención de la guía v2 (sección 3.2), con FX definido como "moneda
cotizada por 1 USD" (ej. COP por USD):

    P_COP = P_USD * FX
    1 + R_COP = (1 + R_USD) * (1 + R_FX)

Solo se implementa conversión contra USD (puente): para pasar de una moneda
X a otra Y se convierte primero X->USD y luego USD->Y. Con X=USD o Y=USD
ese paso se omite.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import yfinance as yf

from src.data import AssetData

# Tickers de Yahoo Finance para pares FX "moneda cotizada por 1 USD".
# Ej.: "COP=X" devuelve cuántos COP vale 1 USD (USD/COP).
FX_TICKER_VS_USD = {
    "COP": "COP=X",
}


def fetch_fx_series(quote_currency: str, start_date, end_date) -> pd.Series:
    """Serie FX (unidades de ``quote_currency`` por 1 USD), ordenada y sin duplicados.

    Solo cubre pares contra USD (lo que necesita este proyecto: USD<->COP
    por ahora). Lanza ``ValueError`` si no hay un ticker configurado para
    esa moneda o si la respuesta del proveedor viene vacía.
    """
    ticker = FX_TICKER_VS_USD.get(quote_currency.upper())
    if ticker is None:
        raise ValueError(
            f"No hay ticker FX configurado para {quote_currency}; agréguelo a FX_TICKER_VS_USD."
        )
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True, actions=False)
    if hist is None or hist.empty:
        raise ValueError(f"Respuesta vacía al descargar la serie FX {ticker}.")
    series = hist["Close"].rename(f"USD{quote_currency.upper()}")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _align_fx_no_lookahead(prices: pd.Series, fx_rates_quote_per_usd: pd.Series) -> pd.Series:
    """Alinea cada fecha de ``prices`` al último dato FX disponible en o antes
    de esa fecha (sin look-ahead). Las fechas anteriores al primer dato FX
    quedan en NaN (se descartan en quien llama)."""
    union_index = prices.index.union(fx_rates_quote_per_usd.index)
    fx_filled = fx_rates_quote_per_usd.reindex(union_index).sort_index().ffill()
    return fx_filled.reindex(prices.index)


def convert_prices_to_usd(prices: pd.Series, fx_rates_quote_per_usd: pd.Series) -> pd.Series:
    """Convierte una serie de precios en ``quote_currency`` a USD.

    ``P_USD = P_quote / FX`` (FX = unidades de quote por 1 USD).
    """
    fx_aligned = _align_fx_no_lookahead(prices, fx_rates_quote_per_usd)
    valid = fx_aligned.notna()
    return (prices[valid] / fx_aligned[valid]).rename(prices.name)


def convert_prices_from_usd(prices_usd: pd.Series, fx_rates_quote_per_usd: pd.Series) -> pd.Series:
    """Convierte una serie de precios en USD a ``quote_currency``.

    ``P_quote = P_USD * FX`` (FX = unidades de quote por 1 USD).
    """
    fx_aligned = _align_fx_no_lookahead(prices_usd, fx_rates_quote_per_usd)
    valid = fx_aligned.notna()
    return (prices_usd[valid] * fx_aligned[valid]).rename(prices_usd.name)


def convert_price_series(
    prices: pd.Series,
    source_currency: str,
    target_currency: str,
    fx_rates_quote_per_usd: "dict[str, pd.Series]",
) -> pd.Series:
    """Convierte ``prices`` de ``source_currency`` a ``target_currency``, pasando
    por USD como moneda puente cuando ninguna de las dos es USD."""
    source_currency = source_currency.upper()
    target_currency = target_currency.upper()
    if source_currency == target_currency:
        return prices

    if source_currency != "USD":
        prices_usd = convert_prices_to_usd(prices, fx_rates_quote_per_usd[source_currency])
    else:
        prices_usd = prices

    if target_currency != "USD":
        return convert_prices_from_usd(prices_usd, fx_rates_quote_per_usd[target_currency])
    return prices_usd


def harmonize_universe_currency(
    assets: "dict[str, AssetData]",
    base_currency: str,
    fx_rates_quote_per_usd: "dict[str, pd.Series]",
) -> "dict[str, AssetData]":
    """Convierte cada ``AssetData`` a ``base_currency``, dejando intactos los que
    ya coinciden y sin tocar los que vienen con ``ok=False`` (ya están
    excluidos por su propio error, antes de esta capa).

    ``fx_rates_quote_per_usd`` es un diccionario ``{moneda: serie FX
    moneda-por-USD}`` con una entrada por cada moneda distinta de
    ``base_currency`` presente en ``assets`` (y por ``base_currency`` mismo
    si ``base_currency != "USD"``, para poder pasar por USD como moneda
    puente). Si falta la serie FX que un activo necesita, ese activo queda
    marcado ``ok=False`` con un motivo explícito — nunca se descarta en
    silencio ni se deja sin convertir con una etiqueta de moneda falsa.
    """
    base_currency = base_currency.upper()
    harmonized: "dict[str, AssetData]" = {}

    for ticker, asset in assets.items():
        if not asset.ok or asset.prices is None or asset.currency is None:
            harmonized[ticker] = asset
            continue

        source_currency = asset.currency.upper()
        if source_currency == base_currency:
            harmonized[ticker] = asset
            continue

        try:
            prices = asset.prices["adjusted_close"]
            converted = convert_price_series(
                prices, source_currency, base_currency, fx_rates_quote_per_usd
            )
        except KeyError as exc:
            harmonized[ticker] = replace(
                asset,
                ok=False,
                prices=None,
                error=f"Falta la serie FX para convertir {source_currency}->{base_currency}: {exc}",
            )
            continue

        if converted.empty:
            harmonized[ticker] = replace(
                asset,
                ok=False,
                prices=None,
                error=(
                    f"Sin fechas FX disponibles para convertir {source_currency}->{base_currency} "
                    f"en el rango de precios de {ticker}."
                ),
            )
            continue

        new_prices = pd.DataFrame({"adjusted_close": converted})
        harmonized[ticker] = replace(
            asset,
            prices=new_prices,
            currency=base_currency,
            source=f"{asset.source} + FX {source_currency}->{base_currency}",
        )

    return harmonized
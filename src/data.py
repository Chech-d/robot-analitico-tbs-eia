"""
Capa de datos (RF-05 a RF-08): descarga, validación, limpieza y remuestreo.

- Descarga precios ajustados con yfinance e informa proveedor, moneda, zona
  horaria y fecha del último dato (RF-05).
- Ordena por fecha, elimina duplicados, descarta precios no positivos y
  filas sin dato (RF-06).
- Remuestrea ANTES de calcular rendimientos, usando el último precio válido
  del periodo para semanal/mensual (RF-07).
- Un ticker inválido, con respuesta vacía o con muestra insuficiente no
  detiene el análisis de los demás activos del lote (RF-08).
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

FREQUENCY_RULES = {
    "Diaria": None,
    "Semanal": "W",
    "Mensual": "ME",
}

FREQUENCY_PERIODS_PER_YEAR = {
    "Diaria": 252,
    "Semanal": 52,
    "Mensual": 12,
}


@dataclass
class AssetData:
    ticker: str
    ok: bool
    prices: Optional[pd.DataFrame] = None  # índice: fecha; columna: adjusted_close
    error: Optional[str] = None
    source: str = "Yahoo Finance (yfinance)"
    currency: Optional[str] = None
    timezone: Optional[str] = None
    dropped_rows: int = 0


def _clean_prices(raw: pd.DataFrame) -> "tuple[pd.DataFrame, int]":
    """Ordena, elimina duplicados y descarta precios no positivos o faltantes."""
    df = raw.copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    before = len(df)
    df = df[df["adjusted_close"].notna()]
    df = df[df["adjusted_close"] > 0]
    dropped = before - len(df)
    return df, dropped


def fetch_asset_data(ticker: str, start_date, end_date) -> AssetData:
    """
    Descarga precios ajustados de un ticker. No lanza excepción ante fallos
    individuales: los reporta en el resultado para que el llamador pueda
    seguir con los demás activos del lote (RF-08).
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return AssetData(ticker=ticker, ok=False, error="Ticker vacío.")

    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=True,  # Close queda ajustado por dividendos/splits
            actions=False,
        )
    except Exception as exc:  # captura cualquier fallo de red o del proveedor
        return AssetData(ticker=ticker, ok=False, error=f"Error al descargar: {exc}")

    if hist is None or hist.empty:
        return AssetData(
            ticker=ticker,
            ok=False,
            error="Respuesta vacía: ticker inexistente, sin datos en el rango o timeout.",
        )

    raw = pd.DataFrame({"adjusted_close": hist["Close"]})
    raw.index.name = "date"

    cleaned, dropped = _clean_prices(raw)

    if cleaned.empty:
        return AssetData(
            ticker=ticker,
            ok=False,
            error="Muestra insuficiente tras la limpieza (sin precios válidos).",
            dropped_rows=dropped,
        )

    currency = None
    timezone = None
    try:
        fast_info = yf_ticker.fast_info
        currency = getattr(fast_info, "currency", None)
        timezone = getattr(fast_info, "timezone", None)
    except Exception:
        pass  # complementario: no bloquea el análisis si no está disponible

    return AssetData(
        ticker=ticker,
        ok=True,
        prices=cleaned,
        currency=currency,
        timezone=timezone,
        dropped_rows=dropped,
    )


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Remuestrea ANTES de calcular rendimientos (RF-07)."""
    rule = FREQUENCY_RULES.get(frequency)
    if rule is None:
        return prices
    resampled = prices.resample(rule).last()
    return resampled.dropna(subset=["adjusted_close"])


def fetch_many(tickers, start_date, end_date, frequency: str):
    """
    Descarga y remuestrea varios tickers. Un fallo individual no detiene a
    los demás (RF-08, T-06).
    """
    results = {}
    for raw_ticker in tickers:
        result = fetch_asset_data(raw_ticker, start_date, end_date)
        if result.ok and result.prices is not None:
            result.prices = resample_prices(result.prices, frequency)
            if result.prices.empty:
                result.ok = False
                result.error = "Sin observaciones tras remuestrear."
        results[result.ticker] = result
    return results
"""
Rendimientos logarítmicos, estadística descriptiva y drawdown
(RF-09 a RF-12, secciones 5.1 a 5.3 de la guía).

Usa EXCLUSIVAMENTE rendimientos logarítmicos: g_t = ln(P_t / P_{t-1}).
Nunca se calcula ni se ofrece una ruta con rendimientos simples.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

FREQUENCY_PERIODS_PER_YEAR = {
    "Diaria": 252,
    "Semanal": 52,
    "Mensual": 12,
}


def log_returns(prices: pd.Series) -> pd.Series:
    """g_t = ln(P_t / P_{t-1}). Única definición de rendimiento admitida."""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def annualize_mean(mean_per_period: float, periods_per_year: int) -> float:
    """μ_anual = m · ḡ (sección 5.2)."""
    return periods_per_year * mean_per_period


def annualize_vol(std_per_period: float, periods_per_year: int) -> float:
    """σ_anual = √m · s_g (sección 5.2)."""
    return (periods_per_year ** 0.5) * std_per_period


@dataclass
class DescriptiveStats:
    n: int
    mean: float
    median: float
    variance: float  # muestral, ddof=1
    std: float
    minimum: float
    p25: float
    p75: float
    maximum: float
    skewness: float  # Fisher, corregida por sesgo
    excess_kurtosis: float  # Fisher, corregida por sesgo
    mean_annualized: float
    std_annualized: float
    last_return_percentile: float  # midrank, sobre 100


def descriptive_stats(returns: pd.Series, periods_per_year: int) -> DescriptiveStats:
    """
    Estadísticas descriptivas de los rendimientos logarítmicos (RF-11, 5.3).
    Cuantiles lineales tipo 7 (equivalente al default de pandas/numpy).
    Asimetría y curtosis de Fisher con corrección de sesgo (default de pandas).
    """
    n = len(returns)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    last_value = float(returns.iloc[-1])

    return DescriptiveStats(
        n=n,
        mean=mean,
        median=float(returns.median()),
        variance=float(returns.var(ddof=1)),
        std=std,
        minimum=float(returns.min()),
        p25=float(returns.quantile(0.25)),
        p75=float(returns.quantile(0.75)),
        maximum=float(returns.max()),
        skewness=float(returns.skew()),
        excess_kurtosis=float(returns.kurtosis()),
        mean_annualized=annualize_mean(mean, periods_per_year),
        std_annualized=annualize_vol(std, periods_per_year),
        last_return_percentile=float(percentileofscore(returns, last_value, kind="mean")),
    )


def recent_window_stats(
    returns: pd.Series, periods_per_year: int, window: Optional[int] = None
) -> "tuple[DescriptiveStats, bool]":
    """
    Estadísticas sobre la ventana reciente min(m, T) (RF-12). Devuelve
    (stats, es_parcial) — es_parcial=True si T < m (ventana truncada).
    """
    m = window or periods_per_year
    t = len(returns)
    effective_window = min(m, t)
    recent = returns.iloc[-effective_window:]
    is_partial = t < m
    return descriptive_stats(recent, periods_per_year), is_partial


def drawdown_series(prices: pd.Series) -> pd.Series:
    """D_t = P_t / max_{s<=t}(P_s) - 1 (RF-12)."""
    running_max = prices.cummax()
    return prices / running_max - 1.0


def max_drawdown(prices: pd.Series) -> float:
    return float(drawdown_series(prices).min())
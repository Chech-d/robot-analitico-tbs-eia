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
from scipy import stats as scipy_stats
from scipy.stats import percentileofscore
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

FREQUENCY_PERIODS_PER_YEAR = {
    "Diaria": 252,
    "Semanal": 52,
    "Mensual": 12,
}

DIAGNOSTICS_ALPHA = 0.05
# Por debajo de este tamaño de muestra, los cuatro contrastes (todos
# asintóticos) dejan de ser confiables y algunos degeneran numéricamente
# (p.ej. bloques con muy pocas observaciones). Se prefiere no mostrarlos.
MIN_DIAGNOSTICS_N = 30


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


# ---------------------------------------------------------------------------
# Diagnósticos de supuestos (sección 5.4 de la guía): normalidad, dependencia,
# homocedasticidad y varianza condicional. No tienen un RF propio; alimentan
# la matriz de advertencia/no señal de RF-14 (walk-forward) y RF-18
# (probabilidades terminales). Todos se evalúan a alfa=5% y ninguno bloquea
# el análisis: un rechazo se muestra como una advertencia informativa, nunca
# como un error.
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticResult:
    name: str
    hypothesis: str
    parameters: str
    statistic: float
    p_value: float
    reject: bool  # True si p_value < alfa (se rechaza H0)
    interpretation: str


def _dependence_lags(n: int) -> int:
    """L = min(10, floor(n/5)), con mínimo 1 (sección 5.4)."""
    return max(1, min(10, n // 5))


def jarque_bera_test(returns: pd.Series) -> DiagnosticResult:
    """Normalidad (Seccion 5.4). H0: los rendimientos siguen una distribución normal."""
    stat, p_value = scipy_stats.jarque_bera(returns.values)
    reject = bool(p_value < DIAGNOSTICS_ALPHA)
    return DiagnosticResult(
        name="Jarque-Bera (normalidad)",
        hypothesis="H0: los rendimientos logarítmicos siguen una distribución normal.",
        parameters=f"n = {len(returns)}, alfa = 5%",
        statistic=float(stat),
        p_value=float(p_value),
        reject=reject,
        interpretation=(
            "Se rechaza normalidad (advertencia): considerar que las colas pueden ser "
            "más pesadas que en una normal."
            if reject
            else "No se rechaza normalidad."
        ),
    )


def ljung_box_test(returns: pd.Series) -> DiagnosticResult:
    """Dependencia serial (sección 5.4). H0: no hay autocorrelación hasta el rezago L."""
    n = len(returns)
    lags = _dependence_lags(n)
    result = acorr_ljungbox(returns.values, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    p_value = float(result["lb_pvalue"].iloc[0])
    reject = bool(p_value < DIAGNOSTICS_ALPHA)
    return DiagnosticResult(
        name="Ljung-Box (dependencia)",
        hypothesis="H0: no hay autocorrelación serial en los rendimientos hasta el rezago L.",
        parameters=f"L = {lags}, n = {n}, alfa = 5%",
        statistic=stat,
        p_value=p_value,
        reject=reject,
        interpretation=(
            "Se rechaza independencia (advertencia): hay autocorrelación serial "
            "detectable en los rendimientos."
            if reject
            else "No se rechaza independencia."
        ),
    )


def brown_forsythe_test(returns: pd.Series) -> DiagnosticResult:
    """
    Homocedasticidad (sección 5.4). H0: la varianza es igual entre tres bloques
    cronológicos consecutivos (equivalente a Levene con centro en la mediana).
    """
    values = returns.values
    n = len(values)
    block_indices = np.array_split(np.arange(n), 3)
    groups = [values[idx] for idx in block_indices if len(idx) > 0]
    stat, p_value = scipy_stats.levene(*groups, center="median")
    reject = bool(p_value < DIAGNOSTICS_ALPHA)
    return DiagnosticResult(
        name="Brown-Forsythe (homocedasticidad)",
        hypothesis=(
            "H0: la varianza de los rendimientos es igual entre los tres "
            "bloques cronológicos."
        ),
        parameters=f"bloques = 3, tamaños = {[len(g) for g in groups]}, alfa = 5%",
        statistic=float(stat),
        p_value=float(p_value),
        reject=reject,
        interpretation=(
            "Se rechaza homocedasticidad (advertencia): la volatilidad no es "
            "estable a lo largo de la muestra."
            if reject
            else "No se rechaza homocedasticidad."
        ),
    )


def arch_lm_test(returns: pd.Series) -> DiagnosticResult:
    """
    Varianza condicional / efectos ARCH (sección 5.4). H0: no hay efectos ARCH
    (varianza condicional constante) hasta el rezago L.
    """
    n = len(returns)
    lags = _dependence_lags(n)
    stat, p_value, _f_stat, _f_p_value = het_arch(
        returns.values, nlags=lags, result_object=False
    )
    reject = bool(p_value < DIAGNOSTICS_ALPHA)
    return DiagnosticResult(
        name="ARCH-LM (varianza condicional)",
        hypothesis=(
            "H0: no hay efectos ARCH (varianza condicional constante) hasta "
            "el rezago L."
        ),
        parameters=f"L = {lags}, n = {n}, alfa = 5%",
        statistic=float(stat),
        p_value=float(p_value),
        reject=reject,
        interpretation=(
            "Se rechaza varianza condicional constante (advertencia): hay "
            "indicios de agrupamiento de volatilidad (clusters)."
            if reject
            else "No se rechaza varianza condicional constante."
        ),
    )


def run_diagnostics(returns: pd.Series) -> "list[DiagnosticResult]":
    """
    Corre los cuatro diagnósticos de supuestos sobre g_t, en orden fijo.
    El llamador debe verificar antes que len(returns) >= MIN_DIAGNOSTICS_N;
    esta función no lo valida para mantenerse simple y testeable.
    """
    return [
        jarque_bera_test(returns),
        ljung_box_test(returns),
        brown_forsythe_test(returns),
        arch_lm_test(returns),
    ]
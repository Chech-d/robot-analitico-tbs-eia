# src/risk_matrix.py
"""Matrices de riesgo conjunto y portafolios de Markowitz (escalamiento v2).

Implementa la sección 4 (covarianza/correlación/riesgo de cartera) y la
sección 5 (conjunto factible, frontera eficiente y las 4 carteras) de la
guía oficial "BOT de Portafolios v2". No sustituye ni modifica los módulos
de v1 (``analytics.py``, ``forecasting.py``, ``risk_rules.py``,
``comparison.py``); es una capa nueva que se apoya en los rendimientos
logarítmicos ya calculados por v1 para la estimación individual, y añade
retornos simples para la agregación de capital en cartera, tal como exige
la sección 4.1 de la guía.

Convenciones:

- Los rendimientos de entrada para ``sample_moments`` son retornos SIMPLES
  por periodo (decimal), uno por activo, en un DataFrame con fechas en el
  índice y un activo por columna. Si se parte de rendimientos logarítmicos
  individuales ``g_it`` (los que ya produce ``analytics.log_returns``), se
  convierten con :func:`simple_from_log_returns` antes de llamar a
  ``sample_moments`` — la media/covarianza de una cartera NO se calcula
  promediando logs (ver :func:`portfolio_log_return_vs_weighted_log_sum`).
- ``mu`` y ``Sigma`` están en la misma frecuencia que los retornos de
  entrada (no anualizados). La anualización (``m*mu``, ``m*Sigma``) es
  responsabilidad de quien llama, como ya lo es en v1 para activos
  individuales.
- Todas las funciones de carteras devuelven pesos como ``pandas.Series``
  indexados por el mismo orden de activos que ``mu``/``Sigma``, para que
  nunca se mezclen pesos de un activo con el riesgo de otro por
  reindexación implícita.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

__all__ = [
    "simple_from_log_returns",
    "portfolio_log_return_vs_weighted_log_sum",
    "sample_moments",
    "correlation_from_covariance",
    "check_covariance_matrix",
    "portfolio_variance",
    "portfolio_volatility",
    "risk_contributions",
    "gmv_weights",
    "gmv_weights_closed_form",
    "tangency_weights",
    "tangency_weights_closed_form",
    "personal_optimal_weights",
    "personal_optimal_weights_closed_form",
    "max_expected_return_weights",
    "efficient_frontier",
]


# ---------------------------------------------------------------------------
# Rendimientos de cartera (sección 4.1)
# ---------------------------------------------------------------------------


def simple_from_log_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Convierte rendimientos logarítmicos individuales a retornos simples.

    ``R_it = exp(g_it) - 1``. Se usa para poder agregar varios activos en
    una cartera; la capa individual de v1 sigue usando ``g_it`` sin cambios.
    """

    return np.exp(log_returns) - 1.0


def portfolio_log_return_vs_weighted_log_sum(
    weights: pd.Series, simple_returns_row: pd.Series
) -> tuple[float, float]:
    """Contraste exigido por la guía (sección 4.1 / prueba P2-6).

    Devuelve ``(log(1 + R_p), suma ponderada de logs individuales)``. Con
    pesos iguales y retornos +10 %/-10 %, el primero es negativo y el
    segundo es exactamente cero — son magnitudes distintas, el rendimiento
    de una cartera NO es el promedio ponderado de los logs de cada activo.
    """

    r_p = float((weights * simple_returns_row).sum())
    log_portfolio_return = float(np.log1p(r_p))
    weighted_log_sum = float((weights * np.log1p(simple_returns_row)).sum())
    return log_portfolio_return, weighted_log_sum


# ---------------------------------------------------------------------------
# Estimación de mu y Sigma (sección 4.2)
# ---------------------------------------------------------------------------


def sample_moments(simple_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Media y matriz de covarianzas muestrales sobre una muestra común.

    Antes de calcular, descarta cualquier fila con un valor faltante en
    cualquier activo (muestra común a todos, ``T`` observaciones), tal como
    exige la guía: ``pandas.cov`` con eliminación por pares puede producir
    una matriz no PSD si se usa sin esta máscara común. Se usa ``ddof=1``
    (igual que ``pandas.DataFrame.cov`` por defecto).
    """

    common = simple_returns.dropna(axis=0, how="any")
    if len(common) < 2:
        raise ValueError(
            "Se necesitan al menos 2 observaciones comunes a todos los activos "
            f"para estimar Sigma; quedaron {len(common)}."
        )
    mu = common.mean(axis=0)
    sigma = common.cov(ddof=1)
    return mu, sigma


def correlation_from_covariance(sigma: pd.DataFrame) -> pd.DataFrame:
    """``C_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)``.

    Si la varianza de un activo es cero, su fila/columna de correlación
    queda en ``NaN`` (indefinida, no es un error) para los pares con ese
    activo; no se fuerza un valor arbitrario.
    """

    std = np.sqrt(np.diag(sigma.values))
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.outer(std, std)
        corr_values = np.where(denom > 0, sigma.values / denom, np.nan)
    return pd.DataFrame(corr_values, index=sigma.index, columns=sigma.columns)


def check_covariance_matrix(sigma: pd.DataFrame, atol: float = 1e-10) -> dict:
    """Diagnóstico de Sigma: simetría, PSD, autovalores y condicionamiento.

    No repara nada silenciosamente — solo informa, como exige la sección
    4.2 de la guía (una regularización requiere diagnóstico y
    justificación explícita del estudiante, no un arreglo automático).
    """

    values = sigma.values
    is_symmetric = bool(np.allclose(values, values.T, atol=atol))
    eigenvalues = np.linalg.eigvalsh((values + values.T) / 2.0)
    min_eig = float(eigenvalues.min())
    is_psd = bool(min_eig >= -atol)
    is_singular = bool(min_eig <= atol)
    max_eig = float(eigenvalues.max())
    condition_number = (
        float("inf") if min_eig <= 0 else float(max_eig / min_eig)
    )
    return {
        "is_symmetric": is_symmetric,
        "eigenvalues": eigenvalues,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "is_psd": is_psd,
        "is_singular": is_singular,
        "condition_number": condition_number,
    }


# ---------------------------------------------------------------------------
# Riesgo de cartera (sección 4.4)
# ---------------------------------------------------------------------------


def _align_weights(weights: pd.Series, sigma: pd.DataFrame) -> np.ndarray:
    aligned = weights.reindex(sigma.index)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"Pesos faltantes para los activos: {missing}")
    return aligned.values.astype(float)


def portfolio_variance(weights: pd.Series, sigma: pd.DataFrame) -> float:
    """``var_p = w' Sigma w``. Una lista de activos sin pesos no es cartera."""

    w = _align_weights(weights, sigma)
    return float(w @ sigma.values @ w)


def portfolio_volatility(weights: pd.Series, sigma: pd.DataFrame) -> float:
    var_p = portfolio_variance(weights, sigma)
    return float(np.sqrt(max(var_p, 0.0)))


def risk_contributions(weights: pd.Series, sigma: pd.DataFrame) -> tuple[pd.Series, float]:
    """``RC_i = w_i*(Sigma w)_i / sigma_p``; ``sum(RC_i) = sigma_p``.

    Si ``sigma_p == 0`` (rama determinista, p.ej. un solo activo con
    varianza cero o pesos que cancelan todo el riesgo), devuelve una
    ``Series`` de ceros en vez de dividir por cero.
    """

    w = _align_weights(weights, sigma)
    sigma_p = portfolio_volatility(weights, sigma)
    if sigma_p <= 0.0:
        rc = pd.Series(0.0, index=sigma.index)
        return rc, sigma_p
    marginal = sigma.values @ w
    rc_values = w * marginal / sigma_p
    return pd.Series(rc_values, index=sigma.index), sigma_p


# ---------------------------------------------------------------------------
# Las 4 carteras (sección 5.1)
# ---------------------------------------------------------------------------


def _default_bounds(n: int) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n


def gmv_weights_closed_form(sigma: pd.DataFrame) -> pd.Series:
    """``GMV = Sigma^-1 1 / (1' Sigma^-1 1)``.

    Control analítico de la sección 5.1: solo válido si Sigma es
    invertible y ninguna restricción de desigualdad está activa en el
    óptimo. Úsese para verificar, no como sustituto del optimizador cuando
    hay cotas/presupuesto adicionales.
    """

    sigma_inv = np.linalg.inv(sigma.values)
    ones = np.ones(len(sigma))
    w = sigma_inv @ ones
    w = w / (ones @ sigma_inv @ ones)
    return pd.Series(w, index=sigma.index)


def tangency_weights_closed_form(mu: pd.Series, sigma: pd.DataFrame, rf: float) -> pd.Series:
    """``w ~ Sigma^-1 (mu - rf*1)``, normalizado a suma 1 (control analítico)."""

    mu_aligned = mu.reindex(sigma.index)
    sigma_inv = np.linalg.inv(sigma.values)
    excess = mu_aligned.values - rf
    raw = sigma_inv @ excess
    total = raw.sum()
    if np.isclose(total, 0.0):
        raise ValueError(
            "La normalización de Sigma^-1(mu-rf*1) no acredita tangencia cuando "
            "la suma es ~0 (ver guía, sección 5.1); revise rf o mu."
        )
    w = raw / total
    return pd.Series(w, index=sigma.index)


def personal_optimal_weights_closed_form(
    mu: pd.Series, sigma: pd.DataFrame, gamma: float
) -> pd.Series:
    """Óptimo personal sin cotas, solo con presupuesto (``sum(w)=1``).

    Maximiza ``U = w'mu - (gamma/2) w'Sigma w`` sujeto a ``1'w=1`` por
    multiplicadores de Lagrange:
    ``w = (1/gamma) Sigma^-1 (mu - lambda*1)``, con
    ``lambda = (1'Sigma^-1 mu - gamma) / (1'Sigma^-1 1)``.
    """

    if gamma <= 0:
        raise ValueError("gamma debe ser positiva (aversión al riesgo del inversionista).")
    mu_aligned = mu.reindex(sigma.index)
    sigma_inv = np.linalg.inv(sigma.values)
    ones = np.ones(len(sigma))
    denom = ones @ sigma_inv @ ones
    lam = (ones @ sigma_inv @ mu_aligned.values - gamma) / denom
    w = (sigma_inv @ (mu_aligned.values - lam * ones)) / gamma
    return pd.Series(w, index=sigma.index)


def _solve_qp_min_variance(
    sigma: pd.DataFrame,
    bounds: list[tuple[float, float]],
    target_return: float | None,
    mu: pd.Series | None,
    w0: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, str]:
    n = len(sigma)
    if w0 is None:
        w0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if target_return is not None:
        if mu is None:
            raise ValueError("Se requiere mu si se fija target_return.")
        mu_values = mu.reindex(sigma.index).values
        constraints.append(
            {"type": "eq", "fun": lambda w, m=mu_values, r=target_return: w @ m - r}
        )

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma.values @ w)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return result.x, bool(result.success), str(result.message)


def gmv_weights(
    sigma: pd.DataFrame, bounds: list[tuple[float, float]] | None = None
) -> pd.Series:
    """Cartera de mínima varianza global, con cotas/presupuesto opcionales.

    Si ``bounds`` es ``None`` usa el control analítico cerrado (sin cotas).
    Si se dan cotas, resuelve el problema de optimización restringido
    (sección 5 de la guía); el resultado converge al cerrado cuando las
    cotas no están activas.
    """

    if bounds is None:
        return gmv_weights_closed_form(sigma)
    n = len(sigma)
    if len(bounds) != n:
        raise ValueError("bounds debe tener una tupla (min,max) por activo.")
    w, success, message = _solve_qp_min_variance(sigma, bounds, None, None)
    if not success:
        raise RuntimeError(f"GMV con cotas no convergió: {message}")
    return pd.Series(w, index=sigma.index)


def tangency_weights(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float,
    bounds: list[tuple[float, float]] | None = None,
) -> pd.Series:
    """Cartera de máximo Sharpe, con cotas/presupuesto opcionales."""

    if bounds is None:
        return tangency_weights_closed_form(mu, sigma, rf)
    n = len(sigma)
    mu_values = mu.reindex(sigma.index).values

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu_values - rf
        vol = np.sqrt(max(float(w @ sigma.values @ w), 1e-18))
        return -ret / vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w0 = np.full(n, 1.0 / n)
    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Tangente con cotas no convergió: {result.message}")
    return pd.Series(result.x, index=sigma.index)


def personal_optimal_weights(
    mu: pd.Series,
    sigma: pd.DataFrame,
    gamma: float,
    bounds: list[tuple[float, float]] | None = None,
) -> pd.Series:
    """Cartera óptima personal (``U=w'mu-(gamma/2)w'Sigma w``), con cotas opcionales."""

    if gamma <= 0:
        raise ValueError("gamma debe ser positiva (aversión al riesgo del inversionista).")
    if bounds is None:
        return personal_optimal_weights_closed_form(mu, sigma, gamma)
    n = len(sigma)
    mu_values = mu.reindex(sigma.index).values

    def neg_utility(w: np.ndarray) -> float:
        return float(-(w @ mu_values) + (gamma / 2.0) * (w @ sigma.values @ w))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w0 = np.full(n, 1.0 / n)
    result = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Óptimo personal con cotas no convergió: {result.message}")
    return pd.Series(result.x, index=sigma.index)


def max_expected_return_weights(
    mu: pd.Series, sigma: pd.DataFrame, bounds: list[tuple[float, float]] | None = None
) -> pd.Series:
    """Cartera de máximo rendimiento esperado dentro de las cotas.

    Es un problema lineal: ``max w'mu`` sujeto a ``1'w=1`` y cotas. Sin
    cotas por activo (``bounds=None``) no está acotado (se concentraría
    todo en el activo de mayor mu) — la guía exige mostrar esto
    explícitamente en vez de ocultarlo, así que aquí se exige ``bounds``.
    """

    n = len(sigma)
    if bounds is None:
        bounds = _default_bounds(n)
    mu_values = mu.reindex(sigma.index).values
    # linprog minimiza, así que se minimiza -mu'w
    result = linprog(
        c=-mu_values,
        A_eq=[np.ones(n)],
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Máximo rendimiento esperado no acotado o infactible: {result.message}")
    return pd.Series(result.x, index=sigma.index)


def efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    bounds: list[tuple[float, float]] | None = None,
    n_points: int = 30,
) -> pd.DataFrame:
    """Rama eficiente (y marca la no eficiente) del conjunto factible.

    Resuelve, para una rejilla de rendimientos objetivo entre el de la GMV
    y el máximo alcanzable, ``min w'Sigma w`` sujeto a ``w'mu=r``. Marca
    cada punto como eficiente (``is_efficient``) si su riesgo no es mayor
    que el de ningún punto con rendimiento objetivo menor o igual —
    distingue la rama eficiente de la rama inferior no eficiente de la
    misma curva de mínima varianza.
    """

    n = len(sigma)
    if bounds is None:
        bounds = _default_bounds(n)
    gmv = gmv_weights(sigma, bounds=bounds)
    mu_values = mu.reindex(sigma.index).values
    r_min = float(gmv.values @ mu_values)
    max_ret_w = max_expected_return_weights(mu, sigma, bounds=bounds)
    r_max = float(max_ret_w.values @ mu_values)
    if np.isclose(r_min, r_max):
        targets = np.array([r_min])
    else:
        targets = np.linspace(r_min, r_max, n_points)

    rows = []
    w0 = gmv.values
    for target in targets:
        w, success, message = _solve_qp_min_variance(sigma, bounds, float(target), mu, w0=w0)
        if not success:
            continue
        w0 = w
        risk = float(np.sqrt(max(w @ sigma.values @ w, 0.0)))
        rows.append({"target_return": float(target), "risk": risk, "weights": pd.Series(w, index=sigma.index)})

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        raise RuntimeError("No se pudo construir ningún punto factible de la frontera.")
    # Rama eficiente: riesgo no creciente al recortar hacia abajo en retorno.
    min_risk_so_far = np.inf
    is_efficient = []
    for risk in frontier.sort_values("target_return")["risk"]:
        if risk <= min_risk_so_far + 1e-9:
            is_efficient.append(True)
            min_risk_so_far = min(min_risk_so_far, risk)
        else:
            is_efficient.append(False)
    frontier = frontier.sort_values("target_return").reset_index(drop=True)
    frontier["is_efficient"] = is_efficient
    return frontier
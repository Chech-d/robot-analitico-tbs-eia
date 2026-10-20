"""Beta, CAPM, apalancamiento (Hamada) y CAL/CML/SML (M7 de la guia v2,
secciones 5.2 y 8).

Beta (seccion 8): se calcula frente a un benchmark seleccionado y
documentado, contrastando la pendiente OLS CON INTERCEPTO de
R_i-rf_t = alpha + beta*(R_M-rf_t) + epsilon contra el calculo manual
Cov(exceso_i,exceso_M)/Var(exceso_M) sobre la MISMA muestra -- ambos
deben coincidir (P2-31). Un benchmark con varianza cero (constante) produce
una salida INDEFINIDA explicita, nunca una division forzada.

Hamada simplificado (seccion 8.1), con deuda de beta declarada (por
defecto cero, "se declara deuda de beta cero y aprovechamiento del escudo
fiscal; ser deuda bancaria senior no prueba ese supuesto"):

    beta_U = [beta_L + beta_D*(1-tax)*D/E] / [1 + (1-tax)*D/E]
    beta_L = beta_U + (beta_U - beta_D)*(1-tax)*D/E

Con beta_D=0 esto se reduce a la forma simplificada de la guia
(beta_U=beta_L/[1+(1-tax)*D/E]). Ambas transformaciones deben compartir
el mismo supuesto de tax/beta_D para poder revertirse exactamente
(P2-32: "Hamada ida/vuelta con igual supuesto de deuda").

CAPM y Ke (seccion 8.2):

    Ke = rf + beta_L * ERP
    Ke_USD = rf_USD + beta_L * ERP_madura + lambda * CRP   (extension pais)
    Ke_COP = (1+Ke_USD) * (1+inflacion_COL) / (1+inflacion_USA) - 1

CAL/CML/SML (seccion 5.2): tres rectas con ejes propios -- CAL/CML tienen
riesgo TOTAL (sigma) en el eje horizontal, SML tiene BETA (riesgo
sistematico), nunca volatilidad total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


def excess_returns(returns: pd.Series, rf_period: "Union[float, pd.Series]") -> pd.Series:
    """R - rf por periodo. rf_period puede ser un escalar (rf
    constante en el periodo) o una serie alineada por fecha (rf variable,
    seccion 8: "Con rf variable, el cociente de retornos brutos no tiene
    por que coincidir con el de excesos")."""
    if isinstance(rf_period, pd.Series):
        rf_period = rf_period.reindex(returns.index)
    return returns - rf_period


@dataclass
class BetaResult:
    beta: float
    beta_manual: float   # Cov(exceso_i,exceso_M)/Var(exceso_M), cruzado contra `beta` (P2-31)
    alpha: float
    se_beta: float
    ci_low: float
    ci_high: float
    r_squared: float
    n: int


def estimate_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_period: "Union[float, pd.Series]" = 0.0,
    confidence: float = 0.95,
) -> BetaResult:
    """Beta frente al benchmark dado (seccion 8): OLS con intercepto sobre
    excesos de rendimiento, con cruce contra el calculo manual
    Cov/Var en la MISMA muestra comun.

    Si el benchmark queda con varianza cero (constante) en la muestra
    comun, lanza ValueError explicito -- nunca se fuerza una division
    por cero ni se devuelve un beta inventado.
    """
    asset_excess = excess_returns(asset_returns, rf_period)
    bench_excess = excess_returns(benchmark_returns, rf_period)
    df = pd.DataFrame({"asset": asset_excess, "bench": bench_excess}).dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"Se necesitan al menos 3 observaciones comunes; hay {n}.")

    bench_var = df["bench"].var(ddof=1)
    if not np.isfinite(bench_var) or np.isclose(bench_var, 0.0, atol=1e-16):
        raise ValueError(
            "El benchmark es constante (varianza cero) en la muestra comun: el beta queda "
            "INDEFINIDO, no se fuerza una division por cero."
        )

    X = sm.add_constant(df["bench"].values)
    model = sm.OLS(df["asset"].values, X).fit()
    alpha, beta = model.params
    se_beta = model.bse[1]
    ci_low, ci_high = model.conf_int(alpha=1 - confidence)[1]
    r_squared = model.rsquared

    cov_manual = df[["asset", "bench"]].cov(ddof=1).loc["asset", "bench"]
    beta_manual = cov_manual / bench_var

    return BetaResult(
        beta=float(beta), beta_manual=float(beta_manual), alpha=float(alpha),
        se_beta=float(se_beta), ci_low=float(ci_low), ci_high=float(ci_high),
        r_squared=float(r_squared), n=n,
    )


def portfolio_beta(weights: pd.Series, asset_betas: pd.Series) -> float:
    """Beta de cartera = suma ponderada de betas individuales (seccion 8:
    "El beta de una cartera con pesos constantes es su suma ponderada solo
    con muestra, benchmark y convenciones compatibles" -- se asume aqui que
    asset_betas ya viene de la misma muestra/benchmark)."""
    asset_betas = asset_betas.reindex(weights.index)
    if asset_betas.isna().any():
        faltantes = asset_betas[asset_betas.isna()].index.tolist()
        raise ValueError(f"Falta el beta de: {faltantes}.")
    return float((weights * asset_betas).sum())


def unlever_beta(beta_L: float, D: float, E: float, tax: float, beta_D: float = 0.0) -> float:
    """Desapalanca (Hamada, sec. 8.1): beta_U=[beta_L+beta_D*(1-tax)*D/E]
    /[1+(1-tax)*D/E]. Con beta_D=0 es la forma simplificada de la
    guia (beta_U=beta_L/[1+(1-tax)*D/E])."""
    apalancamiento = (1 - tax) * D / E
    return (beta_L + beta_D * apalancamiento) / (1 + apalancamiento)


def relever_beta(beta_U: float, D: float, E: float, tax: float, beta_D: float = 0.0) -> float:
    """Reapalanca (Hamada, sec. 8.1): beta_L=beta_U+(beta_U-beta_D)*
    (1-tax)*D/E. Con el mismo tax/beta_D, revierte exactamente a
    unlever_beta (P2-32: "ida/vuelta con igual supuesto de deuda")."""
    return beta_U + (beta_U - beta_D) * (1 - tax) * D / E


def capm_required_return(rf: float, beta_L: float, erp: float) -> float:
    """Ke = rf + beta_L * ERP (CAPM basico, sec. 8.2)."""
    return rf + beta_L * erp


def capm_country_extension(rf_usd: float, beta_L: float, erp_madura: float, lambda_: float, crp: float) -> float:
    """Ke_USD = rf_USD + beta_L*ERP_madura + lambda*CRP (sec. 8.2):
    extension de pais, SEPARADA del CAPM basico -- no se duplica el pais
    entre rf, prima local, CRP y flujos (eso se vigila en como se arman
    los insumos, no dentro de esta formula)."""
    return rf_usd + beta_L * erp_madura + lambda_ * crp


def ke_local_currency(ke_usd: float, inflation_local: float, inflation_usd: float) -> float:
    """Ke_COP = (1+Ke_USD)*(1+inflacion_COL)/(1+inflacion_USA)-1 (sec. 8.2):
    supuesto de CONSISTENCIA NOMINAL, no una prediccion cierta de TRM; no
    se suman las inflaciones directamente."""
    return (1 + ke_usd) * (1 + inflation_local) / (1 + inflation_usd) - 1


def capital_allocation_line(rf: float, mu_tangency: float, sigma_tangency: float, y: float) -> "tuple[float, float]":
    """CAL (sec. 5.2): E(R_c)=rf+y*(mu_T-rf), sigma_c=abs(y)*sigma_T.
    Devuelve (E_Rc, sigma_c) para un nivel de apalancamiento y dado
    (0<=y<=1 en la base sin deuda; y>1 implica apalancarse, seccion 5.2)."""
    e_rc = rf + y * (mu_tangency - rf)
    sigma_c = abs(y) * sigma_tangency
    return e_rc, sigma_c


def optimal_y_unconstrained(mu_tangency: float, rf: float, sigma_tangency: float, gamma: float) -> float:
    """y* = (mu_T-rf)/(gamma*sigma_T^2) (sec. 5.2): optimo SIN
    restricciones, puede dar y*>1 (apalancarse). Si se prohibe deuda, no
    basta con recortar a 1: hay que resolver de nuevo la utilidad en el
    conjunto permitido (ver src.risk_matrix.personal_optimal_weights)."""
    return (mu_tangency - rf) / (gamma * sigma_tangency ** 2)


def security_market_line(rf: float, betas: "Union[float, pd.Series, np.ndarray]", erp: float):
    """SML (sec. 5.2): rendimiento requerido en funcion del BETA (riesgo
    sistematico), nunca de la volatilidad total -- eje horizontal distinto
    al de CAL/CML. rf + beta*erp evaluado para uno o varios betas."""
    return rf + betas * erp
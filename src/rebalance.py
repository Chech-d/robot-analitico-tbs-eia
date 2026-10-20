"""Rebalanceo con informacion nueva (M6 de la guia v2, seccion 7).

Cubre la secuencia verificable de la seccion 7.1: deriva de pesos por
movimiento de precio (sin flujos ni transacciones), decision de rebalancear
o abstenerse (banda de no operacion, evento duplicado, dato obsoleto),
simulacion de las operaciones con costos sobre el NOMINAL REAL negociado
(no sobre la mitad) y reconciliacion de caja.

Convenciones (seccion 7.1), sin flujos (aportes=0) y sin apalancamiento
(cantidades, caja >= 0 siempre):

    w_i,t_pre = w_i,t-1_post * (1+R_i,t) / (1+R_p,t)
    turnover_unilateral = 0.5 * sum(abs(delta_w))

El costo total de una operacion de rebalanceo es
``costo_por_nominal * sum(|nominal comprado| + |nominal vendido|)``: se
cobra sobre el nominal real de cada compra y venta, nunca sobre la mitad
del nominal (error comun). El objetivo se aplica sobre el patrimonio
POSTERIOR a costos (``V_post = V_pre - costos``), lo que hace el problema
implicito (el costo depende del nominal operado, que depende del objetivo
sobre V_post); se resuelve con un punto fijo simple, no con un despeje
algebraico fragil que solo sirva para 2 activos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


def drifted_weights(w_prev_post: pd.Series, simple_returns: pd.Series) -> "tuple[pd.Series, float]":
    """Pesos ANTES de operar, por el solo efecto del movimiento de precios
    (seccion 7.1: "Sin flujos ni transacciones: w_i,t_pre=w_i,t-1_post*
    (1+R_i,t)/(1+R_p,t)").

    Devuelve ``(w_pre, R_p)`` con ``R_p = sum(w_prev_post * simple_returns)``
    (retorno simple de cartera sin flujos, misma convencion de
    ``src.risk_matrix``/seccion 4.1 de la guia).
    """
    simple_returns = simple_returns.reindex(w_prev_post.index)
    r_p = float((w_prev_post * simple_returns).sum())
    w_pre = w_prev_post * (1 + simple_returns) / (1 + r_p)
    return w_pre, r_p


def turnover_unilateral(w_pre: pd.Series, w_target: pd.Series) -> float:
    """``0.5 * sum(abs(w_target - w_pre))`` (seccion 7.1): fraccion del
    patrimonio que hay que mover en un solo sentido; NO es el nominal
    bruto realmente negociado (compra+venta), que se calcula aparte en
    ``simulate_rebalance_to_target`` porque incluye el efecto de los
    costos sobre el objetivo."""
    w_target = w_target.reindex(w_pre.index)
    return 0.5 * float((w_target - w_pre).abs().sum())


@dataclass
class RebalanceResult:
    feasible: bool
    reason: str
    trades: "Optional[pd.Series]" = None          # nominal operado por activo (+compra, -venta)
    new_quantities: "Optional[pd.Series]" = None
    cash_pre: float = 0.0
    cash_post: "Optional[float]" = None
    cost: "Optional[float]" = None
    gross_notional_traded: "Optional[float]" = None  # T = sum(|trade_i|)
    value_pre: "Optional[float]" = None            # patrimonio antes de costos, a precios nuevos
    value_post: "Optional[float]" = None           # patrimonio despues de costos
    iterations: int = 0


def simulate_rebalance_to_target(
    quantities_pre: pd.Series,
    prices_new: pd.Series,
    cash_pre: float,
    target_weights: pd.Series,
    cost_rate: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> RebalanceResult:
    """Simula las operaciones necesarias para llegar a ``target_weights``
    (objetivo sobre el patrimonio POSTERIOR a costos, seccion 7.1).

    No permite caja negativa (sin apalancamiento, "no se permite deuda no
    autorizada"): si el objetivo no es alcanzable sin endeudarse,
    ``feasible=False`` con el motivo explicito en ``reason`` — nunca se
    renormaliza en silencio para ocultarlo.
    """
    assets = quantities_pre.index
    prices_new = prices_new.reindex(assets)
    target_weights = target_weights.reindex(assets).fillna(0.0)

    if prices_new.isna().any():
        faltantes = prices_new[prices_new.isna()].index.tolist()
        return RebalanceResult(feasible=False, reason=f"Faltan precios nuevos para: {faltantes}.")

    v = quantities_pre * prices_new  # valor de cada posicion a precios nuevos, antes de operar
    value_pre = float(v.sum() + cash_pre)

    if target_weights.sum() > 1 + 1e-9:
        return RebalanceResult(
            feasible=False,
            reason=f"Los pesos objetivo suman {target_weights.sum():.6f} > 1 (implicaria apalancamiento).",
            value_pre=value_pre,
        )

    cost = 0.0
    for i in range(max_iter):
        value_post = value_pre - cost
        if value_post < 0:
            return RebalanceResult(
                feasible=False,
                reason="El costo de rebalanceo supera el patrimonio disponible.",
                value_pre=value_pre,
            )
        target_value = target_weights * value_post
        trades = target_value - v
        gross_notional = float(trades.abs().sum())
        new_cost = cost_rate * gross_notional
        if abs(new_cost - cost) < tol:
            cost = new_cost
            break
        cost = new_cost
    else:
        return RebalanceResult(
            feasible=False, reason="El costo de rebalanceo no convergio (revisar cost_rate).", value_pre=value_pre
        )

    value_post = value_pre - cost
    target_value = target_weights * value_post
    trades = target_value - v
    gross_notional = float(trades.abs().sum())
    cash_post = cash_pre - float(trades.sum()) - cost

    if cash_post < -1e-8:
        return RebalanceResult(
            feasible=False,
            reason=f"El rebalanceo dejaria caja negativa ({cash_post:.4f}); no se ejecuta sin autorizar deuda.",
            value_pre=value_pre,
            cost=cost,
        )

    new_quantities = (v + trades) / prices_new

    return RebalanceResult(
        feasible=True,
        reason="ok",
        trades=trades,
        new_quantities=new_quantities,
        cash_pre=cash_pre,
        cash_post=max(cash_post, 0.0),
        cost=cost,
        gross_notional_traded=gross_notional,
        value_pre=value_pre,
        value_post=value_post,
        iterations=i + 1,
    )


def needs_rebalance(w_pre: pd.Series, w_target: pd.Series, band: float) -> "tuple[bool, pd.Series]":
    """Regla por UMBRAL/banda (seccion 7): True si algun activo se desvio
    mas de ``band`` (en peso) de su objetivo. Devuelve tambien la
    desviacion por activo para que la decision quede trazable."""
    w_target = w_target.reindex(w_pre.index).fillna(0.0)
    desviacion = (w_target - w_pre).abs()
    return bool((desviacion > band).any()), desviacion


def is_duplicate_event(event_id: str, seen_event_ids: "set[str]") -> bool:
    """Un evento ya procesado no debe generar una segunda decision
    (seccion 7.1: "Un evento duplicado no produce dos decisiones")."""
    return event_id in seen_event_ids


@dataclass
class DataQualityCheck:
    ok: bool
    stale_tickers: "list[str]" = field(default_factory=list)
    max_skew_seconds: float = 0.0


def check_data_freshness(
    quote_age_seconds: "dict[str, float]", max_quote_age_seconds: float, max_asset_time_skew_seconds: float
) -> DataQualityCheck:
    """Calidad del dato antes de decidir (seccion 7.1: "Datos obsoletos,
    precios asincronicos mas alla del umbral... preservan la ultima
    cartera con estado explicito y suspenden la decision dependiente").

    ``quote_age_seconds``: antiguedad de la cotizacion de cada activo, en
    segundos, al momento de la decision.
    """
    if not quote_age_seconds:
        return DataQualityCheck(ok=True)
    stale = [t for t, age in quote_age_seconds.items() if age > max_quote_age_seconds]
    skew = max(quote_age_seconds.values()) - min(quote_age_seconds.values())
    ok = len(stale) == 0 and skew <= max_asset_time_skew_seconds
    return DataQualityCheck(ok=ok, stale_tickers=stale, max_skew_seconds=skew)
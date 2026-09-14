"""
VaR, niveles de entrada/salida y probabilidades terminales (RF-16 a RF-18,
secciones 5.6 y 5.7 de la guía).

Posición larga únicamente. Reutiliza el modelo y el horizonte ya ajustados
en src/forecasting.py (checklist de la guía, punto 145: "VaR, probabilidades,
entrada, stop-loss y take-profit comparten horizonte y modelo"). Cualquier
falla en las condiciones de 5.7 produce "no señal" — nunca se fuerza una
señal.

Todas las fórmulas fueron verificadas contra el fixture obligatorio de la
guía (sección 8.1: "Modelo A", "Modelo B", "Serie constante" y "Modelo B
con costos").
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.forecasting import ForecastModel, price_quantile, prob_price_above, terminal_moments

# Valores predeterminados congelados de la sección 5.7.
DEFAULT_PL = 0.05
DEFAULT_PU = 0.95
DEFAULT_BR_MIN = 1.00
DEFAULT_CB = 0.0
DEFAULT_CS = 0.0

# Niveles de confianza admitidos para el VaR (RF-16: 95% obligatorio, 99% permitido).
VAR_CONFIDENCE_LEVELS = (0.95, 0.99)


@dataclass
class VarResult:
    confidence: float
    horizon: int
    model_name: str
    capital: float
    var_fraction: float  # VaR fraccional; convención de signo: positivo = pérdida, con piso en cero
    var_dollar: float


def compute_var(model: ForecastModel, p0: float, horizon: int, confidence: float, capital: float) -> VarResult:
    """
    VaR paramétrico individual (RF-16, sección 5.6).
    VaR^frac = max(0, 1 - exp(q_{1-c})), con q_{1-c} el cuantil inferior del
    log-rendimiento acumulado a H (equivalente al cuantil (1-c) del precio
    terminal dividido por P0). El piso en cero es obligatorio: bajo un
    modelo con deriva fuerte, el escenario "adverso" puede seguir siendo una
    ganancia, y el VaR no puede ser negativo.
    """
    q_lower_price = price_quantile(model, p0, horizon, 1 - confidence)
    var_fraction = max(0.0, 1.0 - q_lower_price / p0)
    return VarResult(
        confidence=confidence,
        horizon=horizon,
        model_name=model.name,
        capital=capital,
        var_fraction=var_fraction,
        var_dollar=var_fraction * capital,
    )


@dataclass
class RiskParameters:
    p_l: float = DEFAULT_PL
    p_u: float = DEFAULT_PU
    br_min: float = DEFAULT_BR_MIN
    c_b: float = DEFAULT_CB  # costo proporcional de compra, c_b >= 0
    c_s: float = DEFAULT_CS  # costo proporcional de venta, 0 <= c_s < 1


@dataclass
class SignalResult:
    ok: bool  # True = señal sustentada; False = no señal (nunca se fuerza una señal)
    reasons: "list[str]" = field(default_factory=list)
    entry: float = 0.0
    sl_h: float = 0.0
    tp_h: float = 0.0
    p_be: float = 0.0
    br_bruta: Optional[float] = None
    d_neto: Optional[float] = None
    u_neto: Optional[float] = None
    br_neto: Optional[float] = None


def evaluate_signal(model: ForecastModel, p0: float, horizon: int, params: RiskParameters) -> SignalResult:
    """
    Entrada, stop-loss, take-profit y condición de señal (RF-17, sección 5.7).
    Posición larga: E = P0 (último precio ajustado). SL_H y TP_H son los
    cuantiles p_L y p_U del precio terminal, bajo el mismo modelo y H que el
    pronóstico. Señal solo si se cumplen TODAS las condiciones; cualquier
    falla produce no señal sustentada (con el motivo explícito).
    """
    entry = p0
    sl_h = price_quantile(model, p0, horizon, params.p_l)
    tp_h = price_quantile(model, p0, horizon, params.p_u)
    p_be = entry * (1 + params.c_b) / (1 - params.c_s)
    br_bruta = (tp_h - entry) / (entry - sl_h) if entry > sl_h else None

    reasons: "list[str]" = []

    if not (sl_h < entry < tp_h):
        reasons.append(
            f"No se cumple SL_H < E < TP_H (SL_H={sl_h:.6f}, E={entry:.6f}, TP_H={tp_h:.6f})."
        )
    if not (p_be >= entry):
        reasons.append(f"No se cumple P_BE >= E (P_BE={p_be:.6f}, E={entry:.6f}).")
    if not (p_be < tp_h):
        reasons.append(f"No se cumple P_BE < TP_H (P_BE={p_be:.6f}, TP_H={tp_h:.6f}).")

    d_neto = entry * (1 + params.c_b) - sl_h * (1 - params.c_s)
    u_neto = tp_h * (1 - params.c_s) - entry * (1 + params.c_b)

    if not (d_neto > 0):
        reasons.append(f"No se cumple D_neto > 0 (D_neto={d_neto:.6f}).")
    if not (u_neto > 0):
        reasons.append(f"No se cumple U_neto > 0 (U_neto={u_neto:.6f}).")

    br_neto = None
    if d_neto > 0:
        br_neto = u_neto / d_neto
        if not (br_neto >= params.br_min):
            reasons.append(
                f"No se cumple BR_neto >= BR_min (BR_neto={br_neto:.6f}, BR_min={params.br_min:.6f})."
            )
    else:
        reasons.append("BR_neto no evaluable porque D_neto <= 0 (no se divide por cero).")

    return SignalResult(
        ok=len(reasons) == 0,
        reasons=reasons,
        entry=entry,
        sl_h=sl_h,
        tp_h=tp_h,
        p_be=p_be,
        br_bruta=br_bruta,
        d_neto=d_neto,
        u_neto=u_neto,
        br_neto=br_neto,
    )


@dataclass
class TerminalProbabilities:
    p_be: float
    is_deterministic: bool
    prob_win: float
    prob_lose: float
    prob_neutral: float


def terminal_probabilities(
    model: ForecastModel, p0: float, horizon: int, p_be: float
) -> TerminalProbabilities:
    """
    Probabilidades terminales de ganar y perder frente a P_BE (RF-18,
    sección 5.7), usando la misma distribución y H que el pronóstico.

    Rama determinista (v_H=0): se define P* = P0*exp(m_H) y
    epsilon = 1e-12*max(1, P0); la probabilidad 1 se asigna a ganar si
    P*>P_BE+epsilon, a perder si P*<P_BE-epsilon, o a neutral si queda
    dentro del margen. Nunca se divide por cero.
    """
    m_h, v_h = terminal_moments(model, horizon)
    if v_h <= 0:
        p_star = p0 * np.exp(m_h)
        eps = 1e-12 * max(1.0, p0)
        if p_star > p_be + eps:
            return TerminalProbabilities(p_be=p_be, is_deterministic=True, prob_win=1.0, prob_lose=0.0, prob_neutral=0.0)
        if p_star < p_be - eps:
            return TerminalProbabilities(p_be=p_be, is_deterministic=True, prob_win=0.0, prob_lose=1.0, prob_neutral=0.0)
        return TerminalProbabilities(p_be=p_be, is_deterministic=True, prob_win=0.0, prob_lose=0.0, prob_neutral=1.0)

    prob_win = prob_price_above(model, p0, horizon, p_be)
    prob_lose = 1.0 - prob_win
    return TerminalProbabilities(p_be=p_be, is_deterministic=False, prob_win=prob_win, prob_lose=prob_lose, prob_neutral=0.0)
"""
Modelos homocedásticos y validación walk-forward (RF-13 a RF-15,
secciones 5.5 y 8.1 de la guía).

- Modelo A: caminata aleatoria sin deriva (benchmark obligatorio).
- Modelo B: lognormal con deriva y parámetros constantes.
- Ambos comparten la misma familia: el rendimiento logarítmico acumulado a
  H periodos se asume Normal(m_H, v_H) (homocedástico, sin autocorrelación),
  y el precio terminal P_H = P0 * exp(rendimiento acumulado) (lognormal).
  La única diferencia entre modelos es si la media por periodo se fuerza a
  cero (caminata aleatoria) o se estima de la muestra (lognormal con deriva).
- Walk-forward: últimos 10 orígenes consecutivos, ventana expansiva,
  reestimación solo con datos disponibles en cada origen (sin look-ahead).

Todas las fórmulas fueron verificadas contra el fixture obligatorio de la
guía (sección 8.1, "Modelo A" y "Modelo B").
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Modelos (RF-13)
# ---------------------------------------------------------------------------


@dataclass
class ForecastModel:
    name: str
    mean_per_period: float  # m (por periodo, NO anualizado)
    std_per_period: float  # sigma (por periodo, NO anualizado)
    n_train: int


def fit_random_walk(returns: pd.Series) -> ForecastModel:
    """Caminata aleatoria sin deriva: media forzada a 0 (benchmark obligatorio)."""
    return ForecastModel(
        name="Caminata aleatoria (sin deriva)",
        mean_per_period=0.0,
        std_per_period=float(returns.std(ddof=1)),
        n_train=len(returns),
    )


def fit_lognormal_drift(returns: pd.Series) -> ForecastModel:
    """Lognormal de parámetros constantes con deriva: media y sigma muestrales."""
    return ForecastModel(
        name="Lognormal (con deriva)",
        mean_per_period=float(returns.mean()),
        std_per_period=float(returns.std(ddof=1)),
        n_train=len(returns),
    )


# Único punto de registro de modelos seleccionables desde la interfaz.
FORECAST_MODELS: "dict[str, Callable[[pd.Series], ForecastModel]]" = {
    "Caminata aleatoria (sin deriva)": fit_random_walk,
    "Lognormal (con deriva)": fit_lognormal_drift,
}


# ---------------------------------------------------------------------------
# Distribución terminal y trayectoria (RF-14)
# ---------------------------------------------------------------------------


def terminal_moments(model: ForecastModel, horizon: int) -> "tuple[float, float]":
    """m_H = m * H, v_H = sigma^2 * H (homocedástico, sin autocorrelación)."""
    m_h = model.mean_per_period * horizon
    v_h = (model.std_per_period ** 2) * horizon
    return m_h, v_h


def price_quantile(model: ForecastModel, p0: float, horizon: int, p: float) -> float:
    """
    Cuantil p (0<p<1) del precio terminal P_H, bajo P_H = P0 * exp(g_acumulado),
    g_acumulado ~ Normal(m_H, v_H). Si v_H=0 (rama determinista, RF 5.4.1),
    todos los cuantiles colapsan en la mediana: no se divide por cero.
    """
    m_h, v_h = terminal_moments(model, horizon)
    if v_h <= 0:
        return float(p0 * np.exp(m_h))
    sd_h = v_h ** 0.5
    z_p = norm.ppf(p)
    return float(p0 * np.exp(m_h + z_p * sd_h))


def prob_price_above(model: ForecastModel, p0: float, horizon: int, reference_price: float) -> float:
    """
    P(P_H > reference_price). Rama determinista (v_H=0): 1, 0 o 0.5 según si
    el precio terminal queda por encima, por debajo o exactamente igual al
    precio de referencia (empate con tolerancia numérica).
    """
    m_h, v_h = terminal_moments(model, horizon)
    if v_h <= 0:
        terminal_price = p0 * np.exp(m_h)
        if np.isclose(terminal_price, reference_price, atol=1e-12, rtol=1e-10):
            return 0.5
        return 1.0 if terminal_price > reference_price else 0.0
    sd_h = v_h ** 0.5
    # ln(reference/P0) = umbral de rendimiento acumulado para superar la referencia
    threshold = np.log(reference_price / p0)
    return float(1.0 - norm.cdf((threshold - m_h) / sd_h))


@dataclass
class TerminalDistribution:
    model_name: str
    p0: float
    horizon: int
    mean_per_period: float
    std_per_period: float
    m_h: float
    v_h: float
    is_deterministic: bool  # v_h == 0 (rama determinista, 5.4.1)
    median_price: float
    mean_price: float
    q05_price: float
    q95_price: float
    prob_above_entry: float  # P(P_H > P0)


def terminal_distribution(model: ForecastModel, p0: float, horizon: int) -> TerminalDistribution:
    """Resumen de la distribución terminal a horizonte H (RF-14)."""
    m_h, v_h = terminal_moments(model, horizon)
    is_deterministic = v_h <= 0
    median_price = float(p0 * np.exp(m_h))
    mean_price = float(p0 * np.exp(m_h + v_h / 2))
    return TerminalDistribution(
        model_name=model.name,
        p0=p0,
        horizon=horizon,
        mean_per_period=model.mean_per_period,
        std_per_period=model.std_per_period,
        m_h=m_h,
        v_h=v_h,
        is_deterministic=is_deterministic,
        median_price=median_price,
        mean_price=mean_price,
        q05_price=median_price if is_deterministic else price_quantile(model, p0, horizon, 0.05),
        q95_price=median_price if is_deterministic else price_quantile(model, p0, horizon, 0.95),
        prob_above_entry=prob_price_above(model, p0, horizon, p0),
    )


@dataclass
class ForecastStep:
    step: int  # k = 1..H
    m_k: float
    v_k: float
    median_price: float
    mean_price: float
    q05_price: float
    q95_price: float


def forecast_path(model: ForecastModel, p0: float, horizon: int) -> "list[ForecastStep]":
    """Trayectoria analítica completa de 1 a H (RF-14): un punto por paso."""
    steps = []
    for k in range(1, horizon + 1):
        dist_k = terminal_distribution(model, p0, k)
        steps.append(
            ForecastStep(
                step=k,
                m_k=dist_k.m_h,
                v_k=dist_k.v_h,
                median_price=dist_k.median_price,
                mean_price=dist_k.mean_price,
                q05_price=dist_k.q05_price,
                q95_price=dist_k.q95_price,
            )
        )
    return steps


def periods_until(last_date, target_date, frequency: str) -> int:
    """
    Convierte una fecha objetivo a una cantidad de periodos H en la
    frecuencia elegida (RF-04, RF-14): el usuario puede fijar H como cantidad
    de periodos o como fecha objetivo, y el sistema la convierte de forma
    transparente. No crea observaciones ficticias en fines de semana: para
    frecuencia diaria cuenta días hábiles (lunes a viernes); festivos
    específicos quedan fuera de alcance de este piloto académico.
    Devuelve 0 si la fecha objetivo no es posterior a la última fecha con dato.
    """
    last_date = pd.Timestamp(last_date)
    target_date = pd.Timestamp(target_date)
    # El activo puede traer zona horaria (p.ej. yfinance) y la fecha elegida
    # en el selector no; se comparan "sin reloj" para evitar TypeError.
    if last_date.tzinfo is not None:
        last_date = last_date.tz_localize(None)
    if target_date.tzinfo is not None:
        target_date = target_date.tz_localize(None)
    if target_date <= last_date:
        return 0
    if frequency == "Diaria":
        return int(len(pd.bdate_range(last_date + pd.Timedelta(days=1), target_date)))
    if frequency == "Semanal":
        return int(len(pd.date_range(last_date, target_date, freq="W")))
    if frequency == "Mensual":
        return int(len(pd.date_range(last_date, target_date, freq="ME")))
    return 0


# ---------------------------------------------------------------------------
# Walk-forward (RF-15, sección 5.5)
# ---------------------------------------------------------------------------

WALKFORWARD_ORIGINS = 10
# Constante aditiva de la regla de suficiencia de la sección 5.5 (9 periodos
# de margen adicionales a los 10 orígenes menos uno).
WALKFORWARD_MARGIN = 9


def walkforward_min_train(periods_per_year: int, horizon: int) -> int:
    """n_min = max(2m, 5H) (sección 5.5): mínimo de entrenamiento en el primer origen."""
    return max(2 * periods_per_year, 5 * horizon)


def is_walkforward_sufficient(n_returns: int, periods_per_year: int, horizon: int) -> bool:
    """T >= max(2m,5H) + H + 9 (sección 5.5, RF-04/RF-15): regla de suficiencia temporal."""
    if horizon < 1:
        return False
    n_min = walkforward_min_train(periods_per_year, horizon)
    return n_returns >= n_min + horizon + WALKFORWARD_MARGIN


def max_valid_horizon(n_returns: int, periods_per_year: int) -> int:
    """
    Mayor horizonte H (entero >= 1) que todavía cumple la regla de suficiencia
    T >= max(2m,5H) + H + 9 de la sección 5.5 (RF-04: "el límite operacional
    deberá justificarse por los datos"). Devuelve 0 si ningún H >= 1 cumple.
    """
    h_max = 0
    horizon = 1
    while horizon <= n_returns and is_walkforward_sufficient(n_returns, periods_per_year, horizon):
        h_max = horizon
        horizon += 1
    return h_max


@dataclass
class WalkForwardOrigin:
    origin_index: int  # posición (0-based) dentro de la serie de rendimientos
    n_train: int
    forecast_cum_return: float  # m_H del origen (pronóstico puntual)
    q05_cum_return: float
    q95_cum_return: float
    actual_cum_return: float
    squared_error: float
    absolute_error: float
    in_interval_90: bool
    direction_correct: Optional[bool]  # None si el pronóstico no define dirección (m_H=0)


@dataclass
class WalkForwardResult:
    ok: bool
    model_name: str
    horizon: int
    reason: Optional[str] = None  # motivo si ok=False (RF-15: "validación insuficiente")
    n_origins: int = 0
    rmse: Optional[float] = None
    mae: Optional[float] = None
    coverage_90: Optional[float] = None
    directional_accuracy: Optional[float] = None
    n_directional_defined: int = 0
    origins: "list[WalkForwardOrigin]" = field(default_factory=list)


def walk_forward_validate(
    returns: pd.Series,
    fit_fn: Callable[[pd.Series], ForecastModel],
    model_name: str,
    horizon: int,
    periods_per_year: int,
) -> WalkForwardResult:
    """
    Valida un modelo con los últimos WALKFORWARD_ORIGINS orígenes consecutivos,
    ventana expansiva y reestimación solo con datos disponibles en cada origen
    (sin look-ahead). Suficiencia exacta de la sección 5.5: n_min=max(2m,5H),
    T>=n_min+H+9. Si la muestra no alcanza, devuelve ok=False (RF-15: "con
    muestra insuficiente se declarará validación insuficiente y no señal"),
    sin lanzar excepción.
    """
    n = len(returns)
    values = returns.values

    if horizon < 1:
        return WalkForwardResult(
            ok=False, model_name=model_name, horizon=horizon,
            reason="Horizonte inválido (H debe ser un entero positivo).",
        )

    n_min = walkforward_min_train(periods_per_year, horizon)

    if not is_walkforward_sufficient(n, periods_per_year, horizon):
        return WalkForwardResult(
            ok=False,
            model_name=model_name,
            horizon=horizon,
            reason=(
                f"Validación insuficiente (sección 5.5): se requiere "
                f"T >= max(2m,5H)+H+9 = {n_min}+{horizon}+9 = {n_min + horizon + WALKFORWARD_MARGIN} "
                f"rendimientos y la muestra disponible tiene T={n} para H={horizon}."
            ),
        )

    last_origin = n - horizon  # deja exactamente H rendimientos para evaluar
    first_origin = last_origin - (WALKFORWARD_ORIGINS - 1)

    origins_result: "list[WalkForwardOrigin]" = []
    for origin in range(first_origin, last_origin + 1):
        train = pd.Series(values[:origin])
        model = fit_fn(train)
        m_h, v_h = terminal_moments(model, horizon)
        sd_h = v_h ** 0.5 if v_h > 0 else 0.0

        actual_cum_return = float(values[origin: origin + horizon].sum())
        error = actual_cum_return - m_h

        if sd_h > 0:
            q05 = m_h + norm.ppf(0.05) * sd_h
            q95 = m_h + norm.ppf(0.95) * sd_h
            in_interval = bool(q05 <= actual_cum_return <= q95)
        else:
            q05 = q95 = m_h
            in_interval = bool(np.isclose(actual_cum_return, m_h, atol=1e-12, rtol=1e-10))

        if m_h == 0:
            direction_correct = None  # el modelo no predice una dirección (RF-15: "cuando esté definida")
        else:
            direction_correct = bool(np.sign(m_h) == np.sign(actual_cum_return))

        origins_result.append(
            WalkForwardOrigin(
                origin_index=origin,
                n_train=len(train),
                forecast_cum_return=m_h,
                q05_cum_return=q05,
                q95_cum_return=q95,
                actual_cum_return=actual_cum_return,
                squared_error=error ** 2,
                absolute_error=abs(error),
                in_interval_90=in_interval,
                direction_correct=direction_correct,
            )
        )

    rmse = float(np.sqrt(np.mean([o.squared_error for o in origins_result])))
    mae = float(np.mean([o.absolute_error for o in origins_result]))
    coverage_90 = float(np.mean([o.in_interval_90 for o in origins_result]))

    directional_flags = [o.direction_correct for o in origins_result if o.direction_correct is not None]
    directional_accuracy = float(np.mean(directional_flags)) if directional_flags else None

    return WalkForwardResult(
        ok=True,
        model_name=model_name,
        horizon=horizon,
        n_origins=len(origins_result),
        rmse=rmse,
        mae=mae,
        coverage_90=coverage_90,
        directional_accuracy=directional_accuracy,
        n_directional_defined=len(directional_flags),
        origins=origins_result,
    )
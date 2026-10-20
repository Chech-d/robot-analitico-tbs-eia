"""
Punto de entrada de la aplicación.
Robot analítico para la preselección de activos - Teoría Moderna de
Portafolios, Tech Business School - Universidad EIA.
"""
import json

import pandas as pd
import streamlit as st

from src import access_log
from src.analytics import (
    FREQUENCY_PERIODS_PER_YEAR,
    MIN_DIAGNOSTICS_N,
    descriptive_stats,
    drawdown_series,
    log_returns,
    max_drawdown,
    recent_window_stats,
    run_diagnostics,
)
from src.auth import is_auth_configured, require_login
from src.charts import forecast_chart, price_chart, return_chart, risk_return_map
from src.comparison import (
    MIN_COMPARISON_ASSETS,
    build_comparison,
    comparison_table,
    export_parameters,
    non_dominated_tickers,
    preselect_max_mean_under_risk_limit,
    preselect_max_rvr,
    preselect_min_vol_under_min_mean,
    preselect_non_dominated,
    processed_prices_export,
)
from src.data import FREQUENCY_RULES, fetch_asset_data, fetch_many, resample_prices
from src.forecasting import (
    FORECAST_MODELS,
    WALKFORWARD_ORIGINS,
    forecast_path,
    max_valid_horizon,
    periods_until,
    terminal_distribution,
    walk_forward_validate,
)
from src.risk_rules import (
    DEFAULT_BR_MIN,
    DEFAULT_CB,
    DEFAULT_CS,
    DEFAULT_PL,
    DEFAULT_PU,
    RiskParameters,
    compute_var,
    evaluate_signal,
    terminal_probabilities,
)
from src.fx import fetch_fx_series
from src.universe import build_return_matrix
from src.risk_matrix import (
    check_covariance_matrix,
    correlation_from_covariance,
    efficient_frontier,
    gmv_weights,
    max_expected_return_weights,
    personal_optimal_weights,
    portfolio_volatility,
    sample_moments,
    tangency_weights,
)
from src.black_litterman import (
    View,
    active_views,
    assemble_views,
    black_litterman_posterior,
    check_black_litterman_inputs,
    excess_from_total_view,
    implied_equilibrium_returns,
)
from src.rebalance import (
    drifted_weights,
    needs_rebalance,
    simulate_rebalance_to_target,
    turnover_unilateral,
)

# ---------------------------------------------------------------------------
# MODO DESARROLLO: mientras no tengan credenciales OIDC reales, dejen esto en
# True para poder probar el resto de la app (datos, análisis, forecasting,
# riesgo, comparación) sin quedar bloqueados por el login. Antes de la
# entrega final, si el profesor confirma que el login es obligatorio, pasen
# esto a False y usen las credenciales reales.
# ---------------------------------------------------------------------------
DEV_MODE = True

# ---------------------------------------------------------------------------
# IDENTIDAD DEL EQUIPO (RF-01)
# ---------------------------------------------------------------------------
SYSTEM_NAME = "Pre - Asset Allocation Analytic Bot"
TEAM_NAME = "Equipo DPST"
MEMBERS = [
    "Sergio Delgado Laverde",
    "David Angel Perez",
    "Pedro Velez Uribe",
    "Tomás Giraldo Gomez",
]
VERSION = "0.1.0"
LAST_UPDATE = "2026-09-14"

EXECUTION_POLICY_SUMMARY = (
    "Piloto académico restringido. Posición larga únicamente. Rendimiento "
    "logarítmico exclusivo. El sistema puede emitir 'no señal'. No calcula "
    "portafolios, pesos, covarianzas ni frontera eficiente. Ver la vista "
    "Metodología para la política de ejecución completa."
)

# Descripciones cortas de cada prueba de diagnóstico (sección 5.4), mostradas
# en la app para que quede claro qué mide cada una además del resultado.
DIAGNOSTIC_DESCRIPTIONS = {
    "Jarque-Bera (normalidad)": (
        "Compara la asimetría y la curtosis de los rendimientos con las que "
        "tendría una distribución normal perfecta. Entre más distintas sean, "
        "más alto sale el estadístico y más bajo el p-valor."
    ),
    "Ljung-Box (dependencia)": (
        "Revisa si los rendimientos de distintos días están correlacionados "
        "entre sí (autocorrelación) hasta cierto número de rezagos. Si hay un "
        "patrón que se repite en el tiempo, el estadístico sube y el p-valor "
        "baja."
    ),
    "Brown-Forsythe (homocedasticidad)": (
        "Divide la muestra en bloques cronológicos y compara si la "
        "volatilidad (varianza) es parecida entre esos bloques. Si algún "
        "bloque resulta mucho más volátil que otro, el estadístico sube y el "
        "p-valor baja."
    ),
    "ARCH-LM (varianza condicional)": (
        "Revisa si los periodos de alta volatilidad tienden a agruparse "
        "('volatility clustering'), es decir, si la varianza de hoy depende "
        "de la varianza de días recientes. Si existe ese patrón, el "
        "estadístico sube y el p-valor baja."
    ),
}

# Explicaciones cortas de cada criterio de preselección (RF-21, sección 5.8),
# mostradas justo debajo del selector para que quede claro qué hace cada uno.
CRITERION_EXPLANATIONS = {
    "Máxima media histórica bajo límite de riesgo": (
        "De los activos cuya volatilidad anualizada NO supera el límite que "
        "definas abajo, escoge el que tenga la media histórica anualizada "
        "más alta. Si ningún activo cumple ese límite de riesgo, no se "
        "selecciona ninguno."
    ),
    "Mínima volatilidad bajo media mínima": (
        "De los activos cuya media histórica anualizada sea al menos la que "
        "definas abajo, escoge el que tenga la menor volatilidad anualizada. "
        "Si ningún activo alcanza esa media mínima, no se selecciona "
        "ninguno."
    ),
    "Máxima razón individual media-volatilidad (RVR)": (
        "Calcula, para cada activo POR SEPARADO, su propia razón "
        "media anualizada / volatilidad anualizada (una especie de "
        "'rendimiento por unidad de riesgo' individual) y escoge el activo "
        "con el valor más alto. No combina ni pondera activos entre sí, solo "
        "compara activos individuales uno por uno."
    ),
    "Conjunto no dominado (media-volatilidad)": (
        "Un activo está 'dominado' cuando existe otro activo del lote con "
        "media igual o mayor Y volatilidad igual o menor al mismo tiempo (es "
        "decir, otro activo es mejor o igual en las dos cosas a la vez). "
        "Este criterio devuelve TODOS los activos que no están dominados por "
        "ningún otro, así que puede devolver varios, no necesariamente uno "
        "solo."
    ),
}

# Texto mínimo obligatorio (guía, sección 3.3). No reducir su alcance.
DISCLAIMER_TEXT = (
    "Esta aplicación fue desarrollada exclusivamente como actividad evaluativa del "
    "curso Teoría Moderna de Portafolios de Tech Business School - Universidad EIA. "
    "Sus datos, modelos y resultados tienen fines académicos y educativos. En ningún "
    "momento constituye asesoría financiera, recomendación de inversión ni una "
    "herramienta para tomar decisiones de inversión en la vida real. Es una "
    "herramienta académica que deberá revisarse, validarse y ajustarse, y puede "
    "contener errores, omisiones, rezagos o información incompleta. El sistema no "
    "ejecuta operaciones ni garantiza resultados."
)


def render_header() -> None:
    st.set_page_config(page_title=SYSTEM_NAME, page_icon="📊", layout="wide")
    st.title(f"📊 {SYSTEM_NAME}")
    st.caption(f"{TEAM_NAME} · versión {VERSION} · última actualización {LAST_UPDATE}")
    with st.expander("Equipo e identidad"):
        for member in MEMBERS:
            st.write(f"- {member}")
        st.info(EXECUTION_POLICY_SUMMARY)


def render_disclaimer_gate(user_name: str, user_email: str) -> None:
    """
    Bloquea el análisis hasta que el usuario acepte, por separado, la
    autorización de privacidad y el disclaimer académico (sección 3.3).
    Detiene la ejecución (st.stop) si falta alguna de las dos.

    La primera vez que ambos consentimientos quedan aceptados en esta
    sesión del navegador, registra el acceso (nombre, correo autoinformados
    y ambos consentimientos, con fecha/hora UTC) en un archivo local -
    `data/access_log.sqlite3`, ver `src/access_log.py` -. El curso no
    suministró base de datos, sink ni gestor de secretos institucional
    (sección 7.1, punto 46 de la guía); este registro local es la evidencia
    parcial que se implementó ante esa ausencia. No sustituye el modelo
    completo de sesión/auditoría/purga de la Fase 9 (ver TRACEABILITY.md,
    RF-02, RNF-03).
    """
    st.subheader("Antes de continuar")
    st.warning(DISCLAIMER_TEXT)

    privacy_ok = st.checkbox(
        "Autorizo el tratamiento de mis datos según el aviso de privacidad del "
        "piloto académico (Ley 1581 de 2012).",
        value=False,
        key="consent_privacy",
    )
    disclaimer_ok = st.checkbox(
        "He leído y acepto el disclaimer académico anterior.",
        value=False,
        key="consent_disclaimer",
    )

    if not (privacy_ok and disclaimer_ok):
        st.session_state["access_logged"] = False
        st.stop()

    if not st.session_state.get("access_logged", False):
        access_log.log_access(user_name, user_email, privacy_ok, disclaimer_ok)
        st.session_state["access_logged"] = True


def render_historical_analysis() -> None:
    """
    Análisis histórico de un activo (RF-09 a RF-12): rendimientos
    logarítmicos, gráficos, descriptivos y drawdown.
    """
    result = st.session_state.get("asset_result")
    frequency = st.session_state.get("asset_frequency")
    if result is None or not result.ok:
        return

    st.divider()
    st.subheader(f"Análisis histórico de {result.ticker} (RF-09 a RF-12)")

    prices = result.prices["adjusted_close"]
    returns = log_returns(prices)

    if len(returns) < 2:
        st.warning("Muestra insuficiente para calcular estadísticas (se necesitan al menos 2 rendimientos).")
        return

    periods_per_year = FREQUENCY_PERIODS_PER_YEAR.get(frequency, 252)

    st.latex(r"g_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")

    tab_price, tab_returns, tab_stats, tab_diag = st.tabs(
        ["Precio", "Rendimientos", "Descriptivos", "Diagnósticos"]
    )

    with tab_price:
        st.plotly_chart(
            price_chart(prices, result.ticker, result.currency, frequency, result.source),
            use_container_width=True,
        )
        st.caption(
            f"Último precio: {prices.iloc[-1]:.4f} ({prices.index[-1].date()}) · "
            f"Fuente: {result.source} · Moneda: {result.currency or 'N/D'}"
        )
        with st.expander("Ver datos en tabla (alternativa accesible al gráfico)"):
            st.dataframe(prices.tail(30).rename("precio_ajustado"))

    with tab_returns:
        st.plotly_chart(return_chart(returns, result.ticker, frequency), use_container_width=True)
        st.caption(f"Último log-rendimiento: {returns.iloc[-1]:.6f} ({returns.index[-1].date()})")
        st.info(
            "**¿Qué es un rendimiento logarítmico?** Es la variación del precio "
            "entre un periodo y el anterior, calculada con el logaritmo natural: "
            "g_t = ln(P_t / P_{t-1}). Un valor de 0.02 equivale aproximadamente a "
            "una subida de 2% respecto al periodo anterior, y uno de -0.02 a una "
            "caída de aproximadamente 2%. Se usa el logaritmo en vez del "
            "porcentaje simple porque los rendimientos logarítmicos de varios "
            "periodos se pueden sumar directamente para obtener el rendimiento "
            "acumulado, y porque sus propiedades estadísticas son las que asumen "
            "los modelos de esta guía (RF-10)."
        )
        with st.expander("Ver datos en tabla (alternativa accesible al gráfico)"):
            st.dataframe(returns.tail(30).rename("log_rendimiento"))

    with tab_stats:
        stats = descriptive_stats(returns, periods_per_year)
        recent_stats, is_partial = recent_window_stats(returns, periods_per_year)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Muestra completa**")
            st.write(
                {
                    "n": stats.n,
                    "media": stats.mean,
                    "mediana": stats.median,
                    "varianza": stats.variance,
                    "desv. estándar": stats.std,
                    "mínimo": stats.minimum,
                    "P25": stats.p25,
                    "P75": stats.p75,
                    "máximo": stats.maximum,
                    "asimetría (Fisher)": stats.skewness,
                    "exceso de curtosis (Fisher)": stats.excess_kurtosis,
                    "media anualizada": stats.mean_annualized,
                    "volatilidad anualizada": stats.std_annualized,
                    "percentil (midrank) del último g_t": stats.last_return_percentile,
                }
            )
        with col2:
            label = "Ventana reciente" + (" (PARCIAL, T < m)" if is_partial else "")
            st.markdown(f"**{label}**")
            st.write(
                {
                    "n": recent_stats.n,
                    "media": recent_stats.mean,
                    "desv. estándar": recent_stats.std,
                    "media anualizada": recent_stats.mean_annualized,
                    "volatilidad anualizada": recent_stats.std_annualized,
                }
            )

        dd = drawdown_series(prices)
        st.metric("Máxima caída histórica (drawdown)", f"{max_drawdown(prices):.2%}")
        st.line_chart(dd, height=200)

    with tab_diag:
        st.markdown(
            "Contrastes de los supuestos que sustentan el uso de modelos "
            "**homocedásticos** (sección 5.4 de la guía). Ningún rechazo "
            "bloquea el análisis: se muestran como advertencias informativas, "
            "nunca como error."
        )
        st.caption(
            "**Regla de aceptación/rechazo (para las cuatro pruebas):** cada "
            "una compara su p-valor contra alfa = 5%. Si el p-valor es MENOR "
            "que 5%, se rechaza la hipótesis nula (el supuesto no se cumple "
            "en esta muestra, ícono ⚠️); si es mayor o igual, no se rechaza "
            "(el supuesto es razonable con esta muestra, ícono ✅)."
        )
        if len(returns) < MIN_DIAGNOSTICS_N:
            st.info(
                f"Muestra insuficiente para diagnósticos confiables "
                f"(hay {len(returns)} rendimientos, se recomiendan al menos "
                f"{MIN_DIAGNOSTICS_N}). No se muestran los contrastes."
            )
        else:
            for diag in run_diagnostics(returns):
                icon = "⚠️" if diag.reject else "✅"
                with st.expander(f"{icon} {diag.name}", expanded=diag.reject):
                    st.write(f"**¿Qué revisa esta prueba?** {DIAGNOSTIC_DESCRIPTIONS.get(diag.name, '')}")
                    st.write(f"**Hipótesis nula:** {diag.hypothesis}")
                    st.write(f"**Parámetros:** {diag.parameters}")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Estadístico", f"{diag.statistic:.4f}")
                    col_b.metric("p-valor", f"{diag.p_value:.4f}")
                    if diag.reject:
                        st.warning(diag.interpretation)
                    else:
                        st.success(diag.interpretation)


def render_forecast() -> None:
    """
    Pronóstico homocedástico multihorizonte y validación walk-forward
    (RF-13 a RF-15, secciones 5.5 y 8.1 de la guía).
    """
    result = st.session_state.get("asset_result")
    frequency = st.session_state.get("asset_frequency")
    if result is None or not result.ok:
        return

    prices = result.prices["adjusted_close"]
    returns = log_returns(prices)
    if len(returns) < 2:
        return

    p0 = float(prices.iloc[-1])
    last_date = prices.index[-1]
    periods_per_year = FREQUENCY_PERIODS_PER_YEAR[frequency]
    h_max = max_valid_horizon(len(returns), periods_per_year)
    if h_max < 1:
        st.warning(
            "La muestra disponible no alcanza para validar ningún horizonte H con la "
            "regla de suficiencia de la sección 5.5 (T >= max(2m,5H)+H+9). "
            "Elijan un rango de fechas más amplio."
        )
        return

    st.divider()
    st.subheader(f"Pronóstico de {result.ticker} (RF-13 a RF-15)")
    st.caption(
        "Piloto académico: los modelos son homocedásticos por alcance pedagógico. "
        "Si los diagnósticos de la pestaña anterior muestran volatilidad variable o "
        "colas gruesas, los resultados siguientes se mantienen homocedásticos de "
        "todas formas y deben leerse como condicionales a ese supuesto."
    )

    col_model, col_h_mode = st.columns(2)
    with col_model:
        model_label = st.selectbox("Modelo", list(FORECAST_MODELS.keys()), key="forecast_model")
    with col_h_mode:
        h_mode = st.radio(
            "Definir horizonte H como", ["Cantidad de periodos", "Fecha objetivo"],
            horizontal=True, key="forecast_h_mode",
        )

    if h_mode == "Cantidad de periodos":
        st.caption(
            f"H máximo admitido con los datos actuales: {h_max} periodo(s) "
            "(límite justificado por la regla de suficiencia walk-forward de la "
            "sección 5.5: T >= max(2m,5H)+H+9 — RF-04)."
        )
        horizon = st.number_input(
            f"H (en periodos de frecuencia '{frequency}')", min_value=1, max_value=h_max,
            value=min(10, h_max), step=1, key="forecast_h_periods",
        )
    else:
        target_date = st.date_input(
            "Fecha objetivo", value=last_date + pd.Timedelta(days=14), key="forecast_target_date",
        )
        horizon = periods_until(last_date, target_date, frequency)
        if horizon <= 0:
            st.warning("La fecha objetivo debe ser posterior a la última fecha con dato.")
            return
        if horizon > h_max:
            st.warning(
                f"La fecha objetivo implica H={horizon}, que supera el límite H_max={h_max} "
                "admitido por la regla de suficiencia walk-forward de la sección 5.5 "
                "(T >= max(2m,5H)+H+9). Elijan una fecha objetivo más cercana."
            )
            return
        st.caption(
            f"H convertido a {horizon} periodo(s) de frecuencia '{frequency}' entre "
            f"{last_date.date()} y {target_date}."
        )

    horizon = int(horizon)
    fit_fn = FORECAST_MODELS[model_label]
    model = fit_fn(returns)
    dist = terminal_distribution(model, p0, horizon)
    steps = forecast_path(model, p0, horizon)

    st.latex(r"P_H = P_0 \cdot \exp(g_{1} + g_{2} + \dots + g_{H}), \quad g_{1}+\dots+g_{H} \sim \mathcal{N}(m_H,\, v_H)")
    st.write(
        f"**Modelo:** {dist.model_name} · **Parámetros (por periodo, de la muestra de entrenamiento):** "
        f"media = {dist.mean_per_period:.6f}, sigma = {dist.std_per_period:.6f}, "
        f"n = {model.n_train} · **Supuesto:** homocedástico, sin autocorrelación."
    )
    if dist.is_deterministic:
        st.info(
            "Varianza predictiva igual a cero (rama determinista, sección 5.4.1): "
            "todos los cuantiles coinciden con la mediana."
        )

    tab_traj, tab_terminal, tab_wf, tab_risk = st.tabs(
        ["Trayectoria 1..H", "Distribución terminal", "Validación walk-forward", "Riesgo y niveles"]
    )

    with tab_traj:
        st.plotly_chart(forecast_chart(prices, steps, result.ticker, frequency), use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "paso (k)": [s.step for s in steps],
                    "m_k": [s.m_k for s in steps],
                    "v_k": [s.v_k for s in steps],
                    "mediana": [s.median_price for s in steps],
                    "media": [s.mean_price for s in steps],
                    "Q05": [s.q05_price for s in steps],
                    "Q95": [s.q95_price for s in steps],
                }
            ).set_index("paso (k)")
        )

    with tab_terminal:
        target_label = (last_date + pd.tseries.offsets.BDay(horizon)).date() if frequency == "Diaria" else None
        st.write(f"**Horizonte H = {horizon}** periodo(s)" + (f" · fecha objetivo aprox.: {target_label}" if target_label else ""))
        col1, col2, col3 = st.columns(3)
        col1.metric("Mediana P_H", f"{dist.median_price:.4f}")
        col2.metric("Media P_H", f"{dist.mean_price:.4f}")
        col3.metric("P(P_H > P0)", f"{dist.prob_above_entry:.2%}")
        col4, col5, col6 = st.columns(3)
        col4.metric("Q05 P_H", f"{dist.q05_price:.4f}")
        col5.metric("Q95 P_H", f"{dist.q95_price:.4f}")
        col6.metric("m_H / v_H", f"{dist.m_h:.6f} / {dist.v_h:.6f}")

    with tab_wf:
        st.caption(
            f"Ventana expansiva, últimos {WALKFORWARD_ORIGINS} orígenes consecutivos, "
            "reestimación solo con datos disponibles en cada origen (sin look-ahead). "
            "Objetivo: rendimiento logarítmico acumulado a H."
        )
        wf = walk_forward_validate(returns, fit_fn, model_label, horizon, periods_per_year)
        if not wf.ok:
            st.warning(f"Validación insuficiente (no señal sustentada): {wf.reason}")
        else:
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("RMSE (log-espacio)", f"{wf.rmse:.6f}")
            col_b.metric("MAE (log-espacio)", f"{wf.mae:.6f}")
            col_c.metric("Cobertura IC 90%", f"{wf.coverage_90:.0%}")
            col_d.metric(
                "Exactitud direccional",
                f"{wf.directional_accuracy:.0%}" if wf.directional_accuracy is not None else "N/D",
                help="N/D si el modelo no define una dirección (p.ej. caminata aleatoria, m_H=0).",
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "origen (índice)": [o.origin_index for o in wf.origins],
                        "n entrenamiento": [o.n_train for o in wf.origins],
                        "pronóstico (m_H)": [o.forecast_cum_return for o in wf.origins],
                        "Q05": [o.q05_cum_return for o in wf.origins],
                        "Q95": [o.q95_cum_return for o in wf.origins],
                        "real (acumulado)": [o.actual_cum_return for o in wf.origins],
                        "dentro IC 90%": [o.in_interval_90 for o in wf.origins],
                        "dirección correcta": [
                            "N/D" if o.direction_correct is None else o.direction_correct for o in wf.origins
                        ],
                    }
                ).set_index("origen (índice)")
            )

    with tab_risk:
        st.caption(
            "VaR, probabilidades, entrada, stop-loss y take-profit comparten el mismo "
            f"modelo ({dist.model_name}) y el mismo horizonte (H={horizon}) que las "
            "pestañas anteriores (RF-16 a RF-18, secciones 5.6 y 5.7)."
        )

        col_cap, col_conf, col_brmin = st.columns(3)
        with col_cap:
            capital = st.number_input(
                "Capital a invertir", min_value=0.0, value=10000.0, step=100.0, key="risk_capital",
            )
        with col_conf:
            confidence_label = st.selectbox(
                "Nivel de confianza del VaR", ["95%", "99%"], key="risk_confidence",
            )
            confidence = 0.95 if confidence_label == "95%" else 0.99
        with col_brmin:
            br_min = st.number_input(
                "BR_min (beneficio/riesgo neto mínimo)", min_value=0.0, value=DEFAULT_BR_MIN,
                step=0.1, key="risk_br_min",
            )

        col_cb, col_cs = st.columns(2)
        with col_cb:
            c_b = st.number_input(
                "Costo de compra c_b (fracción, ej. 0.001 = 0.1%)", min_value=0.0, value=DEFAULT_CB,
                step=0.001, format="%.4f", key="risk_cb",
            )
        with col_cs:
            c_s = st.number_input(
                "Costo de venta c_s (fracción, ej. 0.001 = 0.1%)", min_value=0.0, max_value=0.999,
                value=DEFAULT_CS, step=0.001, format="%.4f", key="risk_cs",
            )

        params = RiskParameters(p_l=DEFAULT_PL, p_u=DEFAULT_PU, br_min=br_min, c_b=c_b, c_s=c_s)

        st.markdown(f"**VaR paramétrico individual ({confidence_label}, RF-16)**")
        var_result = compute_var(model, p0, horizon, confidence, capital)
        col_v1, col_v2 = st.columns(2)
        col_v1.metric(f"VaR fraccional ({confidence_label})", f"{var_result.var_fraction:.4%}")
        col_v2.metric("VaR en unidades monetarias", f"{var_result.var_dollar:,.2f}")

        st.divider()
        st.markdown(f"**Entrada, stop-loss y take-profit (p_L={params.p_l:.0%}, p_U={params.p_u:.0%}, RF-17)**")
        signal = evaluate_signal(model, p0, horizon, params)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Entrada (E = P0)", f"{signal.entry:.4f}")
        col_s2.metric("Stop-loss (SL_H)", f"{signal.sl_h:.4f}")
        col_s3.metric("Take-profit (TP_H)", f"{signal.tp_h:.4f}")
        col_s4.metric("Precio de equilibrio (P_BE)", f"{signal.p_be:.4f}")

        col_n1, col_n2, col_n3 = st.columns(3)
        col_n1.metric("Riesgo neto (D_neto)", f"{signal.d_neto:.4f}" if signal.d_neto is not None else "N/D")
        col_n2.metric("Beneficio neto (U_neto)", f"{signal.u_neto:.4f}" if signal.u_neto is not None else "N/D")
        col_n3.metric("BR_neto", f"{signal.br_neto:.4f}" if signal.br_neto is not None else "N/D")
        if signal.br_bruta is not None:
            st.caption(f"BR_bruta (sin costos, referencia): {signal.br_bruta:.4f}")

        if signal.ok:
            st.success("Señal sustentada: se cumplen todas las condiciones de la sección 5.7.")
        else:
            st.warning("No señal (nunca se fuerza una señal). Motivos:")
            for reason in signal.reasons:
                st.write(f"- {reason}")

        st.divider()
        st.markdown("**Probabilidades terminales frente a P_BE (RF-18)**")
        term_probs = terminal_probabilities(model, p0, horizon, signal.p_be)
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Pr(ganar)", f"{term_probs.prob_win:.2%}")
        col_p2.metric("Pr(perder)", f"{term_probs.prob_lose:.2%}")
        col_p3.metric("Pr(neutral)", f"{term_probs.prob_neutral:.2%}" if term_probs.is_deterministic else "N/A")
        if term_probs.is_deterministic:
            st.info(
                "Rama determinista (v_H=0): la probabilidad se concentra por completo "
                "en un único desenlace (ganar, perder o neutral)."
            )


def render_comparison() -> None:
    """
    Comparación de N activos, mapa histórico rendimiento-riesgo, dominancia,
    preselección transparente y exportación (RF-19 a RF-22, sección 5.8).

    Vista independiente del análisis de un solo activo de arriba. No calcula
    pesos, covarianzas, correlaciones conjuntas ni frontera eficiente: es una
    preselección de activos individuales.
    """
    st.divider()
    st.subheader("Comparación de activos (RF-19 a RF-22)")
    st.caption(
        "Vista independiente del análisis de un solo activo. Acepta un mínimo "
        f"simultáneo de {MIN_COMPARISON_ASSETS} activos válidos. No calcula pesos, "
        "covarianzas, correlaciones conjuntas ni frontera eficiente: es una preselección "
        "de activos individuales, no asset allocation."
    )

    default_tickers = (
        "AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, JNJ, XOM, PG, "
        "KO, PEP, WMT, HD, COST, UNH, V, MA, CAT, MCD"
    )
    tickers_text = st.text_area(
        "Tickers a comparar (sepáralos por coma o por línea; agrega o borra los que quieras)",
        value=default_tickers,
        key="comparison_tickers_text",
        height=80,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Fecha inicial", value=pd.Timestamp("2023-01-01"), key="comparison_start")
    with col2:
        end_date = st.date_input("Fecha final", value=pd.Timestamp.today(), key="comparison_end")
    with col3:
        frequency = st.selectbox("Frecuencia", list(FREQUENCY_RULES.keys()), key="comparison_frequency")

    if st.button("Comparar activos"):
        raw_tickers = [
            t.strip().upper() for chunk in tickers_text.split(",") for t in chunk.split("\n")
        ]
        tickers, seen = [], set()
        for t in raw_tickers:
            if t and t not in seen:
                tickers.append(t)
                seen.add(t)

        if len(tickers) < 2:
            st.error("Escribe al menos 2 tickers para comparar.")
        else:
            with st.spinner(f"Descargando y comparando {len(tickers)} activos..."):
                assets = fetch_many(tickers, start_date, end_date, frequency)
                periods_per_year = FREQUENCY_PERIODS_PER_YEAR.get(frequency, 252)
                result = build_comparison(assets, frequency, periods_per_year)
            st.session_state["comparison_result"] = result

    result = st.session_state.get("comparison_result")
    if result is None:
        return

    if not result.ok:
        st.error(f"No se pudo construir la comparación: {result.reason}")
        if result.excluded:
            with st.expander(f"Activos excluidos ({len(result.excluded)})"):
                for e in result.excluded:
                    st.write(f"- **{e.ticker}**: {e.reason}")
        return

    n_valid = len(result.summaries)
    if n_valid < MIN_COMPARISON_ASSETS:
        st.warning(
            f"Hay {n_valid} activo(s) válido(s) en esta comparación; la guía exige que la "
            f"vista acepte al menos {MIN_COMPARISON_ASSETS} simultáneamente. Agrega más "
            "tickers válidos para cumplir ese mínimo."
        )
    else:
        st.success(f"{n_valid} activos válidos comparados (mínimo exigido: {MIN_COMPARISON_ASSETS}).")

    st.caption(
        f"Moneda base: {result.base_currency} · Frecuencia: {result.frequency} · "
        f"Ventana común: {result.common_start.date()} a {result.common_end.date()} "
        f"({result.n_common} observaciones de precio). Las coordenadas del mapa son "
        "históricas y NO cambian con el horizonte H del pronóstico."
    )

    if result.excluded:
        with st.expander(f"Activos excluidos de esta comparación ({len(result.excluded)})"):
            for e in result.excluded:
                st.write(f"- **{e.ticker}**: {e.reason}")

    non_dominated = non_dominated_tickers(result.summaries)
    table = comparison_table(result, non_dominated)

    tab_table, tab_map, tab_select, tab_export = st.tabs(
        ["Tabla comparativa", "Mapa histórico", "Preselección", "Exportar"]
    )

    with tab_table:
        st.dataframe(table)

    with tab_map:
        st.plotly_chart(risk_return_map(result.summaries, non_dominated), use_container_width=True)
        st.caption(
            "Rombos rojos = activos NO dominados (ningún otro activo del lote ofrece a la "
            "vez media igual o mayor y volatilidad igual o menor). Esto no es una frontera "
            "eficiente de portafolios ni una recomendación de la mejor inversión universal."
        )

    with tab_select:
        st.write(
            "Regla de preselección transparente (RF-21). El resultado se expresa como "
            "'mejor según este criterio y estos parámetros', nunca como la mejor inversión "
            "en términos absolutos."
        )
        criterion = st.selectbox(
            "Criterio",
            [
                "Máxima media histórica bajo límite de riesgo",
                "Mínima volatilidad bajo media mínima",
                "Máxima razón individual media-volatilidad (RVR)",
                "Conjunto no dominado (media-volatilidad)",
            ],
            key="comparison_criterion",
        )
        st.caption(CRITERION_EXPLANATIONS[criterion])

        if criterion == "Máxima media histórica bajo límite de riesgo":
            risk_limit = st.number_input(
                "Límite de riesgo (volatilidad anualizada máxima aceptable)",
                min_value=0.0, value=0.30, step=0.01, format="%.4f", key="comparison_risk_limit",
            )
            selected = preselect_max_mean_under_risk_limit(result.summaries, risk_limit)
        elif criterion == "Mínima volatilidad bajo media mínima":
            min_mean = st.number_input(
                "Media histórica anualizada mínima aceptable",
                value=0.0, step=0.01, format="%.4f", key="comparison_min_mean",
            )
            selected = preselect_min_vol_under_min_mean(result.summaries, min_mean)
        elif criterion == "Máxima razón individual media-volatilidad (RVR)":
            selected = preselect_max_rvr(result.summaries)
        else:
            selected = preselect_non_dominated(result.summaries)

        if selected:
            tie_note = " (empate)" if len(selected) > 1 else ""
            st.success(f"Mejor según este criterio: {', '.join(selected)}{tie_note}")
        else:
            st.warning("Ningún activo cumple este criterio con los parámetros elegidos.")

    with tab_export:
        st.write("Descarga los datos procesados de esta comparación (RF-22).")
        params = export_parameters(result)
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.download_button(
                "Precios procesados (CSV)",
                data=processed_prices_export(result).to_csv(index=False),
                file_name="precios_procesados.csv",
                mime="text/csv",
            )
        with col_e2:
            st.download_button(
                "Tabla comparativa (CSV)",
                data=table.reset_index().to_csv(index=False),
                file_name="tabla_comparativa.csv",
                mime="text/csv",
            )
        with col_e3:
            st.download_button(
                "Parámetros (JSON)",
                data=json.dumps(params, indent=2, ensure_ascii=False),
                file_name="parametros_comparacion.json",
                mime="application/json",
            )

def render_portfolios() -> None:
    """
    Matriz de covarianzas/correlaciones y las 4 carteras de Markowitz
    (escalamiento v2: src.fx + src.universe + src.risk_matrix).

    A diferencia de render_comparison() (que es preselección individual),
    aquí SÍ se calculan pesos, covarianzas y carteras — lo que v1 excluía
    explícitamente. Posiciones largas únicamente, sin apalancamiento
    (suma de pesos = 1, 0% a 100% por activo).
    """
    st.divider()
    st.subheader("Portafolios: covarianza y las 4 carteras de Markowitz (v2)")
    st.caption(
        "Construye la matriz de covarianzas/correlaciones del universo elegido y las "
        "4 carteras de la guía (mínima varianza, tangente, óptimo personal y máximo "
        "rendimiento esperado). Admite moneda mixta (USD/COP, con conversión automática) "
        "y acepta ETFs de bonos (TLT, IEF, SHY, etc.) como cualquier otro ticker."
    )

    default_tickers = (
        "AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, JNJ, XOM, PG, "
        "KO, PEP, WMT, HD, COST, UNH, V, MA, CAT, MCD"
    )
    tickers_text = st.text_area(
        "Tickers del universo (agrega o quita los que quieras)",
        value=default_tickers, key="portfolio_tickers_text", height=80,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Fecha inicial", value=pd.Timestamp("2023-01-01"), key="portfolio_start")
    with col2:
        end_date = st.date_input("Fecha final", value=pd.Timestamp.today(), key="portfolio_end")
    with col3:
        frequency = st.selectbox("Frecuencia", list(FREQUENCY_RULES.keys()), key="portfolio_frequency")

    col4, col5, col6 = st.columns(3)
    with col4:
        base_currency = st.selectbox("Moneda base", ["USD", "COP"], key="portfolio_base_currency")
    with col5:
        rf_annual = st.number_input(
            "Tasa libre de riesgo anual (rf)", min_value=0.0, value=0.04, step=0.001,
            format="%.4f", key="portfolio_rf_input",
        )
    with col6:
        gamma = st.number_input(
            "Aversión al riesgo del inversionista (gamma, para el óptimo personal)",
            min_value=0.01, value=5.0, step=0.5, key="portfolio_gamma",
        )

    if st.button("Construir portafolios"):
        raw_tickers = [
            t.strip().upper() for chunk in tickers_text.split(",") for t in chunk.split("\n")
        ]
        tickers, seen = [], set()
        for t in raw_tickers:
            if t and t not in seen:
                tickers.append(t)
                seen.add(t)

        if len(tickers) < 2:
            st.error("Escribe al menos 2 tickers para construir un portafolio.")
        else:
            with st.spinner(f"Descargando {len(tickers)} activos y armando el universo..."):
                assets = fetch_many(tickers, start_date, end_date, frequency)
                periods_per_year = FREQUENCY_PERIODS_PER_YEAR.get(frequency, 252)

                currencies_present = {
                    a.currency.upper() for a in assets.values() if a.ok and a.currency
                }
                fx_rates = {}
                fx_errors = []
                for currency in currencies_present - {base_currency}:
                    try:
                        fx_rates[currency] = fetch_fx_series(currency, start_date, end_date)
                    except ValueError as exc:
                        fx_errors.append(f"{currency}: {exc}")

                matrix, result = build_return_matrix(
                    assets, frequency, periods_per_year, base_currency,
                    fx_rates_quote_per_usd=fx_rates or None,
                )

            st.session_state["portfolio_matrix"] = matrix
            st.session_state["portfolio_comparison"] = result
            st.session_state["portfolio_rf_period"] = rf_annual / periods_per_year
            st.session_state["portfolio_gamma_value"] = gamma
            st.session_state["portfolio_periods_per_year"] = periods_per_year
            if fx_errors:
                st.warning("No se pudo obtener la tasa FX para: " + "; ".join(fx_errors))

    matrix = st.session_state.get("portfolio_matrix")
    result = st.session_state.get("portfolio_comparison")
    if matrix is None or result is None:
        return

    if not result.ok or matrix.empty:
        st.error(f"No se pudo construir el universo: {result.reason}")
        if result.excluded:
            with st.expander(f"Activos excluidos ({len(result.excluded)})"):
                for e in result.excluded:
                    st.write(f"- **{e.ticker}**: {e.reason}")
        return

    if len(matrix.columns) < 2:
        st.warning("Se necesitan al menos 2 activos válidos en el universo para construir carteras.")
        return

    st.success(
        f"{len(matrix.columns)} activos en el universo, moneda base {result.base_currency}, "
        f"ventana común {result.common_start.date()} a {result.common_end.date()} "
        f"({len(matrix)} retornos)."
    )
    if result.excluded:
        with st.expander(f"Activos excluidos de este universo ({len(result.excluded)})"):
            for e in result.excluded:
                st.write(f"- **{e.ticker}**: {e.reason}")

    try:
        mu, sigma = sample_moments(matrix)
    except ValueError as exc:
        st.error(f"No se pudo estimar mu/Sigma: {exc}")
        return

    diag = check_covariance_matrix(sigma)
    corr = correlation_from_covariance(sigma)
    rf_period = st.session_state["portfolio_rf_period"]
    gamma_value = st.session_state["portfolio_gamma_value"]
    periods_per_year = st.session_state["portfolio_periods_per_year"]
    bounds = [(0.0, 1.0)] * len(sigma)

    tab_cov, tab_port, tab_frontier = st.tabs(
        ["Matriz de covarianzas/correlaciones", "Las 4 carteras", "Frontera eficiente"]
    )

    with tab_cov:
        st.write(
            f"Diagnóstico: simétrica={diag['is_symmetric']}, PSD={diag['is_psd']}, "
            f"singular={diag['is_singular']}, número de condición={diag['condition_number']:.2f}"
        )
        st.markdown("**Correlaciones**")
        st.dataframe(corr.style.format("{:.3f}"))
        st.markdown("**Covarianzas (por periodo, sin anualizar)**")
        st.dataframe(sigma.style.format("{:.6f}"))

    with tab_port:
        try:
            w_gmv = gmv_weights(sigma, bounds=bounds)
            w_tan = tangency_weights(mu, sigma, rf_period, bounds=bounds)
            w_per = personal_optimal_weights(mu, sigma, gamma_value, bounds=bounds)
            w_max = max_expected_return_weights(mu, sigma, bounds=bounds)
        except Exception as exc:
            st.error(f"No se pudieron calcular las carteras: {exc}")
        else:
            rows = []
            for name, w in [
                ("Mínima varianza (GMV)", w_gmv),
                ("Tangente (máx. Sharpe)", w_tan),
                (f"Óptimo personal (gamma={gamma_value:g})", w_per),
                ("Máximo rendimiento esperado", w_max),
            ]:
                ret_annual = float(w @ mu) * periods_per_year
                vol_annual = portfolio_volatility(w, sigma) * (periods_per_year ** 0.5)
                rows.append({
                    "cartera": name,
                    "rendimiento anualizado": ret_annual,
                    "riesgo anualizado": vol_annual,
                })
            st.dataframe(pd.DataFrame(rows).set_index("cartera").style.format("{:.2%}"))

            st.markdown("**Composición de cada cartera**")
            weights_table = pd.DataFrame({
                "Mínima varianza (GMV)": w_gmv,
                "Tangente": w_tan,
                "Óptimo personal": w_per,
                "Máximo rendimiento": w_max,
            })
            st.dataframe(weights_table.style.format("{:.2%}"))

    with tab_frontier:
        try:
            frontier = efficient_frontier(mu, sigma, bounds=bounds, n_points=25)
        except Exception as exc:
            st.error(f"No se pudo construir la frontera: {exc}")
        else:
            plot_df = pd.DataFrame({
                "riesgo anualizado": frontier["risk"] * (periods_per_year ** 0.5),
                "rendimiento anualizado": frontier["target_return"] * periods_per_year,
                "eficiente": frontier["is_efficient"],
            })
            st.scatter_chart(plot_df, x="riesgo anualizado", y="rendimiento anualizado", color="eficiente")
            st.caption(
                "Los puntos marcados como eficientes (True) pertenecen a la rama eficiente; "
                "los demás son la rama inferior no eficiente de la misma curva de mínima varianza."
            )
def render_black_litterman() -> None:
    """
    Black-Litterman: prior de equilibrio + opiniones (views) del equipo,
    con su incertidumbre, dan un rendimiento esperado "corregido" (mu_BL)
    que se usa en las mismas carteras de Markowitz (escalamiento v2,
    guía sección 6, src.black_litterman).

    Requiere haber construido portafolios arriba primero (usa la misma
    Sigma y el mismo universo).
    """
    st.divider()
    st.subheader("Black-Litterman: opiniones del equipo (v2)")

    matrix = st.session_state.get("portfolio_matrix")
    comparison = st.session_state.get("portfolio_comparison")
    if matrix is None or comparison is None or not comparison.ok or matrix.empty:
        st.info("Primero construye los portafolios arriba (botón 'Construir portafolios').")
        return

    try:
        mu, sigma = sample_moments(matrix)
    except ValueError as exc:
        st.error(f"No se pudo estimar mu/Sigma: {exc}")
        return

    assets = list(sigma.index)
    rf_period = st.session_state["portfolio_rf_period"]
    periods_per_year = st.session_state["portfolio_periods_per_year"]
    bounds = [(0.0, 1.0)] * len(sigma)

    st.caption(
        "pi (el 'prior') se calcula desde w_ref con pi=delta·Sigma·w_ref (sección 6.1 de la "
        "guía). w_ref debe ser una referencia documentada por el equipo — aquí el valor por "
        "defecto es igual peso (1/N); si usan otra (ej. capitalización de mercado), justifíquenla."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        w_ref_mode = st.selectbox("Pesos de referencia (w_ref)", ["Igual peso (1/N)", "Personalizados"], key="bl_wref_mode")
    with col2:
        delta = st.number_input("Delta (aversión implícita de la referencia)", min_value=0.01, value=2.5, step=0.1, key="bl_delta")
    with col3:
        tau = st.number_input("Tau (escala de incertidumbre del prior)", min_value=0.001, value=0.05, step=0.01, format="%.3f", key="bl_tau")

    if w_ref_mode == "Personalizados":
        w_ref_raw = {}
        cols = st.columns(min(len(assets), 5))
        for i, a in enumerate(assets):
            with cols[i % len(cols)]:
                w_ref_raw[a] = st.number_input(f"w_ref {a}", min_value=0.0, value=1.0 / len(assets), key=f"bl_wref_{a}")
        total = sum(w_ref_raw.values()) or 1.0
        w_ref = pd.Series({a: v / total for a, v in w_ref_raw.items()})
    else:
        w_ref = pd.Series({a: 1.0 / len(assets) for a in assets})

    st.markdown("**Opiniones (views) del equipo**")
    st.session_state.setdefault("bl_views", [])

    with st.form("bl_add_view_form", clear_on_submit=True):
        tipo = st.radio("Tipo de view", ["Absoluta (un activo)", "Relativa (dos activos)"], horizontal=True)
        if tipo == "Absoluta (un activo)":
            activo = st.selectbox("Activo", assets, key="bl_abs_activo")
            q_anual = st.number_input("Rendimiento TOTAL anual esperado para este activo", value=0.10, format="%.4f", key="bl_abs_q")
            p_row = [1.0 if a == activo else 0.0 for a in assets]
        else:
            c1, c2 = st.columns(2)
            with c1:
                activo_mejor = st.selectbox("Activo que se espera que SUPERE", assets, key="bl_rel_mejor")
            with c2:
                opciones_peor = [a for a in assets if a != activo_mejor] or assets
                activo_peor = st.selectbox("Activo de referencia (al que supera)", opciones_peor, key="bl_rel_peor")
            q_anual = st.number_input(
                "Diferencia de rendimiento ANUAL esperada (positivo = el primero supera al segundo)",
                value=0.02, format="%.4f", key="bl_rel_q",
            )
            p_row = [1.0 if a == activo_mejor else (-1.0 if a == activo_peor else 0.0) for a in assets]

        omega_var = st.number_input(
            "Varianza de incertidumbre de la view (Omega) — más alto = menos confianza en el número",
            min_value=0.0, value=0.0001, format="%.6f", key="bl_omega",
        )
        confidence = st.slider("Confianza (0 excluye la view del cálculo)", 0.0, 1.0, 1.0, key="bl_confidence")
        thesis = st.text_area("Tesis / justificación (obligatoria)", key="bl_thesis")
        source = st.text_input("Fuente (opcional)", key="bl_source")
        student_author = st.text_input("Quién la propone", key="bl_author")
        sin_vencimiento = st.checkbox("Sin fecha de vencimiento", value=True, key="bl_sin_venc")
        expires_at = None if sin_vencimiento else st.date_input("Vence el", key="bl_expira").isoformat()

        agregar = st.form_submit_button("Agregar view")
        if agregar:
            if not thesis.strip():
                st.error("La tesis es obligatoria (sección 6.2 de la guía: toda view se justifica).")
            else:
                q_periodo = q_anual / periods_per_year
                q_excess = excess_from_total_view(q_periodo, rf_period, p_row)
                st.session_state["bl_views"].append({
                    "view_id": f"view-{len(st.session_state['bl_views']) + 1}",
                    "student_author": student_author or "Sin especificar",
                    "thesis": thesis.strip(),
                    "assets": assets,
                    "P_row": p_row,
                    "Q_excess": q_excess,
                    "Omega_variance": omega_var,
                    "confidence": confidence,
                    "source": source or None,
                    "expires_at": expires_at,
                })

    if st.session_state["bl_views"]:
        for i, v in enumerate(st.session_state["bl_views"]):
            tipo_str = "relativa" if sum(1 for x in v["P_row"] if x != 0) > 1 else "absoluta"
            cols_v = st.columns([5, 1])
            with cols_v[0]:
                st.write(
                    f"**{v['view_id']}** ({tipo_str}, confianza={v['confidence']:.2f}): {v['thesis']}"
                    + (f" — vence {v['expires_at']}" if v["expires_at"] else "")
                )
            with cols_v[1]:
                if st.button("Eliminar", key=f"bl_del_{i}"):
                    st.session_state["bl_views"].pop(i)
                    st.rerun()
    else:
        st.caption("Sin views todavía — sin views, Black-Litterman conserva el prior (pi) sin cambios.")

    if st.button("Calcular Black-Litterman"):
        view_objects = [View(**v) for v in st.session_state["bl_views"]]
        activas = active_views(view_objects)

        pi = implied_equilibrium_returns(sigma, w_ref, delta)
        if activas:
            P, Q, Omega = assemble_views(activas, assets=assets)
            bl_result = black_litterman_posterior(sigma, pi, tau, P, Q, Omega)
            diag = check_black_litterman_inputs(sigma, P, Omega)
        else:
            bl_result = black_litterman_posterior(sigma, pi, tau, None, None, None)
            diag = check_black_litterman_inputs(sigma)

        st.session_state["bl_pi"] = pi
        st.session_state["bl_result"] = bl_result
        st.session_state["bl_diag"] = diag
        st.session_state["bl_n_activas"] = len(activas)
        st.session_state["bl_n_excluidas"] = len(view_objects) - len(activas)

    bl_result = st.session_state.get("bl_result")
    if bl_result is None:
        return

    if st.session_state.get("bl_n_excluidas", 0) > 0:
        st.caption(
            f"{st.session_state['bl_n_excluidas']} view(s) excluidas del cálculo (confianza 0 o vencidas) "
            f"de {st.session_state['bl_n_excluidas'] + st.session_state['bl_n_activas']} totales."
        )

    with st.expander("Diagnóstico (dimensiones, simetría, PSD)"):
        st.write(st.session_state["bl_diag"])

    comparacion = pd.DataFrame({
        "histórico": mu * periods_per_year,
        "prior (pi)": (bl_result.pi + rf_period) * periods_per_year,
        "posterior (mu_BL)": (bl_result.mu_posterior_excess + rf_period) * periods_per_year,
    })
    st.markdown("**Rendimiento esperado anualizado: histórico vs. prior vs. posterior**")
    st.dataframe(comparacion.style.format("{:.2%}"))

    if bl_result.n_views > 0:
        mu_bl_periodo = bl_result.mu_posterior_excess + rf_period
        gamma_value = st.session_state["portfolio_gamma_value"]
        w_gmv = gmv_weights(sigma, bounds=bounds)
        w_tan_bl = tangency_weights(mu_bl_periodo, sigma, rf_period, bounds=bounds)
        w_per_bl = personal_optimal_weights(mu_bl_periodo, sigma, gamma_value, bounds=bounds)
        w_max_bl = max_expected_return_weights(mu_bl_periodo, sigma, bounds=bounds)

        st.markdown("**Carteras recalculadas con mu_BL (misma Sigma, sección 6.1)**")
        st.session_state["bl_portfolio_weights"] = {
            "Mínima varianza (GMV)": w_gmv,
            "Tangente (BL)": w_tan_bl,
            "Óptimo personal (BL)": w_per_bl,
            "Máximo rendimiento (BL)": w_max_bl,
        }
        st.dataframe(pd.DataFrame(st.session_state["bl_portfolio_weights"]).style.format("{:.2%}"))
def render_rebalance() -> None:
    """
    Rebalanceo con nueva información (escalamiento v2, guía sección 7,
    src.rebalance). Simula invertir un capital inicial en una cartera
    objetivo (al inicio de la ventana común) y ver si, después de que los
    precios se movieron hasta el final de esa ventana, conviene rebalancear
    o abstenerse — con costos cobrados sobre el nominal real negociado.

    Requiere haber construido portafolios arriba primero (usa la misma
    ventana común de precios y, si existen, las carteras ya calculadas).
    """
    st.divider()
    st.subheader("Rebalanceo con nueva información (v2)")

    matrix = st.session_state.get("portfolio_matrix")
    comparison = st.session_state.get("portfolio_comparison")
    if matrix is None or comparison is None or not comparison.ok or not comparison.common_prices:
        st.info("Primero construye los portafolios arriba (botón 'Construir portafolios').")
        return

    try:
        mu, sigma = sample_moments(matrix)
    except ValueError as exc:
        st.error(f"No se pudo estimar mu/Sigma: {exc}")
        return

    assets = list(sigma.index)
    periods_per_year = st.session_state["portfolio_periods_per_year"]
    rf_period = st.session_state["portfolio_rf_period"]
    gamma_value = st.session_state["portfolio_gamma_value"]
    bounds = [(0.0, 1.0)] * len(sigma)

    opciones_cartera = ["Mínima varianza (GMV)", "Tangente", "Óptimo personal", "Máximo rendimiento esperado"]
    bl_weights = st.session_state.get("bl_portfolio_weights")
    if bl_weights:
        opciones_cartera += [f"{nombre} (desde Black-Litterman)" for nombre in bl_weights.keys()]
    opciones_cartera.append("Pesos personalizados")

    cartera_elegida = st.selectbox("Cartera objetivo para el rebalanceo", opciones_cartera, key="rb_cartera_elegida")

    if cartera_elegida == "Pesos personalizados":
        pesos_raw = {}
        cols = st.columns(min(len(assets), 5))
        for i, a in enumerate(assets):
            with cols[i % len(cols)]:
                pesos_raw[a] = st.number_input(f"peso {a}", min_value=0.0, value=1.0 / len(assets), key=f"rb_peso_{a}")
        total = sum(pesos_raw.values()) or 1.0
        pesos_objetivo = pd.Series({a: v / total for a, v in pesos_raw.items()})
        st.caption("Los pesos se normalizan automáticamente para sumar 1 (largo-solo, sin apalancamiento).")
    elif cartera_elegida.endswith("(desde Black-Litterman)"):
        nombre_bl = cartera_elegida.replace(" (desde Black-Litterman)", "")
        pesos_objetivo = bl_weights[nombre_bl]
    else:
        if cartera_elegida == "Mínima varianza (GMV)":
            pesos_objetivo = gmv_weights(sigma, bounds=bounds)
        elif cartera_elegida == "Tangente":
            pesos_objetivo = tangency_weights(mu, sigma, rf_period, bounds=bounds)
        elif cartera_elegida == "Óptimo personal":
            pesos_objetivo = personal_optimal_weights(mu, sigma, gamma_value, bounds=bounds)
        else:
            pesos_objetivo = max_expected_return_weights(mu, sigma, bounds=bounds)

    col1, col2, col3 = st.columns(3)
    with col1:
        capital_inicial = st.number_input("Capital inicial a invertir", min_value=0.0, value=100000.0, step=1000.0, key="rb_capital")
    with col2:
        costo_por_nominal = st.number_input(
            "Costo por nominal operado (fracción, ej. 0.001 = 0.1%)", min_value=0.0, value=0.001, format="%.4f", key="rb_costo"
        )
    with col3:
        banda = st.number_input(
            "Banda de no operación (desviación máxima tolerada por activo)", min_value=0.0, value=0.03, format="%.3f", key="rb_banda"
        )

    if st.button("Simular rebalanceo"):
        common_prices = comparison.common_prices
        precios_iniciales = pd.Series({t: s.iloc[0] for t, s in common_prices.items()}).reindex(assets)
        precios_nuevos = pd.Series({t: s.iloc[-1] for t, s in common_prices.items()}).reindex(assets)

        cantidades_iniciales = capital_inicial * pesos_objetivo.reindex(assets) / precios_iniciales
        simple_returns = precios_nuevos / precios_iniciales - 1
        w_pre, r_p = drifted_weights(pesos_objetivo.reindex(assets), simple_returns)
        disparado, desviacion = needs_rebalance(w_pre, pesos_objetivo, banda)

        st.session_state["rb_result"] = {
            "precios_iniciales": precios_iniciales,
            "precios_nuevos": precios_nuevos,
            "cantidades_iniciales": cantidades_iniciales,
            "w_pre": w_pre,
            "r_p": r_p,
            "disparado": disparado,
            "desviacion": desviacion,
            "pesos_objetivo": pesos_objetivo.reindex(assets),
            "costo_por_nominal": costo_por_nominal,
        }
        if disparado:
            rb = simulate_rebalance_to_target(
                cantidades_iniciales, precios_nuevos, 0.0, pesos_objetivo.reindex(assets), costo_por_nominal
            )
            st.session_state["rb_simulacion"] = rb
        else:
            st.session_state["rb_simulacion"] = None

    rb_result = st.session_state.get("rb_result")
    if rb_result is None:
        return

    st.caption(
        f"Ventana común: {comparison.common_start.date()} (inversión inicial) a "
        f"{comparison.common_end.date()} (hoy). Retorno de cartera en ese periodo: {rb_result['r_p']:.2%}."
    )

    st.markdown("**Pesos por deriva de precios vs. objetivo**")
    tabla_pesos = pd.DataFrame({
        "objetivo": rb_result["pesos_objetivo"],
        "antes de rebalancear (deriva)": rb_result["w_pre"],
        "desviación": rb_result["desviacion"],
    })
    st.dataframe(tabla_pesos.style.format("{:.2%}"))

    if not rb_result["disparado"]:
        st.success(
            "Ningún activo superó la banda de no operación — la política indica ABSTENERSE de "
            "rebalancear (no se justifica pagar comisión por una desviación tan chica)."
        )
        return

    st.warning("Al menos un activo superó la banda — corresponde rebalancear.")
    rb = st.session_state.get("rb_simulacion")
    if rb is None:
        return

    if not rb.feasible:
        st.error(f"No se pudo simular el rebalanceo: {rb.reason}")
        return

    turnover = turnover_unilateral(rb_result["w_pre"], rb_result["pesos_objetivo"])
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Costo total", f"${rb.cost:,.2f}")
    col_b.metric("Nominal bruto operado", f"${rb.gross_notional_traded:,.2f}")
    col_c.metric("Turnover unilateral", f"{turnover:.2%}")
    col_d.metric("Caja final", f"${rb.cash_post:,.2f}")

    st.markdown("**Operaciones (compra positiva, venta negativa)**")
    tabla_trades = pd.DataFrame({
        "nominal operado": rb.trades,
        "cantidades nuevas": rb.new_quantities,
    })
    st.dataframe(tabla_trades.style.format({"nominal operado": "${:,.2f}", "cantidades nuevas": "{:.4f}"}))
    st.caption(f"Patrimonio antes de costos: ${rb.value_pre:,.2f} · después de costos: ${rb.value_post:,.2f}")
def main() -> None:
    render_header()

    if DEV_MODE:
        st.sidebar.warning("MODO DESARROLLO: login OIDC desactivado temporalmente.")
        user_name = st.sidebar.text_input("Nombre (simulado)", value="Estudiante Demo")
        user_email = st.sidebar.text_input("Correo (simulado)", value="demo@eia.edu.co")
    else:
        if not is_auth_configured():
            st.error(
                "Autenticación OIDC no configurada todavía.\n\n"
                "1. Copien `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml`.\n"
                "2. Completen las credenciales de Google/Microsoft que entregue el curso "
                "por el LMS.\n\n"
                "La app se detiene aquí a propósito: no debe quedar disponible sin login real."
            )
            st.stop()

        user = require_login()  # detiene la ejecución si no hay sesión activa
        user_name, user_email = getattr(user, "name", "?"), getattr(user, "email", "?")

    render_disclaimer_gate(user_name, user_email)

    st.success(f"Sesión iniciada como {user_name} ({user_email})")

    st.divider()
    st.subheader("Datos: probar un activo (RF-05 a RF-07)")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_input = st.text_input("Ticker", value="AAPL")
    with col2:
        start_date = st.date_input("Fecha inicial", value=pd.Timestamp("2023-01-01"))
    with col3:
        end_date = st.date_input("Fecha final", value=pd.Timestamp.today())

    frequency = st.selectbox("Frecuencia", list(FREQUENCY_RULES.keys()))

    if st.button("Descargar y limpiar"):
        with st.spinner("Descargando..."):
            result = fetch_asset_data(ticker_input, start_date, end_date)
            if result.ok:
                result.prices = resample_prices(result.prices, frequency)

        if result.ok:
            st.session_state["asset_result"] = result
            st.session_state["asset_frequency"] = frequency
            st.success(
                f"{result.ticker}: {len(result.prices)} observaciones · "
                f"fuente: {result.source} · moneda: {result.currency or 'N/D'} · "
                f"zona horaria: {result.timezone or 'N/D'} · "
                f"último dato: {result.prices.index.max().date()}"
            )
            if result.dropped_rows:
                st.warning(
                    f"Se descartaron {result.dropped_rows} filas inválidas "
                    "(faltantes o precios no positivos)."
                )
            st.dataframe(result.prices.tail(20))
        else:
            st.session_state.pop("asset_result", None)
            st.error(f"{result.ticker}: {result.error}")

    render_historical_analysis()
    render_forecast()
    render_comparison()
    render_portfolios()
    render_black_litterman()
    render_rebalance()

    if not DEV_MODE and st.button("Cerrar sesión"):
        st.logout()


if __name__ == "__main__":
    main()
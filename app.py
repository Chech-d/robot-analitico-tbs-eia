"""
Punto de entrada de la aplicación.
Robot analítico para la preselección de activos - Teoría Moderna de
Portafolios, Tech Business School - Universidad EIA.

EDITEN los valores de identidad de más abajo antes de la primera entrega.
"""
import pandas as pd
import streamlit as st

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
from src.charts import price_chart, return_chart
from src.data import FREQUENCY_RULES, fetch_asset_data, resample_prices

# ---------------------------------------------------------------------------
# MODO DESARROLLO: mientras no tengan credenciales OIDC reales, dejen esto en
# True para poder probar el resto de la app (datos, análisis, forecasting,
# riesgo, comparación) sin quedar bloqueados por el login. Antes de la
# entrega final, si el profesor confirma que el login es obligatorio, pasen
# esto a False y usen las credenciales reales.
# ---------------------------------------------------------------------------
DEV_MODE = True

# ---------------------------------------------------------------------------
# IDENTIDAD DEL EQUIPO (RF-01) — EDITAR ANTES DE ENTREGAR
# ---------------------------------------------------------------------------
SYSTEM_NAME = "EDITAR: Nombre del sistema"
TEAM_NAME = "EDITAR: Nombre del equipo"
MEMBERS = [
    "EDITAR: Integrante 1",
    "EDITAR: Integrante 2",
    "EDITAR: Integrante 3",
]
VERSION = "0.1.0"
LAST_UPDATE = "2026-09-12"

EXECUTION_POLICY_SUMMARY = (
    "Piloto académico restringido. Posición larga únicamente. Rendimiento "
    "logarítmico exclusivo. El sistema puede emitir 'no señal'. No calcula "
    "portafolios, pesos, covarianzas ni frontera eficiente. Ver la vista "
    "Metodología para la política de ejecución completa."
)

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


def render_disclaimer_gate() -> None:
    """
    Bloquea el análisis hasta que el usuario acepte, por separado, la
    autorización de privacidad y el disclaimer académico (sección 3.3).
    Detiene la ejecución (st.stop) si falta alguna de las dos.
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
        st.stop()

    # TODO (Fase 9): registrar ambos consentimientos como eventos separados
    # y persistirlos junto con perfil/sesión/evento/outbox en una transacción.


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

    with tab_returns:
        st.plotly_chart(return_chart(returns, result.ticker, frequency), use_container_width=True)
        st.caption(f"Último log-rendimiento: {returns.iloc[-1]:.6f} ({returns.index[-1].date()})")

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
                    st.write(f"**Hipótesis nula:** {diag.hypothesis}")
                    st.write(f"**Parámetros:** {diag.parameters}")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Estadístico", f"{diag.statistic:.4f}")
                    col_b.metric("p-valor", f"{diag.p_value:.4f}")
                    if diag.reject:
                        st.warning(diag.interpretation)
                    else:
                        st.success(diag.interpretation)


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

    render_disclaimer_gate()

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

    st.write(
        "Aquí seguirá el resto de la aplicación: forecasting, riesgo y "
        "comparación de activos (ver roadmap)."
    )

    if not DEV_MODE and st.button("Cerrar sesión"):
        st.logout()


if __name__ == "__main__":
    main()
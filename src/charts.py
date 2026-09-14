"""
Visualizaciones (RF-09, RF-14) y mapa histórico rendimiento-riesgo
(RF-19 a RF-21).

Este archivo cubre los gráficos de precio y rendimiento del análisis de un
solo activo (Fase 4), la trayectoria de pronóstico (Fase 6) y el mapa
histórico rendimiento-riesgo de la comparación de N activos (Fase 8).
"""
import pandas as pd
import plotly.graph_objects as go


def price_chart(prices, ticker: str, currency: str, frequency: str, source: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines", name=ticker))
    fig.update_layout(
        title=f"Precio ajustado - {ticker} ({frequency}, fuente: {source})",
        xaxis_title="Fecha",
        yaxis_title=f"Precio ajustado ({currency or 'N/D'})",
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def return_chart(returns, ticker: str, frequency: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns.values, mode="lines", name=ticker))
    fig.update_layout(
        title=f"Rendimiento logarítmico - {ticker} ({frequency})",
        xaxis_title="Fecha",
        yaxis_title="g_t (log-rendimiento)",
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def _future_dates(last_date, frequency: str, horizon: int) -> pd.DatetimeIndex:
    """Fechas aproximadas de los próximos `horizon` periodos, solo para el eje del gráfico."""
    if frequency == "Semanal":
        return pd.date_range(start=last_date, periods=horizon + 1, freq="W")[1:]
    if frequency == "Mensual":
        return pd.date_range(start=last_date, periods=horizon + 1, freq="ME")[1:]
    return pd.bdate_range(start=last_date, periods=horizon + 1)[1:]  # Diaria (y default)


def forecast_chart(prices, steps, ticker: str, frequency: str, history_window: int = 60) -> go.Figure:
    """
    Trayectoria de pronóstico (RF-14): cola del histórico + mediana pronosticada
    + banda del intervalo 5%-95% para cada paso de 1 a H.
    """
    fig = go.Figure()
    hist = prices.tail(history_window)
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Histórico"))

    future_dates = _future_dates(hist.index[-1], frequency, len(steps))
    medians = [s.median_price for s in steps]
    q05s = [s.q05_price for s in steps]
    q95s = [s.q95_price for s in steps]

    fig.add_trace(
        go.Scatter(x=future_dates, y=q95s, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=q05s, mode="lines", line=dict(width=0), fill="tonexty",
            name="Intervalo 5%-95% (pronóstico)", fillcolor="rgba(99, 110, 250, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatter(x=future_dates, y=medians, mode="lines", name="Mediana pronosticada", line=dict(dash="dash"))
    )
    fig.update_layout(
        title=f"Pronóstico - {ticker} ({frequency})",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def risk_return_map(summaries, non_dominated) -> go.Figure:
    """
    Mapa histórico rendimiento-riesgo (RF-19/RF-20, sección 5.8): volatilidad
    anualizada en X, media histórica logarítmica anualizada en Y. Las
    coordenadas son históricas (no dependen de H) y se etiqueta cada punto.
    Los activos no dominados se resaltan; esto NO es una frontera eficiente
    de portafolios ni una recomendación de la mejor inversión universal.
    """
    dominated = [s for s in summaries if s.ticker not in non_dominated]
    frontier = [s for s in summaries if s.ticker in non_dominated]

    fig = go.Figure()
    if dominated:
        fig.add_trace(
            go.Scatter(
                x=[s.vol_annual for s in dominated],
                y=[s.mean_annual for s in dominated],
                mode="markers+text",
                text=[s.ticker for s in dominated],
                textposition="top center",
                name="Dominado",
                marker=dict(size=10, color="rgba(99, 110, 250, 0.6)"),
            )
        )
    if frontier:
        fig.add_trace(
            go.Scatter(
                x=[s.vol_annual for s in frontier],
                y=[s.mean_annual for s in frontier],
                mode="markers+text",
                text=[s.ticker for s in frontier],
                textposition="top center",
                name="No dominado",
                marker=dict(size=13, color="rgba(239, 85, 59, 0.9)", symbol="diamond"),
            )
        )
    fig.update_layout(
        title="Mapa histórico rendimiento-riesgo (no es frontera eficiente de portafolios)",
        xaxis_title="Volatilidad histórica anualizada (σ)",
        yaxis_title="Media histórica logarítmica anualizada (ḡ)",
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig
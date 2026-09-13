"""
Visualizaciones (RF-09) y mapa histórico rendimiento-riesgo (RF-19 a RF-21).

El mapa histórico obligatorio se construye en la Fase 8 del roadmap. Este
archivo por ahora cubre los gráficos de precio y rendimiento de un activo.
"""
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
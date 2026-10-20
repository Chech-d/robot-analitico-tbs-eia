"""
Escalamiento v2 - M6: Rebalanceo con informacion nueva (seccion 7 de la guia).

- P2-25: deriva de pesos y cantidades previas correctamente calculadas.
- P2-26: turnover, nominal negociado, costos y caja reconciliados.
- P2-27: evento duplicado no genera una segunda decision.
- P2-28: abstencion por retraso, costo o liquidez queda registrada.

El caso de control de ``rebalanceo_control.json`` (A/B, cantidades
[60,40] a precio 100, precios nuevos [110,95], objetivo 50/50,
costo_por_nominal=0.001) se verifica contra un calculo a mano documentado
en AI_USAGE.md: venta de A por 1401.4, compra de B por 1398.6, costo=2.8,
caja final=0, patrimonio final=10397.2.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rebalance import (
    DataQualityCheck,
    RebalanceResult,
    check_data_freshness,
    drifted_weights,
    is_duplicate_event,
    needs_rebalance,
    simulate_rebalance_to_target,
    turnover_unilateral,
)
from tests.fixtures_path import REBALANCEO_CONTROL

pytestmark = pytest.mark.skipif(not REBALANCEO_CONTROL.exists(), reason="fixture v2 no copiado aun")


def _control():
    with open(REBALANCEO_CONTROL, "r", encoding="utf-8") as f:
        data = json.load(f)
    assets = data["assets"]
    quantities_pre = pd.Series(data["cantidades"], index=assets, dtype=float)
    precios_iniciales = pd.Series(data["precios_iniciales"], index=assets, dtype=float)
    precios_nuevos = pd.Series(data["precios_nuevos"], index=assets, dtype=float)
    target_weights = pd.Series(data["pesos_objetivo"], index=assets, dtype=float)
    return data, assets, quantities_pre, precios_iniciales, precios_nuevos, target_weights


def test_drifted_weights_matches_control_case_p2_25():
    _, assets, quantities_pre, precios_iniciales, precios_nuevos, _ = _control()

    w_prev_post = (quantities_pre * precios_iniciales) / (quantities_pre * precios_iniciales).sum()
    simple_returns = precios_nuevos / precios_iniciales - 1

    w_pre, r_p = drifted_weights(w_prev_post, simple_returns)

    # Calculo a mano: V_prev=10000, V_pre(precios nuevos)=10400 -> R_p=0.04
    assert r_p == pytest.approx(0.04, abs=1e-10)
    # w_A_pre = 6600/10400, w_B_pre = 3800/10400
    assert w_pre["A"] == pytest.approx(6600 / 10400, abs=1e-10)
    assert w_pre["B"] == pytest.approx(3800 / 10400, abs=1e-10)
    assert w_pre.sum() == pytest.approx(1.0, abs=1e-10)


def test_turnover_unilateral_matches_control_case():
    _, assets, quantities_pre, precios_iniciales, precios_nuevos, target_weights = _control()
    w_prev_post = (quantities_pre * precios_iniciales) / (quantities_pre * precios_iniciales).sum()
    simple_returns = precios_nuevos / precios_iniciales - 1
    w_pre, _ = drifted_weights(w_prev_post, simple_returns)

    turnover = turnover_unilateral(w_pre, target_weights)

    # 0.5*(|0.5-0.634615|+|0.5-0.365385|) = 0.134615...
    assert turnover == pytest.approx(0.1346153846, abs=1e-8)


def test_simulate_rebalance_matches_control_case_p2_26():
    data, assets, quantities_pre, precios_iniciales, precios_nuevos, target_weights = _control()

    result = simulate_rebalance_to_target(
        quantities_pre=quantities_pre,
        prices_new=precios_nuevos,
        cash_pre=data["caja"],
        target_weights=target_weights,
        cost_rate=data["costo_por_nominal"],
    )

    assert result.feasible is True
    assert result.value_pre == pytest.approx(10400.0, abs=1e-8)
    assert result.cost == pytest.approx(2.8, abs=1e-6)
    assert result.gross_notional_traded == pytest.approx(2800.0, abs=1e-6)
    assert result.value_post == pytest.approx(10397.2, abs=1e-6)

    # Venta de A, compra de B (cálculo a mano en AI_USAGE.md).
    assert result.trades["A"] == pytest.approx(-1401.4, abs=1e-4)
    assert result.trades["B"] == pytest.approx(1398.6, abs=1e-4)

    assert result.new_quantities["A"] == pytest.approx(47.26, abs=1e-4)
    assert result.new_quantities["B"] == pytest.approx(54.72210526, abs=1e-4)

    # Caja no debe quedar negativa; en este caso de control queda en 0.
    assert result.cash_post == pytest.approx(0.0, abs=1e-6)

    # El costo se cobro sobre el NOMINAL REAL (venta+compra), no sobre la mitad:
    # cost == cost_rate * (|venta|+|compra|), nunca cost_rate*0.5*(|venta|+|compra|).
    assert result.cost == pytest.approx(data["costo_por_nominal"] * result.gross_notional_traded, abs=1e-9)


def test_simulate_rebalance_sin_caja_suficiente_es_no_factible():
    # Pide invertir mas del 100% del patrimonio: no factible sin apalancamiento.
    quantities_pre = pd.Series({"A": 10.0, "B": 10.0})
    prices_new = pd.Series({"A": 10.0, "B": 10.0})
    target_weights = pd.Series({"A": 0.7, "B": 0.7})  # suma 1.4 > 1

    result = simulate_rebalance_to_target(
        quantities_pre=quantities_pre, prices_new=prices_new, cash_pre=0.0,
        target_weights=target_weights, cost_rate=0.001,
    )

    assert result.feasible is False
    assert "apalancamiento" in result.reason.lower()


def test_needs_rebalance_respeta_banda_p2_28():
    w_pre = pd.Series({"A": 0.55, "B": 0.45})
    w_target = pd.Series({"A": 0.5, "B": 0.5})

    # Banda ancha: la desviacion (0.05) no la supera -> no rebalancear (abstencion).
    disparado_ancha, desviacion = needs_rebalance(w_pre, w_target, band=0.10)
    assert disparado_ancha is False
    assert desviacion["A"] == pytest.approx(0.05, abs=1e-10)

    # Banda angosta: si la supera -> rebalancear.
    disparado_angosta, _ = needs_rebalance(w_pre, w_target, band=0.02)
    assert disparado_angosta is True


def test_evento_duplicado_no_genera_segunda_decision_p2_27():
    vistos = {"evento-2026-10-20T09:00:00Z"}

    assert is_duplicate_event("evento-2026-10-20T09:00:00Z", vistos) is True
    assert is_duplicate_event("evento-2026-10-20T09:05:00Z", vistos) is False


def test_abstencion_por_dato_obsoleto_p2_28():
    # Un activo con cotizacion de 400s de antigüedad, limite de 60s -> no ok.
    check = check_data_freshness(
        quote_age_seconds={"A": 10.0, "B": 400.0}, max_quote_age_seconds=60.0, max_asset_time_skew_seconds=120.0
    )
    assert check.ok is False
    assert check.stale_tickers == ["B"]


def test_frescura_de_datos_ok_cuando_todo_dentro_de_limites():
    check = check_data_freshness(
        quote_age_seconds={"A": 5.0, "B": 8.0}, max_quote_age_seconds=60.0, max_asset_time_skew_seconds=30.0
    )
    assert check.ok is True
    assert check.stale_tickers == []


def test_frescura_de_datos_detecta_desincronizacion_entre_activos():
    # Ninguno individualmente obsoleto, pero muy desincronizados entre si.
    check = check_data_freshness(
        quote_age_seconds={"A": 1.0, "B": 55.0}, max_quote_age_seconds=60.0, max_asset_time_skew_seconds=10.0
    )
    assert check.ok is False
    assert check.stale_tickers == []  # ninguno supera el limite individual
    assert check.max_skew_seconds == pytest.approx(54.0, abs=1e-10)
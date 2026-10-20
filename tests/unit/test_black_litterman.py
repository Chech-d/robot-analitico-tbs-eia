"""
Escalamiento v2 - M5: Black-Litterman (seccion 6 de la guia).

- P2-19: BL sin views conserva el prior (mu_posterior == pi, M == B).
- P2-20: Q=P*pi no mueve la media; P/Q mal dimensionados lanzan error.
- P2-21: Confianza cero EXCLUYE la opinion (no implica Omega cero).
- P2-22: Views absoluta/relativa, vencimiento y duplicado.
- P2-23: Incertidumbre de la media (M) separada del riesgo predictivo
  (Sigma != Sigma_predictiva salvo que M sea cero).

El caso de control de ``black_litterman_control.json`` (A/B, delta=3,
tau=0.05) se verifica con valores calculados a mano (ver AI_USAGE.md) y se
usa como referencia numerica fija de P2-19 a P2-23.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.black_litterman import (
    BlackLittermanResult,
    View,
    active_views,
    assemble_views,
    black_litterman_posterior,
    check_black_litterman_inputs,
    excess_from_total_view,
    find_duplicate_views,
    implied_equilibrium_returns,
)
from tests.fixtures_path import BLACK_LITTERMAN_CONTROL

pytestmark = pytest.mark.skipif(not BLACK_LITTERMAN_CONTROL.exists(), reason="fixture v2 no copiado aun")


def _control():
    with open(BLACK_LITTERMAN_CONTROL, "r", encoding="utf-8") as f:
        data = json.load(f)
    assets = data["assets"]
    sigma = pd.DataFrame(data["Sigma"], index=assets, columns=assets)
    w_ref = pd.Series(data["w_ref"], index=assets)
    P = pd.DataFrame(data["P"], index=["v1", "v2"], columns=assets)
    Q = pd.Series(data["Q"], index=["v1", "v2"])
    Omega = pd.DataFrame(data["Omega"], index=["v1", "v2"], columns=["v1", "v2"])
    return data, assets, sigma, w_ref, P, Q, Omega


def test_implied_equilibrium_returns_matches_control_pi():
    _, assets, sigma, w_ref, *_ = _control()
    pi = implied_equilibrium_returns(sigma, w_ref, delta=3)
    # pi = 3 * Sigma @ w_ref = [0.0228, 0.0552] (calculo a mano, ver AI_USAGE.md)
    assert pi["A"] == pytest.approx(0.0228, abs=1e-10)
    assert pi["B"] == pytest.approx(0.0552, abs=1e-10)


def test_posterior_matches_control_case_with_views():
    data, assets, sigma, w_ref, P, Q, Omega = _control()
    pi = implied_equilibrium_returns(sigma, w_ref, delta=data["delta"])

    result = black_litterman_posterior(sigma, pi, tau=data["tau"], P=P, Q=Q, Omega=Omega)

    # Calculo a mano documentado en AI_USAGE.md: mu_BL_exceso = [0.0285, 0.0450]
    assert result.mu_posterior_excess["A"] == pytest.approx(0.0285, abs=1e-6)
    assert result.mu_posterior_excess["B"] == pytest.approx(0.0450, abs=1e-6)
    assert result.n_views == 2

    # M es la incertidumbre de la MEDIA: simetrica, PSD y mas chica que B en la diagonal.
    assert result.M.loc["A", "B"] == pytest.approx(result.M.loc["B", "A"], abs=1e-12)
    assert result.sigma_predictiva.loc["A", "A"] > sigma.loc["A", "A"]


def test_bl_sin_views_conserva_el_prior_p2_19():
    _, assets, sigma, w_ref, *_ = _control()
    pi = implied_equilibrium_returns(sigma, w_ref, delta=3)

    sin_views = black_litterman_posterior(sigma, pi, tau=0.05, P=None, Q=None, Omega=None)

    pd.testing.assert_series_equal(sin_views.mu_posterior_excess, pi, check_names=False)
    pd.testing.assert_frame_equal(sin_views.M, 0.05 * sigma)
    assert sin_views.n_views == 0
    assert sin_views.A is None


def test_q_igual_a_p_pi_no_mueve_la_media_p2_20():
    _, assets, sigma, w_ref, P, _, Omega = _control()
    pi = implied_equilibrium_returns(sigma, w_ref, delta=3)

    # Q = P @ pi (ninguna opinion contradice al prior): la media posterior
    # debe quedar igual a pi, sin importar Omega.
    q_igual_a_prior = pd.Series(P.values @ pi.values, index=P.index)

    result = black_litterman_posterior(sigma, pi, tau=0.05, P=P, Q=q_igual_a_prior, Omega=Omega)

    assert result.mu_posterior_excess["A"] == pytest.approx(pi["A"], abs=1e-10)
    assert result.mu_posterior_excess["B"] == pytest.approx(pi["B"], abs=1e-10)


def test_dimensiones_mal_formadas_lanzan_error_p2_20():
    _, assets, sigma, w_ref, P, Q, Omega = _control()
    pi = implied_equilibrium_returns(sigma, w_ref, delta=3)

    Q_mal_dimensionado = Q.iloc[:1]  # 1 valor para 2 filas de P
    with pytest.raises(ValueError):
        black_litterman_posterior(sigma, pi, tau=0.05, P=P, Q=Q_mal_dimensionado, Omega=Omega)

    Omega_mal_dimensionada = Omega.iloc[:1, :1]
    with pytest.raises(ValueError):
        black_litterman_posterior(sigma, pi, tau=0.05, P=P, Q=Q, Omega=Omega_mal_dimensionada)


def test_excess_from_total_view_absoluta_y_relativa():
    # Absoluta: P_row=[1,0] -> se resta rf una vez.
    assert excess_from_total_view(0.06, rf=0.03, p_row=[1, 0]) == pytest.approx(0.03)
    # Relativa: P_row=[1,-1] -> suma(P_row)=0, no se resta rf.
    assert excess_from_total_view(-0.01, rf=0.03, p_row=[1, -1]) == pytest.approx(-0.01)


def test_confianza_cero_excluye_la_opinion_no_pone_omega_cero_p2_21():
    v_valida = View(
        view_id="v1", student_author="S", thesis="A supera al prior", assets=["A", "B"],
        P_row=[1, 0], Q_excess=0.03, Omega_variance=0.0002, confidence=1.0,
    )
    v_sin_confianza = View(
        view_id="v2", student_author="S", thesis="duda sin evidencia", assets=["A", "B"],
        P_row=[1, -1], Q_excess=-0.01, Omega_variance=0.001, confidence=0.0,
    )

    activas = active_views([v_valida, v_sin_confianza])

    assert [v.view_id for v in activas] == ["v1"]  # v2 se EXCLUYE, no queda con Omega=0


def test_views_vencidas_se_retiran_p2_22():
    ahora = datetime(2026, 10, 20, tzinfo=timezone.utc)
    vigente = View(
        view_id="vigente", student_author="S", thesis="t", assets=["A", "B"], P_row=[1, 0],
        Q_excess=0.03, Omega_variance=0.0002, expires_at="2026-12-31",
    )
    vencida = View(
        view_id="vencida", student_author="S", thesis="t", assets=["A", "B"], P_row=[1, 0],
        Q_excess=0.02, Omega_variance=0.0002, expires_at="2026-09-01",
    )

    activas = active_views([vigente, vencida], as_of=ahora)

    assert [v.view_id for v in activas] == ["vigente"]


def test_views_duplicadas_se_detectan_p2_22():
    v1 = View(view_id="v1", student_author="S", thesis="t1", assets=["A", "B"], P_row=[1, 0],
              Q_excess=0.03, Omega_variance=0.0002)
    v2 = View(view_id="v2", student_author="S", thesis="misma noticia repetida", assets=["A", "B"],
              P_row=[1, 0], Q_excess=0.03, Omega_variance=0.0005)
    v3 = View(view_id="v3", student_author="S", thesis="distinta", assets=["A", "B"], P_row=[1, -1],
              Q_excess=-0.01, Omega_variance=0.001)

    duplicadas = find_duplicate_views([v1, v2, v3])

    assert len(duplicadas) == 1
    assert {duplicadas[0][0].view_id, duplicadas[0][1].view_id} == {"v1", "v2"}


def test_view_absoluta_vs_relativa():
    absoluta = View(view_id="v1", student_author="S", thesis="t", assets=["A", "B"], P_row=[1, 0],
                     Q_excess=0.03, Omega_variance=0.0002)
    relativa = View(view_id="v2", student_author="S", thesis="t", assets=["A", "B"], P_row=[1, -1],
                     Q_excess=-0.01, Omega_variance=0.001)

    assert absoluta.is_relative() is False
    assert relativa.is_relative() is True


def test_assemble_views_reindexa_por_nombre_de_activo():
    v = View(view_id="v1", student_author="S", thesis="t", assets=["B", "A"], P_row=[-1, 1],
             Q_excess=-0.01, Omega_variance=0.001)

    P, Q, Omega = assemble_views([v], assets=["A", "B"])

    assert list(P.columns) == ["A", "B"]
    assert P.loc["v1", "A"] == 1
    assert P.loc["v1", "B"] == -1
    assert Q["v1"] == pytest.approx(-0.01)
    assert Omega.loc["v1", "v1"] == pytest.approx(0.001)


def test_assemble_views_vacio_da_matrices_vacias():
    P, Q, Omega = assemble_views([], assets=["A", "B"])
    assert P.empty and len(P.columns) == 2
    assert Q.empty
    assert Omega.empty


def test_assemble_views_activo_fuera_del_universo_lanza_error():
    v = View(view_id="v1", student_author="S", thesis="t", assets=["A", "C"], P_row=[1, -1],
             Q_excess=0.01, Omega_variance=0.001)
    with pytest.raises(ValueError):
        assemble_views([v], assets=["A", "B"])


def test_m_es_incertidumbre_de_media_separada_del_riesgo_p2_23():
    _, assets, sigma, w_ref, P, Q, Omega = _control()
    pi = implied_equilibrium_returns(sigma, w_ref, delta=3)

    result = black_litterman_posterior(sigma, pi, tau=0.05, P=P, Q=Q, Omega=Omega)

    # Sigma (riesgo) no cambia por si sola; sigma_predictiva es una variante
    # etiquetada aparte (seccion 6.1), nunca reemplaza a Sigma en silencio.
    pd.testing.assert_frame_equal(sigma, pd.DataFrame(_control()[0]["Sigma"], index=assets, columns=assets))
    assert not result.sigma_predictiva.equals(sigma)
    # M es chica frente a Sigma (mas confianza en las views -> M mas chica, nunca negativa en la diagonal).
    assert result.M.loc["A", "A"] >= 0
    assert result.M.loc["B", "B"] >= 0


def test_check_black_litterman_inputs_diagnostics():
    _, assets, sigma, w_ref, P, Q, Omega = _control()

    diag = check_black_litterman_inputs(sigma, P=P, Omega=Omega)

    assert diag["has_views"] is True
    assert diag["dimension_ok"] is True
    assert diag["omega_is_symmetric"] is True
    assert diag["omega_is_psd"] is True


def test_check_black_litterman_inputs_sin_views():
    _, assets, sigma, *_ = _control()
    diag = check_black_litterman_inputs(sigma)
    assert diag["has_views"] is False
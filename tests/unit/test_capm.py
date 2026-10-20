"""
Escalamiento v2 - M7: Beta, CAPM, Hamada y CAL/CML/SML (secciones 5.2 y 8).

- P2-18: CAL, CML y SML con ejes propios (riesgo total vs. beta).
- P2-31: beta manual (Cov/Var) y OLS coinciden; benchmark constante da
  salida INDEFINIDA, no una division forzada.
- P2-32: Hamada ida/vuelta con el mismo supuesto de deuda (tax, beta_D).
- P2-33: CAPM, extension pais, inflacion y moneda sin doble conteo.

El caso "comparable" vs. "referencia" de ``caso_alimentos_retornos.csv``
(6 observaciones anuales, caso Alimentos del Norte) se usa como control
aritmetico del beta (sec. 9 de la guia: "Seis observaciones permiten un
control aritmetico, no acreditar precision para una inversion real").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.capm import (
    BetaResult,
    capital_allocation_line,
    capm_country_extension,
    capm_required_return,
    estimate_beta,
    excess_returns,
    ke_local_currency,
    optimal_y_unconstrained,
    portfolio_beta,
    relever_beta,
    security_market_line,
    unlever_beta,
)
from tests.fixtures_path import CASO_ALIMENTOS_RETORNOS

pytestmark = pytest.mark.skipif(not CASO_ALIMENTOS_RETORNOS.exists(), reason="fixture v2 no copiado aun")


def _retornos_caso():
    df = pd.read_csv(CASO_ALIMENTOS_RETORNOS, index_col="anio")
    return df["comparable"], df["referencia"]


def test_excess_returns_escalar_y_serie():
    r = pd.Series([0.10, 0.05, -0.02], index=[2020, 2021, 2022])
    assert excess_returns(r, 0.03).tolist() == pytest.approx([0.07, 0.02, -0.05])

    rf_serie = pd.Series([0.03, 0.02, 0.01], index=[2020, 2021, 2022])
    assert excess_returns(r, rf_serie).tolist() == pytest.approx([0.07, 0.03, -0.03])


def test_beta_manual_coincide_con_ols_caso_control_p2_31():
    comparable, referencia = _retornos_caso()

    result = estimate_beta(comparable, referencia, rf_period=0.0)

    assert result.n == 6
    assert result.beta == pytest.approx(result.beta_manual, abs=1e-8)
    # Calculo a mano (ver AI_USAGE.md): Cov=0.00848, Var(referencia)=0.0056 -> beta=1.514286
    assert result.beta == pytest.approx(1.5142857142857142, abs=1e-8)
    assert result.r_squared == pytest.approx(0.7643537414965986, abs=1e-8)
    assert result.ci_low < result.beta < result.ci_high


def test_benchmark_constante_da_salida_indefinida_p2_31():
    comparable, _ = _retornos_caso()
    benchmark_constante = pd.Series([0.05] * len(comparable), index=comparable.index)

    with pytest.raises(ValueError, match="(?i)indefinid"):
        estimate_beta(comparable, benchmark_constante, rf_period=0.0)


def test_estimate_beta_requiere_muestra_minima():
    muy_corta = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError):
        estimate_beta(muy_corta, muy_corta, rf_period=0.0)


def test_portfolio_beta_suma_ponderada():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    betas = pd.Series({"A": 1.2, "B": 0.8})

    assert portfolio_beta(weights, betas) == pytest.approx(0.6 * 1.2 + 0.4 * 0.8)


def test_portfolio_beta_falla_si_falta_un_beta():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    betas = pd.Series({"A": 1.2})
    with pytest.raises(ValueError):
        portfolio_beta(weights, betas)


def test_hamada_ida_y_vuelta_mismo_de_p2_32():
    beta_L = 1.3
    beta_U = unlever_beta(beta_L, D=0.6, E=1.0, tax=0.35, beta_D=0.0)
    beta_L_roundtrip = relever_beta(beta_U, D=0.6, E=1.0, tax=0.35, beta_D=0.0)

    assert beta_L_roundtrip == pytest.approx(beta_L, abs=1e-10)


def test_hamada_ida_y_vuelta_con_beta_deuda_no_nula_p2_32():
    beta_L = 1.3
    beta_U = unlever_beta(beta_L, D=0.6, E=1.0, tax=0.35, beta_D=0.2)
    beta_L_roundtrip = relever_beta(beta_U, D=0.6, E=1.0, tax=0.35, beta_D=0.2)

    assert beta_L_roundtrip == pytest.approx(beta_L, abs=1e-10)


def test_hamada_formula_simplificada_con_beta_deuda_cero():
    # beta_U = beta_L / [1+(1-tax)*D/E] (forma simplificada de la seccion 8.1)
    beta_L, D, E, tax = 1.5142857142857142, 0.6, 1.0, 0.35
    beta_U = unlever_beta(beta_L, D, E, tax, beta_D=0.0)
    esperado = beta_L / (1 + (1 - tax) * D / E)
    assert beta_U == pytest.approx(esperado, abs=1e-12)


def test_hamada_caso_alimentos_del_norte_p2_32():
    comparable, referencia = _retornos_caso()
    beta_comparable = estimate_beta(comparable, referencia, rf_period=0.0).beta

    beta_U = unlever_beta(beta_comparable, D=0.6, E=1.0, tax=0.35, beta_D=0.0)
    beta_L_empresa = relever_beta(beta_U, D=0.75, E=1.0, tax=0.35, beta_D=0.0)

    # Calculo a mano documentado en AI_USAGE.md.
    assert beta_U == pytest.approx(1.0894141829393622, abs=1e-8)
    assert beta_L_empresa == pytest.approx(1.6205035971223012, abs=1e-8)


def test_capm_basico_y_extension_pais_sin_doble_conteo_p2_33():
    ke_basico = capm_required_return(rf=0.043, beta_L=1.6205035971223012, erp=0.055)
    ke_usd = capm_country_extension(
        rf_usd=0.043, beta_L=1.6205035971223012, erp_madura=0.055, lambda_=1, crp=0.032
    )
    # La extension pais agrega lambda*CRP una sola vez sobre el CAPM basico.
    assert ke_usd == pytest.approx(ke_basico + 1 * 0.032, abs=1e-12)
    assert ke_usd == pytest.approx(0.16412769784172657, abs=1e-8)


def test_ke_moneda_local_caso_alimentos_p2_33():
    ke_usd = 0.16412769784172657
    ke_cop = ke_local_currency(ke_usd, inflation_local=0.045, inflation_usd=0.025)

    assert ke_cop == pytest.approx(0.18684238462888225, abs=1e-8)


def test_ke_moneda_local_identidad_sin_diferencial_de_inflacion():
    # Si la inflacion local e internacional son iguales, Ke no deberia cambiar
    # (no se duplica el efecto pais metiendo una inflacion que no aporta nada).
    ke_usd = 0.10
    assert ke_local_currency(ke_usd, inflation_local=0.03, inflation_usd=0.03) == pytest.approx(ke_usd, abs=1e-12)


def test_cal_y_optimal_y_p2_18():
    rf, mu_t, sigma_t, gamma = 0.03, 0.12, 0.20, 4.0

    e_rc, sigma_c = capital_allocation_line(rf, mu_t, sigma_t, y=0.5)
    assert e_rc == pytest.approx(rf + 0.5 * (mu_t - rf))
    assert sigma_c == pytest.approx(0.5 * sigma_t)

    y_star = optimal_y_unconstrained(mu_t, rf, sigma_t, gamma)
    assert y_star == pytest.approx((mu_t - rf) / (gamma * sigma_t ** 2))


def test_sml_usa_beta_no_volatilidad_total_p2_18():
    rf, erp = 0.03, 0.06
    betas = pd.Series({"A": 0.8, "B": 1.0, "C": 1.5})

    sml = security_market_line(rf, betas, erp)

    assert sml["B"] == pytest.approx(rf + 1.0 * erp)
    assert sml["A"] < sml["B"] < sml["C"]
    # Un escalar tambien debe funcionar (un solo activo).
    assert security_market_line(rf, 1.0, erp) == pytest.approx(rf + erp)
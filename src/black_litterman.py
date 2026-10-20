"""Black-Litterman: prior de equilibrio, opiniones (views) con incertidumbre
y posterior (M5 de la guia v2, seccion 6).

Convencion (seccion 6.1 de la guia, idéntica al fixture de control
``black_litterman_control.json``), con todo en EXCESOS de rendimiento
(sobre ``rf``), decimal:

    pi = delta * Sigma @ w_ref                          (prior de equilibrio)
    B = tau * Sigma
    A = P @ B @ P.T + Omega
    mu_BL_exceso = pi + B @ P.T @ solve(A, Q - P @ pi)
    M = B - B @ P.T @ solve(A, P @ B)
    Sigma_predictiva = Sigma + M

Sin views (P vacio, 0 filas): mu_BL_exceso = pi y M = B, tal cual exige
P2-19 ("BL sin views conserva el prior") — no es un caso especial oculto,
es la misma formula evaluada con K=0 opiniones.

M es la incertidumbre POSTERIOR DE LA MEDIA, no riesgo de retornos
(seccion 6.1: "M es incertidumbre posterior de la media, no riesgo de
retornos"). La comparacion principal de carteras (ver src.risk_matrix)
debe cambiar mu manteniendo Sigma fija; una variante con
``sigma_predictiva`` se etiqueta aparte, nunca se mezcla sin avisar.

Omega es una decision del estudiante, justificada por view (seccion 6.2:
"cada view registra... confianza justificada"). Este modulo NO deriva
Omega automaticamente a partir de una formula generica de "confianza":
eso convertiria una decision financiera del equipo en un valor por
defecto de la IA, lo que PROMPT_TUTOR_V2 prohibe explicitamente
("la IA implementa, el estudiante decide").
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class View:
    """Una opinion (view) sobre excesos de rendimiento, con la trazabilidad
    minima que exige la seccion 6.2 de la guia (ID, autor, tesis, fuente,
    publicacion, horizonte, confianza justificada, evidencia contraria,
    vencimiento)."""

    view_id: str
    student_author: str
    thesis: str
    assets: "list[str]"      # orden de P_row (debe existir en el universo)
    P_row: "list[float]"     # 1 fila de P: coeficientes de la opinion sobre `assets`
    Q_excess: float          # ya en exceso (ver `excess_from_total_view`)
    Omega_variance: float    # incertidumbre de ESTA opinion, decidida y justificada por el estudiante
    confidence: float = 1.0  # 0 => se excluye (sec. 6.2); no determina Omega automaticamente
    source: Optional[str] = None
    published_at: Optional[str] = None
    known_at: Optional[str] = None
    horizon: Optional[str] = None
    counterevidence: Optional[str] = None
    expires_at: Optional[str] = None
    invalidation: Optional[str] = None

    def is_relative(self) -> bool:
        """True si la view compara activos entre si (mas de un coeficiente
        distinto de cero en P_row); False si es una view absoluta."""
        return sum(1 for x in self.P_row if x != 0) > 1


def excess_from_total_view(q_total: float, rf: float, p_row: "list[float]") -> float:
    """Q_exceso = Q_total - rf*(P*1) (seccion 6.1).

    Para una view ABSOLUTA (un solo 1 en ``p_row``) esto resta ``rf`` una
    vez. Para una view RELATIVA (ej. [1, -1]) ``sum(p_row) == 0`` y
    ``Q_total`` ya esta en exceso (no se le resta rf) — por construccion,
    nunca a mano.
    """
    return q_total - rf * sum(p_row)


def active_views(views: "list[View]", as_of: "Optional[datetime]" = None) -> "list[View]":
    """Views con confianza > 0 y no vencidas a la fecha ``as_of`` (ahora si
    no se indica).

    "Confianza cero EXCLUYE la opinion, no implica Omega cero" (sec. 6.2):
    por eso una view con ``confidence <= 0`` se quita de la lista en vez de
    forzarle Omega a cero (lo que la volveria una certeza, el error opuesto).
    """
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(timezone.utc))
    if as_of_ts.tzinfo is not None:
        as_of_ts = as_of_ts.tz_localize(None)
    kept = []
    for v in views:
        if v.confidence <= 0:
            continue
        if v.expires_at and as_of_ts > pd.Timestamp(v.expires_at):
            continue
        kept.append(v)
    return kept


def find_duplicate_views(views: "list[View]") -> "list[tuple[View, View]]":
    """Pares de views con la misma fila P y el mismo Q_excess: evidencia
    repetida que no deberia contarse dos veces (sec. 6.2: "opiniones que
    repiten la misma noticia no son necesariamente independientes")."""
    duplicates = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            if views[i].P_row == views[j].P_row and views[i].Q_excess == views[j].Q_excess:
                duplicates.append((views[i], views[j]))
    return duplicates


def assemble_views(
    views: "list[View]", assets: "list[str]"
) -> "tuple[pd.DataFrame, pd.Series, pd.DataFrame]":
    """Construye ``(P, Q, Omega)`` a partir de views ya activas (ver
    ``active_views``), reindexando cada ``P_row`` al orden de ``assets``
    (sec. 6.1: "se verifica orden de activos, dimensiones"). Omega se
    construye diagonal (opiniones no correlacionadas entre si) con la
    varianza que cada view declara.

    Devuelve matrices de 0 filas si ``views`` esta vacio.
    """
    if not views:
        return pd.DataFrame(columns=assets), pd.Series(dtype=float), pd.DataFrame()

    rows, q_values, view_ids = [], [], []
    for v in views:
        fuera_de_universo = set(v.assets) - set(assets)
        if fuera_de_universo:
            raise ValueError(
                f"La view {v.view_id!r} referencia activos fuera del universo: "
                f"{sorted(fuera_de_universo)}."
            )
        row_by_asset = dict(zip(v.assets, v.P_row))
        rows.append([row_by_asset.get(a, 0.0) for a in assets])
        q_values.append(v.Q_excess)
        view_ids.append(v.view_id)

    P = pd.DataFrame(rows, index=view_ids, columns=assets)
    Q = pd.Series(q_values, index=view_ids)
    Omega = pd.DataFrame(np.diag([v.Omega_variance for v in views]), index=view_ids, columns=view_ids)
    return P, Q, Omega


def implied_equilibrium_returns(sigma: pd.DataFrame, w_ref: pd.Series, delta: float) -> pd.Series:
    """pi = delta * Sigma @ w_ref (optimizacion inversa desde ``w_ref``,
    sec. 6.1). ``w_ref`` debe venir documentado (capitalizacion de mercado
    o una referencia estrategica justificada) — nunca se asume en silencio
    que un universo parcial o patrimonios de ETFs son "el mercado"."""
    w_ref = w_ref.reindex(sigma.index)
    if w_ref.isna().any():
        faltantes = w_ref[w_ref.isna()].index.tolist()
        raise ValueError(f"w_ref no tiene peso para: {faltantes}.")
    pi = delta * (sigma.values @ w_ref.values)
    return pd.Series(pi, index=sigma.index, name="pi_exceso")


@dataclass
class BlackLittermanResult:
    pi: pd.Series
    mu_posterior_excess: pd.Series
    M: pd.DataFrame        # incertidumbre posterior de la MEDIA (no es riesgo de retornos)
    sigma_predictiva: pd.DataFrame
    A: "Optional[pd.DataFrame]"
    n_views: int


def black_litterman_posterior(
    sigma: pd.DataFrame,
    pi: pd.Series,
    tau: float,
    P: "Optional[pd.DataFrame]" = None,
    Q: "Optional[pd.Series]" = None,
    Omega: "Optional[pd.DataFrame]" = None,
) -> BlackLittermanResult:
    """Posterior de Black-Litterman (sec. 6.1 de la guia).

    Sin views (``P``/``Q``/``Omega`` en ``None`` o con 0 filas):
    ``mu_posterior_excess = pi`` y ``M = B`` — exactamente lo que exige
    P2-19 ("BL sin views conserva el prior").
    """
    assets = sigma.index
    B = tau * sigma

    has_views = P is not None and Q is not None and Omega is not None and len(P) > 0
    if not has_views:
        return BlackLittermanResult(
            pi=pi,
            mu_posterior_excess=pi.copy(),
            M=B.copy(),
            sigma_predictiva=sigma + B,
            A=None,
            n_views=0,
        )

    P = P.reindex(columns=assets).fillna(0.0)
    if len(Q) != len(P):
        raise ValueError(f"Q debe tener {len(P)} opiniones (una por fila de P); tiene {len(Q)}.")
    if Omega.shape != (len(P), len(P)):
        raise ValueError(f"Omega debe ser {len(P)}x{len(P)}; es {Omega.shape}.")

    P_mat, Q_vec, Omega_mat = P.values, Q.values, Omega.values
    B_mat, pi_vec = B.values, pi.reindex(assets).values

    A_mat = P_mat @ B_mat @ P_mat.T + Omega_mat
    rhs = Q_vec - P_mat @ pi_vec
    x = np.linalg.solve(A_mat, rhs)
    mu_post = pi_vec + B_mat @ P_mat.T @ x

    Y = np.linalg.solve(A_mat, P_mat @ B_mat)
    M_mat = B_mat - B_mat @ P_mat.T @ Y

    mu_posterior = pd.Series(mu_post, index=assets, name="mu_BL_exceso")
    M_df = pd.DataFrame(M_mat, index=assets, columns=assets)
    sigma_pred = sigma + M_df
    A_df = pd.DataFrame(A_mat, index=P.index, columns=P.index)

    return BlackLittermanResult(
        pi=pi, mu_posterior_excess=mu_posterior, M=M_df, sigma_predictiva=sigma_pred, A=A_df, n_views=len(P)
    )


def check_black_litterman_inputs(
    sigma: pd.DataFrame, P: "Optional[pd.DataFrame]" = None, Omega: "Optional[pd.DataFrame]" = None
) -> dict:
    """Diagnostico minimo de dimensiones y de Omega (sec. 6.1: "se
    verifica... dimensiones, PSD y condicionamiento")."""
    diag = {"sigma_shape": sigma.shape, "has_views": P is not None and len(P) > 0}
    if P is not None and len(P) > 0:
        diag["P_shape"] = P.shape
        diag["dimension_ok"] = P.shape[1] == sigma.shape[0]
    if Omega is not None and len(Omega) > 0:
        omega_vals = Omega.values
        diag["omega_is_symmetric"] = bool(np.allclose(omega_vals, omega_vals.T))
        eigvals = np.linalg.eigvalsh((omega_vals + omega_vals.T) / 2)
        diag["omega_min_eigenvalue"] = float(eigvals.min())
        diag["omega_is_psd"] = bool(eigvals.min() >= -1e-10)
        diag["omega_condition_number"] = (
            float(np.linalg.cond(omega_vals)) if eigvals.min() > 0 else float("inf")
        )
    return diag
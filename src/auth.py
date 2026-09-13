"""
Autenticación OIDC (RF-02).

Usa la autenticación nativa de Streamlit (st.login / st.logout / st.user),
que delega en Authlib la validación de firma, emisor, audiencia, expiración,
state y nonce. No se reimplementa criptografía aquí (prohibido por la guía,
sección 7.3, punto 62).

IMPORTANTE PARA EL EQUIPO:
La sintaxis exacta de st.login/st.user puede variar entre versiones de
Streamlit. Esta versión del código se escribió con el conocimiento de la
API vigente a comienzos de 2026; antes de usarla en serio, verifiquen contra
la documentación oficial de la versión que fijen en requirements.txt:
  https://docs.streamlit.io/develop/api-reference/user/st.login
  https://docs.streamlit.io/develop/api-reference/user/st.user
  https://docs.streamlit.io/develop/concepts/connections/authentication

Esto es solo el esqueleto de RF-02. Todavía falta (ver Fase 9 del roadmap):
- Allowlist del servidor (bloquear cualquier (iss, sub) no autorizado).
- Rechazar identidad incompleta (falta nombre o correo) como
  "identity_incomplete".
- Registrar sesión lógica, evento de auditoría e intención de notificación
  en una sola transacción al iniciar sesión (puntos 64-66 de la guía).
"""
import streamlit as st


def is_auth_configured() -> bool:
    """True si existe una sección [auth] real en secrets.toml (no el .example)."""
    try:
        return "auth" in st.secrets
    except FileNotFoundError:
        return False


def require_login():
    """
    Bloquea la ejecución (st.stop) hasta que exista una sesión OIDC válida.
    Devuelve st.user cuando el usuario ya inició sesión.
    """
    if not st.user.is_logged_in:
        st.info(
            "Debes iniciar sesión con una cuenta autorizada del curso "
            "(Google o Microsoft) para continuar."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "Iniciar sesión con Google",
                on_click=st.login,
                args=("google",),
                use_container_width=True,
            )
        with col2:
            st.button(
                "Iniciar sesión con Microsoft",
                on_click=st.login,
                args=("microsoft",),
                use_container_width=True,
            )
        st.stop()

    # TODO (Fase 9): validar allowlist del servidor por (iss, sub) y rechazar
    # identidad incompleta antes de devolver el usuario.

    return st.user

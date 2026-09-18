"""
Registro local de accesos y consentimientos.

Contexto (léanlo antes de tocar este archivo):
- La guía (sección 7.1, punto 46) dice que la base de datos, el sink
  institucional y el gestor de secretos "serán suministrados por el curso".
  El profesor confirmó que no va a suministrarlos. Ante esa ausencia, este
  módulo implementa un registro LOCAL mínimo dentro del propio repositorio,
  en vez de dejar el acceso sin ningún rastro.
- Nombre y correo siguen siendo AUTOINFORMADOS por quien usa la app (no hay
  verificación de identidad vía OIDC real, porque esa pieza también depende
  de credenciales que el curso no entregó). Esto NO sustituye el modelo
  completo de la sección 7.3.1 (profiles/consent_events/app_sessions/
  auth_events/outbox con transacción atómica, allowlist, expiración de
  sesión y purga automática por retención) — eso sigue pendiente de la
  Fase 9 y debe seguir documentado como tal en TRACEABILITY.md.
- El archivo `data/access_log.sqlite3` NUNCA debe subirse a git: contiene
  datos personales reales de quienes prueben la app. Ya está cubierto por
  el patrón `*.sqlite3` en `.gitignore`; no lo quiten de ahí.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "access_log.sqlite3"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Crea la tabla de accesos si todavía no existe. Se puede llamar muchas veces."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accesos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                nombre TEXT NOT NULL,
                correo TEXT NOT NULL,
                acepto_privacidad INTEGER NOT NULL,
                acepto_disclaimer INTEGER NOT NULL
            )
            """
        )


def log_access(nombre: str, correo: str, acepto_privacidad: bool, acepto_disclaimer: bool) -> None:
    """
    Inserta una fila nueva con la marca de tiempo UTC actual.
    Quien llama a esta función es responsable de no llamarla más de una vez
    por sesión (ver app.py::render_disclaimer_gate, que usa
    st.session_state para garantizarlo y evitar filas duplicadas en cada
    rerun de Streamlit).
    """
    init_db()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO accesos
                (timestamp_utc, nombre, correo, acepto_privacidad, acepto_disclaimer)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                (nombre or "").strip(),
                (correo or "").strip(),
                int(bool(acepto_privacidad)),
                int(bool(acepto_disclaimer)),
            ),
        )


def read_all() -> list[tuple]:
    """Devuelve todas las filas del registro, ordenadas por fecha (para revisión del equipo)."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT timestamp_utc, nombre, correo, acepto_privacidad, acepto_disclaimer "
            "FROM accesos ORDER BY timestamp_utc"
        )
        return cursor.fetchall()
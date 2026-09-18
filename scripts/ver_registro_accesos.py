"""
Muestra en la terminal el registro local de accesos y consentimientos.

Uso (desde la raíz del repositorio, con el entorno virtual activado):
    python scripts/ver_registro_accesos.py

No requiere instalar nada nuevo: usa sqlite3, que ya viene con Python.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.access_log import DB_PATH, read_all


def main() -> None:
    filas = read_all()

    if not filas:
        print(f"El registro está vacío todavía. Archivo: {DB_PATH}")
        return

    print(f"Registro de accesos ({DB_PATH}):\n")
    encabezado = f"{'Fecha/hora (UTC)':<22} {'Nombre':<28} {'Correo':<28} {'Privacidad':<11} {'Disclaimer':<11}"
    print(encabezado)
    print("-" * len(encabezado))
    for timestamp_utc, nombre, correo, acepto_privacidad, acepto_disclaimer in filas:
        print(
            f"{timestamp_utc:<22} {nombre:<28} {correo:<28} "
            f"{'Sí' if acepto_privacidad else 'No':<11} "
            f"{'Sí' if acepto_disclaimer else 'No':<11}"
        )
    print(f"\nTotal de registros: {len(filas)}")


if __name__ == "__main__":
    main()
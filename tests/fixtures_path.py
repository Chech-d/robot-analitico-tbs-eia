"""
Ubicación de los 3 archivos oficiales del anexo (fixture de 20 activos,
resultados esperados y matriz de casos de prueba) usados por la Fase 10.

En el repositorio real de la estudiante, estos 3 CSV deben copiarse a
tests/fixtures/ (ver instrucciones de la Fase 10). Para pruebas internas de
verificación se puede sobrescribir la ruta con la variable de entorno
TBS_EIA_FIXTURES_DIR.
"""
import os
from pathlib import Path

FIXTURES_DIR = Path(os.environ.get("TBS_EIA_FIXTURES_DIR", Path(__file__).parent / "fixtures"))

FIXTURE_20_PRICES = FIXTURES_DIR / "Fixture_20_activos_sintetico_TBS_EIA.csv"
EXPECTED_RESULTS_20 = FIXTURES_DIR / "Resultados_esperados_20_activos_TBS_EIA.csv"
TEST_MATRIX = FIXTURES_DIR / "Anexo_matriz_casos_prueba_robot_analitico_TBS_EIA_FINAL.csv"
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

# Paquete oficial de escalamiento a v2 (controles sintéticos del profesor,
# sin red, usados para verificar covarianza/Markowitz/Black-Litterman/
# rebalanceo antes de conectar el universo real). Copiar estos archivos del
# paquete "BOT_Portafolios_V2_Estudiantes" a tests/fixtures/v2/.
V2_FIXTURES_DIR = FIXTURES_DIR / "v2"
DOS_ACTIVOS = V2_FIXTURES_DIR / "dos_activos.json"
BLACK_LITTERMAN_CONTROL = V2_FIXTURES_DIR / "black_litterman_control.json"
REBALANCEO_CONTROL = V2_FIXTURES_DIR / "rebalanceo_control.json"
PRECIOS_SINTETICOS_20 = V2_FIXTURES_DIR / "precios_sinteticos_20.csv"
CASO_ALIMENTOS_PARAMETROS = V2_FIXTURES_DIR / "caso_alimentos_parametros.json"
CASO_ALIMENTOS_RETORNOS = V2_FIXTURES_DIR / "caso_alimentos_retornos.csv"
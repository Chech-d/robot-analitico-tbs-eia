# EDITAR: Nombre del sistema

Robot analítico para la preselección de activos — actividad evaluativa del
curso Teoría Moderna de Portafolios, Tech Business School - Universidad EIA.

> Piloto académico restringido. No constituye asesoría financiera ni
> recomendación de inversión. Solo cuentas ficticias del curso.

## Equipo

- EDITAR: Integrante 1
- EDITAR: Integrante 2
- EDITAR: Integrante 3

## Alcance

Analiza el riesgo y rendimiento histórico de un activo, y compara
simultáneamente un mínimo de 20 activos. **No** calcula pesos, covarianzas,
correlaciones conjuntas, frontera eficiente ni recomendaciones de inversión
(es una fase de preselección, no de asset allocation).

## Ejecución local

```bash
# 1. Crear y activar entorno virtual (una sola vez)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar credenciales OIDC (ver .streamlit/secrets.toml.example)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar .streamlit/secrets.toml con las credenciales reales del curso

# 4. Correr la app
streamlit run app.py
```

## Estructura del repositorio

```
app.py                      # Entrada y navegación
src/auth.py                 # OIDC (Google/Microsoft), sesión lógica
src/sessions.py             # app_session, expiración, logout
src/consent.py              # preconsentimiento y consentimientos
src/audit.py                # eventos de auditoría inmutables
src/notifications.py        # outbox de notificaciones
src/notifications_worker.py # worker de entrega (proceso independiente)
src/data.py                 # descarga, validación y remuestreo de precios
src/analytics.py            # rendimientos log, descriptivos, diagnósticos
src/forecasting.py          # modelos homocedásticos y walk-forward
src/risk_rules.py           # VaR, niveles de entrada/salida, probabilidades
src/charts.py               # gráficos y mapa histórico rendimiento-riesgo
src/lifecycle.py            # purga y ciclo de vida de datos
tests/                      # pruebas unitarias e integración (T-01 a T-25)
migrations/                 # esquema de base de datos versionado
```

## Política de ejecución

_Pendiente de completar (ver sección 3.2 de la guía evaluativa): universo,
datos, acceso y privacidad, posición, ventana/frecuencia, rendimiento,
horizonte, modelo, riesgo, entrada/salidas, selección, no operar,
limitaciones._

## Fuente de datos

Precios ajustados descargados con [yfinance](https://pypi.org/project/yfinance/).

## Disclaimer

Ver el texto completo dentro de la aplicación (`app.py`), mostrado antes de
habilitar cualquier análisis.

## Referencias

_Pendiente: completar en formato APA 7 (ver bibliografía de la guía evaluativa)._

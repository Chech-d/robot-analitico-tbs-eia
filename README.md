# Pre - Asset Allocation Analytic Bot

Robot analítico para la preselección de activos — actividad evaluativa del
curso Teoría Moderna de Portafolios, Tech Business School - Universidad EIA.

> Piloto académico restringido. No constituye asesoría financiera ni
> recomendación de inversión. Solo cuentas ficticias del curso.

## Equipo

**Equipo DPST**

- Sergio Delgado Laverde
- David Angel Perez
- Pedro Velez Uribe
- Tomás Giraldo Gomez

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

## Pruebas

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Antes de correr las pruebas, copien los 3 archivos oficiales del anexo de la
guía a `tests/fixtures/` (mismos nombres exactos):

- `Fixture_20_activos_sintetico_TBS_EIA.csv`
- `Resultados_esperados_20_activos_TBS_EIA.csv`
- `Anexo_matriz_casos_prueba_robot_analitico_TBS_EIA_FINAL.csv`

La suite cubre T-01 a T-17 (cálculo de rendimientos, estadísticas
descriptivas, remuestreo, modelos de pronóstico, validación walk-forward,
VaR, niveles de entrada/salida, comparación de 20 activos, dominancia y
preselección), verificados con tolerancia atol=1e-8/rtol=1e-10 contra los
fixtures oficiales de la guía. T-18 a T-24 (seguridad, sesión, workers) están
pendientes de la Fase de seguridad real (ver `TRACEABILITY.md`).

## Estructura del repositorio
app.py # Entrada y navegación
src/auth.py # OIDC (Google/Microsoft), sesión lógica
src/sessions.py # app_session, expiración, logout
src/consent.py # preconsentimiento y consentimientos
src/audit.py # eventos de auditoría inmutables
src/access_log.py # registro local de accesos y consentimientos (evidencia parcial, Fase 9 pendiente)
src/notifications.py # outbox de notificaciones
src/notifications_worker.py # worker de entrega (proceso independiente)
src/data.py # descarga, validación y remuestreo de precios
src/analytics.py # rendimientos log, descriptivos, diagnósticos
src/forecasting.py # modelos homocedásticos y walk-forward
src/risk_rules.py # VaR, niveles de entrada/salida, probabilidades
src/comparison.py # comparación de N activos, dominancia, preselección
src/charts.py # gráficos y mapa histórico rendimiento-riesgo
src/lifecycle.py # purga y ciclo de vida de datos
scripts/ver_registro_accesos.py # imprime en terminal el contenido de data/access_log.sqlite3
tests/ # pruebas unitarias e integración (T-01 a T-25)
tests/fixtures/ # CSV oficiales del anexo (fixture, resultados esperados, matriz)
migrations/ # esquema de base de datos versionado
data/ # generado en tiempo de ejecución; nunca se sube a git


## Política de ejecución

_(Sección 3.2 de la guía evaluativa. Es el contrato lógico del sistema:
describe lo que la aplicación SÍ hace, con qué reglas, y cuándo se abstiene
de dar una conclusión operativa.)_

**Universo.** Acciones individuales negociadas en los mercados cubiertos por
Yahoo Finance (a través de la librería `yfinance`), en la moneda nativa de
cotización de cada ticker. No se realiza conversión de monedas: en la
comparación de activos, cualquier activo cuya moneda no coincida con la
moneda base del lote queda excluido y reportado por separado (RF-20).

**Datos.** Proveedor: Yahoo Finance vía `yfinance`. Se usa el precio de
cierre ajustado (`auto_adjust=True`, ya incorpora dividendos y splits). La
aplicación muestra en pantalla el proveedor, la moneda, la zona horaria y la
fecha del último dato disponible (RF-05). Las filas con precio no positivo o
faltante se descartan explícitamente antes de cualquier cálculo (RF-06).

**Acceso y privacidad.** El acceso está pensado para autenticarse con Google
o Microsoft mediante OIDC, con preconsentimiento y sesión lógica auditada.
**Estado actual: en desarrollo.** El curso confirmó que no va a suministrar
la base de datos, el sink ni el gestor de secretos institucional previstos
en la sección 7.1 (punto 46) de la guía, así que el módulo OIDC completo de
`src/auth.py` sigue siendo un esqueleto y la app corre con `DEV_MODE=True`
en `app.py`, que sustituye el login real por campos de nombre/correo
autoinformados. El disclaimer académico y los dos consentimientos por
separado (autorización de datos y aceptación del disclaimer) sí están
implementados y bloquean el análisis hasta que ambos se acepten
(`app.py::render_disclaimer_gate`).

Como evidencia parcial de acceso ante la ausencia de infraestructura
institucional, la primera vez que una sesión de navegador acepta ambos
consentimientos, se registra localmente (fecha/hora UTC, nombre, correo y
ambas aceptaciones) en `data/access_log.sqlite3` (`src/access_log.py`).
Ese archivo se genera automáticamente, nunca se sube a git (ya está en
`.gitignore`) y se puede revisar con `python scripts/ver_registro_accesos.py`.
Esto **no** sustituye el modelo completo de la sección 7.3.1 (identidad
verificada por OIDC, sesión lógica con expiración, evento de auditoría
inmutable, outbox y purga automática por retención): nombre y correo siguen
sin verificarse contra un proveedor real, y esa parte permanece pendiente
de la Fase de seguridad (ver `TRACEABILITY.md`, RF-02, RNF-03).

**Posición.** Únicamente posición larga para el núcleo obligatorio; el
sistema no opera ni simula posiciones cortas.

**Ventana y frecuencia.** El usuario elige fecha inicial, fecha final y
frecuencia (diaria, semanal o mensual). El remuestreo siempre ocurre ANTES
de calcular rendimientos (RF-07), usando el último precio válido del
periodo. Mínimo de observaciones: al menos 2 rendimientos para mostrar
estadísticas descriptivas; para pronóstico y validación walk-forward, el
número de rendimientos T debe cumplir T ≥ max(2m,5H)+H+9 (sección 5.5,
donde m son los periodos por año de la frecuencia elegida y H el horizonte).

**Rendimiento.** Exclusivamente logarítmico: g_t = ln(P_t / P_{t-1}). Es la
única definición admitida en todo el sistema (RF-10); no existe una ruta de
cálculo con rendimientos simples. La anualización usa μ_anual = m·ḡ y
σ_anual = √m·s_g, con m = 252 (diaria), 52 (semanal) o 12 (mensual).

**Horizonte.** El usuario define H como una cantidad entera positiva de
periodos, o como una fecha objetivo que el sistema convierte de forma
transparente a la frecuencia elegida (RF-04). El límite máximo de H no es un
botón fijo: se calcula en vivo a partir de los datos disponibles, con la
regla de suficiencia walk-forward T ≥ max(2m,5H)+H+9 (sección 5.5), y se
muestra al usuario antes de que elija H.

**Modelo.** Dos especificaciones homocedásticas seleccionables para el mismo
H (RF-13): una caminata aleatoria sin deriva (benchmark obligatorio, media
forzada a cero) y un modelo lognormal de parámetros constantes con deriva
(media y volatilidad muestrales). Ambos comparten la misma familia: el
rendimiento logarítmico acumulado a H periodos se asume Normal(m_H, v_H),
homocedástico y sin autocorrelación.

**Riesgo.** VaR paramétrico individual al 95% (obligatorio) y 99%
(opcional), con capital hipotético ingresado por el usuario, convención de
signo positivo = pérdida, y piso obligatorio en cero (RF-16).

**Entrada y salidas.** Posición larga: entrada E = último precio ajustado
válido. Con parámetros congelados p_L=5%, p_U=95%, BR_min=1.0 y costos de
compra/venta c_b, c_s configurables, se calculan SL_H y TP_H (cuantiles de
la misma distribución y H), el precio de equilibrio P_BE, el riesgo neto
D_neto, el beneficio neto U_neto y la relación beneficio-riesgo neta
BR_neto (sección 5.7).

**Selección.** Cuatro criterios de preselección transparente entre activos
comparados (RF-21): máxima media histórica bajo un límite de riesgo; mínima
volatilidad bajo una media mínima; máxima razón individual
media-volatilidad (RVR); y conjunto no dominado media-volatilidad. Los
cuatro devuelven TODOS los empates dentro de tolerancia y nunca fuerzan una
selección si ningún activo cumple el criterio. El resultado siempre se
presenta como "mejor según este criterio y estos parámetros", nunca como
"la mejor inversión" ni como frontera eficiente de portafolios.

**No operar (no señal).** La aplicación declara explícitamente "no señal" —
y nunca fuerza una conclusión operativa — cuando: los datos son inválidos o
insuficientes; el precio no es positivo; la muestra no alcanza la regla de
suficiencia walk-forward; hay menos de 10 orígenes válidos; no se cumple
`SL_H < E < TP_H`; `P_BE < E` o `P_BE ≥ TP_H`; `D_neto ≤ 0`; `U_neto ≤ 0`; o
`BR_neto < BR_min`. Cada no-señal muestra el motivo (o motivos) exacto que
falló.

**Limitaciones.** Los modelos son homocedásticos por alcance pedagógico
(no incluyen ARCH, GARCH ni EWMA, y el sistema lo advierte explícitamente si
los diagnósticos de la sección 5.4 muestran volatilidad variable o colas
gruesas). Los resultados dependen enteramente de la calidad y suficiencia de
los datos históricos descargados de Yahoo Finance; el comportamiento pasado
no garantiza el comportamiento futuro. El sistema no ejecuta operaciones
reales, no sustituye asesoría financiera profesional y puede contener
errores, omisiones o información incompleta. El módulo de seguridad
completo (login OIDC real, sesión lógica persistente, base de datos,
notificaciones, purga y *workers* independientes) está pendiente de
implementación (ver `TRACEABILITY.md`, RF-02, RNF-02, RNF-03) y actualmente
opera en modo de desarrollo simulado (`DEV_MODE=True`).

## Disclaimer

Ver el texto completo dentro de la aplicación (`app.py`), mostrado antes de
habilitar cualquier análisis.

## Referencias

CFA Institute. (2024a). *CFA Program curriculum 2025: Level I, volume 9:
Portfolio management.*

CFA Institute. (2024b). *CFA Program curriculum 2025: Level III core, volume
1: Asset allocation.*

Congreso de Colombia. (2012, 17 de octubre). *Ley 1581 de 2012, por la cual
se dictan disposiciones generales para la protección de datos personales.*
https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981

Francis, J. C., & Kim, D. (2013). *Modern portfolio theory: Foundations,
analysis, and new developments.* John Wiley & Sons.

Google. (2026, 15 de junio). *OpenID Connect.* Google for Developers.
https://developers.google.com/identity/openid-connect/openid-connect

International Center for Academic Integrity. (2021). *The fundamental
values of academic integrity* (3rd ed.).
https://academicintegrity.org/images/pdfs/20019_ICAI-Fundamental-Values_R12.pdf

Lodderstedt, T., Bradley, J., Labunets, A., & Fett, D. (2025). *Best current
practice for OAuth 2.0 security* (BCP 240, RFC 9700). RFC Editor.
https://doi.org/10.17487/RFC9700

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1),
77-91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x

Miao, F., & Holmes, W. (2023). *Guidance for generative AI in education and
research.* UNESCO. https://unesdoc.unesco.org/ark:/48223/pf0000386693

Microsoft. (2026, 30 de junio). *OpenID Connect (OIDC) on the Microsoft
identity platform.* Microsoft Learn.
https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols-oidc

OWASP Foundation. (2025). *OWASP application security verification
standard* (Version 5.0.0).
https://owasp.org/www-project-application-security-verification-standard/

Souppaya, M., Scarfone, K., & Dodson, D. (2022). *Secure Software
Development Framework (SSDF) version 1.1: Recommendations for mitigating
the risk of software vulnerabilities* (NIST Special Publication 800-218).
National Institute of Standards and Technology.
https://doi.org/10.6028/NIST.SP.800-218
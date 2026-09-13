"""
Capa de datos (RF-05 a RF-08): descarga, validación, limpieza y remuestreo.

Se construye en la Fase 3 del roadmap. Debe:
- Descargar precios ajustados (yfinance) e informar proveedor, moneda, zona
  horaria y fecha del último dato.
- Ordenar por fecha, eliminar duplicados, tratar faltantes explícitamente y
  rechazar precios no positivos.
- Remuestrear (diario/semanal/mensual) ANTES de calcular rendimientos.
- Validar ticker inexistente, respuesta vacía, timeout y muestra
  insuficiente sin detener el análisis de los demás activos del lote.
"""

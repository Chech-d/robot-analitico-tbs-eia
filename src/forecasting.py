"""
Modelos homocedásticos y validación walk-forward (RF-13 a RF-15,
secciones 5.5 y validación de la guía).

Se construye en la Fase 6 del roadmap:
- Modelo A: caminata aleatoria sin deriva (benchmark obligatorio).
- Modelo B: lognormal con deriva y parámetros constantes.
- Walk-forward: últimos 10 orígenes, ventana expansiva, reestimación solo
  con datos disponibles en cada origen (sin look-ahead).
"""

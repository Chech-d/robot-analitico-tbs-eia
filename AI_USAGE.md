# Registro de uso de inteligencia artificial

Registrar aquí **toda interacción sustantiva** de IA (que haya modificado
fórmulas, lógica, modelos, arquitectura, seguridad, privacidad, pruebas o
interpretación). Las sugerencias triviales de autocompletado se pueden
resumir periódicamente en una sola fila.

| Fecha | Responsable | Herramienta / modelo | Propósito | Prompt (resumen fiel) | Archivo o función | Cambios humanos | Validación | Limitaciones |
|---|---|---|---|---|---|---|---|---|
| 2026-09-12 | EDITAR | Claude (Sonnet 5) | Generar el esqueleto inicial del repo: estructura de carpetas, requirements.txt, app.py con identidad/disclaimer/login OIDC stub, README | "Dame el paso a paso para construir el robot analítico según la guía evaluativa" | app.py, src/auth.py, requirements.txt, README.md | Editar nombre del sistema, equipo e integrantes; conectar credenciales OIDC reales; completar TODOs marcados en el código | Pendiente: correr `streamlit run app.py` localmente y verificar que la app carga | El módulo de auth es un esqueleto; falta allowlist, sesión lógica completa y persistencia transaccional (Fase 9) |

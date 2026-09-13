"""
Sesión lógica de aplicación (app_session), independiente de la cookie OIDC.

Pendiente (Fase 9): crear app_session_id aleatorio tras el callback OIDC,
garantizar una única sesión activa por user_id, expirar a los 30 minutos de
inactividad o 8 horas desde created_at, y bloquear todas las vistas al
expirar o hacer logout (puntos 64-65 de la guía).
"""

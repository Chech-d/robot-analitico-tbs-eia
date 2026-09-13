"""
Eventos de auditoría (auth_events): inmutables durante la retención.

Pendiente (Fase 9): insertar auth_events en la misma transacción que
perfil/consentimientos/sesión/outbox; impedir lectura o modificación por
usuarios ordinarios.
"""

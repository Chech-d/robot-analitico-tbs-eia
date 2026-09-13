"""
Worker independiente del proceso web: `python -m src.notifications_worker`.

Pendiente (Fase 9): ejecutarse al menos una vez por minuto vía scheduler,
máximo tres intentos totales con adquisición atómica de fila, entrega
idempotente por event_id contra el sink institucional.
"""

if __name__ == "__main__":
    raise NotImplementedError("Worker de notificaciones pendiente (Fase 9).")

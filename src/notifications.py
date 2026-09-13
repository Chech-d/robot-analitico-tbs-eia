"""
Outbox de notificaciones (notification_outbox).

Pendiente (Fase 9): UNIQUE(session_id, notification_type) para impedir más
de una intención session_started; el mensaje solo incluirá nombre/correo
ficticios, proveedor, UTC, sistema y event_id (nunca IP, tokens, tickers ni
resultados).
"""

# Matriz de trazabilidad RF/RNF - Pruebas - Rúbrica - Evidencia

Completar la columna "Evidencia (commit/archivo)" a medida que se implementa
cada requisito. Base: sección 8.2 de la guía evaluativa.

| RF/RNF | Pruebas | Rúbrica | Entregable esperado | Evidencia (commit/archivo) |
|---|---|---|---|---|
| RF-01 | T-18, T-25 | G1, G8 | D1, D3, D5, D11 | |
| RF-02 | T-18 a T-24 | G10 | D1, D5, D6, D7 | |
| RF-03 | T-04 a T-06 | G2, G7 | D1, D6, D10 | |
| RF-04 | T-07 a T-10 | G5, G8 | D1, D6 | |
| RF-05 | T-03 | G2 | D3, D6 | |
| RF-06 | T-01 a T-03 | G2 | D6 | |
| RF-07 | T-01, T-07 | G2, G3 | D6 | |
| RF-08 | T-03, T-06 | G2 | D6 | |
| RF-09 | T-04, T-25 | G3, G8 | D1, D11 | |
| RF-10 | T-01, T-07 | G3 | D6 | |
| RF-11 | T-01, T-07 | G3, G4, G6 | D6, D10 | |
| RF-12 | T-04, T-12 | G3, G4 | D10 | |
| RF-13 | T-11 | G5 | D6 | |
| RF-14 | T-08, T-09, T-11 | G5 | D6, D10 | |
| RF-15 | T-11 | G5 | D6 | |
| RF-16 | T-15 | G6 | D6 | |
| RF-17 | T-14 | G6 | D6 | |
| RF-18 | T-13, T-14 | G6 | D6 | |
| RF-19 | T-05, T-16 | G7 | D10 | |
| RF-20 | T-16 | G7 | D6, D10 | |
| RF-21 | T-17 | G7 | D10 | |
| RF-22 | T-04, T-05 | G7, G8, G11 | D10, D12 | |
| RNF-01 | T-25 | G8 | D1, D11, D12 | |
| RNF-02 | T-24, T-25 | G9 | D1, D4, D5, D12 | |
| RNF-03 | T-18 a T-24 | G10 | D5, D6, D7, D12 | |
| RNF-04 | T-01 a T-25 | G9 | D6, D12 | |
| RNF-05 | Revisión documental | G11 | D3, D8, D9, D12 | |

## Entregables (D1-D12)

- D1. URL HTTPS restringida a la allowlist.
- D2. Repositorio Git y commit final etiquetado.
- D3. README completo (autores, ejecución, fuente, ecuaciones, supuestos, licencias, disclaimer, APA 7).
- D4. Código modular, dependencias fijadas, `.env.example`/`secrets.toml.example` sin secretos.
- D5. Diagrama, política de ejecución, aviso versionado, inventario de datos, purga.
- D6. Pruebas T-01 a T-25, cobertura, fixtures, mocks, tolerancias.
- D7. Evidencia anonimizada del flujo OIDC/consentimiento/sesión/outbox/worker/purga.
- D8. `AI_USAGE.md`.
- D9. `CONTRIBUTIONS.md`.
- D10. Dos exportaciones de ejemplo (1 activo; 20 activos + 1 inválido).
- D11. Video (máx. 5 min).
- D12. `TRACEABILITY.md` (este archivo) enlazado al commit final.

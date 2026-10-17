# Matriz de cumplimiento — Tutor Christian Johnson (15 de octubre de 2026)

| Observación | Estado de esta entrega (17/oct/2026) | Evidencia / acción |
|---|---|---|
| Corregir numeración 3.8 → 4 y siguientes | **Corregido en Word** | `ENEMDU_2025_Manuscrito_RF_VERIFICADO_17OCT2026.docx` se entrega aparte |
| RF definitivo 0,5308; no confundir con 0,5504 | **Resuelto en documentación y notebook exploratorio** | El 01 etiqueta todas las corridas nuevas como NO OFICIALES |
| Tabla A2 sensibilidad RF definitivo | **PARCIALMENTE RESUELTA — sensibilidad por clase auditada** | La Tabla A2 recoge recall del RF oficial basado en `pred_rf.npy`; el análisis de sensibilidad a umbrales sigue pendiente de probabilidades RF originales |
| Repositorio estable con código completo desde microdatos crudos | **PARCIAL: scripts recuperados** | Los scripts de todas las fases se recuperaron del ZIP y adaptaron a rutas portables; falta ejecutar y comprobar íntegramente el flujo |
| README, instrucciones, entorno, versiones | **PARCIAL** | README e instrucciones listas, captura de entorno provista; versiones históricas pendientes de verificar |
| Referencia Zenodo/OSF con DOI | **PENDIENTE** | Requiere versión validada y subida por el titular |
| Diferenciar estabilidad estadística de reproducibilidad computacional | **CORREGIDO EN TEXTO** | ARI=0,9905 designado estabilidad bootstrap |
| «demuestra» en introducción y aporte original | **CORREGIDO EN TEXTO** | «propone y aplica», aporte socioeconómico explícito |
| Delimitar concepto «protección económica» | **CORREGIDO EN TEXTO** | Dimensión económico-laboral y pensionaria |
| Reforzar discusión de política pública | **CORREGIDO EN TEXTO** | Discusión y recomendaciones actualizadas |
| Extensión a instrucciones de revista | **PENDIENTE DE CONOCER REVISTA** | No recortar arbitrariamente antes de elegir revista |
| Eval. independiente agrupada por posiciones del panel | **PENDIENTE** | Evitar sobreestimar generalización con split por filas; análisis prospectivo |

**Condición para cierre:** ZIP y scripts originales ya recuperados; RF validado desde predicciones; faltan probabilidades para umbrales, ejecutar y cotejar el pipeline completo, verificar tablas y documentar dependencias verdaderas; citar versión fija de GitHub y cerrar verificación integral.

## Comprobaciones efectivas desde el ZIP privado (actualización 17/10/2026)

Se contrastó el CSV preparado y las 45.959 etiquetas K=4 (coincidencia íntegra), las 25 estimaciones bootstrap (idénticas) y los 13.788 índices de TEST (idénticos). `pred_rf.npy` reproduce las métricas de RF auditadas. **El reentrenamiento actual sigue generando accuracy 0,5504 y F1-macro 0,4709**, diferentes de RF histórico 0,5308/0,4588. Esta discordancia es un **pendiente de reproducibilidad**, no un error tipográfico corregido por sustitución.

La Tabla A2 introducida es de sensibilidad por clase (recall); una **tabla de sensibilidad a variaciones de umbral** continúa pendiente de probabilidades de la ejecución histórica. No publicar material con datos individualizados.


## Cambio de cuenta de publicación

Destino autorizado solicitado por el autor: **`Julians30`**. La versión previa `romero-coronel/Heterogeneidad` es un repositorio de otro propietario sobre el que la conexión actual solo tiene lectura. El repositorio actual de `Julians30` es el destino de publicación. El artículo debe citar esta URL y preferiblemente un commit inmutable; no se ha asignado DOI.

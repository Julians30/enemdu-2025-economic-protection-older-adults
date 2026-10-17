# Matriz de cumplimiento — Tutor Christian Johnson (15 de octubre de 2026)

| Observación | Estado de esta entrega (17/oct/2026) | Evidencia / acción |
|---|---|---|
| Corregir numeración 3.8 → 4 y siguientes | **Corregido en Word** | `ENEMDU_2025_Manuscrito_RF_VERIFICADO_17OCT2026.docx` se entrega aparte |
| Métricas de RF histórico y trazabilidad de entrenamiento | **Métricas auditadas; reproducción exacta del ajuste pendiente** | Exactitud y F1 comprobados mediante predicciones archivadas; ejecuciones de contraste identificadas de forma separada |
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

Se contrastó el CSV preparado y las 45.959 etiquetas K=4 (coincidencia íntegra), las 25 estimaciones bootstrap (idénticas) y los 13.788 índices de TEST (idénticos). `pred_rf.npy` reproduce las métricas de RF auditadas. Una ejecución independiente con las dependencias disponibles produjo resultados distintos de los archivados (accuracy 0,5504 y F1-macro 0,4709 frente a 0,5308 y 0,4588, respectivamente). Este ensayo **no forma parte de la estimación de referencia del manuscrito**. Hasta recuperar el entorno y los parámetros efectivos del ajuste original, la reproducibilidad completa de esta etapa permanece sin demostrar. La documentación conserva la divergencia para permitir su investigación; no se confunden estimaciones de distintas ejecuciones.

La Tabla A2 introducida es de sensibilidad por clase (recall); una **tabla de sensibilidad a variaciones de umbral** continúa pendiente de probabilidades de la ejecución histórica. No publicar material con datos individualizados.



### Nota de procedencia de archivos históricos

En el ZIP original, el metadato de fecha del archivo `scripts/05_supervisado.py` es posterior al de `outputs/tablas/pred_rf.npy` y `outputs/tablas/metricas_rf.csv`. Estas marcas **no permiten afirmar que el script haya cambiado sustancialmente** (pueden verse afectadas por copias y empaquetado), pero tampoco acreditan que el código actualmente conservado sea la versión exacta que generó las predicciones originales. Recuperar el commit, registro de parámetros efectivos o artefacto del modelo entrenado permitiría fortalecer la reconstrucción.

## Ubicación del repositorio

Repositorio de código y documentación: **https://github.com/Julians30/enemdu-2025-economic-protection-older-adults**. Para trazabilidad bibliográfica, se recomienda citar una versión fija (commit o liberación archivada con DOI una vez validada). No se ha asignado DOI.

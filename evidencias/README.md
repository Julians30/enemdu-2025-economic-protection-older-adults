# Evidencias agregadas (sin microdatos)

`rf_original_auditoria.json` fue calculado desde el ZIP privado entregado por el autor, cruzando `pred_rf.npy` con las etiquetas del `split_test_idx.npy` y la base segmentada. Incluye hashes SHA-256, confusión y métricas agregadas. **No incluye microdatos, índices individuales ni predicciones unitarias.**

Las cifras de log-loss proceden únicamente del CSV histórico; no se recalcularon por falta de probabilidades oficiales por clase. La comparación de métricas no sustituye la reproducción del entrenamiento histórico.

`reproduccion_etapas_2026_10_17.json` enumera las etapas efectivamente contrastadas y separa la coincidencia de las salidas históricas de la reproducción del entrenamiento.

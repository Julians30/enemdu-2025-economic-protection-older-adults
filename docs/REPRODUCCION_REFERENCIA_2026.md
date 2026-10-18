# Certificación interna de reproducción de referencia (17 octubre 2026)

- Datos analíticos: 45.959 registros; etiquetas K=4 contrastadas con el ZIP histórico; ARI bootstrap (25 repeticiones) = 0,9905.
- TRAIN/TEST por filas fijos: 32.171/13.788. Esta partición **no elimina dependencia de panel**.
- Modelos: logit con escalado dentro de TRAIN; Random Forest con RandomizedSearchCV (3 configuraciones, 2 folds, submuestra TRAIN de 12.000); XGBoost 350 árboles con pesos inversos a prevalencia.
- Random Forest: max_depth=None, min_samples_leaf=5, n_estimators=250, random_state=42, n_jobs=1.
- Python 3.13.5, numpy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, XGBoost 3.1.3; OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1.
- Corridas independientes A y B: resultados completos de Tabla 3, vectores de predicción, matrices de probabilidades (logit/RF/XGB) y vectores OOF RF **idénticos byte a byte** en el mismo entorno controlado. Solo los agregados se publican.
- Evaluación avanzada: TRAIN-OOF 3 folds, selección isotónica en partición TRAIN-OOF 80/20, reajuste sobre OOF TRAIN; TEST nunca se usa para calibración o elección de umbral. Segunda evaluación avanzada coincidió exactamente.
- Los datos y artefactos por registro se preservan en un repositorio privado del autor; los resultados principales solo proceden de la ejecución de referencia documentada y repetida.
- Validación de independencia longitudinal: GroupShuffleSplit complementario n_train=32.237 y n_test=13.722, cero claves compartidas; no valida nueva segmentación fuera de muestra.

# Protocolo vigente — ENEMDU 2025

Para la **nueva versión supervisada de referencia** (auditada el 17 de octubre de 2026), seguir el [README](../README.md), instalar [versiones fijadas](../requirements_referencia_fija.txt) y ejecutar secuencialmente:

1. [Entrenamiento y sensibilidad RF](../scripts/06_modelos_referencia_reproducibles.py)
2. [Calibración y umbrales solo TRAIN-OOF](../scripts/07_evaluacion_avanzada_referencia.py)
3. [Figuras reproducibles](../scripts/08_figuras_avanzadas_referencia.py)

Los códigos 06–08 leen el ZIP privado y reutilizan las etiquetas K=4, ya contrastadas desde los microdatos ENEMDU mediante scripts 01–04. No compartir el ZIP, los identificadores, predicciones y probabilidades individuales ni modelos serializados. Los resultados públicos están en [evidencias](../evidencias/referencia_supervisada_2026/).

**Verificación:** dos ejecuciones independientes, en el entorno Python 3.13.5 con semilla 42 y un hilo, dieron archivos de predicción y matrices probabilísticas idénticos byte a byte. La etapa de evaluación avanzada también se repitió con resultados idénticos. Esta afirmación se limita al entorno y a la ejecución actual y no constituye validación externa de K-means.

**Prueba de dependencia de panel:** consultar [validación agrupada complementaria](VALIDACION_AGRUPADA_PANEL_2026.md). La partición agrupada no sustituye la Tabla 3: usa TRAIN y TEST diferentes y clusters derivados de toda la base.

**DOI:** sin DOI asignado por el momento; la recomendación de depositar una versión permanente no se presenta como completada.

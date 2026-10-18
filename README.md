# ENEMDU 2025 — Economic Protection of Older Adults in Ecuador

**Segmentación económico-laboral y pensionaria de personas de 65 años o más.** Estudio con K-means, evaluación supervisada de tres clasificadores, calibración, sensibilidad a umbrales y explicabilidad. Esta versión documenta una **nueva ejecución supervisada reproducible**, distinta de ciertos resultados históricos archivados.

## Resultados de referencia

- **Muestra y segmentación:** 45.959 registros analíticos; cuatro perfiles. El ARI bootstrap medio = **0,9905** informa estabilidad estadística, **no** validación externa ni reproducibilidad computacional por sí solo.
- **Clasificación supervisada:** partición de referencia por registros 70/30 (TRAIN = 32.171, TEST = 13.788), índices originales fijados, semilla 42. Métricas sobre TEST, **sin ponderación**:

| Clasificador | Accuracy | F1-macro | Log-loss |
|---|---:|---:|---:|
| Regresión logística (variables numéricas escaladas solo en TRAIN) | 0,5157 | 0,4500 | 1,0954 |
| Random Forest (250 árboles; hoja mínima 5) | **0,5504** | **0,4709** | **1,0236** |
| XGBoost (350 árboles; configuración registrada) | 0,5424 | 0,4676 | 1,0426 |

Estas métricas, las probabilidades y las predicciones **se reprodujeron de forma idéntica byte a byte en dos ejecuciones independientes con el entorno fijado y un hilo**. Las evaluaciones avanzadas también se repitieron con resultados idénticos. Solo se publican métricas agregadas.

### Contexto de trazabilidad histórica

Un archivo histórico anterior de Random Forest conserva **accuracy = 0,5308** y **F1-macro = 0,4588** (recalculables desde predicciones archivadas), pero no se ha recuperado la configuración efectiva del entrenamiento que los generó. Por tanto, **esos valores históricos no se mezclan con las tablas de referencia actuales ni se presenta su ajuste como reproducido**. Esta distinción se explica en el apéndice del manuscrito y la [matriz de observaciones](docs/ESTADO_OBSERVACIONES_TUTOR.md).

## Archivos para reproducir

```text
scripts/
  01...05                   preparación, elección K, clustering, bootstrap, supervisión histórica (conservados)
  06_modelos_referencia_reproducibles.py  modelos y sensibilidad RF
  07_evaluacion_avanzada_referencia.py    calibración TRAIN-OOF, umbrales y métricas
  08_figuras_avanzadas_referencia.py      gráficos XGBoost y SHAP
requirements_referencia_fija.txt         entorno auditado Python 3.13.5
evidencias/referencia_supervisada_2026/   tablas agregadas y protocolo auditado
```

**Importante:** los scripts 06–08 leen un ZIP privado que contiene la base analítica segmentada y el split original. Para comprobar también la obtención de esa base desde microdatos, ejecutar primero los scripts portables 01–04 documentados en `scripts/flujo_original_portable/`. Los microdatos se obtienen de manera legítima desde el INEC; **no** se redistribuyen en este repositorio.

### Ejecución de referencia (terminal local)

```bash
python -m pip install -r requirements_referencia_fija.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/06_modelos_referencia_reproducibles.py --zip /ruta/privada/enedmu_adultos_mayores.zip --out /ruta/privada/resultados_modelos
python scripts/07_evaluacion_avanzada_referencia.py --zip /ruta/privada/enedmu_adultos_mayores.zip --models /ruta/privada/resultados_modelos --out /ruta/privada/resultados_avanzados
python scripts/08_figuras_avanzadas_referencia.py --zip /ruta/privada/enedmu_adultos_mayores.zip --models /ruta/privada/resultados_modelos --avanzado /ruta/privada/resultados_avanzados --out /ruta/privada/figuras
```

La semilla es 42 y el flujo usa un único hilo. En Google Colab, la versión de Python del entorno hospedado puede diferir de la versión auditada: **no** inferir identidad byte a byte sin comprobarlo nuevamente.

### Sensibilidad de Random Forest y clase minoritaria

- La Tabla A2 recalculada reporta **recall y falsos negativos por segmento** de Random Forest de referencia.
- La Tabla A3 incluye **cambios de umbral de la clase 2** del mismo modelo, calculados a partir de probabilidades guardadas en forma privada; el umbral elegido se determinó por OOF de TRAIN, nunca sobre TEST.
- La Tabla 5 utiliza un calibrador isotónico seleccionado por Brier Score en una subdivisión de TRAIN-OOF; todos los modelos se reportan de forma coherente. El archivo de resultados públicos y el script describen el procedimiento.

## Límites y alcance

1. La partición principal es por **registros**, no disjunta por posiciones persona–hogar del panel; 6.362 registros TEST comparten clave base con TRAIN. Por separado, se ejecutó validación agrupada supervisada con **cero claves compartidas**. Ambas etiquetas de cluster fueron construidas en el conjunto analítico completo; esto **no** constituye validación externa de K-means.
2. Las claves base de panel se denominan **posiciones persona–hogar**, no individuos físicos únicos.
3. La protección económica observada se limita a las dimensiones **económico-laboral y pensionaria**; no mide todos los ingresos o recursos del hogar.
4. No publicar los microdatos ENEMDU privados, identificadores, índices individuales de partición, modelos serializados ni predicciones/probabilidades por registro. El acceso original a ENEMDU se rige por sus condiciones oficiales.
5. El repositorio publica código y resultados agregados; un DOI permanente (Zenodo/OSF) y la adaptación a una revista concreta dependen de posteriores decisiones editoriales.

**Tutor:** Christian Johnson. **Licencia:** MIT para el código propio sujeto a derechos de terceros; no aplica automáticamente a los microdatos del INEC.

# ENEMDU 2025 — Validación complementaria con separación por posiciones de panel

**Objetivo:** evaluar la sensibilidad de la etapa supervisada cuando ninguna clave base de `id_persona` aparece a la vez en TRAIN y TEST. Esta evaluación **NO sustituye** la partición original por registros del manuscrito ni certifica la reproducción exacta del entrenamiento histórico de Random Forest.

## Material de partida

- ZIP privado `enedmu_adultos_mayores.zip` provisto por los autores. El ZIP no se publica en este repositorio.
- Se leen los microdatos originales de personas ENEMDU 2025 y `outputs/adultos_mayores_con_cluster.csv`. Su correspondencia fila a fila se verifica mediante variables de control.
- La clave base de `id_persona` se obtiene eliminando el sufijo de mes de dos dígitos. **No equivale necesariamente a identidad física de persona**; se refiere a posiciones persona–hogar longitudinales.
- La solución K=4 y sus etiquetas se conservaron de la segmentación realizada en la muestra completa. Por tanto, el experimento aísla grupos solo en el entrenamiento supervisado y **NO es validación externa del K-means**.

## Diseño computacional

- `GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)` con claves base como grupos (la estratificación exacta no está garantizada).
- 45.959 registros; 31.101 claves base; 32.237 TRAIN; 13.722 TEST; **cero claves compartidas**.
- Modelos: logit multinomial balanceado con escalado `StandardScaler` de edad/educación aprendido exclusivamente en TRAIN; Random Forest 250 árboles, `max_depth=None`, `min_samples_leaf=5`, `class_weight=balanced`, semilla 42; XGBoost 350 árboles, profundidad 6, aprendizaje 0,05, submuestreo de observaciones/variables 0,8, pesos inversos a frecuencias de clase.
- Se conservan exactamente las variables predictoras sociodemográficas; no se utilizan ingresos, actividad laboral ni jubilación como predictores.
- Se calcula accuracy, F1-macro, F1-ponderado, kappa, log-loss y métricas por clase; solo se publica salida **agregada**.

## Resultados de esta nueva ejecución

| Modelo | Accuracy | F1-macro | F1-ponderado | Kappa | Log-loss |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0,5197 | 0,4570 | 0,5458 | 0,3381 | 1,0876 |
| Random Forest | 0,5454 | 0,4708 | 0,5622 | 0,3590 | 1,0329 |
| XGBoost | 0,5363 | 0,4641 | 0,5573 | 0,3523 | 1,0479 |

Estas métricas no deben compararse causalmente con los resultados de la prueba por registros porque se utilizaron partición y ajustes diferentes. Un resultado mejor en esta nueva prueba **no demuestra ausencia de optimismo ni corrige retroactivamente** el entrenamiento histórico. Se requiere validación externa para evaluar la generalización de la construcción de clusters en nuevas ediciones de ENEMDU.

## Ejecutar (en entorno privado)

```bash
python -m pip install -r requirements_validacion_2026_10_17.txt
python scripts/validacion_agrupada_panel.py --zip /ruta/privada/enedmu_adultos_mayores.zip --out /ruta/privada/resultados_validacion_agrupada
```

Las dependencias versionadas corresponden a la ejecución de auditoría de octubre de 2026; **no** son una reconstrucción documentada del entorno original de entrenamiento. Las diferencias de biblioteca/versión pueden modificar levemente las estimaciones.

**Resultados principales del manuscrito:** proceden exclusivamente de la ejecución supervisada de referencia (Tabla 3). Este análisis agrupado se reporta por separado y no constituye una segunda estimación sobre la partición TEST principal.

## Privacidad

Solo se difunden código, agregados y notas metodológicas: **no** `id_persona`, particiones por fila, microdatos, probabilidades o predicciones individualizadas.
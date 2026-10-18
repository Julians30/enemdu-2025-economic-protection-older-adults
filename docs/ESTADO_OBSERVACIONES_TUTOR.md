# Estado actualizado de observaciones — Christian Johnson (17/oct/2026)

## Trabajo realizado

| Observación | Estado | Verificación |
|---|---|---|
| Código integral e instrucciones | **ATENDIDA para nueva referencia** | Scripts portables de datos/clustering y `scripts/06…08`; entorno `requirements_referencia_fija.txt`; README actualizado |
| Semillas, configuración, versiones y salida reproducible | **ATENDIDA para nueva referencia supervisada** | Dos ejecuciones independientes, con predicciones y probabilidades byte a byte idénticas en Python 3.13.5, un hilo y semilla 42 |
| A2 sensibilidad RF | **ATENDIDA** | Tabla A2 por segmento; nueva Tabla A3 analiza escenarios de umbral y elección exclusiva en TRAIN-OOF |
| Estructura Resultados 4, Discusión 5, Conclusiones 6 y Referencias 7 | **ATENDIDA** | Manuscrito corregido |
| ARI ≠ reproducibilidad computacional | **ATENDIDA** | ARI bootstrap = 0,9905 indica estabilidad frente al remuestreo |
| Redacción «propone y aplica» y aporte original | **ATENDIDA** | Introducción y conclusión revisadas |
| Delimitación de protección económica | **ATENDIDA** | Alcance económico-laboral y pensionario |
| Interpretación económica y políticas públicas | **ATENDIDA** | Discusión y recomendaciones |
| Sensibilidad a dependencia longitudinal | **ATENDIDA como evaluación supervisada complementaria** | GroupShuffleSplit: 32.237 TRAIN / 13.722 TEST, cero claves compartidas; K-means definido sobre todo el conjunto |
| Extensión de la revista | **CONDICIONAL** | No se ha seleccionado la revista destinataria |
| Archivo permanente con DOI (Zenodo/OSF) | **RECOMENDACIÓN OPCIONAL PENDIENTE** | No inventar DOI; gestionar al depositar versión definitiva |

## Qué significa “reproducible” aquí

La **nueva evaluación supervisada de referencia** —no el ajuste anterior archivado— se reentrenó con configuración y entorno fijados. RF = accuracy **0,5504**, F1-macro **0,4709**; XGBoost = **0,5424/0,4676**; Logit = **0,5157/0,4500**. Los resultados probabilísticos, la calibración isotónica y la selección de umbrales proceden de las mismas ejecuciones y están documentados en [el protocolo](REPRODUCCION_REFERENCIA_2026.md). Los vectores individuales y modelos se conservan privados.

El Random Forest archivado con accuracy **0,5308** se verificó desde sus predicciones pero **no pudo reentrenarse exactamente** con los scripts/entorno históricos conservados. Sus cifras aparecen únicamente como antecedente metodológico en el apéndice, no como desempeño principal. No se debe afirmar que el ajuste histórico está reproducido.

## Límites científicos persistentes

La reproducibilidad interna de la nueva ejecución **no equivale a validación externa** del K-means, ni elimina el riesgo de dependencia en el split principal por registros. Las 31.101 claves base se denominan posiciones persona–hogar; no se afirman personas físicas únicas. Quedan a cargo del equipo autoral la selección de revista y, si lo desean, la generación de un DOI archivado.

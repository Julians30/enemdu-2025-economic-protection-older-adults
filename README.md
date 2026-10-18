# ENEMDU 2025 — Heterogeneidad socioeconómica y protección económica


## Objeto de la investigación

Segmentación en cuatro perfiles empíricos de **protección económica en su dimensión económico-laboral y pensionaria** de adultos mayores en Ecuador, seguida de clasificación a partir de predictores sociodemográficos. Fuente: microdatos ENEMDU 2025 (INEC), sujetos a sus condiciones oficiales de uso.

**Alcance de esta versión:** incluye la trazabilidad de la preparación de datos, la segmentación, la estabilidad y la evaluación histórica a partir de los artefactos conservados. La reconstrucción exacta del entrenamiento supervisado se documenta como una verificación metodológica todavía abierta; por ello no se presenta como una reproducción integral certificada.

## Repositorio del proyecto

Repositorio oficial de código del proyecto: **[Julians30/enemdu-2025-economic-protection-older-adults](https://github.com/Julians30/enemdu-2025-economic-protection-older-adults)**.
Este repositorio contiene código, documentación, configuraciones documentadas y **resultados agregados de auditoría**, pero no contiene los microdatos ENEMDU ni los artefactos por registro. **La publicación de código no certifica automáticamente la reproducibilidad del entrenamiento histórico.** Se prevé archivar una versión con DOI cuando se hayan cerrado las comprobaciones restantes.

## Resultados de referencia y verificación

El análisis documenta cuatro perfiles económico-laborales y pensionarios mediante K-means. En la evaluación supervisada histórica, las métricas de Random Forest fueron **accuracy = 0,5308**, **F1-macro = 0,4588**, **F1-ponderado = 0,5535** y **kappa = 0,3465**. Estas métricas fueron **recalculadas a partir de predicciones y etiquetas archivadas** para los 13.788 registros de prueba y no se confunden con un nuevo entrenamiento.

La preparación de datos, la segmentación y las comprobaciones de estabilidad se contrastaron con los artefactos originales. La **reproducción exacta del ajuste histórico de Random Forest** continúa bajo auditoría y no debe darse por certificada. La trazabilidad de ejecuciones y las diferencias entre ajustes independientes se explican en [la documentación técnica de auditoría](docs/ESTADO_OBSERVACIONES_TUTOR.md). Ningún nuevo ajuste sustituye automáticamente el resultado de referencia.

## Carpeta de notebooks

- `notebooks/01_ENEMDU_Evaluacion_Avanzada_CORREGIDA.ipynb`: permite realizar una evaluación experimental independiente y documentada. Sus métricas no sustituyen las del conjunto de artefactos históricos; exige la partición exacta del ZIP.
- `notebooks/02_Auditar_RF_definitivo_y_tabla_A2.ipynb`: lee el ZIP privado original y verifica el RF oficial **0,5308**, generando sensibilidad por clase y falsos negativos a partir de `pred_rf.npy`. Los umbrales alternativos siguen sin reconstruirse sin probabilidades originales.

**Todos los notebooks se entregan sin salidas preguardadas**, para evitar confusiones con resultados anteriores o publicación involuntaria de datos.

## Instrucciones rápidas (Google Colab)

1. Descargar y conservar de forma privada `enedmu_adultos_mayores.zip`. **No subirlo a GitHub**.
2. Abrir en Colab el notebook `01_...CORREGIDA.ipynb` e instalar paquetes en el entorno real (las dependencias exactas de la ejecución histórica todavía deben recuperarse).
3. Subir el ZIP cuando lo solicite la celda 2. Debe incluir `adultos_mayores_con_cluster.csv` y `split_test_idx.npy`. El cuaderno se detiene si falta el split exacto.
4. Ejecutar celdas secuencialmente. Los resultados generados por este notebook son una **nueva ejecución experimental**. Consultar los archivos de control para conocer su procedencia y no mezclarlos con los resultados archivados.
5. Si están disponibles las probabilidades TEST y TRAIN-OOF del RF definitivo, abrir el notebook `02_...tabla_A2.ipynb`. Con solo `metricas_rf.csv` **NO** es posible reconstruir la tabla A2.
6. Guardar las predicciones en carpeta privada, comprobando que TEST corresponde exactamente a 13.788 registros y que la selección de umbral usa solo TRAIN.
7. Ejecutar `python scripts/01_auditar_insumos.py /ruta/enedmu_adultos_mayores.zip` para auditoría de inventario y `python scripts/02_capturar_entorno.py --output_dir evidencias/entorno_real` en el **mismo entorno que ejecutó el análisis**. Verificar que la interfaz de ejecución coincide con la versión indicada en el manuscrito.
8. Los scripts de depuración, K-means, bootstrap, clasificación y SHAP ya se recuperaron; ejecutar el flujo en un entorno controlado, contrastar resultados, determinar las versiones históricas reales y resolver las discrepancias antes de declarar reproducibilidad integral.

## Organización

```
notebooks/         # 01 exploratorio corregido y 02 auditoría RF oficial
scripts/           # auditoría de ZIP, captura del entorno, validación local
config/            # valores históricamente documentados (no recalculados)
docs/              # observaciones del tutor y pendientes
requirements_por_verificar.in  # paquetes, todavía sin versión confirmada
```

## Control científico y ético

- La unidad observacional son **registros ENEMDU**, no 45.959 personas únicas.
- Las 31.101 claves base corresponden a **posiciones persona–hogar** del panel, no a identidades físicas comprobadas.
- En la partición inicial por filas, 6.362 registros TEST (46,14 %) comparten clave base con TRAIN; no asumir generalización a personas independientes.
- ARI bootstrap K=4 = 0,9905 mide **estabilidad del agrupamiento**, no reproducibilidad informática.
- No atribuir métricas ni gráficos del RF reentrenado a la ejecución definitiva.
- No publicar predicciones individuales, CSV analíticos, modelos serializados ni ZIP originales sin revisar bases legales, condiciones del INEC y privacidad.

## Validación complementaria agrupada por posiciones del panel

Se realizó una comprobación adicional de la fase supervisada con **GroupShuffleSplit** (semilla 42) sobre claves base de `id_persona` sin el sufijo de mes: 32.237 registros en TRAIN, 13.722 en TEST y **cero claves compartidas**. Los tres clasificadores se reentrenaron con atributos sociodemográficos y se publicaron únicamente **métricas agregadas**, código y metadatos (sin registros individuales).

Esta evaluación **no sustituye** el TEST histórico de la Tabla 3 ni valida externamente el K-means (las etiquetas se construyeron originalmente con toda la muestra). Consulta [protocolo, resultados e interpretación](docs/VALIDACION_AGRUPADA_PANEL_2026.md) y [script de ejecución](scripts/validacion_agrupada_panel.py). Se dispone también de un notebook de Google Colab en el Drive privado del proyecto.

## Estado de la Tabla A2

**Verificada como sensibilidad de clasificación por segmentos (recall por clase)** con los vectores originales `pred_rf.npy` y `split_test_idx.npy`. La sensibilidad a cambios de umbral no está demostrada por falta de probabilidades del RF original. La anterior A2 de XGBoost pasa a A3 en el manuscrito revisado.

## Referencia al repositorio

La URL pública corresponde a este repositorio. Para citar una versión inmutable, use el enlace al commit concreto de GitHub. El DOI de Zenodo/OSF todavía no existe y no se ha inventado.

## Evidencia de auditoría del archivo original (17/oct/2026)

- Se recibieron scripts originales, microdatos 2025, base segmentada e índices/predicciones históricos. **Los microdatos NO se incluyen aquí**.
- RF definitivo recalculado desde etiquetas: accuracy=**0,530751** (0,5308), F1-macro=**0,458765** (0,4588), F1-ponderado=**0,553518**, kappa=**0,346520**.
- Segmento minoritario 2: 324 casos reales, 163 verdaderos positivos, 161 falsos negativos, recall **50,31 %**, precisión **8,34 %**. La tabla A2 es **sensibilidad observada por clase** a la regla original, NO curva sensibilidad-umbral.
- `evidencias/rf_original_auditoria.json` guarda métricas agregadas, confusión y hashes. El log-loss **1,0464** solo consta en el archivo `metricas_rf.csv`: no fue posible recalcularlo desde etiquetas sin probabilidades del RF.
- `scripts/flujo_original_portable/` incluye versiones con rutas portables de los scripts recuperados (lógica de modelado conservada; no garantizan resultados idénticos bajo bibliotecas actuales).
- `requirements_validacion_2026_10_17.txt` especifica el entorno disponible para las comprobaciones de auditoría; NO identifica el entorno histórico.

### Comprobaciones y relanzamiento (no equivalen a reproducción integral certificada)

```bash
python -m pip install -r requirements_validacion_2026_10_17.txt
python scripts/01_auditar_insumos.py /ruta/enedmu_adultos_mayores.zip
python scripts/04_verificar_rf_oficial.py --zip /ruta/enedmu_adultos_mayores.zip --salida resultados_rf_auditoria
# Relanzamiento del proceso desde ENEMDU crudo en un directorio privado externo al repositorio:
python scripts/ejecutar_flujo_portable.py --zip /ruta/enedmu_adultos_mayores.zip --workspace /directorio_privado/enemdu_analisis --fase preparar
# Opciones posteriores: segmentar, bootstrap, supervisado_rf, completo.
```

El ejecutor crea **nuevos resultados en un workspace privado** y nunca debe reemplazar los vectores RF del ZIP original. El estudio usa partición original por registros; una evaluación con agrupamiento longitudinal es una extensión metodológica pendiente.


## Publicación por el autor

Consulte `docs/PROTOCOLO_REPRODUCIBILIDAD.md` antes de ejecutar o reutilizar este contenido. El repositorio **no** debe incluir `enedmu_adultos_mayores.zip`, la base analítica, predicciones individuales, contraseñas, tokens ni documentación con datos personales. El material publicable se limita a código, configuraciones, documentación y resultados **agregados**. Los scripts portables se proporcionan para validación, no como prueba de que el ajuste histórico de RF pueda reconstruirse idéntico.

## Licencia y procedencia

El repositorio incluye la licencia MIT seleccionada por su propietario para el código cuya redistribución está autorizada. La licencia **no** sustituye las condiciones oficiales de uso de los microdatos del INEC ni concede derechos sobre código de terceros sin su autorización. Compruebe los créditos de los colaboradores antes de reutilizar o relicenciar material ajeno.
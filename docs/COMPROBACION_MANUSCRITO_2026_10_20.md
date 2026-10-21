# Correspondencia manuscrito–repositorio (20 de octubre de 2026)

## Alcance

La comprobación corresponde al manuscrito sobre heterogeneidad socioeconómica y protección económica de adultos mayores en Ecuador a partir de la **ENEMDU Anual 2025**, cuya versión editorial reconoce a **José Julián Coronel Reyes** y **Fidel Eduardo Romero León** como autores.

Se trabajó con los artefactos del proyecto conservados en el espacio privado de los autores. **No se incluyen el ZIP original, microdatos, identificadores, índices de partición, modelos serializados ni predicciones individuales** en el repositorio público.

## Comprobaciones ejecutadas

| Objeto | Comprobación y resultado |
| --- | --- |
| Archivo anual de personas | 334.786 registros en el archivo original de referencia; 45.959 registros con edad ≥ 65 años |
| Expansión de la población analítica | Suma de `fexp` = **1.855.076,4174** en los registros de 65 años o más; estimación ponderada referida a ENEMDU Anual 2025 |
| Claves base de panel | 31.101 posiciones persona–hogar: 16.243 observadas una vez y 14.858 observadas dos veces; no son necesariamente personas físicas distintas |
| Partición principal por registros | 32.171 TRAIN / 13.788 TEST; **6.362** registros de TEST comparten clave base con TRAIN |
| Modelos principales, TEST por registros | Regresión logística accuracy **0,5157**, F1-macro **0,4500**; Random Forest **0,5504** / **0,4709**; XGBoost **0,5424** / **0,4676** |
| Validación supervisada agrupada | 32.237 TRAIN / 13.722 TEST, **cero** claves base compartidas |
| F1-macro en evaluación agrupada | Regresión logística **0,4570**; Random Forest **0,4708**; XGBoost **0,4641** |

Los modelos principales y la validación agrupada se volvieron a ejecutar a partir de los artefactos originales del estudio. Las métricas se cotejaron con los archivos agregados ya publicados en este repositorio:

- [Tabla 3, modelos canónicos](../evidencias/referencia_supervisada_2026/TABLA3_MODELOS_CANONICOS.csv).
- [Validación agrupada complementaria](../evidencias/validacion_agrupada_panel_2026.csv).
- [Protocolo de validación agrupada](VALIDACION_AGRUPADA_PANEL_2026.md).
- [Protocolo de reproducción de referencia](REPRODUCCION_REFERENCIA_2026.md).

## Criterios de interpretación

1. La suma de los pesos `fexp` justifica la cifra aritmética aproximada de **1,86 millones** utilizada en el texto, pero **no** debe interpretarse como el conteo de personas únicas entrevistadas. La inferencia poblacional está sujeta a las reglas de diseño, ponderación y estimación de la ENEMDU Anual 2025.
2. La segmentación K-means (`K=4`) se construyó inicialmente sobre el conjunto analítico completo. La evaluación agrupada separa claves base **solo en la fase supervisada** y no constituye validación externa de los clusters.
3. Los resultados de TRAIN/TEST original y de prueba agrupada provienen de particiones distintas. No son comparaciones pareadas ni constituyen evidencia causal.
4. La verificación realizada corresponde al material original privado y al entorno controlado de referencia. El control automatizado de GitHub verifica sintaxis y coherencia de agregados, **no** reentrena los modelos y **no** certifica reproducibilidad integral a partir de una descarga pública nueva del INEC.
5. Los archivos históricos o versiones previas no deben mezclarse con la nueva ejecución canónica de Tabla 3. El repositorio no tiene un DOI asignado; cualquier depósito futuro debe corresponder a una versión cerrada y verificable.

## Autoría y citación

**Autores del manuscrito:** José Julián Coronel Reyes ([ORCID](https://orcid.org/0000-0002-7883-5388)) y Fidel Eduardo Romero León ([ORCID](https://orcid.org/0009-0007-9825-1850)); **afiliación compartida:** Escuela de Posgrado, Maestría en Economía con mención en Inteligencia Artificial Aplicada, Universidad Estatal de Milagro (UNEMI), Ecuador.

El manuscrito aún no dispone de datos bibliográficos definitivos (revista, volumen, páginas o DOI), por lo que **no** deben inventarse ni publicarse como confirmados. La autoría del artículo no implica por sí sola la asignación de autoría individual sobre todas las líneas de software del repositorio.

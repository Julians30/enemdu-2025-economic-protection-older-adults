# Protocolo de reproducción y verificación (ENEMDU 2025)

**Repositorio:** https://github.com/Julians30/enemdu-2025-economic-protection-older-adults

## Fuentes y resguardo

1. Descargue los microdatos ENEMDU 2025 desde fuentes oficiales del INEC, cumpliendo los términos de acceso y uso aplicables. Los microdatos y predicciones individuales **no se publican** en este repositorio.
2. Conserve el paquete original `enedmu_adultos_mayores.zip` en una carpeta privada fuera del repositorio. Este paquete incluye archivos intermedios y salidas históricas que no equivalen al dataset distribuible.
3. Trabaje en un entorno privado de Python. `requirements_validacion_2026_10_17.txt` recoge versiones utilizadas para algunas comprobaciones, **no** las versiones históricas originales; `requirements.txt` es un entorno de trabajo sugerido, no un lockfile históricamente validado.

## Verificación trazable

```bash
python -m pip install -r requirements_validacion_2026_10_17.txt
python scripts/01_auditar_insumos.py /RUTA_PRIVADA/enedmu_adultos_mayores.zip
python scripts/04_verificar_rf_oficial.py --zip /RUTA_PRIVADA/enedmu_adultos_mayores.zip --salida /RUTA_PRIVADA/auditoria_rf
python scripts/ejecutar_flujo_portable.py --zip /RUTA_PRIVADA/enedmu_adultos_mayores.zip --workspace /RUTA_PRIVADA/analisis --fase preparar
python scripts/ejecutar_flujo_portable.py --zip /RUTA_PRIVADA/enedmu_adultos_mayores.zip --workspace /RUTA_PRIVADA/analisis --fase segmentar
python scripts/ejecutar_flujo_portable.py --zip /RUTA_PRIVADA/enedmu_adultos_mayores.zip --workspace /RUTA_PRIVADA/analisis --fase bootstrap
```

Use el ejecutor `--fase supervisado_rf` exclusivamente para contrastar **otra ejecución**; **no** es la reproducción certificada de Random Forest 0,5308. No mezcle datos generados con los vectores y métricas históricas. Para análisis de sensibilidad a umbrales RF, deben recuperarse probabilidades OOF TRAIN y TEST del RF histórico, **no** utilizar probabilidades del modelo reentrenado.

## Comprobaciones realizadas y limitaciones

- Preparación: 45.959 registros ENEMDU 65+, verificados contra los datos históricos aportados.
- K=4: se verificó coincidencia íntegra de las 45.959 etiquetas y 25 réplicas bootstrap (ARI promedio 0,9905).
- Partición: 13.788 registros TEST; corresponde al split original **por filas**, susceptible de dependencia longitudinal entre train/test.
- Predicciones históricas RF: accuracy 0,5308; F1-macro 0,4588; kappa 0,3465, recalculados de las etiquetas. Log-loss 1,0464 consta en un archivo histórico y **no** puede recalcularse sin probabilidades.
- Reentrenamiento posterior: accuracy 0,5504; F1-macro 0,4709. La diferencia con las salidas originales no se ha conciliado.

**Estado editorial:** material de auditoría y código público disponibles; reproducibilidad completa del resultado RF original y sensibilidad a umbrales pendientes. Referenciar mediante enlace a commit específico después de publicarse, sin atribuir DOI inexistente.

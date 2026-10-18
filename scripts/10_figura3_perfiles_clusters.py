# -*- coding: utf-8 -*-
"""Figura 3 | ENEMDU 2025: perfiles de clusters con normalizacion original 0-1.

Muestra una matriz de perfiles y una alternativa en paneles. Requiere, en un ZIP
privado, outputs/adultos_mayores_con_cluster.csv de la ejecucion original.
No redistribuir este ZIP ni los microdatos en GitHub.
"""
from pathlib import Path
import io
import json
import zipfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd

# Solo para reproducir valores de la figura historica: medias sin fexp.
COLUMNAS = {
    "p72a_bin": "Recibe jubilación",
    "empleo_bin": "Con empleo",
    "p72b_num_w": "Monto de jubilación",
    "ingrl_num_w": "Ingreso laboral",
}
SEGMENTOS = {
    1: "Jubilados con escasa actividad laboral",
    2: "Trabajadores de mayores ingresos",
    3: "Trabajadores de menores ingresos",
    4: "Inactivos sin pensión ni empleo",
}
COLORES = {1: "#21618C", 2: "#008F89", 3: "#D78C2B", 4: "#8064A2"}
ETIQUETAS = ["Recibe\njubilación", "Con\nempleo", "Monto de\njubilación", "Ingreso\nlaboral"]


def leer_zip(zip_archivo):
    zip_archivo = Path(zip_archivo)
    if not zip_archivo.exists():
        raise FileNotFoundError(f"No se encuentra el ZIP: {zip_archivo}")
    with zipfile.ZipFile(zip_archivo) as z:
        candidatos = [n for n in z.namelist() if n.endswith("/outputs/adultos_mayores_con_cluster.csv") or n == "adultos_mayores_con_cluster.csv"]
        if len(candidatos) != 1:
            raise ValueError(f"Debe existir exactamente un outputs/adultos_mayores_con_cluster.csv; encontrados: {candidatos}")
        with z.open(candidatos[0]) as f:
            tabla = pd.read_csv(f, usecols=[*COLUMNAS.keys(), "cluster"])
    return tabla


def calcular_perfiles(df):
    requeridas = list(COLUMNAS) + ["cluster"]
    if any(c not in df for c in requeridas):
        raise ValueError("Columnas faltantes: " + str([c for c in requeridas if c not in df]))
    if df[requeridas].isna().any().any():
        raise ValueError("Existen datos faltantes en las variables de segmentación o los identificadores de cluster")
    ids = set(df["cluster"].astype(int).unique())
    if ids != {1, 2, 3, 4}:
        raise ValueError(f"Se esperaban los clusters 1–4; aparecen: {ids}")
    for c in ("p72a_bin", "empleo_bin"):
        if not df[c].isin([0, 1]).all():
            raise ValueError(f"{c} debe ser una variable binaria 0/1")
    if (df[["p72b_num_w", "ingrl_num_w"]] < 0).any().any():
        raise ValueError("Los montos usados en el radar deben ser no negativos")
    # Reproduce la representacion del archivo 03_clustering.py original:
    # variables binarias -> promedio (proporción); montos winsorizados ->
    # promedio por cluster / maximo global de la variable winsorizada.
    medias = df.groupby("cluster")[list(COLUMNAS)].mean().reindex([1, 2, 3, 4])
    perfiles = medias.copy()
    maximos = {}
    for c in ("p72b_num_w", "ingrl_num_w"):
        maximo = float(df[c].max())
        if not np.isfinite(maximo) or maximo <= 0:
            raise ValueError(f"Máximo inválido para {c}: {maximo}")
        perfiles[c] = medias[c] / maximo
        maximos[c] = maximo
    if (perfiles.to_numpy() < -1e-12).any() or (perfiles.to_numpy() > 1 + 1e-12).any():
        raise ValueError("Los valores normalizados deben situarse en [0, 1]")
    tamanos = df.groupby("cluster").size().reindex([1, 2, 3, 4]).astype(int)
    return perfiles, tamanos, maximos


def _guardar(fig, carpeta, nombre):
    # Guarda 600 dpi y versiones vectoriales; sin mostrar perfiles por registro.
    out = []
    for ext in ("png", "pdf", "svg"):
        destino = Path(carpeta) / f"{nombre}.{ext}"
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(destino, facecolor="white", bbox_inches="tight", pad_inches=0.30, **kwargs)
        out.append(destino)
    plt.close(fig)
    return out


def figura_heatmap(perfiles, tamanos, carpeta):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(14.8, 7.6), facecolor="white")
    fig.text(0.055, 0.945, "ENEMDU 2025   /   SEGMENTACIÓN SOCIOECONÓMICA", fontsize=11, color="#4A6A7D", weight="bold")
    fig.text(0.055, 0.885, "Perfiles normalizados de los cuatro clusters", fontsize=21, weight="bold", color="#162D42")
    fig.text(0.055, 0.838, "Variables originales de segmentación  ·  K-means (K = 4)  ·  N = 45.959 registros", fontsize=11.2, color="#516372")
    ax = fig.add_axes([0.34, 0.235, 0.515, 0.51])
    valores = perfiles[list(COLUMNAS)].to_numpy()
    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(valores, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(4), labels=ETIQUETAS)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=14, labelsize=11.5)
    def etiqueta(i):
        return f"S{i}  {SEGMENTOS[i]}\n      n = {tamanos[i]:,}".replace(",", ".")
    ax.set_yticks(range(4), labels=[etiqueta(i) for i in range(1, 5)])
    ax.tick_params(axis="y", length=0, labelsize=11, pad=19)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for edge in ax.spines.values():
        edge.set_visible(False)
    for i in range(4):
        for j in range(4):
            val = valores[i, j]
            ax.text(j, i, f"{val:.3f}".replace(".", ","), ha="center", va="center",
                    fontsize=15.5, color=("white" if val >= .53 else "#152B3B"), weight="bold")
    bar = fig.add_axes([0.885, 0.26, 0.019, 0.46])
    cb = fig.colorbar(im, cax=bar, ticks=[0, .25, .5, .75, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10, length=0, pad=8)
    cb.ax.set_yticklabels(["0,00", "0,25", "0,50", "0,75", "1,00"])
    cb.ax.set_title("Escala\n0–1", fontsize=10.5, color="#445E70", pad=11)
    fig.text(0.055, 0.132, "Nota. Las variables binarias muestran la proporción media dentro de cada cluster (0–1).", fontsize=10.25, color="#4D5D6D")
    fig.text(0.055, 0.097, "Los montos corresponden a medias de valores winsorizados al P99 divididas por el máximo global respectivo.", fontsize=10.25, color="#4D5D6D")
    fig.text(0.055, 0.062, "Gráfico descriptivo no ponderado. Esta escala 0–1 no equivale a los centroides z-score empleados para ajustar K-means.", fontsize=10.25, color="#4D5D6D")
    return _guardar(fig, carpeta, "Figura_3_Clusters_Heatmap_ENEMDU2025")


def figura_facetas(perfiles, tamanos, carpeta):
    fig, axs = plt.subplots(2, 2, figsize=(15.5, 9), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.965, top=0.74, bottom=0.16, hspace=0.43, wspace=0.32)
    fig.text(0.055, 0.948, "ENEMDU 2025   /   SEGMENTACIÓN SOCIOECONÓMICA", fontsize=11, color="#4A6A7D", weight="bold")
    fig.text(0.055, 0.900, "Perfil de cada cluster en las variables de segmentación", fontsize=19, weight="bold", color="#162D42")
    fig.text(0.055, 0.853, "Valores normalizados en escala común de 0 a 1  ·  K-means (K = 4)", fontsize=11, color="#516372")
    labs = ["Jubilación (sí)", "Empleo (sí)", "Monto de jubilación", "Ingreso laboral"]
    for idx, ax in enumerate(axs.ravel(), start=1):
        vals = perfiles.loc[idx, list(COLUMNAS)].to_numpy(dtype=float)
        pos = np.arange(4)
        color = COLORES[idx]
        bars = ax.barh(pos, vals, height=0.52, color=color, edgecolor="none", zorder=3)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.135)
        ax.set_yticks(pos, labels=labs)
        ax.tick_params(axis="y", labelsize=10.5, length=0, pad=7)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0", "0,25", "0,50", "0,75", "1"], fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5EAF0", linewidth=1)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"S{idx}  {SEGMENTOS[idx]}\n n = {int(tamanos[idx]):,}".replace(",", "."), loc="left", fontsize=11.3, weight="bold", color="#223C4A", pad=12)
        for rect, v in zip(bars, vals):
            ax.text(v + .024, rect.get_y() + rect.get_height() / 2, f"{v:.3f}".replace(".", ","), va="center", ha="left", fontsize=10.6, weight="bold", color="#283E51")
    fig.text(0.055, 0.070, "Normalización conforme al radar original: medias de variables binarias y medias de montos winsorizados al P99 / máximo global.", fontsize=10.3, color="#4D5D6D")
    fig.text(0.055, 0.038, "Perfiles descriptivos no ponderados; la estandarización z-score corresponde únicamente al ajuste de K-means.", fontsize=10.3, color="#4D5D6D")
    return _guardar(fig, carpeta, "Figura_3_Clusters_Paneles_ENEMDU2025")


def generar_figura3(ruta_zip, carpeta_salida):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    df = leer_zip(ruta_zip)
    perfiles, tamanos, maximos = calcular_perfiles(df)
    archivos = figura_heatmap(perfiles, tamanos, carpeta_salida)
    archivos += figura_facetas(perfiles, tamanos, carpeta_salida)
    resumen = {
        "figura": "Figura 3 — perfiles normalizados de las variables de segmentación",
        "n_registros": int(len(df)),
        "n_por_cluster": {str(i): int(tamanos[i]) for i in range(1,5)},
        "variables": list(COLUMNAS),
        "regla": "binarias: media; montos winsorizados P99: media / máximo global de variable winsorizada; sin ponderar",
        "maximos_denominadores_usd": maximos,
        "valores_normalizados": {str(i): {c: round(float(perfiles.loc[i,c]),9) for c in COLUMNAS} for i in range(1,5)},
        "advertencia": "No presentar los valores 0–1 como centroides z-score de K-means."
    }
    evidencia = carpeta_salida / "metadatos_figura3.json"
    evidencia.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    archivos.append(evidencia)
    caption = carpeta_salida / "pie_figura_3.txt"
    caption.write_text(
        "Figura 3. Perfiles de los cuatro segmentos en las variables normalizadas de segmentación (ENEMDU 2025).\n\n"
        "Nota. Los valores representan proporciones medias de las variables binarias de jubilación y empleo; "
        "para los montos de jubilación e ingreso laboral, las medias de los valores winsorizados al percentil 99 "
        "se dividieron por el máximo observado en cada variable winsorizada. "
        "Se reproduce la transformación descriptiva 0–1 de la figura original; "
        "el entrenamiento de K-means se efectuó sobre variables estandarizadas mediante z-score. "
        "Los perfiles mostrados no están ponderados por fexp. Fuente: elaboración propia con base en ENEMDU 2025.",
        encoding="utf-8"
    )
    archivos.append(caption)
    return archivos, perfiles, tamanos


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Reproduce la Figura 3 de ENEMDU desde el ZIP privado del estudio')
    ap.add_argument('--zip', required=True, help='ZIP original almacenado fuera del repositorio público')
    ap.add_argument('--out', required=True, help='Directorio privado para las figuras y metadatos')
    args = ap.parse_args()
    files, perfiles, sizes = generar_figura3(args.zip, args.out)
    print('Registros:', int(sizes.sum()), 'Clusters:', len(sizes))
    print('Figuras y metadatos generados:')
    for f in files: print(' -', f)
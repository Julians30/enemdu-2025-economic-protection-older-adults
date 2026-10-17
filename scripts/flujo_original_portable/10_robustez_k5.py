from pathlib import Path
# -*- coding: utf-8 -*-
"""Robustez: K=5. Compara la partición K=5 con la K=4 del artículo."""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
OUT = os.path.join(RUTA, 'outputs')
TAB = os.path.join(OUT, 'tablas')

am = pd.read_csv(os.path.join(OUT, 'adultos_mayores_preparado.csv'))
k4 = pd.read_csv(os.path.join(OUT, 'adultos_mayores_con_cluster.csv'))
assert len(am) == len(k4)

Xc = am[['p72a_bin', 'p72b_num_w', 'empleo_bin', 'ingrl_num_w']].copy()
Xc.columns = ['jubila_bin', 'monto_jub', 'empleo_bin', 'ingr_lab']
X_scaled = StandardScaler().fit_transform(Xc)

km5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_scaled)
lab5 = km5.labels_

tab = pd.crosstab(lab5, k4['cluster'], margins=False)
tab.to_csv(os.path.join(TAB, 'tabla_robustez_k5_cruzada.csv'),
           encoding='utf-8-sig')
print('=== Cruzada K=5 (filas) vs K=4 (columnas) ===')
print(tab)

# Perfil de los 5 clusters en escala original
Xc['k5'] = lab5
prof = Xc.groupby('k5')[['jubila_bin', 'monto_jub', 'empleo_bin',
                         'ingr_lab']].mean().round(3)
prof['n'] = Xc.groupby('k5').size()
print('\n=== Centros K=5 (medias, escala original) ===')
print(prof.to_string())

# ¿Cuál segmento K=4 se subdivide?
Xc['k4'] = k4['cluster']
sub = Xc.groupby(['k4', 'k5']).size().unstack(fill_value=0)
sub.to_csv(os.path.join(TAB, 'tabla_robustez_k5_subdivision.csv'),
           encoding='utf-8-sig')
print('\n=== Subdivisión de cada segmento K=4 por K=5 ===')
print(sub)

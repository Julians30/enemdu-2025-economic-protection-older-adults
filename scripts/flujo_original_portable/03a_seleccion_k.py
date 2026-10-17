from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Selección de K (codo, silueta, Davies-Bouldin).
Silhouette/DB se calculan sobre submuestra fija (costo O(n^2)).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
am = pd.read_csv(os.path.join(RUTA, 'outputs', 'adultos_mayores_preparado.csv'))
FIG = os.path.join(RUTA, 'outputs', 'figuras')
TAB = os.path.join(RUTA, 'outputs', 'tablas')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

Xc = am[['p72a_bin','p72b_num_w','empleo_bin','ingrl_num_w']].copy()
Xc.columns = ['jubila_bin','monto_jub','empleo_bin','ingr_lab']
X_scaled = StandardScaler().fit_transform(Xc)

rng_sel = np.random.RandomState(42)
idx_sel = rng_sel.choice(X_scaled.shape[0], size=8000, replace=False)
X_sel = X_scaled[idx_sel]

K_range = range(2, 11)
inertias, sil_scores, db_scores = [], [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labs = km.fit_predict(X_scaled)
    labs_sel = km.predict(X_sel)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sel, labs_sel))
    db_scores.append(davies_bouldin_score(X_sel, labs_sel))
    print(f"  K={k} | inercia={km.inertia_:.1f} | silueta={sil_scores[-1]:.4f} | DB={db_scores[-1]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(list(K_range), inertias, 'bo-')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inercia'); axes[0].set_title('Método del codo')
axes[1].plot(list(K_range), sil_scores, 'ro-')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Coeficiente de silueta'); axes[1].set_title('Silueta')
axes[2].plot(list(K_range), db_scores, 'go-')
axes[2].set_xlabel('K'); axes[2].set_ylabel('Davies-Bouldin'); axes[2].set_title('Davies-Bouldin (menor=mejor)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_seleccion_k.png'), dpi=300); plt.close()

sel_k = pd.DataFrame({'K': list(K_range), 'Inercia': inertias, 'Silueta': sil_scores, 'Davies_Bouldin': db_scores}).round(4)
sel_k.to_csv(os.path.join(TAB,'tabla_seleccion_k.csv'), index=False, encoding='utf-8-sig')
print("\n=== SELECCIÓN DE K ===\n", sel_k.to_string(index=False))

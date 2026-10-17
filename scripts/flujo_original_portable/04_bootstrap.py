from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Estabilidad del clustering (bootstrap ARI).
Ejecutar después de 03_clustering.py. Usa las mismas variables escaladas.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import os

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
TAB = os.path.join(RUTA, 'outputs', 'tablas')
os.makedirs(TAB, exist_ok=True)

am = pd.read_csv(os.path.join(RUTA, 'outputs', 'adultos_mayores_preparado.csv'))
Xc = am[['p72a_bin','p72b_num_w','empleo_bin','ingrl_num_w']].copy()
Xc.columns = ['jubila_bin','monto_jub','empleo_bin','ingr_lab']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xc)

K = 4
rng = np.random.RandomState(42)
n_bs = 25
n_sub = 8000
idx_sub = rng.choice(len(X_scaled), size=n_sub, replace=False)
X_sub = X_scaled[idx_sub]

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_sub)
labels = kmeans.predict(X_sub)

ari_list = []
for b in range(n_bs):
    idx = rng.choice(len(X_sub), size=len(X_sub), replace=True)
    km_b = KMeans(n_clusters=K, random_state=42, n_init=3).fit(X_sub[idx])
    labels_bs = km_b.predict(X_sub)
    ari_list.append(adjusted_rand_score(labels, labels_bs))
    if (b+1) % 5 == 0:
        print(f"  bootstrap {b+1}/{n_bs} | ARI medio: {np.mean(ari_list):.4f}")

print(f"\n=== ESTABILIDAD BOOTSTRAP (ARI) ===\nMedia: {np.mean(ari_list):.4f} | SD: {np.std(ari_list):.4f} | Min: {np.min(ari_list):.4f}")
pd.DataFrame({'ARI_bootstrap': ari_list}).to_csv(os.path.join(TAB,'tabla_bootstrap_ari.csv'), index=False)
print("Guardado: outputs/tablas/tabla_bootstrap_ari.csv")

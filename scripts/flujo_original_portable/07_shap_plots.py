from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Fase 3 (paso 2): figuras SHAP a partir de shap_values_xgb.npz.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
TAB = os.path.join(RUTA, 'outputs', 'tablas')
FIG = os.path.join(RUTA, 'outputs', 'figuras')
os.makedirs(FIG, exist_ok=True)

z = np.load(os.path.join(TAB, 'shap_values_xgb.npz'), allow_pickle=True)
sv = z['sv']                     # (4, n, f)
y = z['y']
feat_names = list(z['features'])

nombres = {1: '(1) Jubilados sin actividad laboral', 2: '(2) Trabajadores de mayores ingresos',
           3: '(3) Trabajadores de menores ingresos', 4: '(4) Inactivos sin pensión ni empleo'}
colores = {1: '#2E86AB', 2: '#F4A261', 3: '#E76F51', 4: '#6A994E'}

# Bar chart global
df_imp = pd.read_csv(os.path.join(TAB, 'tabla_shap_importancia.csv'), index_col=0)
plt.figure(figsize=(9, 7))
df_imp.head(12).iloc[::-1]['global'].plot(kind='barh', color='#2E86AB')
plt.xlabel('Media |SHAP| (impacto global)')
plt.title('Importancia SHAP global — XGBoost (4 clusters)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_shap_importancia_global.png'), dpi=300); plt.close()

# Beeswarm por cluster (2x2), submuestra de 600 filas para render
rng = np.random.RandomState(7)
idx_plot = rng.choice(sv.shape[1], size=600, replace=False)
# La correspondencia se recupera con los índices globales guardados por 06_shap; no usar filas iniciales de TEST.
# reusamos los valores SHAP como proxy solo para colorear por signo con datos reales:
# necesitamos los valores de las features; se recalculan en 06_shap (guardar X). Aquí
# se cargan desde el pipeline:
import sys
from importlib import util
spec = util.spec_from_file_location('sup', os.path.join(os.path.dirname(__file__), '05_supervisado.py'))
sup = util.module_from_spec(spec); spec.loader.exec_module(sup)
df, X, y_full = sup.construir_features()
_, idx_test = sup.split_cv(X, y_full)
idx_global_sh = z['idx_test']  # índices globales de las 1500 observaciones usadas para SHAP
X_sh = X.iloc[idx_global_sh].iloc[idx_plot]
assert list(X_sh.columns) == feat_names, "nombres de features no coinciden"

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
for i, c in enumerate([1, 2, 3, 4]):
    ax = axes[i // 2][i % 2]
    shap.summary_plot(sv[c - 1][idx_plot], X_sh, feature_names=feat_names,
                      show=False, max_display=12, plot_size=None)
    plt.sca(ax)
    ax.set_title(f'Cluster {c}: {nombres[c]}', color=colores[c])
plt.suptitle('SHAP beeswarm por cluster — ENEMDU 2025 (adultos mayores)', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(FIG, 'fig_shap_beeswarm_clusters.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Figuras SHAP guardadas en outputs/figuras/")

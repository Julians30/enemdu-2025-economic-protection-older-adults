from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Fase 3: Interpretación SHAP del mejor modelo (XGBoost).
Paso 1: calcula SHAP sobre submuestra de test y guarda npz + tabla de importancia.
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
TAB = os.path.join(RUTA, 'outputs', 'tablas')

import sys
from importlib import util
spec = util.spec_from_file_location('sup', os.path.join(os.path.dirname(__file__), '05_supervisado.py'))
sup = util.module_from_spec(spec)
spec.loader.exec_module(sup)

df, X, y = sup.construir_features()
_, idx_test = sup.split_cv(X, y)
Xte = X.iloc[idx_test]
yte = y[idx_test]

model = joblib.load(os.path.join(TAB, 'modelo_xgb.joblib'))

# Submuestra fija de test para SHAP
rng = np.random.RandomState(42)
idx_sh = rng.choice(len(Xte), size=1500, replace=False)
X_sh = Xte.iloc[idx_sh]
y_sh = yte[idx_sh]

explainer = shap.TreeExplainer(model)
sv = np.asarray(explainer.shap_values(X_sh))  # (n, f, 4)
sv = np.moveaxis(sv, -1, 0)                   # (4, n, f)
print("SHAP shape:", sv.shape)

# Importancia global: media de |SHAP| por clase y global
feat_names = list(X_sh.columns)
importancia = {}
for c in range(4):
    importancia[f'cluster_{c+1}'] = np.abs(sv[c]).mean(axis=0)
importancia['global'] = np.abs(sv).mean(axis=(0, 1))
df_imp = pd.DataFrame(importancia, index=feat_names).sort_values('global', ascending=False)
df_imp.to_csv(os.path.join(TAB, 'tabla_shap_importancia.csv'), encoding='utf-8-sig')
print("\n=== IMPORTANCIA SHAP (media |SHAP|) ===\n", df_imp.head(12).round(4).to_string())

np.savez(os.path.join(TAB, 'shap_values_xgb.npz'), sv=sv, y=y_sh,
         idx_test=idx_test[idx_sh], features=np.array(feat_names, dtype=str))
print("\nGuardado: outputs/tablas/shap_values_xgb.npz + tabla_shap_importancia.csv")

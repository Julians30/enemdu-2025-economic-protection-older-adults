from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Fase 2: Clasificación supervisada de los 4 clusters.
Predictores: sociodemográficos (NO incluye variables de ingreso/empleo usadas en clustering).
Uso:
  python 05_supervisado.py logit
  python 05_supervisado.py rf
  python 05_supervisado.py xgb
  python 05_supervisado.py comparar
"""
import argparse, os, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             log_loss)
import statsmodels.stats.contingency_tables as sct

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
OUT = os.path.join(RUTA, 'outputs')
TAB = os.path.join(OUT, 'tablas')
os.makedirs(TAB, exist_ok=True)

def construir_features():
    df = pd.read_csv(os.path.join(OUT, 'adultos_mayores_con_cluster.csv'))
    df = df.dropna(subset=['p02','p03','p04','p10a_edu','area','prov','p15']).reset_index(drop=True)
    y = df['cluster'].astype(int).values
    X = pd.DataFrame(index=df.index)
    X['sexo_mujer'] = (df['p02'] == 2).astype(int)
    X['edad'] = df['p03'].astype(float)
    X['jefe_hogar'] = (df['p04'] == 1).astype(int)
    X['educ'] = df['p10a_edu'].astype(float)
    X['area_rural'] = (df['area'] == 2).astype(int)
    etnia = pd.get_dummies(df['p15'].astype(str), prefix='etnia', dtype=int)
    prov = pd.get_dummies(df['prov'].astype(str), prefix='prov', dtype=int)
    X = pd.concat([X, etnia, prov], axis=1)
    return df, X, y

def split_cv(X, y):
    idx_file = os.path.join(TAB, 'split_test_idx.npy')
    if os.path.exists(idx_file):
        idx_test = np.load(idx_file)
        if idx_test.shape != (13788,) or len(set(idx_test.tolist()))!=13788:
            raise ValueError('Partición guardada incompatible; abortar.')
    else:
        _, idx_test = train_test_split(np.arange(len(y)), test_size=0.30,
                                       stratify=y, random_state=42)
        np.save(idx_file, idx_test)
    idx_train = np.setdiff1d(np.arange(len(y)), idx_test)
    return idx_train, idx_test

def metricas(y_true, y_pred, y_proba):
    return {'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_proba, labels=[1, 2, 3, 4])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['logit','rf','xgb','comparar'])
    args = ap.parse_args()

    if args.model == 'comparar':
        comparar()
        return

    t0 = time.time()
    df, X, y = construir_features()
    idx_train, idx_test = split_cv(X, y)
    Xtr, Xte = X.iloc[idx_train], X.iloc[idx_test]
    ytr, yte = y[idx_train], y[idx_test]
    print(f"Train={len(Xtr):,} Test={len(Xte):,} | y_test: {pd.Series(yte).value_counts().to_dict()}")

    rng = np.random.RandomState(42)
    wtr = pd.Series(ytr).value_counts(normalize=True).to_dict()
    sw = np.array([1.0/wtr[v] for v in ytr])
    sub = rng.choice(len(Xtr), size=min(12000, len(Xtr)), replace=False)

    if args.model == 'logit':
        best = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0,
                                  class_weight='balanced', n_jobs=2)
        print("  [logit] C=1.0 (ajuste directo, sin búsqueda)")
        best.fit(Xtr, ytr)
    elif args.model == 'rf':
        rf = RandomForestClassifier(n_estimators=250, class_weight='balanced',
                                    random_state=42, n_jobs=2)
        rs = RandomizedSearchCV(rf, {'max_depth': [8, 14, None], 'min_samples_leaf': [2, 5, 10]},
                                n_iter=3, cv=2, scoring='f1_macro', random_state=42, n_jobs=2).fit(Xtr.iloc[sub], ytr[sub])
        best = rs.best_estimator_
        print("  [rf]", rs.best_params_, "| CV f1_macro:", round(rs.best_score_, 4))
        best.fit(Xtr, ytr)
    else:  # xgb (requiere clases 0-based)
        xgb = XGBClassifier(objective='multi:softprob', n_estimators=350, learning_rate=0.05,
                            max_depth=6, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=8, tree_method='hist', eval_metric='mlogloss')
        ytr_x = ytr - 1
        print("  [xgb] params fijos: depth=6, lr=0.05, n=350 (sin búsqueda)")
        best = xgb
        best.fit(Xtr, ytr_x, sample_weight=sw)

    y_proba = best.predict_proba(Xte)
    y_pred = best.predict(Xte)
    if args.model == 'xgb':
        y_pred = y_pred + 1  # devolver etiquetas al rango 1-4
    m = metricas(yte, y_pred, y_proba)
    m['tiempo_seg'] = round(time.time() - t0, 1)
    pd.DataFrame([m]).round(4).to_csv(os.path.join(TAB, f'metricas_{args.model}.csv'), index=False, encoding='utf-8-sig')
    np.save(os.path.join(TAB, f'pred_{args.model}.npy'), y_pred)
    joblib.dump(best, os.path.join(TAB, f'modelo_{args.model}.joblib'))
    print(f"  [{args.model}] test: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"kappa={m['kappa']:.4f} logloss={m['log_loss']:.4f} | {m['tiempo_seg']}s")

def comparar():
    rows = []
    preds = {}
    for mod in ['logit', 'rf', 'xgb']:
        m = pd.read_csv(os.path.join(TAB, f'metricas_{mod}.csv')).iloc[0]
        m['modelo'] = mod
        rows.append(m)
        preds[mod] = np.load(os.path.join(TAB, f'pred_{mod}.npy'))
    res = pd.DataFrame(rows).set_index('modelo')[['accuracy','f1_macro','f1_weighted','kappa','log_loss']]
    res.to_csv(os.path.join(TAB, 'tabla_comparacion_modelos.csv'), encoding='utf-8-sig')
    print("\n=== COMPARACIÓN DE MODELOS (test 30%) ===\n", res.round(4).to_string())

    df, X, y = construir_features()
    _, idx_test = split_cv(X, y)
    yte = y[idx_test]
    ranking = res['f1_macro'].sort_values(ascending=False)
    mejor, segundo = ranking.index[0], ranking.index[1]
    ok1 = preds[mejor] == yte
    ok2 = preds[segundo] == yte
    disc = np.array([[np.sum(ok1 & ok2), np.sum(ok1 & ~ok2)],
                     [np.sum(~ok1 & ok2), np.sum(~ok1 & ~ok2)]])
    tb = sct.mcnemar(disc, exact=True)
    print(f"\nMcNemar {mejor} vs {segundo} (filas={mejor} correcto, columnas={segundo} correcto):")
    print(disc)
    print(f"p-valor McNemar (exacto): {tb.pvalue:.6f}")
    with open(os.path.join(TAB, 'mcnemar_resultado.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{mejor} vs {segundo}\n")
        f.write(str(disc) + "\n")
        f.write(f"p-value: {tb.pvalue:.3e} (exacto, dos colas)\n")

if __name__ == '__main__':
    main()

from pathlib import Path
# -*- coding: utf-8 -*-
"""Odds ratios del logit multinomial para el artículo.

Mismo esquema de datos y partición que 05_supervisado.py, pero con un
logit multinomial estándar (statsmodels, no penalizado) sobre los cinco
predictores sociodemográficos centrales, lo que permite inferencia (IC 95%,
p-valores). La edad se expresa en décadas. Categoría de referencia: segmento 1.
Se verifica el mapeo columna->segmento mediante predicciones.
Salidas:
  outputs/tablas/tabla_odds_ratios_logit.csv
  outputs/tablas/tabla_odds_ratios_resumen.md
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
OUT = os.path.join(RUTA, 'outputs')
TAB = os.path.join(OUT, 'tablas')


def main():
    df = pd.read_csv(os.path.join(OUT, 'adultos_mayores_con_cluster.csv'))
    df = df.dropna(subset=['p02', 'p03', 'p04', 'p10a_edu', 'area',
                           'prov', 'p15']).reset_index(drop=True)
    y = df['cluster'].astype(int).values
    X = pd.DataFrame(index=df.index)
    X['sexo_mujer'] = (df['p02'] == 2).astype(int)
    X['edad_decadas'] = df['p03'].astype(float) / 10.0
    X['jefe_hogar'] = (df['p04'] == 1).astype(int)
    X['educ'] = df['p10a_edu'].astype(float)
    X['area_rural'] = (df['area'] == 2).astype(int)

    idx_test = np.load(os.path.join(TAB, 'split_test_idx.npy'))
    idx_train = np.setdiff1d(np.arange(len(y)), idx_test)
    Xtr, ytr = X.iloc[idx_train], y[idx_train]

    exog = sm.add_constant(Xtr, has_constant='add')
    modelo = sm.MNLogit(ytr, exog).fit(method='bfgs', disp=False,
                                       maxiter=2000)
    if not modelo.mle_retvals.get('converged'):
        raise RuntimeError('MNLogit no convergió')

    # Verificar mapeo columna -> segmento mediante predicción en entrenamiento
    pred_col = modelo.predict(exog).idxmax(axis=1)
    cm = pd.crosstab(ytr, pred_col)
    print('Crosstab y vs columna predicha (debe ser diagonal):')
    print(cm)

    coef = modelo.params
    se = modelo.bse
    pv = modelo.pvalues

    filas = []
    for col in coef.columns:
        for var in coef.index:
            if var == 'const':
                continue
            b = coef.loc[var, col]
            s = se.loc[var, col]
            filas.append({
                'columna': int(col),
                'variable': var,
                'beta': b,
                'OR': float(np.exp(b)),
                'OR_ic95_inf': float(np.exp(b - 1.96 * s)),
                'OR_ic95_sup': float(np.exp(b + 1.96 * s)),
                'p': float(pv.loc[var, col]),
            })
    t = pd.DataFrame(filas)
    t['OR'] = t['OR'].round(3)
    t['OR_ic95_inf'] = t['OR_ic95_inf'].round(3)
    t['OR_ic95_sup'] = t['OR_ic95_sup'].round(3)
    t['p_fmt'] = t['p'].map(lambda p: '<0.001' if p < 0.001
                            else '%.3f' % p)
    t.to_csv(os.path.join(TAB, 'tabla_odds_ratios_logit.csv'), index=False,
             encoding='utf-8-sig')

    print('\n=== OR resumen (base: segmento 1) ===')
    lines = ['| Segmento | Variable | OR | IC 95% | p |',
             '|---|---|---|---|---|']
    for _, r in t.iterrows():
        seg = int(r['columna']) + 2
        lines.append('| %d | %s | %.3f | %.3f\u2013%.3f | %s |' % (
            seg, r['variable'], r['OR'], r['OR_ic95_inf'],
            r['OR_ic95_sup'], r['p_fmt']))
        print('Seg%d %-12s OR=%.3f (%.3f-%.3f) p=%s' % (
            seg, r['variable'], r['OR'], r['OR_ic95_inf'],
            r['OR_ic95_sup'], r['p_fmt']))
    with open(os.path.join(TAB, 'tabla_odds_ratios_resumen.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\nGuardado: tabla_odds_ratios_logit.csv y '
          'tabla_odds_ratios_resumen.md')


if __name__ == '__main__':
    main()

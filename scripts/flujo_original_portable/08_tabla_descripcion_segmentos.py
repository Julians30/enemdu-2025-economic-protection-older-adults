from pathlib import Path
# -*- coding: utf-8 -*-
"""Genera la tabla descriptiva sociodemográfica por segmento (para el artículo).

Fuente: outputs/adultos_mayores_con_cluster.csv (ENEMDU 2025, 45,959 personas de 65+).
Salida: outputs/tablas/tabla_descriptivo_por_segmento.csv
"""
import os
import pandas as pd

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
OUT = os.path.join(RUTA, 'outputs')
TAB = os.path.join(OUT, 'tablas')
os.makedirs(TAB, exist_ok=True)

d = pd.read_csv(os.path.join(OUT, 'adultos_mayores_con_cluster.csv'))
n = len(d)
W = d['fexp']

print('N total (sin ponderar):', n)
print('N total (ponderado, miles): %.1f' % (W.sum() / 1e6))


def wpct(mask, w):
    """% ponderado de una máscara booleana."""
    return 100.0 * mask.mul(w).sum() / w.sum()


rows = []
for c in [1, 2, 3, 4]:
    s = d[d['cluster'] == c]
    w = s['fexp']
    rows.append({
        'segmento': int(c),
        'n': int(len(s)),
        'pct_total_sin_pond': 100.0 * len(s) / n,
        'pct_total_pond': 100.0 * w.sum() / W.sum(),
        'edad_media': (s['p03'] * w).sum() / w.sum(),
        'pct_mujeres': wpct(s['p02'] == 2, w),
        'pct_jefe_hogar': wpct(s['p04'] == 1, w),
        'pct_urbano': wpct(s['area'] == 1, w),
        'pct_sin_instruccion': wpct(s['p10a_edu'] == 0, w),
        'pct_basica': wpct(s['p10a_edu'] == 1, w),
        'pct_media': wpct(s['p10a_edu'] == 2, w),
        'pct_superior': wpct(s['p10a_edu'] == 3, w),
        'pct_indigena': wpct(s['p15'] == 1, w),
        'pct_pobre': wpct(s['pobreza'] == 1, w),
        'pct_pobreza_extrema': wpct(s['epobreza'] == 1, w),
    })

t = pd.DataFrame(rows).set_index('segmento')
t.to_csv(os.path.join(TAB, 'tabla_descriptivo_por_segmento.csv'),
         encoding='utf-8-sig')
print(t.round(1).to_string())
print('\nGuardado: outputs/tablas/tabla_descriptivo_por_segmento.csv')

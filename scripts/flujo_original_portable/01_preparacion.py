from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Preparación de datos: adultos mayores (65+).
Genera outputs/adultos_mayores_preparado.csv con todas las variables codificadas.
"""
import pandas as pd
import numpy as np
import os

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
RUTA_CSV = os.path.join(RUTA, 'BDDenemdu_personas_2025_anual.csv')
OUT = os.path.join(RUTA, 'outputs')
os.makedirs(OUT, exist_ok=True)

cols = ['area','p02','p03','p04','p10a','p15','p72a','p72b','ingrl','empleo',
        'prov','pobreza','epobreza','fexp','condact','secemp','ingpc']
df = pd.read_csv(RUTA_CSV, sep=';', decimal=',', usecols=cols, low_memory=False)
print(f"Total personas: {len(df):,}")

# ============ FILTRO POBLACIÓN OBJETIVO ============
am = df[df['p03'] >= 65].copy().reset_index(drop=True)
print(f"Adultos mayores 65+: {len(am):,}")

# ============ VARIABLES DE CLUSTERING ============
# p72a: recibe jubilación/pensión  (1=Si, 2=No)
am['p72a_bin'] = (am['p72a'] == '1').astype(int)

# p72b: monto de jubilación (blanks -> NaN; 999999 = no responde -> NaN)
am['p72b_num'] = pd.to_numeric(am['p72b'].replace({' ': np.nan, '': np.nan}), errors='coerce')
am.loc[am['p72b_num'] >= 999999, 'p72b_num'] = np.nan
am.loc[am['p72a_bin'] == 0, 'p72b_num'] = 0.0   # sin jubilación => monto 0
# Imputar monto no reportado entre jubilados con la mediana del monto positivo
MED_P72B = am.loc[am['p72b_num'].between(1, 500000), 'p72b_num'].median()
am.loc[(am['p72a_bin'] == 1) & (am['p72b_num'].isna()), 'p72b_num'] = MED_P72B

# empleo: población con empleo (1=Si, NA=No)
am['empleo_bin'] = (am['empleo'] == '1').astype(int)

# ingrl: ingreso laboral (-1 y 999999 = no reporta => NaN)
am['ingrl_num'] = pd.to_numeric(am['ingrl'].replace({' ': np.nan, '': np.nan}), errors='coerce')
am.loc[am['ingrl_num'] < 0, 'ingrl_num'] = np.nan
am.loc[am['ingrl_num'] >= 999999, 'ingrl_num'] = np.nan
am.loc[am['empleo_bin'] == 0, 'ingrl_num'] = 0.0   # sin empleo => ingreso 0

# Imputar NA de ingrl entre ocupados con la mediana del ingreso positivo de ocupados
emp_pos = am.loc[(am['empleo_bin'] == 1) & (am['ingrl_num'].between(1, 500000)), 'ingrl_num']
MED_INGRL = emp_pos.median()
am.loc[(am['empleo_bin'] == 1) & (am['ingrl_num'].isna()), 'ingrl_num'] = MED_INGRL
print(f"Mediana p72b (jubilados, >0): {MED_P72B:.2f} | Mediana ingrl (ocupados, >0): {MED_INGRL:.2f}")
print("Missings restantes clustering:", am[['p72a_bin','p72b_num','empleo_bin','ingrl_num']].isna().sum().to_dict())

# Winsorizar outliers al percentil 99 (positivos)
for c in ['p72b_num', 'ingrl_num']:
    p99 = am.loc[am[c] > 0, c].quantile(0.99)
    am[c + '_w'] = am[c].clip(upper=p99)
    print(f"{c}: p99 = {p99:.2f}  (n>p99: {(am[c] > p99).sum()})")

# ============ VARIABLES PREDICTORAS ============
am['p02'] = am['p02'].astype(int)                       # 1=Hombre, 2=Mujer
am['p03'] = am['p03'].astype(int)                       # edad
am['area'] = am['area'].astype(int)                     # 1=Urbano, 2=Rural
am['prov'] = am['prov'].astype(int)                     # 1..24
am['p04'] = am['p04'].astype(int)                       # parentesco
am['p15'] = pd.to_numeric(am['p15'], errors='coerce').fillna(8).astype(int)  # etnia

# p10a -> nivel instrucción 0-3 (mapeo validado con nnivins)
m_edu = {1: 0, 2: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3}
am['p10a_edu'] = am['p10a'].astype(int).map(m_edu)

# pobreza / epobreza (missing -> 0, mayoritaria)
am['pobreza'] = pd.to_numeric(am['pobreza'], errors='coerce').fillna(0).astype(int)
am['epobreza'] = pd.to_numeric(am['epobreza'], errors='coerce').fillna(0).astype(int)

# ============ GUARDAR ============
am.to_csv(os.path.join(OUT, 'adultos_mayores_preparado.csv'), index=False, encoding='utf-8-sig')

# Resumen
print("\n=== RESUMEN POBLACIÓN 65+ (n =", len(am), ") ===")
w = am['fexp'].sum()
print(f"Población ponderada: {w/1000:,.1f} mil")
print(f"Edad media: {am['p03'].mean():.1f}  |  % mujeres: {(am['p02']==2).mean()*100:.1f}  |  % urbano: {(am['area']==1).mean()*100:.1f}")
print(f"% con jubilación: {am['p72a_bin'].mean()*100:.2f}  |  % ocupados: {am['empleo_bin'].mean()*100:.2f}")
print(f"% pobreza: {am['pobreza'].mean()*100:.2f}  |  % pobreza extrema: {am['epobreza'].mean()*100:.2f}")
print(f"Jubilación media (jubilados): {am.loc[am['p72a_bin']==1,'p72b_num'].mean():.2f}")
print(f"Ingreso laboral medio (ocupados): {am.loc[am['empleo_bin']==1,'ingrl_num'].mean():.2f}")
print(f"Ponderado: % con jubilación: {(am.loc[am['p72a_bin']==1,'fexp'].sum()/w)*100:.2f} | % ocupados: {(am.loc[am['empleo_bin']==1,'fexp'].sum()/w)*100:.2f}")
print("\nGuardado en", os.path.join(OUT, 'adultos_mayores_preparado.csv'))

from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Fase 1: Segmentación con K-means.
1) VIF y selección de K (codo, silueta, Davies-Bouldin)
2) K-means K=4, perfiles, radar, cruzadas y guardado.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import os

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
am = pd.read_csv(os.path.join(RUTA, 'outputs', 'adultos_mayores_preparado.csv'))
FIG = os.path.join(RUTA, 'outputs', 'figuras')
TAB = os.path.join(RUTA, 'outputs', 'tablas')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Variables de clustering (winsorizadas al P99)
Xc = am[['p72a_bin','p72b_num_w','empleo_bin','ingrl_num_w']].copy()
Xc.columns = ['jubila_bin','monto_jub','empleo_bin','ingr_lab']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xc)

# ============ 1. VIF ============
Xv = sm.add_constant(Xc)
vifs = pd.DataFrame({'Variable': Xc.columns,
                     'VIF': [variance_inflation_factor(Xv.values, i+1) for i in range(Xc.shape[1])]})
vifs.to_csv(os.path.join(TAB,'tabla_vif.csv'), index=False, encoding='utf-8-sig')
print("\n=== VIF ===\n", vifs.round(3).to_string(index=False))

# ============ 3. K-MEANS CON K=4 ============
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
Xc['cluster_raw'] = labels

# ============ 4. INTERPRETACIÓN Y NOMBRES ============
prof = Xc.groupby('cluster_raw')[['jubila_bin','monto_jub','empleo_bin','ingr_lab']].mean().round(3)
print("\n=== Centros (media, escala original) ===\n", prof.to_string())

# Mapeo genérico basado en los centros observados:
#  - cluster de pensionados puros (jubila>0.5, empleo<=0.5)  -> 1
#  - cluster sin nada (ambas <=0.5)                          -> 4
#  - clusters trabajadores (empleo>0.5) ordenados por ingreso: menor -> 3, mayor -> 2
def clasificar(r):
    if r['jubila_bin'] > 0.5 and r['empleo_bin'] <= 0.5: return 1
    if r['jubila_bin'] <= 0.5 and r['empleo_bin'] <= 0.5: return 4
    if r['empleo_bin'] > 0.5: return None  # se resuelve por rango de ingreso
    return 4

nuevo = {i: clasificar(prof.loc[i]) for i in prof.index}
workers = {i: r for i, r in nuevo.items() if r is None}
w_ingr = {i: prof.loc[i, 'ingr_lab'] for i in workers}
w_orden = sorted(w_ingr.items(), key=lambda kv: kv[1])  # menor -> mayor
if len(w_orden) == 2:
    nuevo[w_orden[0][0]] = 3   # trabajador de menores ingresos (sin pensión)
    nuevo[w_orden[1][0]] = 2   # trabajador de mayores ingresos (con pensión)
elif len(w_orden) == 1:
    nuevo[w_orden[0][0]] = 3
elif len(w_orden) > 2:
    nw = len(w_orden)
    for j, (i, _) in enumerate(w_orden):
        nuevo[i] = 3 if j < nw/2 else 2

Xc['cluster'] = Xc['cluster_raw'].map(nuevo)
nombres = {1:'(1) Jubilados sin actividad laboral', 2:'(2) Trabajadores de mayores ingresos',
           3:'(3) Trabajadores de menores ingresos', 4:'(4) Inactivos sin pensión ni empleo'}
Xc['cluster_nombre'] = Xc['cluster'].map(nombres)
print("\n=== Mapeo cluster_raw -> cluster ===\n", {k: nuevo[k] for k in sorted(nuevo)})
print("\n=== TAMAÑO DE CLUSTERS ===\n", Xc['cluster'].value_counts().sort_index().to_string())


# ============ 5. TABLA DE PERFILES ============
am['cluster'] = Xc['cluster'].values
am['cluster_nombre'] = Xc['cluster_nombre'].values

perfil = am.groupby('cluster_nombre', observed=True).apply(lambda g: pd.Series({
    'n': len(g), '% del total': len(g)/len(am)*100, '% (ponderado)': g['fexp'].sum()/am['fexp'].sum()*100,
    'Recibe jubilación (%)': g['p72a_bin'].mean()*100,
    'Monto jubilación (media, USD)': g['p72b_num'].mean(),
    'Monto jubilación (mediana, USD)': g['p72b_num'].median(),
    'Con empleo (%)': g['empleo_bin'].mean()*100,
    'Ingreso laboral (media, USD)': g['ingrl_num'].mean(),
    'Ingreso laboral (mediana, USD)': g['ingrl_num'].median(),
    'Ingreso total (media, USD)': (g['p72b_num']+g['ingrl_num']).mean()})).round(2)
perfil.to_csv(os.path.join(TAB,'tabla_perfil_clusters.csv'), encoding='utf-8-sig')
print("\n=== PERFIL DE CLUSTERS ===\n", perfil.to_string())

perfil_sd = am.groupby('cluster_nombre', observed=True).apply(lambda g: pd.Series({
    'SD monto jubilación': g['p72b_num'].std(), 'SD ingreso laboral': g['ingrl_num'].std()})).round(2)
perfil_sd.to_csv(os.path.join(TAB,'tabla_perfil_clusters_sd.csv'), encoding='utf-8-sig')

# ============ 6. RADAR CHART ============
norm = Xc.groupby('cluster')[['jubila_bin','empleo_bin','monto_jub','ingr_lab']].mean()
monto_max = Xc['monto_jub'].max(); ingr_max = Xc['ingr_lab'].max()
norm['monto_jub'] = norm['monto_jub']/monto_max
norm['ingr_lab'] = norm['ingr_lab']/ingr_max
norm.columns = ['Recibe jubilación','Con empleo','Monto jubilación (norm)','Ingreso laboral (norm)']

labels_radar = list(norm.columns)
angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
angles += angles[:1]
colores = ['#2E86AB','#F4A261','#E76F51','#6A994E']
fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
for i, idx in enumerate([1,2,3,4]):
    vals = norm.loc[idx].tolist() + [norm.loc[idx].tolist()[0]]
    ax.plot(angles, vals, 'o-', color=colores[i], linewidth=2, label=nombres[idx].split(') ',1)[1])
    ax.fill(angles, vals, alpha=0.12, color=colores[i])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_radar, size=10)
ax.set_ylim(0, 1.05)
ax.set_title('Radar chart: perfiles de los 4 clusters (variables normalizadas)')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_radar_clusters.png'), dpi=300, bbox_inches='tight'); plt.close()

# ============ 7. TABLAS CRUZADAS ============
def ct(var, labels_map=None, nombre=''):
    g = am.groupby([var,'cluster_nombre'], observed=True).apply(lambda x: x['fexp'].sum()).unstack(fill_value=0)
    pct = g.div(g.sum(axis=0), axis=1)*100
    if labels_map:
        pct.index = [labels_map.get(v, v) for v in pct.index]
    pct.to_csv(os.path.join(TAB, f'tabla_cruzada_{nombre}.csv'), encoding='utf-8-sig')
    print(f"\n=== Cruzada por {nombre} (ponderado, % dentro de cluster) ===\n", pct.round(1).to_string())

ct('p02', {1:'Hombre', 2:'Mujer'}, 'sexo')
ct('area', {1:'Urbano', 2:'Rural'}, 'area')
ct('p10a_edu', {0:'Sin instrucción', 1:'Básica', 2:'Media/Bachillerato', 3:'Superior'}, 'educacion')
ct('pobreza', {0:'No pobre', 1:'Pobre'}, 'pobreza')
ct('epobreza', {0:'No pobreza extrema', 1:'Pobreza extrema'}, 'pobreza_extrema')

# ============ 8. GUARDAR ============
am.to_csv(os.path.join(RUTA, 'outputs', 'adultos_mayores_con_cluster.csv'), index=False, encoding='utf-8-sig')
print("\nGuardado: outputs/adultos_mayores_con_cluster.csv")

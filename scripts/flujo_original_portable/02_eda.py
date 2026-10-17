from pathlib import Path
# -*- coding: utf-8 -*-
"""
ENEMDU 2025 - Análisis Exploratorio de Datos (EDA).
Genera tablas y figuras en outputs/.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

RUTA = os.environ.get('ENEMDU_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
am = pd.read_csv(os.path.join(RUTA, 'outputs', 'adultos_mayores_preparado.csv'))
FIG = os.path.join(RUTA, 'outputs', 'figuras')
TAB = os.path.join(RUTA, 'outputs', 'tablas')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

w = am['fexp']
W = w.sum()

# ============ 1. ESTADÍSTICAS DESCRIPTIVAS ============
def wt(x, vals):
    return (am.loc[am[x]==vals,'fexp'].sum()/W)*100

desc = {
    'n (sin ponderar)': len(am),
    'Población proyectada (miles)': round(W/1000,1),
    'Edad media (años)': round((am['p03']*w).sum()/W,1),
    '% mujeres': round(wt('p02',2),1),
    '% urbano': round(wt('area',1),1),
    '% con jubilación (p72a=1)': round(wt('p72a_bin',1),2),
    '% con empleo': round(wt('empleo_bin',1),2),
    'Jubilación media (USD, jubilados)': round((am.loc[am['p72a_bin']==1,'p72b_num']*am.loc[am['p72a_bin']==1,'fexp']).sum()/am.loc[am['p72a_bin']==1,'fexp'].sum(),2),
    'Ingreso laboral medio (USD, ocupados)': round((am.loc[am['empleo_bin']==1,'ingrl_num']*am.loc[am['empleo_bin']==1,'fexp']).sum()/am.loc[am['empleo_bin']==1,'fexp'].sum(),2),
    '% en pobreza': round(wt('pobreza',1),2),
    '% en pobreza extrema': round(wt('epobreza',1),2),
    '% indígena': round(wt('p15',1),1),
}
tab_desc = pd.DataFrame({'Indicador': list(desc.keys()), 'Valor': list(desc.values())})
tab_desc.to_csv(os.path.join(TAB,'tabla_eda_descriptivos.csv'), index=False, encoding='utf-8-sig')
print(tab_desc.to_string(index=False))

# Informalidad entre ocupados
secemp_labels = {'1':'Sector formal', '2':'Sector informal', '3':'Servicio doméstico', '4':'No clasificado'}
am['secemp_cat'] = am['secemp'].map(secemp_labels)
emp = am[am['empleo_bin']==1]
tab_sec = emp.groupby('secemp_cat', dropna=False)['fexp'].agg(['count','sum']).rename(columns={'count':'n','sum':'fexp_sum'})
tab_sec['% (no pond.)'] = tab_sec['n']/len(emp)*100
tab_sec['% (pond.)'] = tab_sec['fexp_sum']/emp['fexp'].sum()*100
tab_sec = tab_sec[['n','% (no pond.)','% (pond.)']].round(2)
tab_sec.to_csv(os.path.join(TAB,'tabla_secemp.csv'), encoding='utf-8-sig')
print("\n=== Sector de empleo (ocupados) ===\n", tab_sec.to_string())
print(f"Informalidad: {tab_sec.loc['Sector informal','% (pond.)']:.1f}% (pond.) | {tab_sec.loc['Sector informal','% (no pond.)']:.1f}% (no pond.)")

# Perfil etario
am['grupo_edad'] = pd.cut(am['p03'], [64,69,74,79,120], labels=['65-69','70-74','75-79','80+'])

# ============ 2. FIGURA: distribución de edad por sexo ============
fig, ax = plt.subplots(figsize=(7,4.5))
for sex, color, lab in [(1,'#2E86AB','Hombres'), (2,'#E07A5F','Mujeres')]:
    sub = am[am['p02']==sex]
    ages, bins = np.histogram(sub['p03'], bins=range(65,101,2), weights=sub['fexp'])
    ax.bar((bins[:-1]+bins[1:])/2, ages/W*100, width=1.8, color=color, alpha=0.65, label=lab, edgecolor='white')
ax.set_xlabel('Edad (años)'); ax.set_ylabel('% de la población 65+ (ponderado)')
ax.set_title('Distribución de edades de la población adulta mayor por sexo')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_edad_sexo.png'), dpi=300); plt.close()

# ============ 3. FIGURA: histogramas de ingresos ============
fig, axes = plt.subplots(1,2, figsize=(10,4))
for ax, c, tit, xlab in [(axes[0],'p72b_num','Monto de jubilación/pensión (USD/mes)','Monto mensual (USD)'),
                          (axes[1],'ingrl_num','Ingreso laboral (USD/mes)','Ingreso mensual (USD)')]:
    sub = am[am[c]>0]
    sns.histplot(sub[c], bins=50, ax=ax, color='#2E86AB', kde=True)
    ax.set_title(tit); ax.set_xlabel(xlab); ax.set_ylabel('Frecuencia')
    ax.axvline(sub[c].median(), color='red', ls='--', label=f'Mediana: ${sub[c].median():.0f}')
    ax.axvline(sub[c].mean(), color='green', ls='--', label=f'Media: ${sub[c].mean():.0f}')
    ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_hist_ingresos.png'), dpi=300); plt.close()

# ============ 4. FIGURA: barras p72a y empleo ============
fig, axes = plt.subplots(1,2, figsize=(8,4))
pens = am.groupby('p72a_bin').apply(lambda g: g['fexp'].sum()/W*100)
emp = am.groupby('empleo_bin').apply(lambda g: g['fexp'].sum()/W*100)
axes[0].bar(['No recibe','Recibe jubilación'], pens.reindex([0,1]), color=['#E07A5F','#2E86AB'], edgecolor='white')
axes[0].set_title('Cobertura de jubilación/pensión'); axes[0].set_ylabel('% (ponderado)')
for i,v in enumerate(pens.reindex([0,1])): axes[0].text(i, v+1, f'{v:.1f}%', ha='center')
axes[1].bar(['No ocupado','Ocupado'], emp.reindex([0,1]), color=['#E07A5F','#2E86AB'], edgecolor='white')
axes[1].set_title('Condición de actividad'); axes[1].set_ylabel('% (ponderado)')
for i,v in enumerate(emp.reindex([0,1])): axes[1].text(i, v+1, f'{v:.1f}%', ha='center')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_barras_actividad.png'), dpi=300); plt.close()

# ============ 5. MATRIZ DE CORRELACIÓN (clustering) ============
Xc = am[['p72a_bin','p72b_num','empleo_bin','ingrl_num']].rename(columns={
    'p72a_bin':'Recibe jubilación','p72b_num':'Monto jubilación','empleo_bin':'Empleo','ingrl_num':'Ingreso laboral'})
corr = Xc.corr()
tab_corr = corr.round(3); tab_corr.to_csv(os.path.join(TAB,'tabla_correlacion.csv'), encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size':11})
ax.set_title('Matriz de correlación de variables de clustering')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_correlacion.png'), dpi=300); plt.close()

# ============ 6. OUTLIERS ============
out_info = []
for c in ['p72b_num','ingrl_num']:
    q99 = am.loc[am[c]>0,c].quantile(0.99)
    q999 = am.loc[am[c]>0,c].quantile(0.999)
    out_info.append({'Variable':c,'P99':round(q99,2),'P99.9':round(q999,2),'Max':am[c].max(),
                     'n>P99':int((am[c]>q99).sum())})
pd.DataFrame(out_info).to_csv(os.path.join(TAB,'tabla_outliers.csv'), index=False, encoding='utf-8-sig')
print("\n=== OUTLIERS ===\n", pd.DataFrame(out_info).to_string(index=False))

# ============ 7. COBERTURA POR PROVINCIA ============
prov_nombres = {1:'Azuay',2:'Bolívar',3:'Cañar',4:'Carchi',5:'Cotopaxi',6:'Chimborazo',7:'El Oro',
                8:'Esmeraldas',9:'Guayas',10:'Imbabura',11:'Loja',12:'Los Ríos',13:'Manabí',
                14:'Morona Santiago',15:'Napo',16:'Pastaza',17:'Pichincha',18:'Tungurahua',
                19:'Zamora Chinchipe',20:'Galápagos',21:'Sucumbíos',22:'Orellana',
                23:'Santo Domingo',24:'Santa Elena'}
am['prov_nombre'] = am['prov'].map(prov_nombres)
tab_prov = am.groupby('prov_nombre').apply(lambda g: pd.Series({
    'n': len(g), '% con jubilación (pond.)': (g.loc[g['p72a_bin']==1,'fexp'].sum()/g['fexp'].sum())*100,
    '% ocupados (pond.)': (g.loc[g['empleo_bin']==1,'fexp'].sum()/g['fexp'].sum())*100,
    '% pobreza (pond.)': (g.loc[g['pobreza']==1,'fexp'].sum()/g['fexp'].sum())*100})).round(2)
tab_prov.to_csv(os.path.join(TAB,'tabla_provincia.csv'), encoding='utf-8-sig')
print("\n=== Por provincia ===\n", tab_prov.to_string())

prov_pens = tab_prov.sort_values('% con jubilación (pond.)', ascending=False)
fig, ax = plt.subplots(figsize=(8,8))
colors = plt.cm.RdYlGn(prov_pens['% con jubilación (pond.)']/max(prov_pens['% con jubilación (pond.)']))
ax.barh(prov_pens.index, prov_pens['% con jubilación (pond.)'], color=colors, edgecolor='white')
ax.set_xlabel('% con jubilación/pensión (ponderado)'); ax.set_ylabel('')
ax.set_title('Cobertura de jubilación/pensión entre adultos mayores por provincia')
for i,v in enumerate(prov_pens['% con jubilación (pond.)']): ax.text(v+0.5, i, f'{v:.1f}', va='center', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_provincia_jubilacion.png'), dpi=300); plt.close()

print("\nEDA completado. Figuras en", FIG)

tab_edad = am.groupby('grupo_edad').apply(lambda g: pd.Series({
    'n': len(g), '% (pond.)': g['fexp'].sum()/W*100,
    '% mujeres': (g.loc[g['p02']==2,'fexp'].sum()/g['fexp'].sum())*100,
    '% con jubilación': (g.loc[g['p72a_bin']==1,'fexp'].sum()/g['fexp'].sum())*100,
    '% con empleo': (g.loc[g['empleo_bin']==1,'fexp'].sum()/g['fexp'].sum())*100})).round(2)
tab_edad.to_csv(os.path.join(TAB,'tabla_edad.csv'), encoding='utf-8-sig')
print("\n=== Por grupo de edad ===\n", tab_edad.to_string())

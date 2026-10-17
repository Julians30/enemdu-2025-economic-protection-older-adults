#!/usr/bin/env python3
"""Relanza scripts originales adaptados en workspace PRIVADO (no sobreescribe evidencia)."""
import argparse,subprocess,os,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--zip',required=True);p.add_argument('--fase',default='preparar',choices=['preparar','segmentar','bootstrap','supervisado_rf','completo']);args=p.parse_args()
w=Path(args.workspace).expanduser().resolve();w.mkdir(parents=True,exist_ok=True)
archivo=w/'BDDenemdu_personas_2025_anual.csv'
if not archivo.exists():
 with zipfile.ZipFile(args.zip) as z:
  hits=[i for i in z.infolist() if i.filename.endswith('/BDDenemdu_personas_2025_anual.csv')]
  if len(hits)!=1 or hits[0].file_size>300_000_000:raise ValueError('Archivo CSV original no encontrado o demasiado grande')
  with z.open(hits[0]) as src,archivo.open('wb') as dest:
   import shutil;shutil.copyfileobj(src,dest)
print('Área privada:',w)
base=Path(__file__).parent/'flujo_original_portable'
orden=['01_preparacion.py','02_eda.py','03a_seleccion_k.py','03_clustering.py','04_bootstrap.py','05_supervisado.py','06_shap.py','07_shap_plots.py','08_tabla_descripcion_segmentos.py','09_odds_ratios_logit.py','10_robustez_k5.py']
if args.fase=='preparar':orden=orden[:1]
elif args.fase=='segmentar':orden=orden[:4]
elif args.fase=='bootstrap':orden=orden[:5]
elif args.fase=='supervisado_rf':orden=orden[:6]
if args.fase=='supervisado_rf':orden=['01_preparacion.py','03_clustering.py','04_bootstrap.py','05_supervisado.py']
if args.fase=='completo':
 print('ADVERTENCIA: el reentrenamiento puede diferir del original 0,5308. No usar sus nuevas métricas como oficiales sin auditar.')
orden_procesos=[]
for s in orden:
 if s=='05_supervisado.py':
  for m in (['rf'] if args.fase!='completo' else ['logit','rf','xgb','comparar']):orden_procesos.append([sys.executable,str(base/s),'--model',m])
 else:orden_procesos.append([sys.executable,str(base/s)])
for c in orden_procesos:
 print('Ejecutando', ' '.join(c),flush=True)
 env=os.environ.copy();env['ENEMDU_PROJECT_ROOT']=str(w)
 subprocess.run(c,env=env,check=True)

#!/usr/bin/env python3
"""Comprueba RF histórico desde el ZIP privado sin extraer datos. No reentrena."""
from pathlib import Path
import argparse,zipfile,io,json,hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,confusion_matrix,precision_recall_fscore_support
p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--salida',default='resultados_rf_auditoria');args=p.parse_args()
with zipfile.ZipFile(args.zip) as z:
 def obtener(nombre):
  hits=[f.filename for f in z.infolist() if not f.is_dir() and f.filename.endswith('/'+nombre)]
  if len(hits)!=1:raise ValueError(f'Falta o se duplica {nombre}')
  return z.read(hits[0])
 a=pd.read_csv(io.BytesIO(obtener('adultos_mayores_con_cluster.csv')))
 inds=np.load(io.BytesIO(obtener('split_test_idx.npy')),allow_pickle=False)
 rf_bytes=obtener('pred_rf.npy'); pr=np.load(io.BytesIO(rf_bytes),allow_pickle=False)
 assert len(a)==45959 and len(inds)==len(pr)==13788
 assert len(np.unique(inds))==len(inds) and inds.min()>=0 and inds.max()<len(a)
 yt=a.iloc[inds]['cluster'].astype(int).to_numpy()
 acc=accuracy_score(yt,pr);f1m=f1_score(yt,pr,average='macro');f1p=f1_score(yt,pr,average='weighted');kp=cohen_kappa_score(yt,pr)
 assert abs(acc-0.5308)<0.00005 and abs(f1m-0.4588)<0.00005
 assert abs(f1p-0.5535)<0.00005 and abs(kp-0.3465)<0.00005
 c=confusion_matrix(yt,pr,labels=[1,2,3,4]);prec,rec,f1,support=precision_recall_fscore_support(yt,pr,labels=[1,2,3,4],zero_division=0)
 t=pd.DataFrame([dict(segmento=i+1,n=int(support[i]),vp=int(c[i,i]),fn=int(support[i]-c[i,i]),sensibilidad_recall=float(rec[i]),precision=float(prec[i]),f1=float(f1[i])) for i in range(4)])
 d=Path(args.salida);d.mkdir(exist_ok=True,parents=True)
 t.to_excel(d/'Tabla_A2_RF_sensibilidad.xlsx',index=False)
 resultado=dict(n_test=len(yt),accuracy=acc,f1_macro=f1m,f1_weighted=f1p,kappa=kp,pred_rf_sha256=hashlib.sha256(rf_bytes).hexdigest(),matriz_confusion=c.tolist(),nota='Verificación de etiquetas guardadas; no certifica reentrenamiento histórico, calibración ni variación de umbral.')
 (d/'RF_auditoria.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(resultado,ensure_ascii=False,indent=2));print('Tabla A2:',d/'Tabla_A2_RF_sensibilidad.xlsx')

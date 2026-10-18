#!/usr/bin/env python3
"""Verificación reproducible ENEMDU 2025 sin publicar microdatos.

Modo público: coteja los resultados agregados publicados contra el manifiesto.
Modo privado: recalcula las métricas desde las predicciones y probabilidades de TEST,
usa etiquetas originales del ZIP y comprueba matrices de confusión. Nunca escribe
registros ni vectores individuales en el repositorio.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

CLASSES=(1, 2, 3, 4)
KEYS=('accuracy', 'f1_macro', 'f1_weighted', 'kappa', 'log_loss', 'brier_multiclase')


def ensure(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as fd:
        return list(csv.DictReader(fd))


def close(actual, expected, key, atol=1e-7):
    ensure(abs(float(actual)-float(expected))<=atol,
           f'Diferencia en {key}: {actual} != {expected}')


def public_check(public):
    path=public/'AUDITORIA_REPRODUCIBILIDAD.json'
    with open(path, encoding='utf-8') as fd: meta=json.load(fd)
    rows=read_csv(public/'TABLA3_MODELOS_CANONICOS.csv')
    ensure(len(rows)==3,'La Tabla 3 debe incluir exactamente tres modelos')
    ensure({x['modelo'] for x in rows}=={'rf','logit','xgb'},'Nombres de modelo inesperados')
    saved={d['modelo']:d for d in meta['metrics']}
    for r in rows:
        for k in KEYS: close(r[k], saved[r['modelo']][k], f'{r["modelo"]}/{k}')
    a3=read_csv(public/'TABLA_A2_SENSIBILIDAD_UMBRAL_RF.csv')
    ensure(len(a3)>=2 and any(row['escenario']=='argmax' for row in a3),'Faltan escenarios RF')
    selected=float(meta['rf_threshold_selected_train_oof'])
    ensure(any(row['escenario']=='umbral_train_oof' and abs(float(row['t2'])-selected)<1e-9 for row in a3),
           'El umbral OOF seleccionado no aparece en la Tabla A3')
    print('OK: integridad de Tabla 3, manifiesto y sensibilidad RF (agregados públicos)')
    return meta, rows


def private_check(zip_path, models, meta, rows, advanced=None):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix
    ensure(hashlib.sha256(zip_path.read_bytes()).hexdigest()==meta['zip_sha256'],
           'El ZIP no coincide con la huella de datos auditados')
    with zipfile.ZipFile(zip_path) as z:
        base='enedmu_adultos_mayores/'
        df=pd.read_csv(io.BytesIO(z.read(base+'outputs/adultos_mayores_con_cluster.csv')))
        idx=np.load(io.BytesIO(z.read(base+'outputs/tablas/split_test_idx.npy')))
    ensure(len(df)==45959 and len(idx)==13788,'Tamaños de muestra no esperados')
    y=df['cluster'].astype(int).to_numpy()[idx]
    ensure(set(np.unique(y))==set(CLASSES),'Etiquetas de los clusters inesperadas')
    expected={r['modelo']:r for r in rows}
    for name in ('logit','rf','xgb'):
        yhat=np.load(models/f'pred_{name}.npy',allow_pickle=False)
        p=np.load(models/f'proba_{name}.npy',allow_pickle=False)
        ensure(yhat.shape==(len(y),),f'{name}: etiquetas incompletas')
        ensure(p.shape==(len(y),4),f'{name}: matriz de probabilidad inesperada')
        ensure(np.isfinite(p).all() and (p>=0).all() and (p<=1+1e-6).all(),f'{name}: probabilidad inválida')
        ensure(np.max(np.abs(p.sum(axis=1)-1))<2e-5,f'{name}: filas de probabilidades no normalizadas')
        ensure(np.array_equal(yhat, np.asarray(CLASSES)[np.argmax(p,axis=1)]),f'{name}: argmax distinto de las etiquetas')
        oh=(y[:,None]==np.asarray(CLASSES)[None,:]).astype(float)
        metrics=dict(accuracy=accuracy_score(y,yhat),f1_macro=f1_score(y,yhat,average='macro'),
                     f1_weighted=f1_score(y,yhat,average='weighted'),kappa=cohen_kappa_score(y,yhat),
                     log_loss=log_loss(y,p,labels=CLASSES),brier_multiclase=np.mean(np.sum((oh-p)**2,axis=1)))
        for k in KEYS:close(metrics[k],expected[name][k],f'{name}/{k}')
        cm=confusion_matrix(y,yhat,labels=CLASSES)
        source=pd.read_csv(models/f'confusion_{name}.csv',index_col=0).to_numpy()
        ensure(np.array_equal(cm,source),f'{name}: la matriz de confusión no coincide')
        if advanced is not None:
            pcal=np.load(advanced/f'prob_cal_{name}.npy',allow_pickle=False)
            ensure(pcal.shape==(len(y),4),'Dimensiones calibradas erróneas')
            ensure(np.isfinite(pcal).all() and np.max(np.abs(pcal.sum(axis=1)-1))<2e-5,
                   'Probabilidades calibradas incorrectas')
        print(f'OK: {name} — {len(y)} casos, seis métricas, probabilidades y confusión verificadas')
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--public',type=Path,required=True,help='Carpeta con CSV y JSON agregados publicados')
    ap.add_argument('--zip',type=Path,help='ZIP original: permanece PRIVADO')
    ap.add_argument('--models',type=Path,help='Directorio privado con pred_*.npy y proba_*.npy')
    ap.add_argument('--advanced',type=Path,help='Directorio privado con probabilidades calibradas')
    a=ap.parse_args()
    meta,rows=public_check(a.public)
    ensure(bool(a.zip)==bool(a.models),'--zip y --models deben proporcionarse juntos')
    if a.zip:private_check(a.zip,a.models,meta,rows,a.advanced)
    print('VERIFICACIÓN FINAL COMPLETADA. Ninguna fila privada fue exportada.')


if __name__=='__main__':main()
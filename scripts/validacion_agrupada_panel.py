#!/usr/bin/env python3
"""Evaluación complementaria: separación estricta de posiciones persona–hogar ENEMDU.

Requiere ZIP privado del estudio. No publica id_persona, microdatos ni predicciones.
Réplica nueva y diferente del test histórico por filas; NO sustituye sus métricas.
"""
from __future__ import annotations
import argparse, json, sys, hashlib, platform
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, log_loss, precision_recall_fscore_support, confusion_matrix)

def _suffix(z, suffix):
    names=[name for name in z.namelist() if name.endswith(suffix)]
    if len(names)!=1: raise ValueError(f'Se esperaba un archivo {suffix}, encontrados {len(names)}')
    return names[0]

def main(zip_path:Path, output_dir:Path, models_requested=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as z:
        with z.open(_suffix(z,'BDDenemdu_personas_2025_anual.csv')) as f:
            original=pd.read_csv(f,sep=';',usecols=['id_persona','p03','p02','p04','prov','p15'],low_memory=False)
        with z.open(_suffix(z,'outputs/adultos_mayores_con_cluster.csv')) as f:
            df=pd.read_csv(f,low_memory=False,encoding='utf-8-sig')
    base=original.loc[original['p03']>=65].reset_index(drop=True)
    if len(base)!=len(df) or not all((base[k].astype(str).values==df[k].astype(str).values).all() for k in ['p02','p03','p04','prov','p15']):
        raise AssertionError('No se pudo certificar la alineación fila por fila entre microdatos y base segmentada')
    if df[['p02','p03','p04','p10a_edu','area','prov','p15','cluster']].isna().any().any():
        raise ValueError('Valores faltantes en variables supervisadas; verificar versión de base')
    # id_persona contiene un sufijo de dos dígitos del mes de visita; clave base no identifica necesariamente la misma persona física.
    ids=base.id_persona.astype('string')
    if (ids.str.len()<3).any():raise ValueError('Identificadores de panel inválidos')
    grupos=ids.str[:-2].to_numpy()
    X=pd.DataFrame(index=df.index)
    X['sexo_mujer']=(df.p02==2).astype('int8')
    X['edad']=df.p03.astype(float)
    X['jefe_hogar']=(df.p04==1).astype('int8')
    X['educ']=df.p10a_edu.astype(float)
    X['area_rural']=(df.area==2).astype('int8')
    X=pd.concat([X,pd.get_dummies(df.p15.astype(str),prefix='etnia',dtype='int8'),pd.get_dummies(df.prov.astype(str),prefix='prov',dtype='int8')],axis=1)
    y=df.cluster.astype(int).to_numpy()
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=42).split(X,y,groups=grupos))
    intersection=np.intersect1d(np.unique(grupos[train]),np.unique(grupos[test]))
    assert len(intersection)==0
    proportions={str(i):{'train':float((y[train]==i).mean()),'test':float((y[test]==i).mean())} for i in [1,2,3,4]}
    sw_freq=pd.Series(y[train]).value_counts(normalize=True).to_dict()
    weights=np.array([1./sw_freq[k] for k in y[train]])
    models={
      'Regresión logística':(Pipeline([('scale',ColumnTransformer([('scale_num',StandardScaler(),['edad','educ'])],remainder='passthrough')),('clf',LogisticRegression(solver='lbfgs',C=1,max_iter=2000,class_weight='balanced'))]),False),
      'Random Forest':(RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=4),False)
    }
    try:
        import xgboost
        from xgboost import XGBClassifier
        models['XGBoost']=(XGBClassifier(objective='multi:softprob',n_estimators=350,learning_rate=.05,max_depth=6,subsample=.8,colsample_bytree=.8,random_state=42,tree_method='hist',eval_metric='mlogloss',n_jobs=4),True)
    except ImportError:
        xgboost=None
    if models_requested is not None:
        models={k:v for k,v in models.items() if k.lower().replace(" ","_") in models_requested}
    results=[]; per_class=[]
    for label,(model,is_xgb) in models.items():
        yfit=y[train]-1 if is_xgb else y[train]
        kwargs={'sample_weight':weights} if is_xgb else {}
        model.fit(X.iloc[train],yfit,**kwargs)
        pred=model.predict(X.iloc[test]); pred=pred+1 if is_xgb else pred
        prob=model.predict_proba(X.iloc[test]);
        row={'modelo':label,'accuracy':accuracy_score(y[test],pred),'f1_macro':f1_score(y[test],pred,average='macro'), 'f1_ponderado':f1_score(y[test],pred,average='weighted'),'kappa':cohen_kappa_score(y[test],pred),'log_loss':log_loss(y[test],prob,labels=[1,2,3,4]),'n_test':len(test),'n_train':len(train),'split':'GroupShuffleSplit, test_size=0.30, semilla=42, grupos=id_persona sin sufijo mes'}
        results.append(row)
        prec,rec,f1,sup=precision_recall_fscore_support(y[test],pred,labels=[1,2,3,4],zero_division=0)
        for i in range(4): per_class.append({'modelo':label,'segmento':i+1,'precision':prec[i],'recall':rec[i],'f1':f1[i],'n_real':int(sup[i])})
        print(f'{label}: accuracy={row["accuracy"]:.4f}; F1macro={row["f1_macro"]:.4f} | train={len(train)} test={len(test)}',flush=True)
    result_df=pd.DataFrame(results)
    result_df.to_csv(output_dir/'validacion_agrupada_resumen.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(per_class).to_csv(output_dir/'validacion_agrupada_por_segmento.csv',index=False,encoding='utf-8-sig')
    metadata={'proposito':'Evaluación de sensibilidad complementaria, NO reemplaza test histórico','n_registros':len(y),'n_claves_base':len(np.unique(grupos)),'train':len(train),'test':len(test),'grupos_compartidos_train_test':len(intersection),'clases':proportions,'semilla':42,'train_index_sha256':hashlib.sha256(train.tobytes()).hexdigest(),'test_index_sha256':hashlib.sha256(test.tobytes()).hexdigest(),'nota':'Las claves base identifican posiciones persona–hogar, no identidades personales verificadas. Split no estratificado: comparar las prevalencias. Datos, identificadores y predicciones no publicados.','logit_escalado_solo_train':'edad y educ estandarizadas con scaler ajustado solo sobre entrenamiento','configuracion_rf':'250 árboles, max_depth=None, min_samples_leaf=5, balanced, seed=42; hiperparámetros provenientes de reentrenamiento exploratorio, NO configuración histórica RF verificada','xgboost_disponible':xgboost is not None,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'xgboost':getattr(xgboost,'__version__',None),'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest()}
    (output_dir/'metadatos_validacion_agrupada.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ARCHIVOS',output_dir,'grupos disjuntos',len(intersection)==0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--models',default='regresión_logística,random_forest,xgboost');a=p.parse_args();main(a.zip,a.out,set(a.models.split(',')))
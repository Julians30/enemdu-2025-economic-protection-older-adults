#!/usr/bin/env python3
"""ENEMDU 2025: evaluación SUPERVISADA DE REFERENCIA, nueva versión trazable.
No reemplazar por silencio los artefactos históricos. Usa un ZIP privado del autor.
El split fijo conserva el diseño de referencia por filas, sujeto a su limitación longitudinal.
La selección de umbral C2 se hace por OOF de TRAIN; TEST se toca una sola vez.
"""
import argparse, io, zipfile, hashlib, sys, json, os, platform, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, xgboost, scipy, joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CLASSES=np.array([1,2,3,4]); SEED=42

def sha(b):return hashlib.sha256(b).hexdigest()
def load(z):
    with zipfile.ZipFile(z) as h:
        df=pd.read_csv(io.BytesIO(h.read('enedmu_adultos_mayores/outputs/adultos_mayores_con_cluster.csv')))
        test=np.load(io.BytesIO(h.read('enedmu_adultos_mayores/outputs/tablas/split_test_idx.npy')))
    return df, test

def features(df):
    filtered=df.dropna(subset=['p02','p03','p04','p10a_edu','area','prov','p15']).reset_index(drop=True)
    if len(filtered)!=len(df):raise RuntimeError('Se perdieron filas; verificar split original e índices')
    y=filtered.cluster.astype(int).to_numpy()
    X=pd.DataFrame(index=filtered.index)
    X['sexo_mujer']=(filtered.p02==2).astype(int)
    X['edad']=filtered.p03.astype(float)
    X['jefe_hogar']=(filtered.p04==1).astype(int)
    X['educ']=filtered.p10a_edu.astype(float)
    X['area_rural']=(filtered.area==2).astype(int)
    X=pd.concat([X,pd.get_dummies(filtered.p15.astype(str),prefix='etnia',dtype=int),pd.get_dummies(filtered.prov.astype(str),prefix='prov',dtype=int)],axis=1)
    if sorted(np.unique(y).tolist())!=CLASSES.tolist():raise AssertionError('Clases cambiaron')
    return X,y

def met(y,pr,p):
    oh=(y[:,None]==CLASSES[None,:]).astype(float)
    return dict(accuracy=float(accuracy_score(y,pr)),f1_macro=float(f1_score(y,pr,average='macro')),f1_weighted=float(f1_score(y,pr,average='weighted')),kappa=float(cohen_kappa_score(y,pr)),log_loss=float(log_loss(y,p,labels=CLASSES)),brier_multiclase=float(np.mean(np.sum((oh-p)**2,axis=1))))

def rule(p,t):
    pr=CLASSES[p.argmax(axis=1)].copy()
    if t is None:return pr
    non=np.array([0,2,3]);pr=CLASSES[non[np.argmax(p[:,non],axis=1)]]
    pr[p[:,1]>=t]=2
    return pr

def measure_c2(y,p,t):
    yp=rule(p,t)
    prec,rec,f,_=precision_recall_fscore_support(y,yp,labels=CLASSES,zero_division=0)
    return dict(t2=None if t is None else float(t),accuracy=float(accuracy_score(y,yp)),f1_macro=float(f1_score(y,yp,average='macro')),precision_c2=float(prec[1]),recall_c2=float(rec[1]),f1_c2=float(f[1]),predichos_c2=int(np.sum(yp==2)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--out',required=True);ap.add_argument('--fast',action='store_true',help='Solo validación de depuración (no publicable)');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    b=Path(a.zip).read_bytes();df,test=load(a.zip);X,y=features(df)
    train=np.setdiff1d(np.arange(len(y)),test)
    if len(y)!=45959 or len(train)!=32171 or len(test)!=13788:raise AssertionError('Tamaños inesperados')
    xt,xv=X.iloc[train],X.iloc[test];yt,yv=y[train],y[test]
    rs0=np.random.RandomState(SEED);sub=rs0.choice(len(train),size=12000,replace=False)
    results=[]; details={}; models={}
    rfbase=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=SEED,n_jobs=1)
    if a.fast: params={'max_depth':None,'min_samples_leaf':5}; score=None
    else:
        search=RandomizedSearchCV(rfbase,{'max_depth':[8,14,None],'min_samples_leaf':[2,5,10]},n_iter=3,cv=2,scoring='f1_macro',random_state=SEED,n_jobs=1)
        search.fit(xt.iloc[sub],yt[sub]);params=search.best_params_;score=float(search.best_score_)
    config={"rf_tuning":params,"rf_cv_score":score,"rf_selection":"RandomizedSearchCV(3 configs, 2 folds; stratified subsample of 12000 from TRAIN)","train_indices":"from original split_test_idx.npy; not published", "seed":SEED}
    for name,m in [('logit',Pipeline([('scale',ColumnTransformer([('scale_num',StandardScaler(),['edad','educ'])],remainder='passthrough')),('clf',LogisticRegression(solver='lbfgs',max_iter=3000,C=1,class_weight='balanced'))])),
                   ('rf',RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=SEED,n_jobs=1,**params)),
                   ('xgb',XGBClassifier(objective='multi:softprob',n_estimators=350,learning_rate=.05,max_depth=6,subsample=.8,colsample_bytree=.8,random_state=SEED,n_jobs=1,tree_method='hist',eval_metric='mlogloss'))]:
        yy=yt-1 if name=='xgb' else yt
        weights=None
        if name=='xgb':
            v,c=np.unique(yt,return_counts=True);ww=dict(zip(v,len(yt)/c));weights=np.array([ww[k] for k in yt])
        st=time.monotonic()
        if weights is not None:m.fit(xt,yy,sample_weight=weights)
        else:m.fit(xt,yy)
        p=m.predict_proba(xv)
        pred=(np.asarray(m.predict(xv)).astype(int)+1) if name=='xgb' else np.asarray(m.predict(xv)).astype(int)
        mm=met(yv,pred,p); results.append(dict(modelo=name,**mm))
        np.save(out/f'pred_{name}.npy',pred);np.save(out/f'proba_{name}.npy',p)
        joblib.dump(m,out/f'modelo_{name}.joblib',compress=3)
        conf=confusion_matrix(yv,pred,labels=CLASSES)
        pd.DataFrame(conf,index=[f'real_{i}' for i in CLASSES],columns=[f'pred_{i}' for i in CLASSES]).to_csv(out/f'confusion_{name}.csv')
        print(name,mm,'segundos',round(time.monotonic()-st,1),flush=True)
        details[name]={'metrics':mm,'confusion':conf.tolist()};models[name]=m
    pd.DataFrame(results).to_csv(out/'TABLA3_MODELOS_CANONICOS.csv',index=False,float_format='%.8f')
    # Curva de RF elegida sin consultar TEST: probabilidades OOF del TRAIN.
    rf=models['rf'];cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    st=time.monotonic();poof=cross_val_predict(clone(rf),xt,yt,cv=cv,method='predict_proba',n_jobs=1)
    grid=np.round(np.linspace(.02,.50,49),4)
    oo=[measure_c2(yt,poof,t) for t in grid]
    pd.DataFrame(oo).to_csv(out/'RF_CURVA_UMBRAL_SOLO_TRAIN_OOF.csv',index=False,float_format='%.8f')
    best=max(oo,key=lambda v:(v['f1_c2'],v['f1_macro'],-abs(v['t2']-.25)))['t2']
    tests=[dict(escenario='argmax',**measure_c2(yv,models['rf'].predict_proba(xv),None))]
    for t in [0.05,0.10,0.15,0.20,0.25,0.30,best]:tests.append(dict(escenario='umbral_train_oof' if t==best else 'umbral_predefinido',**measure_c2(yv,models['rf'].predict_proba(xv),t)))
    pd.DataFrame(tests).to_csv(out/'TABLA_A2_SENSIBILIDAD_UMBRAL_RF.csv',index=False,float_format='%.8f')
    print('RF A2 seleccionado train OOF',best,'OOF sec',round(time.monotonic()-st,1),flush=True)
    # SOLO AGREGA no microdatos ni predicciones publicas
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'xgboost':xgboost.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__}
    audit={'type':'nueva_ejecucion_referencia_2026-10-17','historical_rf_accuracy':0.5308,'historical_rf_status':'salidas verificadas; no entrenamiento idéntico','zip_sha256':sha(b),'semilla':SEED,'test_n':len(test),'train_n':len(train),'num_features':X.shape[1],'feature_names':X.columns.tolist(),'model_config':config,'environment':versions,'metrics':results,'rf_threshold_selected_train_oof':best,'scope':'clasificadores desde etiquetas K-means del dataset original: no revalida clusters externos','test_indices_and_predictions_are_private':True}
    (out/'AUDITORIA_REPRODUCIBILIDAD.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    public=out/'PUBLICABLE';public.mkdir(exist_ok=True)
    for fn in ['TABLA3_MODELOS_CANONICOS.csv','TABLA_A2_SENSIBILIDAD_UMBRAL_RF.csv','RF_CURVA_UMBRAL_SOLO_TRAIN_OOF.csv','AUDITORIA_REPRODUCIBILIDAD.json']:
        (public/fn).write_bytes((out/fn).read_bytes())
    print('TERMINADO resultados',out,flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Métricas avanzadas y selección TRAIN-OOF para nueva ejecución canónica ENEMDU.
No usar TEST para elegir calibradores ni umbrales; publicar solo resultados agregados.
"""
import argparse,zipfile,io,json,platform
from pathlib import Path
import numpy as np,pandas as pd,joblib,sklearn,xgboost
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score,f1_score,cohen_kappa_score,log_loss,roc_auc_score,average_precision_score,precision_recall_fscore_support)
from scipy.special import softmax
from importlib.machinery import SourceFileLoader

CL=np.array([1,2,3,4]);SEED=42

def brier(y,p):return float(np.mean(np.sum(((np.asarray(y)[:,None]==CL[None,:])-p)**2,axis=1)))
def eval_metrics(y,p):
    pred=CL[np.argmax(p,axis=1)]
    out={'accuracy':float(accuracy_score(y,pred)),'f1_macro':float(f1_score(y,pred,average='macro')),'f1_weighted':float(f1_score(y,pred,average='weighted')),'kappa':float(cohen_kappa_score(y,pred)),'log_loss':float(log_loss(y,p,labels=CL)),'brier':brier(y,p),'roc_auc_macro':float(roc_auc_score((y[:,None]==CL).astype(int),p,average='macro')),'average_precision_macro':float(average_precision_score((y[:,None]==CL).astype(int),p,average='macro'))};return out

def normalize(p):
    p=np.maximum(p,1e-12);return p/p.sum(axis=1,keepdims=True)
def fit_calib(name,p,y):
    if name=='raw':return None
    est=[]
    for j,k in enumerate(CL):
        iscls=(y==k).astype(int)
        if name=='platt':
            # Platt-like monotone sigmoid applied to base probability; logit transformed via liblinear.
            lr=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=SEED)
            lr.fit(p[:,j].reshape(-1,1),iscls);est.append(lr)
        else:
            iso=IsotonicRegression(out_of_bounds='clip');iso.fit(p[:,j],iscls);est.append(iso)
    return est

def apply_calib(p,name,model):
    if name=='raw':return p
    pred=np.column_stack([m.predict_proba(p[:,j].reshape(-1,1))[:,1] if name=='platt' else m.predict(p[:,j]) for j,m in enumerate(model)])
    return normalize(pred)

def rule(p,t=None):
    if t is None:return CL[p.argmax(axis=1)]
    t0=CL[np.array([0,2,3])[p[:,[0,2,3]].argmax(axis=1)]]
    t0[p[:,1]>=t]=2;return t0

def scenario(y,p,t):
    pred=rule(p,t); prec,rec,f1,sup=precision_recall_fscore_support(y,pred,labels=CL,zero_division=0)
    return {'threshold':None if t is None else float(t),'accuracy':float(accuracy_score(y,pred)),'f1_macro':float(f1_score(y,pred,average='macro')),'precision_C2':float(prec[1]),'recall_C2':float(rec[1]),'f1_C2':float(f1[1]),'pred_C2':int(sum(pred==2))}

def main():
    args=argparse.ArgumentParser();args.add_argument('--zip',required=True);args.add_argument('--models',required=True);args.add_argument('--out',required=True);a=args.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    src=SourceFileLoader('reference',str(Path(__file__).with_name('06_modelos_referencia_reproducibles.py'))).load_module()
    df,tt=src.load(a.zip);X,y=src.features(df);tr=np.setdiff1d(np.arange(len(y)),tt)
    xt=X.iloc[tr];yt=y[tr];xv=X.iloc[tt];yv=y[tt]
    # OOF base predictions; calibration assessment uses split within OOF solely TRAIN.
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    cv_oof={} ;base_scores=[];cal_scores=[];thresh=[];perclass=[]
    for name in ['logit','rf','xgb']:
        model=joblib.load(Path(a.models)/f'modelo_{name}.joblib')
        poof=np.zeros((len(tr),4),dtype=float)
        for k,(fit,valid) in enumerate(cv.split(xt,yt)):
            mm=clone(model)
            if name=='xgb':
                vc=pd.Series(yt[fit]).value_counts(normalize=True).to_dict()
                wt=np.array([1/vc[c] for c in yt[fit]])
                mm.fit(xt.iloc[fit],yt[fit]-1,sample_weight=wt)
            else:mm.fit(xt.iloc[fit],yt[fit])
            poof[valid]=mm.predict_proba(xt.iloc[valid])
        ptest=np.load(Path(a.models)/f'proba_{name}.npy')
        ra,val=train_test_split(np.arange(len(tr)),test_size=.20,random_state=SEED,stratify=yt)
        methods=['raw','platt','isotonic'];scores=[]
        for method in methods:
            cm=fit_calib(method,poof[ra],yt[ra]);pred=apply_calib(poof[val],method,cm)
            scores.append((brier(yt[val],pred),method))
        selected=min(scores,key=lambda t:t[0])[1]
        cal_selection=fit_calib(selected,poof[ra],yt[ra])
        poof_sel=apply_calib(poof[val],selected,cal_selection)
        cm_final=fit_calib(selected,poof,yt)
        calibrated=apply_calib(ptest,selected,cm_final)
        np.save(out/f'prob_cal_{name}.npy',calibrated)
        sc_raw=eval_metrics(yv,ptest);sc_cal=eval_metrics(yv,calibrated)
        cal_scores.append({'modelo':name,'metodo':selected,**{k+'_crudo':v for k,v in sc_raw.items()},**{k+'_calibrado':v for k,v in sc_cal.items()}})
        pred2=CL[ptest.argmax(axis=1)];pr,rec,ff,sup=precision_recall_fscore_support(yv,pred2,labels=CL,zero_division=0)
        A=(yv[:,None]==CL[None,:]).astype(int)
        for j,k in enumerate(CL):
            from sklearn.metrics import roc_auc_score,average_precision_score
            perclass.append(dict(modelo=name,segmento=int(k),precision=float(pr[j]),recall=float(rec[j]),f1=float(ff[j]),roc_auc=float(roc_auc_score(A[:,j],ptest[:,j])),average_precision=float(average_precision_score(A[:,j],ptest[:,j])),n_real=int(sup[j])))
        thresholds=np.round(np.arange(.02,.501,.01),2)
        grid=[scenario(yt[val],poof_sel,t) for t in thresholds]
        pd.DataFrame(grid).to_csv(out/f'CURVA_UMBRAL_{name}_TRAIN_OOF.csv',index=False,float_format='%.8f')
        best_f1=max(grid,key=lambda x:(x['f1_C2'],x['f1_macro']))['threshold']
        qualified=[o for o in grid if o['recall_C2']>=.70]
        rec70=max(qualified,key=lambda x:(x['precision_C2'],x['f1_macro']))['threshold'] if qualified else None
        for label,t in [('argmax',None),('max_F1_C2_train',best_f1),('precision_sujeto_recall70_train',rec70)]:
            if label!='argmax' and t is None:continue
            thresh.append(dict(modelo=name,escenario=label,**scenario(yv,calibrated,t)))
        print(name,'selected calibration',selected,'brier',round(sc_cal['brier'],4),'thresholds',best_f1,rec70,flush=True)
        if name=='xgb':
            fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
            for j,(ax,k) in enumerate(zip(axs.flat,CL)):
                for lbl,p,col in [('Sin calibrar',ptest,'#61708A'),('Calibrado',calibrated,'#087F8C')]:
                    bins=np.linspace(0,1,11);x,ybars=[],[]
                    for l,r in zip(bins[:-1],bins[1:]):
                        mask=(p[:,j]>=l)&(p[:,j]<r if r<1 else p[:,j]<=r)
                        if mask.sum()>=15:
                            x.append(float(p[mask,j].mean()));ybars.append(float((yv[mask]==k).mean()))
                    ax.plot(x,ybars,marker='o',lw=1.7,label=lbl,color=col)
                ax.plot([0,1],[0,1],'--',c='#BBBBBB',lw=1.0)
                ax.set_title(f'Segmento {k}');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Probabilidad predicha');ax.set_ylabel('Frecuencia observada');ax.grid(alpha=.15)
            axs.flat[0].legend(frameon=False);fig.suptitle('Calibración de XGBoost – nueva ejecución de referencia',fontweight='bold',fontsize=13)
            fig.savefig(out/'FIGURA4_CALIBRACION_XGB_REFERENCIA.png',dpi=600,bbox_inches='tight');fig.savefig(out/'FIGURA4_CALIBRACION_XGB_REFERENCIA.svg',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(cal_scores).to_csv(out/'TABLA5_CALIBRACION_REFERENCIA.csv',index=False,float_format='%.8f')
    pd.DataFrame(thresh).to_csv(out/'TABLA6_UMBRALES_REFERENCIA.csv',index=False,float_format='%.8f')
    pd.DataFrame(perclass).to_csv(out/'TABLA4_METRICAS_SEGMENTO_REFERENCIA.csv',index=False,float_format='%.8f')
    note={'calibracion':'3-fold StratifiedKFold TRAIN OOF; 80/20 estratificado sobre TRAIN-OOF para selección raw/platt/isotonic; ajuste final sobre todo TRAIN OOF; ningún acceso a TEST para selección','umbrales':'seleccionados sobre 20% TRAIN-OOF predicciones calibradas usando calibrador entrenado en 80% TRAIN-OOF; aplicado calibrador reentrenado con OOF TRAIN total a TEST','semilla':SEED,'split_original':True,'test':len(tt),'train':len(tr)}
    (out/'PROTOCOLO_AVANZADO.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FIN evaluacion avanzada',out,flush=True)
if __name__=='__main__': main()

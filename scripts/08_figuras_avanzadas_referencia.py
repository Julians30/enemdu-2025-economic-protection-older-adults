#!/usr/bin/env python3
"""Figuras de XGBoost y RF, fuentes privadas; publicación solo PNG/SVG."""
from pathlib import Path
import importlib.util, argparse,joblib,shap,numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,precision_recall_curve
from sklearn.preprocessing import minmax_scale
C=['#0B4F77','#06918A','#D48A17','#855DB0']

def save(fig,out,name):
    for ext in ['png','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=600 if ext=='png' else None, bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    a=argparse.ArgumentParser();a.add_argument('--zip',required=True);a.add_argument('--models',required=True);a.add_argument('--avanzado',required=True);a.add_argument('--out',required=True);r=a.parse_args()
    out=Path(r.out);out.mkdir(parents=True,exist_ok=True)
    p=Path(__file__).with_name('06_modelos_referencia_reproducibles.py'); spec=importlib.util.spec_from_file_location('mod_ref',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    df,idx=m.load(r.zip);X,y=m.features(df);xte=X.iloc[idx];yte=y[idx]
    pcal=np.load(Path(r.avanzado)/'prob_cal_xgb.npy');pr=1+pcal.argmax(1);raw=np.load(Path(r.models)/'proba_xgb.npy')
    conf=confusion_matrix(yte,pr,labels=[1,2,3,4]);pd.DataFrame(conf).to_csv(out/'CONFUSION_XGB_CALIBRADO.csv',index=False,header=False)
    # FIG6
    fig,ax=plt.subplots(figsize=(7,5.6),layout='constrained');im=ax.imshow(conf,cmap='Blues');
    for i in range(4):
        for j in range(4): ax.text(j,i, f'{conf[i,j]:,}'.replace(',','.'),va='center',ha='center',fontsize=11,color='white' if conf[i,j]>.6*conf.max() else '#122B40')
    ax.set_xticks(range(4),[f'S{z}' for z in range(1,5)]);ax.set_yticks(range(4),[f'S{z}' for z in range(1,5)]);ax.set_xlabel('Segmento predicho');ax.set_ylabel('Segmento real');ax.set_title('XGBoost calibrado — matriz de confusión',fontweight='semibold');fig.colorbar(im,ax=ax,shrink=.85,label='Observaciones');save(fig,out,'FIGURA6_CONFUSION_XGB_ISOTONIC')
    # FIG A1
    fig,ax=plt.subplots(figsize=(7.6,5.6),layout='constrained')
    for i,c in enumerate(C):
        from sklearn.metrics import roc_curve,auc
        fpr,tpr,_=roc_curve((yte==i+1).astype(int),pcal[:,i]);ax.plot(fpr,tpr,color=c,lw=2,label=f'S{i+1}, AUC = {auc(fpr,tpr):.3f}')
    ax.plot([0,1],[0,1],'--',color='#AAAAAA',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Tasa de falsos positivos',ylabel='Sensibilidad',title='ROC one-vs-rest · XGBoost calibrado');ax.legend(loc='lower right',frameon=False);ax.grid(alpha=.15);save(fig,out,'FIGURA_A1_ROC_XGB_ISOTONIC')
    # FIG5: TRAIN OOF optimization - don't use TEST to choose threshold
    tab=pd.read_csv(Path(r.avanzado)/'CURVA_UMBRAL_xgb_TRAIN_OOF.csv');ts=pd.read_csv(Path(r.avanzado)/'TABLA6_UMBRALES_REFERENCIA.csv');sel=float(ts.loc[(ts.modelo=='xgb')&(ts.escenario=='max_F1_C2_train'),'threshold'].iloc[0]);fig,ax=plt.subplots(figsize=(7.6,5.2),layout='constrained')
    ax.plot(tab.threshold,tab.f1_C2,lw=2,color=C[0],label='F1 del segmento 2');ax.plot(tab.threshold,tab.recall_C2,lw=1.7,color=C[1],label='Sensibilidad segmento 2');ax.axvline(sel,color=C[2],ls='--',label=f'Umbral seleccionado en TRAIN: {sel:.2f}')
    ax.set(xlabel='Umbral de decisión de la clase 2 (TRAIN OOF)',ylabel='Métrica (0–1)',title='Sensibilidad al umbral: selección en TRAIN OOF');ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15);save(fig,out,'FIGURA5_UMBRAL_XGB_TRAIN_OOF')
    # SHAP from a deterministic subset of original TEST
    model=joblib.load(Path(r.models)/'modelo_xgb.joblib');rng=np.random.RandomState(42);positions=rng.choice(len(idx),size=500,replace=False);xs=xte.iloc[positions];z=np.asarray(shap.TreeExplainer(model).shap_values(xs));assert z.shape==(500,X.shape[1],4)
    global_rank=np.mean(np.abs(z),axis=(0,2));ordering=np.argsort(global_rank)[::-1][:12]
    fig,ax=plt.subplots(figsize=(9.4,6.2),layout='constrained');names=[X.columns[k] for k in ordering[::-1]];values=[global_rank[k] for k in ordering[::-1]]
    ax.barh(names,values,color=C[0],alpha=.88);ax.set_xlabel('Media de |SHAP| sobre cuatro segmentos');ax.set_title('Importancia global de los atributos · XGBoost',fontweight='semibold');ax.grid(axis='x',alpha=.14);save(fig,out,'FIGURA7_SHAP_GLOBAL')
    dfimp=pd.DataFrame({'variable':X.columns,'media_abs_SHAP':global_rank}).sort_values('media_abs_SHAP',ascending=False);dfimp.to_csv(out/'SHAP_IMPORTANCIA_GLOBAL.csv',index=False)
    # Small-multiple beeswarm on selected features to avoid misaligned sample-order
    ordering=np.argsort(global_rank)[::-1][:7];fig,axes=plt.subplots(2,2,figsize=(13,9.5),layout='constrained')
    for c,ax in enumerate(axes.flat):
        for j,k in enumerate(ordering):
            vals=z[:,k,c]; v=xs.iloc[:,k].to_numpy();
            # Normalize color positions deterministically using rank-based percentiles.
            vcol=pd.Series(v).rank(pct=True).to_numpy()
            jit=np.random.RandomState(42+k+c*100).uniform(-.25,.25,size=len(vals))
            ax.scatter(vals,np.full(len(vals),j)+jit,c=vcol,cmap='viridis',s=7,alpha=.45,linewidths=0,vmin=0,vmax=1)
        ax.axvline(0,color='#999999',lw=.7);ax.set_yticks(range(len(ordering)),X.columns[ordering]);ax.invert_yaxis();ax.set_xlabel('Contribución SHAP al modelo');ax.set_title(f'Segmento {c+1}');ax.grid(axis='x',alpha=.13)
    fig.suptitle('Contribuciones SHAP por segmento (muestra TEST de 500 registros)',fontweight='semibold');save(fig,out,'FIGURA8_SHAP_POR_SEGMENTO')
    print('OK',[(p.name,p.stat().st_size) for p in out.glob('*.png')])
if __name__=='__main__':main()

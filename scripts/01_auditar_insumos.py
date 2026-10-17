#!/usr/bin/env python3
"""Auditoría local de nombres y hashes del ZIP original, sin extraer microdatos."""
import sys, json, zipfile, hashlib
from pathlib import Path

def main(filename):
    p=Path(filename)
    if not p.exists() or not zipfile.is_zipfile(p):
        raise SystemExit('ERROR: falta el ZIP original o no tiene formato ZIP válido.')
    mandatory=['adultos_mayores_con_cluster.csv']
    recommended=['split_test_idx.npy','metricas_logit.csv','metricas_rf.csv','metricas_xgb.csv','shap_values_xgb.npz']
    scripts=['01_','02_','03_','04_','05_','06_','07_']
    with zipfile.ZipFile(p,'r') as z:
        paths=[i.filename for i in z.infolist() if not i.is_dir()]
        unsafe=[x for x in paths if x.startswith('/') or '..' in Path(x).parts]
        if unsafe:raise SystemExit('ERROR: se han detectado rutas inseguras en ZIP.')
    sha=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    found={s: [x for x in paths if Path(x).name==s] for s in mandatory+recommended}
    group_scripts={s:[x for x in paths if Path(x).name.startswith(s) and x.endswith('.py')] for s in scripts}
    report={'zip_name':p.name,'bytes':p.stat().st_size,'sha256':sha.hexdigest(),
            'archivos_total':len(paths),'insumos':found,'scripts_por_prefijo':group_scripts,
            'obligatorios_presentes':all(found[s] for s in mandatory),
            'obligatorios_ausentes':[s for s in mandatory if not found[s]],
            'observacion':'Este informe es inventario, NO una ejecución reproducida ni validación de los resultados.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['obligatorios_presentes'] else 2
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Uso: python scripts/01_auditar_insumos.py /ruta/enedmu_adultos_mayores.zip')
    sys.exit(main(sys.argv[1]))

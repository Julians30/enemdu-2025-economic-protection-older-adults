#!/usr/bin/env python3
"""Validación ESTÁTICA de publicación: evita filtrar microdatos y revisa el entorno.

No reproduce los entrenamientos. La coherencia de métricas públicas se verifica
por separado mediante scripts/09_verificar_entrega_referencia.py.
"""
from pathlib import Path
import ast
import csv
import json

root=Path(__file__).resolve().parents[1]
private_extensions={'.zip','.xlsx','.sav','.dta','.parquet','.npy','.npz','.joblib','.pkl','.pickle'}
private_dirs={'data','datos','raw','private','inputs','outputs','resultados'}
allowed_aggregates={
'evidencias/validacion_agrupada_panel_2026.csv',
'evidencias/referencia_supervisada_2026/TABLA3_MODELOS_CANONICOS.csv',
'evidencias/referencia_supervisada_2026/TABLA4_POR_CLASE.csv',
'evidencias/referencia_supervisada_2026/TABLA5_CALIBRACION.csv',
'evidencias/referencia_supervisada_2026/TABLA6_UMBRALES.csv',
'evidencias/referencia_supervisada_2026/TABLA_A3_SENSIBILIDAD_RF_UMBRAL.csv',
}
forbidden_csv_headers={'id_persona','cedula','cédula','nombres','apellidos','direccion',
                       'nombre','telefono','email','correo','person_id','row_id','indice'}
errors=[]
for file in sorted(root.rglob('*')):
    if not file.is_file() or '.git' in file.parts or '__pycache__' in file.parts:
        continue
    rel=file.relative_to(root)
    name=rel.as_posix()
    if file.suffix.lower() in private_extensions or any(s in private_dirs for s in rel.parts[:-1]):
        errors.append(f'Archivo privado/no permitido: {name}')
    if file.suffix.lower()=='.csv':
        if name not in allowed_aggregates:
            errors.append(f'CSV sin autorización explícita: {name}')
        elif file.stat().st_size>100000:
            errors.append(f'CSV agregado demasiado grande: {name}')
        else:
            with file.open(encoding='utf-8-sig',newline='') as fd:
                reader=csv.reader(fd)
                head=next(reader,[])
                if forbidden_csv_headers.intersection({x.casefold().strip() for x in head}):
                    errors.append(f'Columna personal no permitida: {name}')
                if sum(1 for _ in reader)>200:
                    errors.append(f'CSV contiene demasiadas filas para ser agregado: {name}')
    if file.suffix=='.py':
        try: ast.parse(file.read_text(encoding='utf-8'),filename=name)
        except SyntaxError as exc: errors.append(f'Sintaxis {name}: {exc}')
    if file.suffix=='.ipynb':
        try:
            notebook=json.loads(file.read_text(encoding='utf-8'))
            for i,cell in enumerate(notebook.get('cells',[])):
                meta=cell.get('metadata',{})
                if any(k in meta for k in ('executionInfo','user','widget','base_uri')):
                    errors.append(f'Metadatos de ejecución {name}, celda {i}')
                if cell.get('cell_type')=='code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                    errors.append(f'Salida de ejecución {name}, celda {i}')
        except Exception as exc: errors.append(f'Notebook ilegible {name}: {exc}')
try:
    c=json.loads((root/'config/parametros_documentados.json').read_text(encoding='utf-8'))
    if abs(float(c['historico_rf']['accuracy'])-0.5308)>1e-9:
        errors.append('Se modificó el desempeño del RF histórico')
    if abs(float(c['nueva_referencia']['rf']['accuracy'])-0.5504)>1e-9:
        errors.append('Se modificó el RF de referencia')
    if c['validacion_externa_clustering'] is not False:
        errors.append('No se puede declarar validación externa de clusters')
    if c['reproducibilidad_supervisada_interna_verificada'] is not True:
        errors.append('El estado de verificación interna cambió sin revisión')
except Exception as exc:
    errors.append(f'Configuración inválida: {exc}')
if errors:
    for e in errors:print('ERROR:',e)
    raise SystemExit(1)
print('OK: sintaxis, notebooks, privacidad y CSV públicos autorizados.')
print('Nota: comprobación estática, no reentrenamiento científico.')

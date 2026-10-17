#!/usr/bin/env python3
"""Verificar estructura y controles editoriales sin ejecutar modelos ni falsificar resultados."""
import json,sys
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parent.parent
cfg=json.loads((root/'config'/'parametros_documentados.json').read_text(encoding='utf-8'))
assert cfg['rf_definitivo_metrica_documentada']['accuracy']==0.5308
assert cfg['tabla_A2_RF_oficial_comprobada'] is False
for f in sorted((root/'notebooks').glob('*.ipynb')):
 nb=nbformat.read(f,as_version=4)
 nbformat.validate(nb)
 assert all(c.get('execution_count') is None and not c.get('outputs',[]) for c in nb.cells if c.cell_type=='code'),f'{f}: contiene salidas'
 assert len(nb.cells)>5
 print('OK:',f.name,len(nb.cells),'celdas')
for required in ('README.md','.gitignore','requirements_por_verificar.in','docs/ESTADO_OBSERVACIONES_TUTOR.md'):
 assert (root/required).exists(),required
print('CONTROLES DE PAQUETE OK. No equivale a ejecución del flujo analítico.')

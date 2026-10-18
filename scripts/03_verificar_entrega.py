#!/usr/bin/env python3
"""Verificador estructural de la entrega activa; no entrena modelos."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/"config"/"parametros_documentados.json").read_text(encoding="utf-8"))
assert abs(cfg["nueva_referencia"]["rf"]["accuracy"]-0.5504)<1e-9
assert cfg["reproducibilidad_supervisada_interna_verificada"] is True
assert cfg["validacion_externa_clustering"] is False
for target in ["README.md","requirements_referencia_fija.txt","scripts/06_modelos_referencia_reproducibles.py","scripts/07_evaluacion_avanzada_referencia.py","scripts/08_figuras_avanzadas_referencia.py"]:
    assert (root/target).is_file(),target
print("OK: estructura y configuración de la evaluación supervisada de referencia.")
print("Esta prueba es estática; los resultados científicos se verifican en scripts/09_verificar_entrega_referencia.py.")

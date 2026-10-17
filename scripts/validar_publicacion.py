#!/usr/bin/env python3
"""Comprobaciones estáticas de publicación; no reproducen las métricas."""
from pathlib import Path
import ast
import json

root = Path(__file__).resolve().parents[1]
extensions = {".zip", ".csv", ".xlsx", ".sav", ".dta", ".parquet", ".npy", ".npz", ".joblib", ".pkl", ".pickle"}
private_dirs = {"data", "datos", "raw", "private", "inputs", "outputs", "resultados"}
errors = []
for file in sorted(root.rglob("*")):
    if not file.is_file() or ".git" in file.parts or "__pycache__" in file.parts:
        continue
    rel = file.relative_to(root)
    if file.suffix.lower() in extensions or any(x in private_dirs for x in rel.parts[:-1]):
        errors.append(f"Archivo privado/no permitido: {rel}")
    if file.suffix == ".py":
        try:
            ast.parse(file.read_text(encoding="utf-8"), filename=str(rel))
        except SyntaxError as exc:
            errors.append(f"Sintaxis {rel}: {exc}")
    if file.suffix == ".ipynb":
        try:
            notebook = json.loads(file.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                metadata = cell.get("metadata", {})
                if any(k in metadata for k in ("executionInfo", "user", "widget", "base_uri")):
                    errors.append(f"Metadatos de ejecución {rel}, celda {index}")
                if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                    errors.append(f"Salida de ejecución {rel}, celda {index}")
        except Exception as exc:
            errors.append(f"Notebook ilegible {rel}: {exc}")
try:
    config = json.loads((root / "config/parametros_documentados.json").read_text(encoding="utf-8"))
    if config["rf_definitivo_metrica_documentada"]["accuracy"] != 0.5308:
        errors.append("Accuracy oficial RF diferente a 0,5308")
    if config["reproducibilidad_integral_demostrada"] is not False:
        errors.append("Reproducibilidad integral no debe declararse demostrada")
except Exception as exc:
    errors.append(f"Configuración inválida: {exc}")
if errors:
    for item in errors:
        print("ERROR:", item)
    raise SystemExit(1)
print("OK: sintaxis, metadatos de notebooks y exclusión de datos.")
print("Esta validación NO reproduce las métricas del estudio.")

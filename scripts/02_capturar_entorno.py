#!/usr/bin/env python3
"""Captura del entorno real de la ejecución. NO simula las versiones originales."""
import sys, subprocess, argparse, platform, json, pathlib, hashlib, datetime

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output_dir',required=True)
    p.add_argument('--code_dir',default='.')
    args=p.parse_args()
    out=pathlib.Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'requirements_exactos.txt').write_text(packages,encoding='utf-8')
    codes=pathlib.Path(args.code_dir)
    files=[]
    for path in sorted(codes.rglob('*')):
        if path.is_file() and path.suffix in {'.py','.ipynb','.json','.yml','.yaml'}:
            if out in path.parents:continue
            files.append({'path':str(path.relative_to(codes)),
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    env={'fecha_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'python':sys.version,'platform':platform.platform(),'executable':sys.executable,
         'nota':'Capturado durante la ejecución actual; no garantiza coincidencia con 2025 si se ejecuta en otro ambiente.',
         'archivos_codigo':files}
    (out/'entorno_real.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Entorno capturado en',out)
if __name__=='__main__':main()

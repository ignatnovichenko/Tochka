#!/usr/bin/env python3
"""Build a deterministic ZIP containing code, small public data and evidence, not weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_ROOTS = {'runs/verified_import', 'runs/spreadsheet_recheck', 'runs/e2e_qwen_cot', 'runs/business_neural_smoke'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent/'nir_mvp_release.zip')
    args = parser.parse_args()
    output = args.output.resolve()
    excluded_runs = set()
    for mp in (ROOT/'runs').glob('*/manifest.json'):
        try:
            state=json.loads(mp.read_text()).get('state')
        except (ValueError,OSError):
            raise RuntimeError('Invalid run manifest during packaging') from None
        if state in {'running','active','interrupted'}:
            excluded_runs.add(mp.parent.relative_to(ROOT).as_posix())
    files = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.resolve() == output or path.name == 'RELEASE_MANIFEST.json':
            continue
        if {'__pycache__', '.venv', '.git', 'wheel_check'} & set(path.parts) or path.suffix in {'.pyc', '.safetensors', '.gguf', '.so', '.whl', '.dll', '.dylib', '.pt', '.pth'} or path.name == '.env':
            continue
        if any(relative == prefix or relative.startswith(prefix+'/') for prefix in EXCLUDE_ROOTS | excluded_runs):
            continue
        files.append({'path':relative,'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest=ROOT/'RELEASE_MANIFEST.json'
    manifest.write_text(json.dumps({'name':'nir-eval-mvp','version':'0.1.0',
                                   'includes_model_weights':False,'includes_auth_profile':False,
                                   'excluded_unfinished_runs':sorted(excluded_runs | EXCLUDE_ROOTS),'files':files},indent=2)+'\n')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for relative in [entry['path'] for entry in files]+['RELEASE_MANIFEST.json']:
            info=zipfile.ZipInfo('mvp/'+relative,(2020,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o600 << 16
            archive.writestr(info,(ROOT/relative).read_bytes(),compresslevel=6)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC validation failed')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                      'files':len(files)+1},indent=2))


if __name__=='__main__':main()

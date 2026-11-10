#!/usr/bin/env python3
"""Check bundled CPU/GPU measurements; no inference, downloads or network calls."""
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/performance'
GPU=BASE/'colab_gpu_speed_probe'


def main():
    archive=GPU/'NIR_Qwen7B_T4_py312_results.zip'
    expected='af09dabc9942f310bc8b81d6487d02f0b972fe86776ead638b1506298868c172'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
        raise ValueError('Original GPU export ZIP hash mismatch')
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:raise ValueError('Corrupt GPU export ZIP')
        for item in source.infolist():
            name=PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe ZIP entry')
            if item.is_dir():continue
            exported=GPU/'artifacts'/item.filename
            if source.read(item)!=exported.read_bytes():raise ValueError('Exported artifact differs: '+item.filename)
    subprocess.run([sys.executable,str(BASE/'compare_cpu_gpu_probe.py'),
                    str(GPU/'artifacts/gpu_speed_probe'),'--offload-log',str(GPU/'artifacts/model_load.log'),
                    '--output',str(ROOT/'runs/hardware_speed_comparison')],check=True)


if __name__=='__main__':main()

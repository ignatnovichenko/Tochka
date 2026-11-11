"""Bounded application-source fingerprints; never scan installed dependencies."""
import hashlib
import os
from pathlib import Path


def source_hashes():
    root = Path(__file__).parent
    paths = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {'node_modules','__pycache__'}]
        paths.extend(Path(directory)/name for name in files if name.endswith('.py'))
    paths += [root/'promptfoo_runtime'/name for name in ('package.json','package-lock.json','worker.mjs')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}

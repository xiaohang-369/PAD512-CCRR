"""Integrity verification and small JSON/log return export; no model export."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,entry in manifest['files'].items():
        p=root/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
            raise ValueError(f'Package file mismatch: {name}')
    print('PACKAGE_VERIFIED files='+str(len(manifest['files'])),flush=True)


def export(run):
    files=sorted(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log'))
    archive=run/'rank-zero-return.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(run)),recursive=False)
    print('RETURN_ARCHIVE='+str(archive),flush=True)
    print('ARCHIVE_BYTES='+str(archive.stat().st_size),flush=True)
    print('ARCHIVE_SHA256='+sha(archive),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('verify','export'))
    ap.add_argument('path',type=Path);args=ap.parse_args()
    verify(args.path) if args.mode=='verify' else export(args.path)

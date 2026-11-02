"""Verify exact historical server assets without loading models or changing files."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check(bindings):
    errors=[]
    for name,expected in bindings.items():
        p=Path(name)
        if not p.is_file():errors.append(dict(path=name,error='missing'))
        elif digest(p)!=expected:errors.append(dict(path=name,error='sha256_mismatch'))
    return dict(passed=not errors,checked_files=len(bindings),errors=errors,optimizer_updates=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bindings',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();report=check(json.loads(args.bindings.read_text())['files'])
    if args.output:
        with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()

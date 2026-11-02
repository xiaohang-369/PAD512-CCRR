"""Read-only fit coverage audit. No non-fit pixels/features are used for loss."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import torch
from ccrr.data.pad import preprocess
from ccrr_v4.rank_identity import fit_spec


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = args.root / 'protocol_audit_20261009'
    files = [root/'roles'/f'{r}.json' for r in ('fit','calibration','val_tune','val_select','test')]
    files += [root/'preprocessing_inventory.json', args.root/'raw/metadata.csv']
    before = {str(p): sha(p) for p in files}
    roles = {p.stem: json.loads(p.read_text()) for p in files[:5]}
    rows = roles['fit']
    inventory = {r['img_id']: r for r in json.loads(files[5].read_text())}
    hashes = {k: r['rgb512_sha256'] for k,r in inventory.items()}
    with files[6].open() as f:
        meta = {r['img_id']:r for r in csv.DictReader(f)}
    for row in rows:
        assert all(row[k] == meta[row['img_id']][k] for k in ('patient_id','lesion_id','diagnostic'))
    def verify(row):
        p = args.root/row['source_relpath']
        before_raw = sha(p)
        actual = preprocess(p)[-1]
        assert actual == hashes[row['img_id']], row['img_id']
        assert sha(p) == before_raw
        return row['img_id'], before_raw
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = dict(pool.map(verify, rows))
    spec = fit_spec(rows, role='fit', content_hashes=hashes)
    raw_groups = defaultdict(list)
    pixel_groups = defaultdict(list)
    for r in rows:
        raw_groups[raw[r['img_id']]].append(r['img_id'])
        pixel_groups[hashes[r['img_id']]].append(r['img_id'])
    overlap = {}
    for role, other in roles.items():
        if role == 'fit': continue
        overlap[role] = {
            k: len({r[k] for r in rows} & {r[k] for r in other})
            for k in ('img_id','patient_id')
        }
        overlap[role]['rgb512_hash'] = len({hashes[r['img_id']] for r in rows} & {hashes[r['img_id']] for r in other})
    valid = spec['valid'].tolist()
    coverage = []
    for i,r in enumerate(rows):
        coverage.append(dict(index=i, img_id=r['img_id'], patient_id=r['patient_id'],
            lesion_id=r['lesion_id'], valid=valid[i], positives=int(spec['positive'][i].sum()),
            negatives=int(spec['negative'][i].sum()), fixed_weight=float(spec['weights'][i])))
    protocol_path = root/'pair_protocol/fit_p2.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_valid = {q['query'] for q in protocol['queries']}
    report = dict(fit_images=len(rows), fit_patients=len({r['patient_id'] for r in rows}),
        eligible_images=sum(r.get('reid_observation_eligible','1')=='1' for r in rows),
        valid_queries=sum(valid), valid_patients=len({r['patient_id'] for r,v in zip(rows,valid) if v}),
        positive_lesion_groups=len({(r['patient_id'],r['lesion_id']) for r,v in zip(rows,valid) if v}),
        invalid_queries=len(rows)-sum(valid), directed_positive_pairs=int(spec['positive'].sum()),
        positive_count_histogram=dict(Counter(r['positives'] for r in coverage if r['valid'])),
        excluded_reason='no_legal_cross_image_same_lesion_positive',
        raw_duplicate_groups=[v for v in raw_groups.values() if len(v)>1],
        rgb512_duplicate_groups=[v for v in pixel_groups.values() if len(v)>1],
        metadata_rows_match=True, actual_fit_preprocessing_hashes_verified=len(raw),
        role_overlap=overlap, fit_protocol_queries=len(protocol_valid),
        fit_protocol_query_symmetric_difference=len(protocol_valid ^ {r['img_id'] for r,v in zip(rows,valid) if v}),
        input_sha256={**before, str(protocol_path):sha(protocol_path)},
        parents_unchanged=before=={str(p):sha(p) for p in files},
        source_pixels_read_roles=['fit'], optimizer_updates=0)
    (args.output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'queries.json').write_text(json.dumps(coverage,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()

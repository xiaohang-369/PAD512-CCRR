"""Exercise the complete server probe with tiny models, not server assets."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from ccrr_v4.verify import state_digest


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_full_zero_entry_uses_no_optimizer_and_reports_weighted_medical_gradients(tmp_path,monkeypatch):
    v4=Path(__file__).resolve().parents[2]
    fixtures=load(v4/'tests/test_phase2_g1.py','rank_tiny_fixture')
    probe=load(v4/'rank_preflight/scripts/full_zero.py','rank_full_probe')
    cfg,ctx,_,_=fixtures.fixture()
    rows=[dict(img_id=str(i),patient_id=str(i//2),lesion_id='lesion') for i in range(80)]
    audit=tmp_path/'data/protocol_audit_20261009';audit.mkdir(parents=True)
    (audit/'preprocessing_inventory.json').write_text(json.dumps([
        dict(img_id=str(i),rgb512_sha256=str(i)) for i in range(80)]))
    class Data:
        root=tmp_path/'data'
        def __init__(self):self.rows=rows
        def __len__(self):return 80
        def class_weights(self):return torch.ones(6)
    ctx.update(baseline=deepcopy(ctx['model']).eval().requires_grad_(False),
        datasets={'fit':Data()},arrays={'fit':np.random.default_rng(7).integers(0,256,(80,8,8,3),dtype=np.uint8)},parents={})
    bank=dict(rgb=torch.randn(80,6),g1=torch.randn(80,6),target=torch.randn(80,6).softmax(1),
        labels=torch.arange(80)%6,rects=[(0,0,8,8)]*80)
    bank['rgb']=torch.nn.functional.normalize(bank['rgb'],dim=1)
    bank['g1']=torch.nn.functional.normalize(bank['g1'],dim=1)
    def build(ctx,cfg,out,*args):
        torch.save(bank,out/'bank-zero_only-rank0.pt');return bank
    monkeypatch.setattr(probe,'load_context',lambda *args:ctx)
    monkeypatch.setattr(probe,'build_fit_bank',build)
    cfgpath=tmp_path/'config.json';cfgpath.write_text(json.dumps(cfg))
    output=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['probe','--config',str(cfgpath),'--device','cpu','--output',str(output)])
    # Any optimizer construction is an error, even with tiny models.
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',lambda *a,**k:(_ for _ in ()).throw(AssertionError('optimizer forbidden')))
    before=state_digest(ctx['model']);probe.main()
    report=json.loads((output/'report.json').read_text())
    assert report['optimizer_updates']==0 and not report['optimizer_constructed']
    assert len(report['panels'])==9 and report['all_gradient_query_and_gallery_roles']==['fit']
    assert report['panels'][0]['E1']['l2']['medical_0.1']>0
    assert state_digest(ctx['model'])==before

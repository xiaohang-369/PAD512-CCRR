"""Real fit-only CPU zero-update probe; no optimizer is constructed.

Uses exact E0/D0 epoch120 and public ResNet50. Missing medical separator is
explicitly not substituted. Subset scores are not historical val_tune Rank-1.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from ccrr.data.pad import PADData
from ccrr.teachers.adapters import IdentityModel
from ccrr_v4.original import OriginalGrayscale
from ccrr_v4.phase2_objectives import digital_supervision, retrieval_confusion
from ccrr_v4.rank_identity import fit_spec, rank_loss, vector_summary
from ccrr_v4.losses import author_loss
from ccrr_v4.phase1.vgg import AuthorVGG
from ccrr_v4.verify import digest, state_digest


def flat(grads, params):
    return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g,p in zip(grads,params)]).float().cpu()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--resnet',type=Path,required=True)
    ap.add_argument('--vgg',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=32)
    ap.add_argument('--gallery',type=int,default=96)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(20261102)
    paths=[args.checkpoint,args.resnet,args.vgg]
    hashes={str(p):digest(p) for p in paths}
    assert hashes[str(args.checkpoint)]=='5d7f8c2bed98356166bbecebbc87a7fb4324e75126a347e6173788a7fcbef387'
    assert hashes[str(args.resnet)].startswith('11ad3fa6')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert ck['epoch']==120 and ck['step']==10800
    model=OriginalGrayscale().eval();model.load_state_dict(ck['model'],strict=True)
    weights=ck['binding']['config']['stages'][1]['weights'];del ck
    # Same embedding construction as ccrr_v2.reid.initialize; importing that
    # training module needlessly loads SciPy metrics on the local CPU runtime.
    reid=IdentityModel(863)
    state=torch.load(args.resnet,map_location='cpu',weights_only=True)
    reid.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
    reid.eval().requires_grad_(False)
    before={'ED':state_digest(model),'resnet':state_digest(reid)}
    data=PADData(args.root,'fit')
    inventory=json.loads((args.root/'protocol_audit_20261009/preprocessing_inventory.json').read_text())
    spec=fit_spec(data.rows,role='fit',content_hashes={r['img_id']:r['rgb512_sha256'] for r in inventory})
    rng=np.random.default_rng(20261102)
    valid=np.flatnonzero(spec['valid'].numpy());rng.shuffle(valid)
    # Patient-diverse deterministic sample, then include every legal positive.
    chosen=[];seen=set()
    for i in valid:
        patient=data.rows[i]['patient_id']
        if patient not in seen:chosen.append(int(i));seen.add(patient)
        if len(chosen)==args.queries:break
    gallery=set(chosen)
    for i in chosen:gallery.update(torch.where(spec['positive'][i])[0].tolist())
    for i in rng.permutation(len(data)):
        if len(gallery)>=args.gallery:break
        gallery.add(int(i))
    gallery=sorted(gallery);lookup={i:j for j,i in enumerate(gallery)}
    pos=spec['positive'][:,gallery];neg=spec['negative'][:,gallery]
    embeddings=[]; started=time.monotonic()
    with torch.no_grad():
        for t,i in enumerate(gallery):
            item=data[i]
            embeddings.append(reid(item['x'][None],[item['rect']])['embedding'][0])
            if (t+1)%16==0:print(f'RGB_BANK {t+1}/{len(gallery)}',flush=True)
    bank=torch.stack(embeddings).detach();bank_before=bank.clone()
    records=[];mini_gray={};feature_examples=[]
    for k,i in enumerate(chosen):
        item=data[i];x=item['x'][None]*2-1;rects=[item['rect']]
        with torch.no_grad():g=model.encoder(x)
        leaf=g.detach().requires_grad_()
        q=reid(digital_supervision(leaf),rects)['embedding']
        pn=(pos[i:i+1],neg[i:i+1])
        with torch.no_grad():
            s=F.normalize(q,dim=1)@bank.T
            anchor=s.masked_fill(~pn[1],-torch.inf).max(1).values
        entry={'fit_index':i,'subset_gallery_size':len(gallery),'variants':{}}
        for mode in ['live','stop_negative','anchor']:
            loss,stats=rank_loss(q,bank,*pn,mode=mode,anchor=anchor)
            grad=torch.autograd.grad(loss,leaf,retain_graph=True)[0].detach()
            qgrad=torch.autograd.grad(loss,q,retain_graph=True)[0].detach()
            # Embedding-only descent direction; NOT a model or image update.
            dq=-1e-3*qgrad/qgrad.norm().clamp_min(1e-12)
            with torch.no_grad():
                _,after=rank_loss(q.detach()+dq,bank,*pn,mode=mode,anchor=anchor)
            entry['variants'][mode]=dict(loss=float(loss),a=float(stats['a']),b=float(stats['b']),
                gap=float(stats['gap']),positive_top1=bool(stats['strict_positive_top1']),
                gray_gradient_l2=float(grad.norm()),q_gradient_l2=float(qgrad.norm()),
                feature_direction_delta_a=float(after['a']-stats['a']),
                feature_direction_delta_b=float(after['b']-stats['b']))
            if k<2:mini_gray[k,mode]=grad
        old=retrieval_confusion(q,bank,[lookup[i]],.1)
        old_grad=torch.autograd.grad(old,leaf)[0].detach()
        entry['old_cross_rgb_only']=dict(loss=float(old),gray_gradient_l2=float(old_grad.norm()))
        if k<2:
            mini_gray[k,'old_cross_rgb']=old_grad
            feature_examples.append((i,x,g.detach(),rects))
        records.append(entry)
        print(f'FIT_QUERY {k+1}/{len(chosen)} gray_vjp_ok',flush=True)
        del q,leaf,grad,old_grad
    # Fixed mini-batch of two real images, exact full E1 parameter gradients.
    x=torch.cat([e[1] for e in feature_examples]);g=model.encoder(x)
    eparams=list(model.encoder.parameters());dparams=list(model.decoder.parameters())
    vectors={};gvectors={}
    for mode in ['live','stop_negative','anchor','old_cross_rgb']:
        cotangent=torch.cat([mini_gray[k,mode] for k in range(2)])/2
        cotangent*=.1 # diagnostic reference weight only
        vectors['0.1_'+mode]=flat(torch.autograd.grad(g,eparams,grad_outputs=cotangent,retain_graph=True),eparams)
        gvectors['0.1_'+mode]=cotangent.flatten().cpu()
    vgg=AuthorVGG.from_file(args.vgg,hashes[str(args.vgg)])
    with torch.no_grad(): target=vgg(x)
    restored=model.decoder(g)
    total,terms=author_loss(x,g,restored,target,vgg(g),2,weights)
    for name,loss in [('author_total',total),('3_reconstruction',3*terms['mse'])]:
        gvectors[name]=torch.autograd.grad(loss,g,retain_graph=True)[0].detach().flatten().cpu()
        vectors[name]=flat(torch.autograd.grad(loss,eparams,retain_graph=True),eparams)
    dgrad=flat(torch.autograd.grad(total,dparams),dparams)
    # Actual continuous-vs-digital D1 diagnostics at initialization; no PNG file claim.
    with torch.no_grad():
        qg=digital_supervision(g)*2-1
        png_value_psnr=-10*(((model.decoder(qg)-x)/2).square().mean((1,2,3))).log10()
    after={'ED':state_digest(model),'resnet':state_digest(reid)}
    report=dict(optimizer_updates=0,optimizer_constructed=False,device='cpu',torch_version=torch.__version__,
        query_role='fit',gallery_role='fit',queries=len(chosen),query_patients=len(seen),gallery=len(gallery),
        seed=20261102,input_sha256=hashes,models_unchanged=before==after,
        gallery_unchanged=torch.equal(bank,bank_before),all_parameter_grad_buffers_empty=all(p.grad is None for p in model.parameters()),
        frozen_resnet_grads_empty=all(p.grad is None for p in reid.parameters()),
        parents_unchanged=hashes=={str(p):digest(p) for p in paths},
        mini_batch_fit_indices=chosen[:2],e1_gradients=vector_summary(vectors),gray_gradients=vector_summary(gvectors),
        D1_author_gradient_l2=float(dgrad.norm()),author_terms={k:float(v) for k,v in terms.items()},
        D1_digital_value_psnr=png_value_psnr.tolist(),
        medical_gradient_status='not_run_missing_bound_separator_8500_heads_stats',
        limitations=['subset fit gallery, not historical evaluation','old comparison is cross RGB only, not historical two-bank KL',
            'feature direction is infinitesimal geometry test, not a trained-model result',
            'no PNG save/reload or full medical supervision in this local probe'],
        elapsed_seconds=time.monotonic()-started,records=records)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()

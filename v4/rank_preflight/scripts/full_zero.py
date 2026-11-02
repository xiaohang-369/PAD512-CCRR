"""Server-side zero-update diagnostic ONLY. Requires original bound V4 assets.

No optimizer, checkpoint export, validation fitting, or training mode exists.
Reads val_tune metadata/baseline bindings through load_context for consistency;
all generated images and gradients in this script are strictly fit-only.
"""
import argparse
import json
from pathlib import Path
import torch
from ccrr_v4.phase2_g1 import load_context, build_fit_bank, check_frozen, verify_files
from ccrr_v4.phase2_baseline import numeric_policy
from ccrr_v4.phase2_objectives import digital_supervision, medical_objective, retrieval_confusion, validate_config
from ccrr_v4.phase1.data import batch_tensor
from ccrr_v4.phase1.optim import ContinuousBatches
from ccrr_v4.losses import author_loss
from ccrr_v4.rank_identity import fit_spec, rank_loss, vector_summary
from ccrr_v4.verify import read, state_digest, digest


def flatten(grads,params):
    return torch.cat([(v.detach().float() if v is not None else torch.zeros_like(p,dtype=torch.float32)).flatten().cpu()
                      for v,p in zip(grads,params)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--device',required=True,help='Explicit device, e.g. cuda:0')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tau',type=float,default=.1)
    ap.add_argument('--margin',type=float,default=0.)
    ap.add_argument('--batch-size',type=int,choices=(1,2,4,8,16),default=2)
    ap.add_argument('--fit-queries',type=int,default=16)
    args=ap.parse_args()
    cfg=read(args.config);validate_config(cfg)
    if args.fit_queries<args.batch_size or args.fit_queries%args.batch_size:
        raise ValueError('fit-queries must be a positive multiple of batch-size')
    if 'resume' in cfg or cfg['update_modules']!=['E1','D1']:
        raise ValueError('Fresh joint E0/D0 copy only; no old E1/Adam state')
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    ctx=load_context(cfg,device,0)
    ctx['class_weights']=ctx['datasets']['fit'].class_weights().to(device)
    before=state_digest(ctx['model'])
    assert before==state_digest(ctx['baseline'])
    bank=build_fit_bank(ctx,cfg,args.output,device,0,1,'zero_only')
    rgb_hash=digest(args.output/'bank-zero_only-rank0.pt')
    data=ctx['datasets']['fit']
    inv=read(data.root/'protocol_audit_20261009/preprocessing_inventory.json')
    spec=fit_spec(data.rows,role='fit',content_hashes={r['img_id']:r['rgb512_sha256'] for r in inv})
    spec={k:v.to(device) for k,v in spec.items()}
    with torch.no_grad():
        initial_scores=bank['g1']@bank['rgb'].T
        anchors=initial_scores.masked_fill(~spec['negative'],-torch.inf).max(1).values
    stream=ContinuousBatches(len(data),16,cfg['seed'])
    panels=[stream.next()[:args.batch_size]]
    valid=torch.where(spec['valid'])[0].cpu().tolist()
    perm=torch.randperm(len(valid),generator=torch.Generator().manual_seed(20261102)).tolist()
    if args.fit_queries>len(valid):raise ValueError('Too few legal fit queries')
    panels.extend([[valid[i] for i in perm[start:start+args.batch_size]]
                   for start in range(0,args.fit_queries,args.batch_size)])
    eparams=list(ctx['model'].encoder.parameters())
    dparams=list(ctx['model'].decoder.parameters())
    reports=[]
    for panel_id,ids in enumerate(panels):
        x=batch_tensor(ctx['arrays']['fit'],ids,device)
        with numeric_policy(False):g=ctx['model'].encoder(x)
        rects=[bank['rects'][i] for i in ids]
        cotangents={};values={};vectors={};stats={}
        # Free teacher graphs before allocating decoder/VGG, as in production.
        with numeric_policy(True):
            leaf=g.detach().requires_grad_()
            raw,cov,_=ctx['panderm'](digital_supervision(leaf),rects)
            logits=ctx['teachers']['8500'](raw,cov,1)[0]
            med=medical_objective(logits,bank['labels'][ids],bank['target'][ids],ctx['class_weights'],.5)
            cotangents['medical_0.1']=torch.autograd.grad(.1*med,leaf)[0].detach()
            values['medical_0.1']=float(.1*med)
            del leaf,raw,cov,logits,med
            leaf=g.detach().requires_grad_()
            q=ctx['reid'](digital_supervision(leaf),rects)['embedding']
            old=(retrieval_confusion(q,bank['rgb'],ids,.1)+retrieval_confusion(q,bank['g1'],ids,.1))/2
            cotangents['old_identity_0.1']=torch.autograd.grad(.1*old,leaf,retain_graph=True)[0].detach()
            values['old_identity_0.1']=float(.1*old)
            for mode in ('live','stop_negative','anchor'):
                loss,st=rank_loss(q,bank['rgb'],spec['positive'][ids],spec['negative'][ids],
                    mode=mode,tau=args.tau,margin=args.margin,anchor=anchors[ids],weights=spec['weights'][ids])
                cotangents['rank_'+mode]=torch.autograd.grad(loss,leaf,retain_graph=True)[0].detach()
                values['rank_'+mode]=float(loss)
                stats[mode]={k:v.detach().cpu().tolist() for k,v in st.items()}
            del leaf,q,old,loss,st
        for name,cotangent in cotangents.items():
            vectors[name]=flatten(torch.autograd.grad(g,eparams,grad_outputs=cotangent,retain_graph=True),eparams)
        with numeric_policy(False):
            restored=ctx['model'].decoder(g)
            with torch.no_grad():target=ctx['vgg'](x)
            total,terms=author_loss(x,g,restored,target,ctx['vgg'](g),2,ctx['author_cfg']['stages'][1]['weights'])
            losses={'author_total':total,'reconstruction_3':3*terms['mse']}
            for name,loss in losses.items():
                cotangents[name]=torch.autograd.grad(loss,g,retain_graph=True)[0].detach()
                vectors[name]=flatten(torch.autograd.grad(loss,eparams,retain_graph=True),eparams)
                values[name]=float(loss)
            dgrad=flatten(torch.autograd.grad(total,dparams),dparams)
        weighted={}
        for mode in ('live','stop_negative','anchor'):
            for lam in (.001,.003,.01,.03,.1):
                name=f'{mode}_lambda_{lam}'
                v=lam*vectors['rank_'+mode]
                weighted[name]=vector_summary({'rank_weighted':v,'medical_0.1':vectors['medical_0.1'],
                    'author_total':vectors['author_total'],'reconstruction_3':vectors['reconstruction_3']})
        check_frozen(ctx)
        assert state_digest(ctx['model'])==before
        assert all(p.grad is None for p in ctx['model'].parameters())
        record=dict(panel=panel_id,fit_indices=ids,query_role='fit',losses=values,
            valid_queries=int(spec['valid'][ids].sum()),E1=vector_summary(vectors),
            gray=vector_summary({k:v.flatten().float().cpu() for k,v in cotangents.items()}),
            weighted_diagnostics=weighted,D1_author_l2=float(dgrad.norm()),ranking=stats)
        reports.append(record)
        (args.output/f'panel-{panel_id}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(f'ZERO_PANEL {panel_id} fit={len(ids)} unchanged=True',flush=True)
        del x,g,restored,target,total,terms,losses,vectors,cotangents
    verify_files(ctx['parents'])
    assert digest(args.output/'bank-zero_only-rank0.pt')==rgb_hash
    result=dict(optimizer_updates=0,optimizer_constructed=False,model_unchanged=True,
        frozen_models_unchanged=True,parents_unchanged=True,fit_bank_unchanged=True,
        all_gradient_query_and_gallery_roles=['fit'],panels=reports,
        config_sha256=digest(args.config),parents=ctx['parents'],tau=args.tau,margin=args.margin,
        diagnostic_batch_size=args.batch_size,
        full_batch16_cuda_verified=device.type=='cuda' and args.batch_size==16,
        scientific_acceptance='not_evaluated')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

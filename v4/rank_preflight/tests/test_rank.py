from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from ccrr_v4.rank_identity import fit_spec, score_loss, rank_loss, vector_summary
from ccrr_v4.phase2_objectives import digital_supervision


def masks():
    return torch.tensor([[True,False,False]]), torch.tensor([[False,True,True]])


def test_long_vector_cosines_are_bounded_and_identical_vectors_match():
    a=torch.linspace(1e-8,1.,100000)
    s=vector_summary({'a':a,'b':a.clone(),'zero':torch.zeros_like(a)})
    assert s['cosine']['a__b']==pytest.approx(1.,abs=1e-12)
    assert s['cosine']['a__zero'] is None
    with pytest.raises(ValueError):vector_summary({'bad':torch.tensor([float('nan')])})


@pytest.mark.parametrize('mode',['live','stop_negative','anchor'])
def test_analytic_direction_and_detached_reference(mode):
    s=torch.tensor([[.8,.6,.1]],dtype=torch.float64,requires_grad=True)
    anchor=torch.tensor([.6],dtype=torch.float64,requires_grad=True)
    loss,_=score_loss(s,*masks(),mode=mode,anchor=anchor,tau=.1)
    loss.backward()
    assert s.grad[0,0]>0
    assert (s.grad[0,1]<0) if mode=='live' else (s.grad[0,1]==0)
    assert s.grad[0,2]==0 and anchor.grad is None


def test_live_loss_really_can_be_lowered_by_only_raising_negative():
    a=torch.tensor([[.8,.6,.1]])
    b=torch.tensor([[.8,.9,.1]])
    assert score_loss(b,*masks())[0]<score_loss(a,*masks())[0]
    anchor=torch.tensor([.6])
    torch.testing.assert_close(score_loss(a,*masks(),mode='anchor',anchor=anchor)[0],
                               score_loss(b,*masks(),mode='anchor',anchor=anchor)[0])


def test_live_scores_pass_finite_difference_gradcheck_away_from_ties():
    s=torch.tensor([[.8,.6,.1]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:score_loss(z,*masks())[0],(s,))


def test_stop_positive_can_stall_at_exact_cosine_alignment():
    q=torch.tensor([[1.,0.]],requires_grad=True)
    bank=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
    loss,_=rank_loss(q,bank,*masks(),mode='stop_negative')
    assert torch.equal(torch.autograd.grad(loss,q)[0],torch.zeros_like(q))


def test_fit_weights_full_dataset_equal_patient_average():
    rows=[dict(img_id=str(i),patient_id=p,lesion_id='x')
          for i,p in enumerate(['a','a','b','b','b','c'])]
    spec=fit_spec(rows,role='fit',content_hashes={str(i):str(i) for i in range(6)})
    per=torch.tensor([1.,3.,2.,4.,6.,999.])
    actual=(spec['weights']*per).mean()
    torch.testing.assert_close(actual,torch.tensor(((1+3)/2+(2+4+6)/3)/2))


@pytest.mark.parametrize('mode',['live','stop_negative','anchor'])
def test_geometry_can_raise_negative_even_without_negative_gradient(mode):
    q=torch.tensor([[.8,.6]],dtype=torch.float64,requires_grad=True)
    bank=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],dtype=torch.float64,requires_grad=True)
    loss,old=rank_loss(q,bank,*masks(),mode=mode,anchor=torch.tensor([.6]))
    grad=torch.autograd.grad(loss,q)[0]
    moved=F.normalize(q.detach()-.001*grad,dim=1)
    _,new=rank_loss(moved,bank,*masks(),mode=mode,anchor=torch.tensor([.6]))
    assert new['a']<old['a'] and new['b']>old['b'] and bank.grad is None


def test_masks_source_views_other_lesion_ineligible_and_fit_guard():
    rows=[dict(img_id=str(i),patient_id=p,lesion_id=l) for i,(p,l) in enumerate(
        [('a','x'),('a','x'),('a','z'),('b','y'),('a','x'),('a','x')])]
    rows[4]['source_image_id']='0'; rows[5]['reid_observation_eligible']='0'
    hashes={str(i):str(i) for i in range(6)}
    spec=fit_spec(rows,role='fit',content_hashes=hashes)
    assert spec['positive'][0,1] and not spec['positive'][0,4]
    assert not spec['positive'][0,2] and not spec['negative'][0,2]
    assert spec['negative'][0,3] and not spec['positive'][0,5]
    hashes['1']='0'
    assert not fit_spec(rows,role='fit',content_hashes=hashes)['positive'][0,1]
    with pytest.raises(ValueError):fit_spec(rows,role='val_tune',content_hashes=hashes)


def test_empty_shard_is_finite_differentiable_zero():
    s=torch.randn(2,3,requires_grad=True); p=torch.zeros_like(s,dtype=torch.bool)
    loss,_=score_loss(s,p,~p,weights=torch.zeros(2));loss.backward()
    assert loss==0 and torch.equal(s.grad,torch.zeros_like(s))


def test_fixed_weights_and_unequal_valid_counts_match_ddp_average():
    s=torch.tensor([[.8,.6,.1],[.7,.4,.2],[.9,.4,.1],[.5,.3,.1]],requires_grad=True)
    p=masks()[0].expand(4,-1).clone(); n=masks()[1].expand(4,-1).clone()
    p[2:]=False; w=torch.tensor([2.,2.,0.,0.])
    full=score_loss(s,p,n,weights=w)[0]
    shards=sum(score_loss(s[i:i+2],p[i:i+2],n[i:i+2],weights=w[i:i+2])[0] for i in [0,2])/2
    torch.testing.assert_close(full,shards)
    torch.testing.assert_close(torch.autograd.grad(full,s)[0],torch.autograd.grad(shards,s)[0])


def test_rank_tie_subgradient_and_extreme_tau_finite():
    s=torch.tensor([[.7,.7,.2]],requires_grad=True)
    for tau in [.03,.1,.2]:
        loss,stats=score_loss(s,*masks(),tau=tau)
        assert stats['ties'].item() and torch.isfinite(loss)
    p=torch.tensor([[True,True,False]]);n=~p
    score_loss(s,p,n)[0].backward();assert torch.isfinite(s.grad).all()


@pytest.mark.parametrize('tau',[0.,-1.,float('nan')])
def test_invalid_temperature_rejected(tau):
    with pytest.raises(ValueError):score_loss(torch.ones(1,3),*masks(),tau=tau)


def test_ste_half_derivative_and_staged_vjp_matches_direct_no_update():
    torch.manual_seed(42)
    enc=nn.Conv2d(3,1,1); direct=deepcopy(enc)
    teacher=nn.Sequential(nn.Flatten(),nn.Linear(16,3)).eval().requires_grad_(False)
    x=torch.rand(1,3,4,4);bank=torch.randn(3,3)
    before={k:v.clone() for k,v in enc.state_dict().items()}
    g=enc(x);leaf=g.detach().requires_grad_()
    loss,_=rank_loss(teacher(digital_supervision(leaf)),bank,*masks())
    vjp=torch.autograd.grad(loss,leaf)[0]
    g.backward(vjp)
    g2=direct(x);rank_loss(teacher(digital_supervision(g2)),bank,*masks())[0].backward()
    for a,b in zip(enc.parameters(),direct.parameters()):torch.testing.assert_close(a.grad,b.grad)
    assert all(torch.equal(v,before[k]) for k,v in enc.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    g3=torch.rand(1,1,4,4,requires_grad=True)
    torch.testing.assert_close(torch.autograd.grad(digital_supervision(g3).sum(),g3)[0],torch.full_like(g3,.5))

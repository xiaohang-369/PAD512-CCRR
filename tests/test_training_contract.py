import torch
import numpy as np
from ccrr.representation.inn import INN
from ccrr.representation.purifier import Purifier
from ccrr.representation.purifier import SemanticAttacker
from ccrr.evaluation.attacks import ReconstructionAttack
from ccrr.codec.entropy import EntropyModel
from ccrr.training.losses import reconstruction, saturation, gray_loss
from ccrr.tensor_ops import q8, ste_round, public_adapter
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.data.pad import PairSampler
from ccrr.evaluation.gates import deployment_gate
from ccrr.runtime.preflight import inspect


def test_representation_training_update_and_resume(tmp_path):
    torch.manual_seed(4)
    m = INN()
    entropy = EntropyModel()
    opt = optimizer([m, entropy], [1e-4, 1e-3])
    x = torch.rand(1, 3, 32, 32)
    v = torch.ones(1, 1, 32, 32)
    before = m.levels[0][0].net[-1].weight.detach().clone()
    u, z = m(x)
    g = q8(u.sigmoid())
    n = ste_round(z * 32)
    rec = m.inverse(public_adapter(g), n / 32)
    loss = (
        reconstruction(x, rec, v)
        + 0.1 * gray_loss(g, x, v)
        + 1e-3 * saturation(u, v)
        + 0.01 * entropy.rate(z, 1, 1 / 32).mean()
    )
    opt.zero_grad()
    loss.backward()
    schedule(opt, 0, 10000)
    update(opt)
    assert not torch.equal(before, m.levels[0][0].net[-1].weight)
    path = tmp_path / "c.pt"
    save_checkpoint(
        path,
        {"inn": m, "entropy": entropy},
        opt,
        1,
        "C",
        {"numpy_score": np.float64(3)},
    )
    m2 = INN()
    e2 = EntropyModel()
    o2 = optimizer([m2, e2], [1e-4, 1e-3])
    state = load_checkpoint(path, {"inn": m2, "entropy": e2}, o2)
    assert state["step"] == 1
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))


def test_freeze_adversary_keeps_generator_pair_gradient():
    p = Purifier()
    att = SemanticAttacker()
    v = torch.ones(1, 1, 14, 14)
    for w in att.parameters():
        w.requires_grad_(False)
    a = p(torch.randn(1, 768, 14, 14), v)["semantic"]
    b = p(torch.randn(1, 768, 14, 14), v)["semantic"]
    loss = -torch.nn.functional.binary_cross_entropy_with_logits(
        att(a, v, b, v), torch.ones(1)
    )
    loss.backward()
    assert p.encoder.start.weight.grad.abs().sum() > 0 and all(
        w.grad is None for w in att.parameters()
    )


def test_pair_sampling_never_fakes_patient_or_lesion_relations():
    class Data:
        rows = [
            {
                "patient_id": f"p{p}",
                "lesion_id": str(l),
                "diagnostic": "ACK",
                "region": "ARM",
                "reid_observation_eligible": "1",
            }
            for p in range(5)
            for l in range(2)
            for _ in range(2)
        ]

    d = Data()
    sampler = PairSampler(d)
    pairs = sampler.sample(np.random.default_rng(3))
    assert len(pairs) == 16 and sum(p[2] for p in pairs) == 8
    for a, b, s, scenario in pairs:
        ra, rb = d.rows[a], d.rows[b]
        if s:
            assert ra["patient_id"] == rb["patient_id"] and a != b
        else:
            assert ra["patient_id"] != rb["patient_id"]
        if scenario == "p1":
            assert ra["lesion_id"] != rb["lesion_id"]
        if scenario == "p2":
            assert ra["lesion_id"] == rb["lesion_id"]


def test_untrained_or_missing_evidence_cannot_pass_gate():
    assert not deployment_gate({})["passed"]
    report = inspect(
        {
            "device": "cpu",
            "weights": {
                "panderm": {
                    "path": None,
                    "sha256": None,
                    "source": None,
                    "source_sha256": None,
                }
            },
        }
    )
    assert not report["ready_for_teacher_training"]


def test_reconstruction_attacks_have_public_input_gradients():
    for latent in (False, True):
        model = ReconstructionAttack(latent=latent)
        g = torch.rand(1, 1, 32, 32, requires_grad=True)
        out = model(g)
        out.square().mean().backward()
        assert out.shape[1] == (2 if latent else 3) and g.grad.abs().sum() > 0

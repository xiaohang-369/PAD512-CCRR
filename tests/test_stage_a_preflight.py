"""Dependency/gate fixtures exercise refusal logic, not actual model loading."""

from types import SimpleNamespace
import pytest
from ccrr.runtime import preflight
from ccrr.cli import main
from ccrr.runtime import commands


def config():
    return {
        "device": "cpu",
        "weights": {},
        "profile": {"size": 512},
        "seed": 20261008,
        "batch_teacher": 16,
        "batch_pairs": 16,
        "microbatch": 1,
        "bootstrap_repetitions": 2000,
        "budgets": {"clinical": 5000, "mfm_probe": 2000, "reid": 10000},
    }


def test_stage_a_does_not_request_j_models_and_missing_a_stays_failure(monkeypatch):
    monkeypatch.setattr(
        preflight, "verify_dataset", lambda *a, **kw: {"verified": True}
    )
    a = preflight.inspect(config(), "A")
    j = preflight.inspect(config(), "J")
    assert set(a["weights"]) == {"efficientnet_b0", "resnet50", "panderm"}
    assert set(j["weights"]) == {"resnet50", "panderm", "swin_t", "convnext_tiny"}
    assert not a["ready_for_stage"] and not a["teacher_capability_validated"]
    assert any("efficientnet_b0" in m for m in a["missing"])
    assert not any("swin_t" in m or "convnext" in m for m in a["missing"])


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("seed", 1, "seed"),
        ("batch_teacher", 1, "batch"),
        ("microbatch", 2, "microbatch"),
        ("bootstrap_repetitions", 20, "Bootstrap"),
        ("resume_interval", 0, "interval"),
    ],
)
def test_a_protocol_configuration_rejected(monkeypatch, key, value, reason):
    monkeypatch.setattr(
        preflight, "verify_dataset", lambda *a, **kw: {"verified": True}
    )
    c = config()
    c[key] = value
    assert any(reason in s for s in preflight.inspect(c, "A")["missing"])


def test_a_data_failure_propagates(monkeypatch):
    def bad(*a, **kw):
        raise ValueError("changed frozen metadata")

    monkeypatch.setattr(preflight, "verify_dataset", bad)
    r = preflight.inspect(config(), "A")
    assert any("changed frozen metadata" in s for s in r["missing"])


def test_cli_preflight_stage_and_resume_are_dispatched(monkeypatch):
    seen = []
    monkeypatch.setattr(
        commands, "preflight", lambda a: seen.append((a.command, a.stage))
    )
    monkeypatch.setattr(
        commands, "train", lambda a: seen.append((a.command, a.stage, a.resume))
    )
    main(["preflight", "--config", "unused", "--stage", "A"])
    main(["train", "--config", "unused", "--stage", "A", "--resume"])
    assert seen == [("preflight", "A"), ("train", "A", True)]


def test_resume_other_stages_rejected_before_allocating_models(monkeypatch):
    monkeypatch.setattr(commands, "load_config", lambda p: config())
    from ccrr.errors import CCRRError

    with pytest.raises(CCRRError, match="only for A"):
        commands.train(SimpleNamespace(config="unused", stage="J", resume=True))

"""Explicit A–J dispatch, prerequisites and final study entry."""

import json
from ccrr.data.pad import PADData
from ccrr.teachers.adapters import (
    PanDerm,
    freeze,
)
from ccrr.representation.supervision import TeacherSupervision
from ccrr.runtime.support import write_json
from ccrr.runtime.checkpoints import load_checkpoint
from ccrr.errors import require
import ccrr.calibration.workflow as calibration
import ccrr.training.communication as communication
import ccrr.training.joint as joint
import ccrr.training.purifier as purifier
import ccrr.training.representation as representation
import ccrr.training.teachers as teachers


def run(experiment, stage="all"):
    try:
        if stage == "all":
            teachers.stage_a(experiment)
            purifier.stage_b(experiment)
            if not experiment.system.fixed_gray:
                representation.train_representation(experiment, "C")
                representation.train_representation(experiment, "D")
            representation.train_representation(experiment, "E")
            communication.stage_f(experiment)
            calibration.stage_g(experiment)
            joint.joint(experiment, "H")
            if experiment.cfg.get("joint_b"):
                joint.joint(experiment, "I")
            return final_evaluation(experiment)
        require(
            stage in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
            "Unknown stage",
        )
        if stage != "A":
            restore_dependencies(experiment, stage)
        return {
            "A": (lambda: teachers.stage_a(experiment)),
            "B": (lambda: purifier.stage_b(experiment)),
            "C": lambda: representation.train_representation(experiment, "C"),
            "D": lambda: representation.train_representation(experiment, "D"),
            "E": lambda: representation.train_representation(experiment, "E"),
            "F": (lambda: communication.stage_f(experiment)),
            "G": (lambda: calibration.stage_g(experiment)),
            "H": lambda: joint.joint(experiment, "H"),
            "I": lambda: joint.joint(experiment, "I"),
            "J": (lambda: final_evaluation(experiment)),
        }[stage]()
    except Exception as exc:
        write_json(
            experiment.output / "stopped.json",
            {
                "stage": stage,
                "type": type(exc).__name__,
                "reason": str(exc),
                "training_complete": False,
            },
        )
        raise
    finally:
        experiment.allocator.close()


def restore_dependencies(experiment, stage):
    if stage == "B":
        from ccrr.training.b_resources import verify_teachers

        root, record = verify_teachers(experiment.cfg)
        require(
            record["fit_patient_mapping"] == experiment.fit.pid,
            "Frozen ReID patient mapping mismatch",
        )
        experiment.acceptance = dict(record["acceptance"])
        for name, model in (
            ("clinical", experiment.clinical),
            ("reid", experiment.reid),
        ):
            load_checkpoint(root / "checkpoints" / (name + ".pt"), {"model": model})
            freeze(model)
        item = experiment.cfg["weights"]["panderm"]
        experiment.mfm = PanDerm(
            item["source"], item["source_sha256"], item["path"], item["sha256"]
        ).to(experiment.device)
        freeze(experiment.mfm)
        load_checkpoint(
            root / "checkpoints/mfm_probe.pt", {"purifier": experiment.purifier}
        )
        from ccrr.training.b_initialization import after_frozen_a_restore

        after_frozen_a_restore(experiment)
        return
    experiment.acceptance = json.loads(
        (experiment.output / "acceptance.json").read_text()
    )
    required = {
        "B": ["A-clinical", "A-mfm", "A-reid"],
        "C": [],
        "D": ["A-clinical", "A-mfm", "A-reid", "B", "C"],
        "E": ["D"],
        "F": ["E"],
        "G": ["E"],
        "H": ["E"],
        "I": ["H"],
        "J": ["H"],
    }[stage]
    for dep in required:
        require(
            experiment.acceptance.get(dep, {}).get("passed"),
            f"Missing accepted dependency {dep}",
        )
    if stage != "C":
        load_checkpoint(
            experiment.output / "checkpoints/clinical.pt",
            {"model": experiment.clinical},
        )
        freeze(experiment.clinical)
        load_checkpoint(
            experiment.output / "checkpoints/reid.pt", {"model": experiment.reid}
        )
        freeze(experiment.reid)
        e = experiment.cfg["weights"]["panderm"]
        experiment.mfm = PanDerm(
            e["source"], e["source_sha256"], e["path"], e["sha256"]
        ).to(experiment.device)
        if stage == "B":
            load_checkpoint(
                experiment.output / "checkpoints/mfm_probe.pt",
                {"purifier": experiment.purifier},
            )
        else:
            load_checkpoint(
                experiment.output / "checkpoints/purifier.pt",
                {"purifier": experiment.purifier},
            )
            freeze(experiment.purifier)
            tau = json.loads(
                (experiment.output / "identity_threshold.json").read_text()
            )["tau"]
            experiment.supervision = TeacherSupervision(
                experiment.clinical,
                experiment.mfm,
                experiment.reid,
                experiment.purifier,
                experiment.weights,
                tau,
            )
        teachers.initialize_base(experiment, experiment.pub_attacker, "resnet50")
    previous = {
        "D": "C",
        "E": "D",
        "F": "E",
        "G": "F",
        "H": "F",
        "I": "H",
        "J": "H",
    }.get(stage)
    if previous:
        load_checkpoint(
            experiment.output / "checkpoints" / f"{previous}.pt",
            {"system": experiment.system},
        )
        experiment.bind_snapshot("resume-" + stage)
    if stage in ("H", "I", "J"):
        calibration.stage_g(experiment, final=stage == "J")


def final_evaluation(experiment):
    require(
        experiment.acceptance.get("H", {}).get("passed"),
        "Final evaluation requires accepted frozen generator",
    )
    experiment.data["test"] = PADData(
        experiment.cfg["data_root"], "test", experiment.profile.size
    )
    from ccrr.training.final_study import FinalStudy

    study = FinalStudy(experiment)
    return study.run()

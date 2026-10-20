import torch

from ccrr_v3 import reid, runtime
from conftest import cohort


def test_fresh_p2_reference_and_frozen_targets_without_p1(tmp_path, cfg, monkeypatch):
    roles = []

    class FakePAD:
        def __init__(self, root, role):
            roles.append(role)
            self.root, self.role = tmp_path, role
            self.rows = cohort(role)["rows"]
            self.patients = sorted({r["patient_id"] for r in self.rows})
            p = tmp_path / "protocol_audit_20261009/pair_protocol"
            p.mkdir(parents=True, exist_ok=True)
            runtime.write(p / f"{role}_p2.json", cohort(role)["protocols"]["p2"])

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            gen = torch.Generator().manual_seed(i)
            return dict(
                x=torch.rand(3, 8, 8, generator=gen),
                v=torch.ones(1, 8, 8),
                legal=torch.ones(1),
                rect=(0, 0, 8, 8),
                label=i // 2,
                patient=row["patient_id"],
                patient_index=self.patients.index(row["patient_id"]),
                id=row["img_id"],
                metadata={},
            )

    class TinyReference(torch.nn.Module):
        def __init__(self, patients):
            super().__init__()
            self.model = torch.nn.Linear(3, 6)
            self.classifier = torch.nn.Linear(6, patients)

        def forward(self, x, rect):
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            e = self.model(x.mean((2, 3)))
            return dict(embedding=e, logits=self.classifier(e))

    monkeypatch.setattr(reid, "PADData", FakePAD)
    monkeypatch.setattr(
        reid, "initialize", lambda resources, patients: TinyReference(patients)
    )
    train = tmp_path / "reference"
    train.mkdir()
    reid.train({"data_root": str(tmp_path)}, cfg, train, torch.device("cpu"))
    assert set(runtime.read(train / "reference_report.json")) == {
        "rgb_rgb",
        "gray_rgb",
        "gray_gray",
    }
    assert runtime.read(train / "scope.json")["p1_required"] is False
    checkpoint = runtime.digest(train / "reference.pt")
    output = tmp_path / "targets"
    output.mkdir()
    reid.targets({"data_root": str(tmp_path)}, cfg, output, torch.device("cpu"), train)
    assert runtime.digest(train / "reference.pt") == checkpoint
    assert runtime.read(output / "capability.json")["teacher_qualified"] is False
    assert not set(roles) - {"fit", "val_tune", "val_select"}

"""Explicit JSON configuration, with narrowly scoped runtime overrides."""

import json
from pathlib import Path


FIELDS = {
    "schema",
    "model",
    "seed",
    "per_device_batch_size",
    "image_size",
    "devices",
    "cpu_threads",
    "verify_supervisor",
    "resources_file",
}


def load_verification(path, devices=None, resources=None):
    cfg = json.loads(Path(path).read_text())
    if set(cfg) != FIELDS or cfg["schema"] != "ccrr-v4-verification-1":
        raise ValueError("Invalid verification config fields/schema")
    if cfg["model"] != "author-5dc2a0f-port1":
        raise ValueError("Unregistered architecture")
    if devices is not None:
        cfg["devices"] = devices
    if resources is not None:
        cfg["resources_file"] = str(Path(resources).resolve())
        cfg["verify_supervisor"] = True
    if not isinstance(cfg["devices"], list) or not cfg["devices"]:
        raise ValueError("Require one or more devices")
    for device in cfg["devices"]:
        if device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("Devices must be cpu or cuda:<process-local-index>")
    if len(set(cfg["devices"])) != len(cfg["devices"]):
        raise ValueError("Duplicate devices")
    for key in ("seed", "per_device_batch_size", "cpu_threads"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    shape = cfg["image_size"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or n < 4 or n % 4 for n in shape)
    ):
        raise ValueError("image_size must contain two positive multiples of four")
    if type(cfg["verify_supervisor"]) is not bool:
        raise ValueError("verify_supervisor must be boolean")
    if cfg["verify_supervisor"] and not cfg["resources_file"]:
        raise ValueError("Provide a resource file for real supervisor verification")
    return cfg

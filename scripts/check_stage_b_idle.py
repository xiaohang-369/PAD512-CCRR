"""Read-only server process/GPU guard; never signal a process."""

import os
from pathlib import Path
import subprocess


def main():
    conflicts = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = [
                x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x
            ]
            if not args or not (
                Path(args[0]).name.startswith("python") or Path(args[0]).name == "ccrr"
            ):
                continue
            names = {Path(a).name for a in args}
            probe = any(
                n.startswith(("profile_b_r3", "run_b_r3", "probe_stage_b"))
                and n.endswith(".py")
                for n in names
            )
            training = "train" in args and ("ccrr.cli" in args or "ccrr" in names)
            stage = args[args.index("--stage") + 1] if "--stage" in args else "all"
            if probe or (training and stage in ("B", "all")):
                conflicts.append({"pid": int(proc.name), "argv": args})
        except (FileNotFoundError, ProcessLookupError):
            continue
    if conflicts:
        raise SystemExit(
            "Existing B/profile process; no new job started: " + str(conflicts)
        )
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    busy = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if busy:
        raise SystemExit("GPU 1 is occupied; no process killed. PIDs: " + busy)
    print("STAGE_B_IDLE_CONFIRMED", gpu, flush=True)


if __name__ == "__main__":
    main()

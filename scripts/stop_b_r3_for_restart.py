"""Send safe SIGTERM only to the verified old R3 worker; preserve every artifact."""

import argparse
import os
from pathlib import Path
import signal


def matches_worker(argv, result):
    if not any(Path(a).name == "check_stage_b_short_multi.py" for a in argv):
        return False
    try:
        return Path(argv[argv.index("--out") + 1]).resolve() == result.resolve()
    except (ValueError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    result = run / "result"
    pid = int((result / "worker.pid").read_text())
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise SystemExit("Linux pidfd support required; no signal sent")
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        print("Worker already exited. Inspect console.log/status.json before restart.")
        return
    try:
        proc = Path(f"/proc/{pid}")
        argv = [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        if not matches_worker(argv, result):
            raise SystemExit(
                "Worker command/output path does not match old R3; no signal sent"
            )
        if proc.stat().st_uid != os.getuid():
            raise SystemExit("Worker is owned by another user; no signal sent")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        print(
            f"Safe pause requested for verified worker {pid}. No files changed.",
            flush=True,
        )
        # Never SIGKILL: a scoring block may need time to finish and commit.
        import select

        poll = select.poll()
        poll.register(fd, select.POLLIN)
        for _ in range(6):
            if poll.poll(5000):
                print(
                    "Worker exited. Check final log/status and confirm the GPU is free."
                )
                return
            print("Waiting for the existing scoring/update boundary...", flush=True)
        print(
            "Worker is still stopping; do not start a replacement yet. Check the log."
        )
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()

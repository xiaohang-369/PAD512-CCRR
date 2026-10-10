"""Argument parsing and command dispatch only."""

import argparse
import json
import sys
from ccrr.errors import CCRRError
from ccrr.runtime import commands


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ccrr",
        description="Complete capability-conditioned medical representation research system",
    )
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("inspect")
    x.add_argument("--config")
    x = sub.add_parser("train")
    x.add_argument("--config", required=True)
    x.add_argument(
        "--stage",
        default="all",
        choices=["all", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    x.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last durable A optimizer boundary",
    )
    x = sub.add_parser("preflight")
    x.add_argument("--config", required=True)
    x.add_argument(
        "--stage",
        choices=["all", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        default="all",
    )
    x = sub.add_parser("audit-data")
    x.add_argument("--root", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("export")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--cdf", required=True)
    x.add_argument("--config", required=True)
    x.add_argument("--evidence", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("receive")
    x.add_argument("--manifest", required=True)
    x.add_argument("--image", required=True)
    x.add_argument("--key-c")
    x.add_argument("--key-m")
    x.add_argument("--out", required=True)
    x.add_argument("--device", default="cpu")
    x = sub.add_parser("send")
    x.add_argument("--manifest", required=True)
    x.add_argument("--image", required=True)
    x.add_argument("--metadata", required=True)
    x.add_argument("--clinical-checkpoint", required=True)
    x.add_argument("--nonce-db", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--device", default="cpu")
    x = sub.add_parser("smoke")
    x.add_argument("--out", required=True)
    x.add_argument("--size", type=int, choices=(256, 512), default=512)
    args = p.parse_args(argv)
    try:
        return getattr(commands, args.command.replace("-", "_"))(args)
    except (CCRRError, FileNotFoundError, ValueError) as exc:
        print(
            json.dumps(
                {"status": getattr(exc, "status", "Error"), "detail": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Server-user entry point; never dispatches a subsequent V4 phase."""

import argparse
import os
from .data import prepare
from .runner import run


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", required=True)
    prep.add_argument("--data-root", required=True)
    prep.add_argument("--vgg", required=True)
    prep.add_argument("--vgg-sha256", required=True)
    prep.add_argument("--output", required=True)
    for command in ("preflight", "train"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--prepared", required=True)
        p.add_argument("--output", required=True)
        if command == "train":
            p.add_argument(
                "--resume",
                help="Explicit epoch checkpoint; write to a NEW run directory",
            )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.data_root, args.vgg, args.vgg_sha256, args.output)
    else:
        run(
            args.config,
            args.prepared,
            args.output,
            args.command,
            getattr(args, "resume", None),
        )


if __name__ == "__main__":
    main()

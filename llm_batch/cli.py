"""`llm-batch` command-line entry point."""

from __future__ import annotations

import argparse

from llm_batch.poller import Poller
from llm_batch.storage.file import FileStorage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="llm-batch")
    sub = parser.add_subparsers(dest="command", required=True)

    poll = sub.add_parser("poll", help="poll SENT batch jobs and advance pipelines")
    poll.add_argument("state_dir", help="root directory of pipeline state")
    poll.add_argument("--sweep-orphans", type=int, default=None, metavar="DAYS",
                      help="also delete provider-side orphans older than DAYS")
    poll.add_argument("--max-batch-age-days", type=int, default=3,
                      help="mark jobs stuck in SENT past this age as FAILED")

    args = parser.parse_args(argv)

    if args.command == "poll":
        backend = FileStorage(args.state_dir)
        poller = Poller(backend, max_batch_age_days=args.max_batch_age_days)
        poller.poll_all()
        if args.sweep_orphans is not None:
            poller.sweep_orphans(older_than_days=args.sweep_orphans)
        return 0

    return 1

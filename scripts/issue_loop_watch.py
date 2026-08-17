#!/usr/bin/env python3
"""Show what the agent started by `scripts/issue_loop.sh` is doing right now.

The loop reports only after an agent exits, because `claude -p --output-format json`
writes its report at the end. This reads the live session transcript instead, so a
run in progress is visible, and follows the switch to a new transcript when the loop
moves to the next issue.

    scripts/issue_loop_watch.py            # follow live until interrupted
    scripts/issue_loop_watch.py --dump 20  # print the last 20 events and exit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

POLL_SECONDS = 2.0
TAIL_LINES_ON_ATTACH = 40
TOOL_ARG_KEYS = ("command", "file_path", "pattern", "description")


def repo_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def transcript_dir(repo_path: str, home: str | None = None) -> str:
    """Directory holding this repository's Claude Code session transcripts.

    Claude Code names it after the absolute project path with punctuation
    folded to hyphens: `/Users/x/src/repo` becomes `-Users-x-src-repo`, and a
    dot folds too — `…/dontsendfile.com` lives in `…-dontsendfile-com`.

    Only `/` and `.` are folded here because only those two are confirmed. Any
    other punctuation a checkout path might carry would silently resolve to a
    directory that does not exist, so an unmatched name falls back to scanning
    the projects directory for the entry that agrees once both sides have every
    non-alphanumeric character folded.
    """
    home = home if home is not None else os.path.expanduser("~")
    projects = os.path.join(home, ".claude", "projects")
    folded = repo_path.replace(os.sep, "-").replace(".", "-")
    candidate = os.path.join(projects, folded)
    if os.path.isdir(candidate) or not os.path.isdir(projects):
        return candidate
    wanted = _fold_all(repo_path)
    for entry in sorted(os.listdir(projects)):
        if _fold_all(entry) == wanted:
            return os.path.join(projects, entry)
    return candidate


def _fold_all(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", name)


def newest_transcript(directory: str, exclude: tuple[str, ...] = ()) -> str | None:
    candidates = [
        path
        for path in glob.glob(os.path.join(directory, "*.jsonl"))
        if not any(token in path for token in exclude)
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def format_events(raw_line: str, max_text: int = 300) -> list[str]:
    """One display line per tool call or assistant message in a transcript line.

    Anything else (user turns, tool results, malformed lines) yields nothing: the
    point is to see what the agent decided, not to replay its inputs.
    """
    try:
        record = json.loads(raw_line)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(record, dict) or record.get("type") != "assistant":
        return []
    message = record.get("message") or {}
    events: list[str] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            payload = block.get("input") or {}
            argument = next(
                (str(payload[key]) for key in TOOL_ARG_KEYS if payload.get(key)), ""
            )
            events.append(
                f"[TOOL] {block.get('name')}: {argument[:150]}".replace("\n", " ")
            )
        elif block.get("type") == "text" and (block.get("text") or "").strip():
            events.append(f"[SAY]  {block['text'][:max_text]}".replace("\n", " "))
    return events


def dump(directory: str, count: int, exclude: tuple[str, ...]) -> int:
    path = newest_transcript(directory, exclude)
    if path is None:
        print(f"no session transcript under {directory}", file=sys.stderr)
        return 1
    print(f">>> {os.path.basename(path)}")
    events: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            events.extend(format_events(line))
    print("\n".join(events[-count:]))
    return 0


def follow(directory: str, exclude: tuple[str, ...]) -> int:
    current: str | None = None
    handle = None
    try:
        while True:
            newest = newest_transcript(directory, exclude)
            if newest is not None and newest != current:
                if handle is not None:
                    handle.close()
                current = newest
                handle = open(current, encoding="utf-8")
                recent = handle.readlines()[-TAIL_LINES_ON_ATTACH:]
                print(f"\n>>> now watching {os.path.basename(current)}", flush=True)
                for line in recent:
                    for event in format_events(line):
                        print(event, flush=True)
            if handle is not None:
                for line in handle:
                    for event in format_events(line):
                        print(event, flush=True)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        return 0
    finally:
        if handle is not None:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        type=int,
        metavar="N",
        help="print the last N events of the newest session and exit",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="comma-separated session ids to ignore (e.g. the session you are typing in)",
    )
    args = parser.parse_args()

    exclude = tuple(token for token in args.exclude.split(",") if token)
    directory = transcript_dir(repo_root())
    if args.dump is not None:
        return dump(directory, args.dump, exclude)
    return follow(directory, exclude)


if __name__ == "__main__":
    sys.exit(main())

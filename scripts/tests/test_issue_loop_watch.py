"""Tests for the issue-loop transcript viewer.

A wrong transcript-directory rule makes the viewer watch nothing while looking
healthy, so the naming cases below pin the folding that was measured against
real checkouts — including the dotted path that the first version got wrong.
A malformed transcript line must never take the viewer down mid-run either.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from issue_loop_watch import format_events, newest_transcript, transcript_dir


def assistant_line(*blocks: dict) -> str:
    return json.dumps({"type": "assistant", "message": {"content": list(blocks)}})


class TranscriptDirTest(unittest.TestCase):
    def test_project_path_becomes_a_hyphenated_directory_name(self):
        self.assertEqual(
            transcript_dir("/Users/dev/Documents/libjpeg-turbo-rs", home="/Users/dev"),
            "/Users/dev/.claude/projects/-Users-dev-Documents-libjpeg-turbo-rs",
        )

    def test_a_dot_in_the_path_folds_to_a_hyphen_too(self):
        # Measured against a real checkout: `~/Documents/Projects/dontsendfile.com`
        # keeps its transcripts in `-Users-…-Projects-dontsendfile-com`. Folding
        # only the separator pointed the viewer at a directory that never exists,
        # so it reported no session forever on any dotted checkout path.
        self.assertEqual(
            transcript_dir("/h/src/repo-v0.7", home="/h"),
            "/h/.claude/projects/-h-src-repo-v0-7",
        )

    def test_a_nested_checkout_keeps_every_path_segment(self):
        # Worktrees differ only in the trailing segment, so truncating any part
        # of the path would collide two checkouts onto one transcript directory.
        self.assertNotEqual(
            transcript_dir("/src/repo", home="/h"),
            transcript_dir("/src/repo-fix", home="/h"),
        )

    def test_an_unfolded_punctuation_class_is_recovered_by_scanning(self):
        # Only `/` and `.` are confirmed to fold, so a path carrying anything
        # else must still resolve rather than silently watch nothing.
        with tempfile.TemporaryDirectory() as home:
            projects = Path(home) / ".claude" / "projects"
            projects.mkdir(parents=True)
            (projects / "-src-my-repo").mkdir()
            self.assertEqual(
                transcript_dir("/src/my_repo", home=home),
                str(projects / "-src-my-repo"),
            )


class NewestTranscriptTest(unittest.TestCase):
    def test_picks_the_most_recently_written_session(self):
        with tempfile.TemporaryDirectory() as directory:
            older = os.path.join(directory, "older.jsonl")
            newer = os.path.join(directory, "newer.jsonl")
            Path(older).write_text("{}\n", encoding="utf-8")
            time.sleep(0.01)
            Path(newer).write_text("{}\n", encoding="utf-8")
            self.assertEqual(newest_transcript(directory), newer)

    def test_excluded_sessions_are_never_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            wanted = os.path.join(directory, "agent.jsonl")
            Path(wanted).write_text("{}\n", encoding="utf-8")
            time.sleep(0.01)
            skipped = os.path.join(directory, "mine.jsonl")
            Path(skipped).write_text("{}\n", encoding="utf-8")
            self.assertEqual(newest_transcript(directory, exclude=("mine",)), wanted)

    def test_an_empty_directory_yields_no_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(newest_transcript(directory))


class FormatEventsTest(unittest.TestCase):
    def test_a_tool_call_reports_its_name_and_leading_argument(self):
        line = assistant_line(
            {"type": "tool_use", "name": "Bash", "input": {"command": "cargo test"}}
        )
        self.assertEqual(format_events(line), ["[TOOL] Bash: cargo test"])

    def test_a_tool_call_without_a_known_argument_still_reports_its_name(self):
        line = assistant_line({"type": "tool_use", "name": "ToolSearch", "input": {}})
        self.assertEqual(format_events(line), ["[TOOL] ToolSearch: "])

    def test_assistant_text_is_flattened_to_one_line(self):
        line = assistant_line({"type": "text", "text": "first\nsecond"})
        self.assertEqual(format_events(line), ["[SAY]  first second"])

    def test_long_text_is_truncated(self):
        line = assistant_line({"type": "text", "text": "x" * 500})
        self.assertEqual(len(format_events(line)[0]), len("[SAY]  ") + 300)

    def test_blank_text_is_not_an_event(self):
        self.assertEqual(format_events(assistant_line({"type": "text", "text": "  "})), [])

    def test_one_line_can_carry_several_events(self):
        line = assistant_line(
            {"type": "text", "text": "running the suite"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "cargo test"}},
        )
        self.assertEqual(len(format_events(line)), 2)

    def test_non_assistant_records_are_ignored(self):
        self.assertEqual(format_events(json.dumps({"type": "user"})), [])

    def test_a_malformed_line_is_ignored_rather_than_raising(self):
        # A transcript is appended to while being read, so a torn final line is
        # normal; the viewer must survive it.
        self.assertEqual(format_events('{"type": "assist'), [])
        self.assertEqual(format_events(""), [])


if __name__ == "__main__":
    unittest.main()

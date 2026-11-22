#!/usr/bin/env python3
"""
Tests for the run-level policy — the setting the CLI and the GUI share.

The design question this answers: how do two front ends agree about whether to
read a book, without either having to know the other exists?

The answer is that `ask` is a *policy value*, not a separate interactive mode,
and asking is a *capability the caller supplies*. So:

    CLI, interactive      supplies a terminal prompt  -> asks
    CLI, --noninteractive supplies nothing            -> falls back to SKIP
    GUI                   supplies a dialog           -> asks in a window
    a cron job            supplies nothing            -> falls back to SKIP

Precedence: a CLI flag, then a per-run override (what a GUI passes), then the
YAML, then the built-in default.

**Every fallback resolves to the cheap option.** A typo in the config, an
unreadable file, a missing file, an unknown policy name — none of them may
resolve towards hours of GPU. That is the invariant most of these tests exist
to protect.
"""

from pathlib import Path

import pytest

from src.processing.policy import (
    DEFAULT_IMPLAUSIBLE_CHAPTER_PAGES,
    BookPolicy,
    BookSettings,
    load_book_settings,
    resolve_book_settings,
)

SHIPPED_CONFIG = Path(__file__).parent.parent / "config"


def write_policy(tmp_path: Path, body: str) -> Path:
    (tmp_path / "processing_policy.yaml").write_text(body, encoding="utf-8")
    return tmp_path


class TestReadingTheFile:
    def test_the_shipped_config_is_valid_and_asks(self):
        """What a human at a terminal should get out of the box."""
        settings = load_book_settings(SHIPPED_CONFIG)
        assert settings.policy is BookPolicy.ASK

    @pytest.mark.parametrize("name,expected", [
        ("skip", BookPolicy.SKIP),
        ("ask", BookPolicy.ASK),
        ("all", BookPolicy.ALL),
        ("chapters", BookPolicy.CHAPTERS),
        ("ASK", BookPolicy.ASK),
        ("  ask  ", BookPolicy.ASK),
    ])
    def test_policy_names(self, tmp_path, name, expected):
        root = write_policy(tmp_path, f"books:\n  policy: {name}\n")
        assert load_book_settings(root).policy is expected

    def test_chapters_and_ceiling_are_read(self, tmp_path):
        root = write_policy(tmp_path,
                            "books:\n  policy: chapters\n  chapters: [3, 1, 3]\n"
                            "  max_pages: 40\n")
        settings = load_book_settings(root)

        assert settings.chapters == [1, 3]        # sorted, deduplicated
        assert settings.max_pages == 40

    def test_the_misclassified_chapter_threshold_is_configurable(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  implausible_chapter_pages: 120\n")
        assert load_book_settings(root).implausible_chapter_pages == 120


class TestEveryFallbackIsCheap:
    """The invariant: nothing unclear may resolve towards hours of GPU."""

    def test_a_missing_file_reads_no_bodies(self, tmp_path):
        settings = load_book_settings(tmp_path)
        assert settings.policy is BookPolicy.SKIP
        assert "not found" in settings.source

    def test_a_malformed_file_reads_no_bodies(self, tmp_path):
        root = write_policy(tmp_path, "books: [this is not a mapping\n")
        assert load_book_settings(root).policy is BookPolicy.SKIP

    def test_an_unknown_policy_name_reads_no_bodies(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: read_everything_please\n")
        assert load_book_settings(root).policy is BookPolicy.SKIP

    def test_an_empty_file_reads_no_bodies(self, tmp_path):
        root = write_policy(tmp_path, "")
        assert load_book_settings(root).policy is BookPolicy.SKIP

    def test_a_non_numeric_ceiling_is_ignored_not_guessed(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  max_pages: soon\n")
        assert load_book_settings(root).max_pages is None

    def test_a_non_numeric_chapter_is_dropped(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  chapters: [3, 'four', 5]\n")
        assert load_book_settings(root).chapters == [3, 5]


class TestPrecedence:
    def test_a_cli_flag_beats_the_file(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: skip\n")
        settings = resolve_book_settings(root, cli_chapters=[3])

        assert settings.policy is BookPolicy.CHAPTERS
        assert settings.chapters == [3]
        assert settings.source == "--chapters"

    def test_chapters_with_no_numbers_means_the_whole_book(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: skip\n")
        assert resolve_book_settings(root, cli_chapters=[]).policy is BookPolicy.ALL

    def test_a_per_run_override_beats_the_file(self, tmp_path):
        """How a GUI supplies a choice without writing the config."""
        root = write_policy(tmp_path, "books:\n  policy: skip\n")
        settings = resolve_book_settings(root, overrides={"policy": "chapters",
                                                          "chapters": [2, 4]})

        assert settings.policy is BookPolicy.CHAPTERS
        assert settings.chapters == [2, 4]

    def test_a_cli_flag_beats_a_per_run_override(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: skip\n")
        settings = resolve_book_settings(root, cli_chapters=[9],
                                         overrides={"policy": "all"})

        assert settings.policy is BookPolicy.CHAPTERS
        assert settings.chapters == [9]

    def test_the_file_is_used_when_nothing_overrides_it(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: all\n")
        assert resolve_book_settings(root).policy is BookPolicy.ALL

    def test_a_cli_ceiling_beats_the_file(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  max_pages: 40\n")
        assert resolve_book_settings(root, cli_max_pages=10).max_pages == 10


class TestNoninteractive:
    def test_ask_is_downgraded_to_skip_when_nobody_is_watching(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: ask\n")
        settings = resolve_book_settings(root, noninteractive=True)

        assert settings.policy is BookPolicy.SKIP
        assert "downgraded" in settings.source

    def test_an_explicit_expensive_policy_is_respected_unattended(self, tmp_path):
        """
        `--noninteractive` means "do not ask", not "do not work".

        Someone who set `policy: all` in the config and scheduled the run
        meant it; downgrading that would be second-guessing an explicit
        instruction, which is a different thing from filling a gap safely.
        """
        root = write_policy(tmp_path, "books:\n  policy: all\n")
        assert resolve_book_settings(root, noninteractive=True).policy is BookPolicy.ALL

    def test_a_cli_request_is_respected_unattended(self, tmp_path):
        root = write_policy(tmp_path, "books:\n  policy: ask\n")
        settings = resolve_book_settings(root, cli_chapters=[3], noninteractive=True)

        assert settings.policy is BookPolicy.CHAPTERS


class TestTheAskCapability:
    def test_only_the_ask_policy_wants_a_prompt(self):
        assert BookSettings(policy=BookPolicy.ASK).wants_prompt() is True
        for policy in (BookPolicy.SKIP, BookPolicy.ALL, BookPolicy.CHAPTERS):
            assert BookSettings(policy=policy).wants_prompt() is False

    def test_the_built_in_default_is_the_cheap_one(self):
        settings = BookSettings()
        assert settings.policy is BookPolicy.SKIP
        assert settings.implausible_chapter_pages == DEFAULT_IMPLAUSIBLE_CHAPTER_PAGES

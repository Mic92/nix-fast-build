import re

import pytest

from nix_fast_build.term import (
    MAX_LINE_LEN,
    fold_markers,
    is_ignored_eval_line,
    sanitize_line,
    want_color,
)


def test_is_ignored_eval_line_colored() -> None:
    # nix colors "error (ignored):" on a TTY, which broke the plain match.
    line = sanitize_line(
        "\x1b[31;1merror (ignored):\x1b[0m SQLite database '/cache/x.sqlite' is busy"
    )
    assert is_ignored_eval_line(line)
    assert not is_ignored_eval_line("evaluation warning: 'system' has been renamed")


def test_sanitize_passthrough() -> None:
    assert sanitize_line("plain text") == "plain text"
    assert sanitize_line("ünïcode 漢字") == "ünïcode 漢字"


def test_sanitize_cr_overwrite() -> None:
    # Progress bars rewrite the line; keep the final state.
    assert sanitize_line("10%\r50%\r100%") == "100%"
    # Trailing CR (e.g. CRLF already split): keep last non-empty segment.
    assert sanitize_line("done\r") == "done"


def test_sanitize_keeps_sgr_appends_reset() -> None:
    out = sanitize_line("\x1b[31merror\x1b[0m rest")
    assert "\x1b[31m" in out
    assert out.endswith("\x1b[0m")


def test_sanitize_strips_dangerous_escapes() -> None:
    # Cursor movement, clear screen, OSC title: stripped.
    assert sanitize_line("a\x1b[2Jb") == "ab"
    assert sanitize_line("a\x1b[3Ab") == "ab"
    assert sanitize_line("a\x1b]0;evil title\x07b") == "ab"
    # Unterminated OSC must not swallow output of later lines.
    assert sanitize_line("a\x1b]0;unterminated") == "a"


def test_sanitize_stray_esc_dropped() -> None:
    # Lone or partial ESC not matched as a full sequence must not leak:
    # the terminal would pair it with our following output.
    assert sanitize_line("tail\x1b") == "tail"
    assert "\x1b" not in sanitize_line("a\x1b")
    # Kept SGR still intact, including intermediate bytes before m.
    assert "\x1b[31m" in sanitize_line("x\x1b[31my\x1b")
    assert "\x1b[0 m" in sanitize_line("x\x1b[0 my")


def test_sanitize_truncation_no_partial_sgr() -> None:
    # Truncation must not cut an SGR sequence in half: an unterminated
    # CSI makes the terminal swallow the following text.
    for offset in range(1, 5):
        line = "x" * (MAX_LINE_LEN - offset) + "\x1b[31mred\x1b[0m"
        out = sanitize_line(line)
        head = out.split(" …[line truncated]")[0]
        assert not re.search("\x1b[^m]*$", head), f"offset {offset}: {head[-8:]!r}"


def test_sanitize_controls_and_tabs() -> None:
    assert sanitize_line("a\tb") == "a   b"
    assert sanitize_line("a\x07\x08b") == "ab"


def test_sanitize_caps_length() -> None:
    out = sanitize_line("x" * (MAX_LINE_LEN + 100))
    assert len(out) < MAX_LINE_LEN + 50
    assert out.endswith("[line truncated]")


@pytest.mark.parametrize(
    ("env", "isatty", "expected"),
    [
        ({}, True, True),
        ({}, False, False),
        ({"NO_COLOR": "1"}, True, False),
        ({"FORCE_COLOR": "1"}, False, True),
        ({"CLICOLOR_FORCE": "1"}, False, True),
        # NO_COLOR wins over force
        ({"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),
        # Actions CI renders ANSI even though stderr is a pipe
        ({"GITHUB_ACTIONS": "true"}, False, True),
        ({"GITHUB_ACTIONS": "true", "NO_COLOR": "1"}, False, False),
    ],
)
def test_want_color(
    env: dict[str, str],
    isatty: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE", "GITHUB_ACTIONS"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert want_color(isatty) is expected


def test_fold_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert fold_markers() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert fold_markers() is True

import io
import logging
import os
import re

import pytest

from nix_fast_build import tty_renderer
from nix_fast_build.log_format import BuildLogLine
from nix_fast_build.renderer import BuildOutput
from nix_fast_build.tty_renderer import (
    CSI,
    Display,
    DisplayLogHandler,
    Mode,
    TTYRenderer,
)

DRV = "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-x.drv"
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_renderer() -> tuple[TTYRenderer, io.StringIO, FakeClock]:
    out = io.StringIO()
    clock = FakeClock()
    return TTYRenderer(out, clock=clock), out, clock


def feed(build: BuildOutput, *lines: str) -> None:
    for line in lines:
        build.on_event(BuildLogLine(line=line, activity=None))


def plain(lines: list[str]) -> str:
    return ANSI.sub("", "\n".join(lines))


# ── Display ──────────────────────────────────────────────────────────


def test_display_ephemeral_overwrites_in_place() -> None:
    out = io.StringIO()
    d = Display(out)
    d.ephemeral(["a", "b"])
    d.ephemeral(["c"])
    text = out.getvalue()
    # Second paint moves up 2 lines, then clears the leftover line.
    assert f"{CSI}2F" in text
    assert f"{CSI}J" in text
    assert d.ephemeral_lines == 1


def test_display_permanent_above_ephemeral() -> None:
    out = io.StringIO()
    d = Display(out)
    d.ephemeral(["status"])
    d.permanent("event")
    text = out.getvalue()
    # Permanent line printed, ephemeral repainted afterwards.
    assert text.index("event") < text.rindex("status")
    assert d.ephemeral_lines == 1


def test_display_suspend_queues_permanent() -> None:
    out = io.StringIO()
    d = Display(out)
    d.ephemeral(["status"])
    d.suspend()
    before = out.getvalue()
    d.permanent("while paging")
    assert out.getvalue() == before  # nothing written while suspended
    assert d.resume() == 1
    assert "while paging" in out.getvalue()


def test_display_sync_markers() -> None:
    out = io.StringIO()
    d = Display(out)
    d.ephemeral(["x"])
    assert out.getvalue().startswith(f"{CSI}?2026h")
    assert out.getvalue().endswith(f"{CSI}?2026l")


# ── build lifecycle ──────────────────────────────────────────────────


def test_lifecycle_and_failure_extract() -> None:
    r, out, clock = make_renderer()
    good = r.start_build("good", DRV)
    bad = r.start_build("bad", DRV)
    feed(bad, *(f"l{i}" for i in range(10)))
    clock.now += 65
    r.finish_build(good, 0)
    r.finish_build(bad, 1)
    assert r.succeeded == {good}
    assert r.failed == [bad]
    assert not r.running
    text = ANSI.sub("", out.getvalue())
    assert "✔ good  1m05s" in text
    assert "✘ bad  1m05s  rc=1" in text
    # Extract: last 5 lines only.
    assert "bad> l5" in text
    assert "bad> l4" not in text
    assert f"nix log {DRV}" in text


def test_abort_is_silent() -> None:
    r, out, _clock = make_renderer()
    b = r.start_build("x", DRV)
    before = out.getvalue()
    r.abort_build(b)
    assert out.getvalue() == before
    assert not r.running


def test_render_normal_rows() -> None:
    r, _out, clock = make_renderer()
    a = r.start_build("pkgs.alpha", DRV)
    clock.now += 5
    r.start_build("pkgs.beta", DRV)
    feed(a, "compiling foo.c")
    text = plain(r.render_normal())
    assert "pkgs.alpha" in text
    assert "compiling foo.c" in text
    assert "pkgs.beta" in text
    assert "[f] logs" in text


# ── browser ──────────────────────────────────────────────────────────


def fail_n(r: TTYRenderer, n: int) -> list[BuildOutput]:
    builds = []
    for i in range(n):
        b = r.start_build(f"pkg-{i:02d}", DRV)
        feed(b, f"log of {i}")
        r.finish_build(b, 1)
        builds.append(b)
    return builds


def test_browser_pinned_order_and_new_tag() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 2)
    r.on_key("f")
    assert r.mode is Mode.LIST
    assert [b.attr for b in r.pinned] == ["pkg-00", "pkg-01"]
    # New failure while open: appended, tagged new, rows don't shift.
    late = r.start_build("late", DRV)
    r.finish_build(late, 1)
    text = plain(r.render_list())
    assert [b.attr for b in r.pinned] == ["pkg-00", "pkg-01", "late"]
    assert "new" in text


def test_browser_clamp_and_flash() -> None:
    r, _out, clock = make_renderer()
    fail_n(r, 2)
    r.on_key("f")
    r.on_key("k")
    assert "top" in r.flash_text
    r.on_key("j")
    r.on_key("j")
    assert r.cursor == 1
    assert "bottom" in r.flash_text
    # Flash expires.
    clock.now += 2
    assert r._flash_line() is None


def test_browser_paging() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, r.PAGE + 2)
    r.on_key("f")
    r.render_list()
    assert r._pages() == 2
    r.on_key("n")
    assert r.page == 1
    assert r.cursor == r.PAGE
    r.on_key("n")
    assert "last page" in r.flash_text
    r.on_key("p")
    assert r.page == 0


def test_browser_filter_subsequence() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 3)
    extra = r.start_build("checks.deadnix", DRV)
    r.finish_build(extra, 1)
    r.on_key("f")
    r.on_key("/")
    assert r.filter_input
    for ch in "ddnx":
        r.on_key(ch)
    assert [b.attr for b in r._filtered()] == ["checks.deadnix"]
    # Esc clears filter (layered).
    r.on_key("\x1b")
    assert not r.filter_input
    assert r.filter == ""
    assert r.mode is Mode.LIST


def test_browser_digit_and_dump() -> None:
    r, out, _clock = make_renderer()
    fail_n(r, 2)
    r.on_key("f")
    r.on_key("d")  # toggle to dump mode
    assert r.dump_action
    r.on_key("9")
    assert "no entry 9" in r.flash_text
    r.on_key("1")
    assert r.last_viewed is not None
    assert "dumped" in r.flash_text
    text = ANSI.sub("", out.getvalue())
    assert "pkg-00> log of 0" in text


def test_dump_log_to_scrollback() -> None:
    r, out, _clock = make_renderer()
    [b] = fail_n(r, 1)
    r._dump_log(b, "failed", running=False)
    text = ANSI.sub("", out.getvalue())
    assert "log: pkg-00 (failed" in text
    assert "pkg-00> log of 0" in text
    assert f"nix log {DRV}" in text


def test_pager_cmd() -> None:
    r, _out, _clock = make_renderer()
    [b] = fail_n(r, 1)
    cmd = r._pager_cmd(b, "failed", running=False)
    assert cmd[0] == "less"
    assert cmd[-1] == "+G"
    cmd = r._pager_cmd(b, "build, running", running=True)
    assert cmd[-1] == "+F"


def test_exit_browser_keys() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 1)
    for key in ("f", "q", "\x1b"):
        r.on_key("f")
        assert r.mode is Mode.LIST
        r.on_key(key)
        assert r.mode is Mode.NORMAL


def test_unknown_key_flashes() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 1)
    r.on_key("f")
    r.on_key("z")
    assert "unknown key" in r.flash_text


def test_display_log_handler() -> None:
    out = io.StringIO()
    d = Display(out)
    d.ephemeral(["status"])
    h = DisplayLogHandler(d)
    logger = logging.getLogger("nfb-test")
    logger.addHandler(h)
    logger.warning("multi\nline")
    logger.removeHandler(h)
    text = out.getvalue()
    assert "WARNING:nfb-test:multi" in text
    assert d.ephemeral_lines == 1  # region repainted after the log lines


def test_browser_succeeded_label() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 1)
    b = r.start_build("slow", DRV)
    r.on_key("f")  # pins failed + running
    r.finish_build(b, 0)  # succeeds while browser open
    text = plain(r.render_list())
    assert "✔ done" in text
    assert text.count("✘ failed") == 1


def test_render_normal_clamps_to_terminal_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    r, _out, _clock = make_renderer()
    for i in range(30):
        b = r.start_build(f"pkg-{i:02d}", DRV)
        feed(b, "output")
    monkeypatch.setattr(
        tty_renderer.shutil, "get_terminal_size", lambda: os.terminal_size((80, 24))
    )
    lines = r.render_normal()
    assert len(lines) <= 24 - 1
    assert any("more" in line for line in lines)


def test_render_list_adapts_to_terminal_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 10)
    r.on_key("f")
    monkeypatch.setattr(
        tty_renderer.shutil, "get_terminal_size", lambda: os.terminal_size((80, 10))
    )
    lines = r.render_list()
    assert len(lines) <= 10 - 1
    # Paging still covers all entries.
    assert r._pages() * r._page_size() >= 10

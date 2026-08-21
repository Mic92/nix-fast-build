import asyncio
import io
import logging
import os
import re

import pyte
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


# ── Display (against a pyte terminal emulator) ───────────────────────

WIDTH, HEIGHT = 60, 12


def emulate(chunks: str) -> pyte.HistoryScreen:
    screen = pyte.HistoryScreen(WIDTH, HEIGHT, history=1000)
    pyte.Stream(screen).feed(chunks)
    return screen


def buffer_lines(screen: pyte.HistoryScreen) -> list[str]:
    """Scrollback plus visible rows, as plain text."""
    history = [
        "".join(line[x].data for x in range(WIDTH)).rstrip()
        for line in screen.history.top
    ]
    return history + [row.rstrip() for row in screen.display]


def make_display(out: io.StringIO) -> Display:
    return Display(out, size=lambda: os.terminal_size((WIDTH, HEIGHT)))


def test_verdicts_scroll_and_region_stays_at_bottom() -> None:
    """Verdicts land in scrollback exactly once and in order, while the
    live region only ever exists as the last rows of the screen."""
    out = io.StringIO()
    display = make_display(out)
    verdicts = []
    for i in range(40):
        batch = [f"verdict-{i}-a", f"verdict-{i}-b"]
        verdicts += batch
        display.frame(batch, ["HEADER", f"running-{i}", f"gist-{i}"])
    display.close()

    lines = buffer_lines(emulate(out.getvalue()))
    kept = [line for line in lines if line.startswith("verdict-")]
    assert kept == verdicts
    assert sum("HEADER" in line for line in lines) == 0  # cleared on close


def test_region_visible_while_running() -> None:
    out = io.StringIO()
    display = make_display(out)
    for i in range(20):
        display.frame([f"verdict-{i}"], ["HEADER", f"running-{i}"])

    screen = emulate(out.getvalue())
    lines = buffer_lines(screen)
    # The region exists exactly once, at the bottom of the visible screen.
    assert sum("HEADER" in line for line in lines) == 1
    visible = [row.rstrip() for row in screen.display]
    assert "HEADER" in visible[-2]
    assert visible[-1] == "running-19"
    # No verdict was lost or duplicated by the region redraws.
    kept = [line for line in lines if line.startswith("verdict-")]
    assert kept == [f"verdict-{i}" for i in range(20)]


def test_region_growth_does_not_eat_verdicts() -> None:
    """Reserving more bottom rows must not overwrite existing verdicts."""
    out = io.StringIO()
    display = make_display(out)
    display.frame(["verdict-0"], ["HEADER"])
    display.frame(["verdict-1"], ["HEADER", "run-a", "run-b", "run-c"])
    display.frame([], ["HEADER"])
    display.close()

    lines = buffer_lines(emulate(out.getvalue()))
    kept = [line for line in lines if line.startswith("verdict-")]
    assert kept == ["verdict-0", "verdict-1"]
    assert sum("run-a" in line for line in lines) == 0


def test_region_starts_at_the_prompt_row_without_a_gap() -> None:
    """Output continues right below the shell prompt instead of jumping
    to the bottom of the screen."""
    out = io.StringIO()
    display = Display(out, size=lambda: os.terminal_size((WIDTH, HEIGHT)), origin=4)
    display.frame(["verdict-0", "verdict-1"], ["HEADER", "running"])

    visible = [row.rstrip() for row in emulate(out.getvalue()).display]
    assert visible[3:7] == ["verdict-0", "verdict-1", "HEADER", "running"]
    assert all(not row for row in visible[7:])


def test_resize_keeps_single_region() -> None:
    """A SIGWINCH shows up as a changed size() result on the next frame.
    The renderer must re-anchor the region without duplicating it."""
    size = {"cols": WIDTH, "rows": HEIGHT}
    out = io.StringIO()
    display = Display(out, size=lambda: os.terminal_size((size["cols"], size["rows"])))
    screen = pyte.HistoryScreen(WIDTH, HEIGHT, history=1000)
    stream = pyte.Stream(screen)

    def frame(batch: list[str], region: list[str]) -> None:
        mark = out.tell()
        display.frame(batch, region)
        stream.feed(out.getvalue()[mark:])

    for i in range(10):
        frame([f"verdict-{i}"], ["HEADER", f"running-{i}"])
    # The terminal shrinks between two frames.
    size.update(cols=40, rows=8)
    screen.resize(8, 40)
    for i in range(10, 16):
        frame([f"verdict-{i}"], ["HEADER", f"running-{i}"])

    lines = buffer_lines(screen)
    assert sum("HEADER" in line for line in lines) == 1
    visible = [row.rstrip() for row in screen.display]
    assert "HEADER" in visible[-2]
    assert visible[-1] == "running-15"
    # Verdicts printed after the resize are neither lost nor duplicated.
    post = [
        line for line in lines if line.startswith("verdict-1") and line != "verdict-1"
    ]
    assert post == [f"verdict-{i}" for i in range(10, 16)]


def test_resize_burst_never_erases_verdicts() -> None:
    """While the terminal is being resized (a burst of size changes, one
    per frame) nothing may be erased and the region stays hidden until
    the size settles again."""
    size = {"cols": WIDTH, "rows": HEIGHT}
    out = io.StringIO()
    display = Display(out, size=lambda: os.terminal_size((size["cols"], size["rows"])))
    screen = pyte.HistoryScreen(WIDTH, HEIGHT, history=1000)
    stream = pyte.Stream(screen)

    def frame(batch: list[str], region: list[str]) -> None:
        mark = out.tell()
        display.frame(batch, region)
        stream.feed(out.getvalue()[mark:])

    frame(["verdict-0", "verdict-1"], ["HEADER", "running"])
    resize_output_start = out.tell()
    for i, rows in enumerate((11, 10, 9)):  # drag in progress
        size.update(rows=rows)
        screen.resize(rows, WIDTH)
        frame([f"verdict-{2 + i}"], ["HEADER", "running"])
    burst = out.getvalue()[resize_output_start:]
    assert "HEADER" not in burst  # region not redrawn mid-resize
    assert "\x1b[J" not in burst  # nothing erased
    assert "\x1b[2J" not in burst

    # Once the size settles the visible screen is rebuilt from the model:
    # most recent verdicts on top, region below, no stale copies.
    frame(["verdict-5"], ["HEADER", "running"])
    frame(["verdict-6"], ["HEADER", "running"])
    visible = [row.rstrip() for row in screen.display]
    shown = [line for line in visible if line.startswith("verdict-")]
    assert shown == [f"verdict-{i}" for i in range(7)]
    assert visible.index("HEADER") == len(shown)
    assert sum("HEADER" in line for line in visible) == 1


def test_long_verdicts_wrap_without_corrupting_region() -> None:
    out = io.StringIO()
    display = make_display(out)
    long_line = "verdict-long " + "x" * (2 * WIDTH)
    for _ in range(5):
        display.frame([long_line], ["HEADER", "running"])

    lines = buffer_lines(emulate(out.getvalue()))
    assert sum("HEADER" in line for line in lines) == 1
    assert sum(line.startswith("verdict-long") for line in lines) == 5


def test_display_permanent_above_ephemeral() -> None:
    out = io.StringIO()
    d = make_display(out)
    d.ephemeral(["status"])
    d.permanent("event")
    visible = [row.rstrip() for row in emulate(out.getvalue()).display]
    assert visible.index("event") < visible.index("status")


def test_display_suspend_queues_permanent() -> None:
    out = io.StringIO()
    d = make_display(out)
    d.ephemeral(["status"])
    d.suspend()
    before = out.getvalue()
    d.permanent("while paging")
    assert out.getvalue() == before  # nothing written while suspended
    assert d.resume() == 1
    visible = [row.rstrip() for row in emulate(out.getvalue()).display]
    assert "while paging" in visible


def test_display_sync_markers() -> None:
    out = io.StringIO()
    d = make_display(out)
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


def test_render_summary_counts_and_elapsed() -> None:
    r, _out, clock = make_renderer()
    good = r.start_build("good", DRV)
    bad = r.start_build("bad", DRV)
    aborted = r.start_build("aborted", DRV)
    clock.now += 65
    r.finish_build(good, 0)
    r.finish_build(bad, 1)
    r.abort_build(aborted)
    text = ANSI.sub("", r.render_summary())
    assert "✔ 1 succeeded" in text
    assert "✘ 1 failed" in text
    assert "⏹ 1 aborted" in text
    assert "1m05s" in text


def test_render_summary_omits_zero_counts() -> None:
    r, _out, _clock = make_renderer()
    good = r.start_build("good", DRV)
    r.finish_build(good, 0)
    text = ANSI.sub("", r.render_summary())
    assert "✔ 1 succeeded" in text
    assert "failed" not in text
    assert "aborted" not in text


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
    d = make_display(out)
    d.ephemeral(["status"])
    h = DisplayLogHandler(d)
    logger = logging.getLogger("nfb-test")
    logger.addHandler(h)
    logger.warning("multi\nline")
    logger.removeHandler(h)
    visible = [row.rstrip() for row in emulate(out.getvalue()).display]
    assert "WARNING:nfb-test:multi" in visible
    assert "line" in visible
    # Region repainted below the log lines.
    assert visible.index("status") > visible.index("line")


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


def test_wait_until_idle() -> None:
    async def scenario() -> None:
        r, _out, _clock = make_renderer()
        # Not engaged: returns immediately.
        await asyncio.wait_for(r.wait_until_idle(), timeout=1)
        assert not r.all_done
        # Browser open: waits until the user leaves.
        fail_n(r, 1)
        r.on_key("f")
        waiter = asyncio.create_task(r.wait_until_idle())
        await asyncio.sleep(0.05)
        assert not waiter.done()
        assert r.all_done
        assert "finished" in "".join(r.render_list())
        r.on_key("q")
        await asyncio.wait_for(waiter, timeout=1)

    asyncio.run(scenario())


def test_arrow_keys_navigate_not_escape() -> None:
    r, _out, _clock = make_renderer()
    fail_n(r, 3)
    r.on_key("f")
    # Down arrow moves the cursor; must NOT act as Esc and close the list.
    r.feed_bytes(b"\x1b[B")
    assert r.mode is Mode.LIST
    assert r.cursor == 1
    r.feed_bytes(b"\x1b[A")
    assert r.cursor == 0
    # Unknown CSI (e.g. Home) swallowed entirely, no Esc, no stray keys.
    r.feed_bytes(b"\x1b[1~")
    assert r.mode is Mode.LIST
    assert r.cursor == 0
    # Two arrows in one read batch both processed.
    r.feed_bytes(b"\x1b[B\x1b[B")
    assert r.cursor == 2
    # Bare ESC still exits.
    r.feed_bytes(b"\x1b")
    assert r.mode is Mode.NORMAL

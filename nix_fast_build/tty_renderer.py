"""Interactive (TTY) renderer for nix build logs.

Superconsole model: finished-build verdicts and failure extracts go to
normal terminal scrollback (permanent lines), while a region at the
bottom is redrawn in place showing the running builds. 'f' opens a log
browser over failed and running builds; logs open in $PAGER (running
builds are followed live via `less +F`) or are dumped to scrollback.

The terminal is only put into cbreak mode (no raw mode, no alternate
screen), so output stays in scrollback and Ctrl-C keeps working.
"""

import asyncio
import contextlib
import logging
import os
import shlex
import shutil
import signal
import sys
import tempfile
import termios
import time
import tty
from collections import deque
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import IO, Any

from .renderer import BuildOutput, fmt_duration
from .term import clip_ansi, subseq_match, trunc_middle

CSI = "\x1b["
DIM = f"{CSI}2m"
RED = f"{CSI}31m"
GREEN = f"{CSI}32m"
YELLOW = f"{CSI}33m"
BOLD = f"{CSI}1m"
RESET = f"{CSI}0m"

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"

EXTRACT_LINES = 5  # failure extract printed to scrollback


class DisplayLogHandler(logging.Handler):
    """Routes log records through the display.

    Anything written to stderr behind the display's back lands inside the
    ephemeral region and breaks its cursor-up anchor, leaving stale rows.
    """

    def __init__(self, display: "Display") -> None:
        super().__init__()
        self.display = display
        self.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.display.permanent(*self.format(record).splitlines())


class Mode(Enum):
    NORMAL = auto()
    LIST = auto()


class Display:
    """Owns the terminal: permanent scrollback lines + ephemeral region.

    Flicker avoidance: every update is composed into one buffer and emitted
    with a single write, wrapped in the synchronized-output escape (DEC 2026)
    so capable terminals paint it atomically. Ephemeral lines are overwritten
    in place with clear-to-EOL instead of erasing the whole region first.
    """

    def __init__(self, out: IO[str]) -> None:
        self.out = out
        self.ephemeral_lines = 0
        self.last_ephemeral: list[str] = []
        self.suspended = False
        self._pending: list[str] = []  # permanent lines queued while suspended

    def _emit(self, buf: str) -> None:
        self.out.write(f"{CSI}?2026h{buf}{CSI}?2026l")
        self.out.flush()

    def _compose_ephemeral(self, lines: list[str]) -> str:
        """Cursor sits below old region; overwrite it line by line."""
        width = shutil.get_terminal_size().columns
        buf = f"{CSI}{self.ephemeral_lines}F" if self.ephemeral_lines else ""
        for line in lines:
            # Cell-aware clip (CJK/emoji count as 2): an overwide line would
            # wrap and break the cursor-up math for the whole region.
            # RESET re-applied because the clip may drop a trailing reset.
            buf += f"{clip_ansi(line, width)}{RESET}{CSI}K\n"
        if len(lines) < self.ephemeral_lines:
            buf += f"{CSI}J"  # old region was taller: drop leftover lines
        self.ephemeral_lines = len(lines)
        self.last_ephemeral = lines
        return buf

    def permanent(self, *lines: str) -> None:
        if self.suspended:
            # Another program (pager) owns the terminal: queue for later.
            self._pending.extend(lines)
            return
        # Erase ephemeral, print permanent lines, repaint ephemeral: one write.
        buf = f"{CSI}{self.ephemeral_lines}F{CSI}J" if self.ephemeral_lines else ""
        self.ephemeral_lines = 0
        buf += "".join(f"{line}\n" for line in lines)
        buf += self._compose_ephemeral(self.last_ephemeral)
        self._emit(buf)

    def ephemeral(self, lines: list[str]) -> None:
        if self.suspended:
            return
        self._emit(self._compose_ephemeral(lines))

    def suspend(self) -> None:
        """Clear our region and stop touching the terminal."""
        self.ephemeral([])
        self.last_ephemeral = []
        self.suspended = True

    def resume(self) -> int:
        """Re-take the terminal; flush events that happened meanwhile.

        Returns the number of flushed lines so the caller can tell the
        user what they missed.
        """
        self.suspended = False
        # The pager left the cursor anywhere; start fresh on a new line.
        self.ephemeral_lines = 0
        pending, self._pending = self._pending, []
        if pending:
            self.permanent(*pending)
        return len(pending)


class TTYRenderer:
    PAGE = 6  # max browser rows per page (each row is 2 lines)
    BUFFER_LINES = 10_000

    def __init__(
        self,
        out: IO[str],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.display = Display(out)
        self.clock = clock
        self.started_at = clock()
        self.mode = Mode.NORMAL
        self.running: list[BuildOutput] = []  # insertion order = start order
        self.failed: list[BuildOutput] = []
        self.succeeded: set[BuildOutput] = set()
        self.spin = 0
        self.dump_action = False  # False = pager, True = dump to scrollback
        self.page = 0
        self.filter = ""
        self.filter_input = False
        self.pager_active = False
        # termios attr list as returned by tcgetattr (mixed int/list).
        self.cooked_termios: list[Any] = []
        self.bg_tasks: set[asyncio.Task[None]] = set()
        self._render_task: asyncio.Task[None] | None = None
        self._saved_handlers: list[logging.Handler] | None = None
        self._stopping = False
        # LIST mode state: order pinned on entry so rows don't shift under
        # the user's finger; later arrivals are appended and tagged "new".
        self.pinned: list[BuildOutput] = []
        self.new_builds: set[BuildOutput] = set()
        self.cursor = 0
        self.last_viewed: BuildOutput | None = None
        self.all_done = False  # builds finished, only the browser keeps us up
        self._idle = asyncio.Event()
        self.flash_text = ""
        self.flash_until = 0.0

    # ── Renderer protocol ────────────────────────────────────────────

    def start_build(self, attr: str, drv_path: str) -> BuildOutput:
        now = self.clock()
        build = BuildOutput(
            attr=attr,
            drv_path=drv_path,
            renderer=self,
            started_at=now,
            last_output_at=now,
            lines=deque(maxlen=self.BUFFER_LINES),
        )
        self.running.append(build)
        return build

    def finish_build(self, build: BuildOutput, rc: int) -> None:
        with contextlib.suppress(ValueError):
            self.running.remove(build)
        stamp = f"{DIM}{time.strftime('%H:%M:%S')}{RESET}"
        duration = fmt_duration(build.elapsed())
        if rc == 0:
            self.succeeded.add(build)
            self.display.permanent(f"{stamp} {GREEN}✔  {build.attr}{RESET}  {duration}")
            return
        self.failed.append(build)
        tail = list(build.lines)[-EXTRACT_LINES:]
        self.display.permanent(
            f"{stamp} {RED}✘  {build.attr}{RESET}  {duration}  rc={rc}",
            f"{DIM}── error extract (full log: [f] or nix log {build.drv_path}) ──{RESET}",
            *(f"{DIM}{build.attr}>{RESET} {line}" for line in tail),
        )

    def abort_build(self, build: BuildOutput) -> None:
        with contextlib.suppress(ValueError):
            self.running.remove(build)

    def emit_live_line(self, build: BuildOutput, line: str) -> None:
        """Host protocol; TTY mode never streams (rows are already live)."""

    def log_line(self, line: str) -> None:
        self.display.permanent(line)

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        fd = sys.stdin.fileno()
        self.cooked_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        loop.add_reader(fd, self.on_stdin_readable)
        # Hide the cursor: it strobes across the region during redraws.
        self._write_ctl(HIDE_CURSOR)
        loop.add_signal_handler(signal.SIGWINCH, self._on_winch)
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        root.handlers = [DisplayLogHandler(self.display)]
        self._render_task = asyncio.create_task(self._render_loop(), name="tty-render")

    def _engaged(self) -> bool:
        return self.mode is Mode.LIST or self.pager_active or bool(self.bg_tasks)

    def _signal_if_idle(self) -> None:
        """Called at every disengage point (browser exit, pager close)."""
        if self.all_done and not self._engaged():
            self._idle.set()

    async def wait_until_idle(self) -> None:
        """Builds are done: if the user is in the log browser or a pager,
        keep the UI alive until they leave instead of yanking it away."""
        if not self._engaged():
            return
        self.all_done = True
        self.flash("builds finished — leave browser ([q/Esc]) to exit")
        await self._idle.wait()

    async def stop(self) -> None:
        if self._stopping:
            return
        # Also keeps a cancelled pager's cleanup from re-attaching the
        # key reader and cbreak mode behind our back.
        self._stopping = True
        loop = asyncio.get_running_loop()
        fd = sys.stdin.fileno()
        # Detach input first: a keystroke arriving while we await the
        # cancellations below could spawn a fresh pager task.
        with contextlib.suppress(ValueError, OSError):
            loop.remove_reader(fd)
            loop.remove_signal_handler(signal.SIGWINCH)
        tasks = [t for t in (self._render_task, *self.bg_tasks) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._render_task = None
        if self.cooked_termios:
            termios.tcsetattr(fd, termios.TCSADRAIN, self.cooked_termios)
        if self._saved_handlers is not None:
            logging.getLogger().handlers = self._saved_handlers
            self._saved_handlers = None
        self.display.ephemeral([])
        self._write_ctl(SHOW_CURSOR)

    def _write_ctl(self, seq: str) -> None:
        self.display.out.write(seq)
        self.display.out.flush()

    def _on_winch(self) -> None:
        # Terminal resized: old region may have rewrapped, making the
        # cursor-up anchor wrong. Abandon it (stale lines become inert
        # scrollback) and let the next tick draw a fresh region.
        self.display.ephemeral_lines = 0

    async def _render_loop(self) -> None:
        while True:
            self.spin += 1
            if not self.pager_active:
                lines = (
                    self.render_list()
                    if self.mode is Mode.LIST
                    else self.render_normal()
                )
                self.display.ephemeral(lines)
            await asyncio.sleep(0.25)

    # ── rendering ────────────────────────────────────────────────────

    def _running_sorted(self) -> list[BuildOutput]:
        """Longest-elapsed first; ties keep start order (no jitter)."""
        return sorted(self.running, key=lambda b: b.started_at)

    def header(self) -> str:
        elapsed = fmt_duration(self.clock() - self.started_at)
        done = f" · {BOLD}finished{RESET}" if self.all_done else ""
        return (
            f" {BOLD}BUILD{RESET} {GREEN}✔{len(self.succeeded)}{RESET} "
            f"{RED}✘{len(self.failed)}{RESET} ⏵{len(self.running)}"
            f"   {elapsed}{done}"
        )

    def render_normal(self) -> list[str]:
        lines = [self.header()]
        spin = SPINNER[self.spin % len(SPINNER)]
        # A region taller than the terminal breaks the cursor-up anchor
        # (the cursor can't move above the top), so clamp the build rows.
        # Reserve 3 lines: header, footer, and one against off-by-one
        # terminal quirks.
        budget = max(2, shutil.get_terminal_size().lines - 3)
        running = self._running_sorted()
        for shown, b in enumerate(running):
            rows = 2 if b.lines else 1
            if budget - rows < (1 if shown < len(running) - 1 else 0):
                lines.append(f"     {DIM}… +{len(running) - shown} more{RESET}")
                break
            budget -= rows
            phase = b.phase or "build"
            lines.append(
                f" {YELLOW}{spin}{RESET} {trunc_middle(b.attr, 40):<40} "
                f"{phase:<12} {fmt_duration(b.elapsed()):>7}"
            )
            if b.lines:
                lines.append(f"     {DIM}{b.lines[-1]}{RESET}")
        badge = f" ({RED}✘{len(self.failed)}{RESET}{DIM})" if self.failed else ""
        reopen = " · [o] last log" if self.last_viewed else ""
        footer = f"{DIM} [f] logs{badge}{reopen} · [Ctrl-C] abort{RESET}"
        lines.append(self._flash_line() or footer)
        return lines

    def _browsable(self) -> list[BuildOutput]:
        """Failures first (triage priority), then running."""
        return self.failed + self._running_sorted()

    def _enter_list(self) -> None:
        self.mode = Mode.LIST
        self.pinned = list(self._browsable())
        self.new_builds = set()
        self.cursor = 0
        self.page = 0

    def _refresh_pinned(self) -> None:
        """Append builds that appeared after the list was opened.

        Existing rows never move; finished builds stay selectable (their
        captured log is still useful).
        """
        known = set(self.pinned)
        for b in self._browsable():
            if b not in known:
                self.pinned.append(b)
                self.new_builds.add(b)

    def _filtered(self) -> list[BuildOutput]:
        self._refresh_pinned()
        if not self.filter:
            return self.pinned
        return [b for b in self.pinned if subseq_match(self.filter, b.attr)]

    def _page_size(self) -> int:
        # Adapt to terminal height like render_normal: header, separator
        # and footer need 4 lines, each row takes 2.
        return max(1, min(self.PAGE, (shutil.get_terminal_size().lines - 4) // 2))

    def _pages(self) -> int:
        return max(1, -(-len(self._filtered()) // self._page_size()))

    def _page_slice(self) -> list[BuildOutput]:
        size = self._page_size()
        self.page = min(self.page, self._pages() - 1)
        return self._filtered()[self.page * size : (self.page + 1) * size]

    def flash(self, msg: str) -> None:
        self.flash_text = msg
        self.flash_until = self.clock() + 1.5

    def _flash_line(self) -> str | None:
        if self.clock() < self.flash_until:
            return f" {YELLOW}{self.flash_text}{RESET}"
        return None

    def _status_label(self, b: BuildOutput) -> str:
        if b in self.running:
            return f"{YELLOW}⏵ {b.phase or 'build'}{RESET}"
        if b in self.succeeded:
            return f"{GREEN}✔ done{RESET}"
        return f"{RED}✘ failed{RESET}"

    def _action_label(self, b: BuildOutput | None) -> str:
        if self.dump_action:
            return "dump to scrollback"
        if b is not None and b in self.running:
            return "less +F (live)"
        return "less"

    def render_list(self) -> list[str]:
        filtered = self._filtered()
        self.cursor = max(0, min(self.cursor, len(filtered) - 1))
        if filtered:
            self.page = self.cursor // self._page_size()
        pages = self._pages()
        page_info = f" page {self.page + 1}/{pages}" if pages > 1 else ""
        match_info = (
            f" · {len(filtered)}/{len(self.pinned)} match" if self.filter else ""
        )
        lines = [
            self.header(),
            f" {DIM}── logs{page_info}{match_info} ──────────────────{RESET}",
        ]
        visible = self._page_slice()
        for i, b in enumerate(visible, 1):
            selected = self.page * self._page_size() + i - 1 == self.cursor
            marker = f"{BOLD}▸{RESET}" if selected else " "
            new = f" {YELLOW}new{RESET}" if b in self.new_builds else ""
            lines.append(
                f" {marker}{BOLD}{i}{RESET}  {trunc_middle(b.attr, 40):<40} "
                f"{fmt_duration(b.elapsed()):>7}  {self._status_label(b)}{new}"
            )
            gist = b.lines[-1] if b.lines else "(no output yet)"
            lines.append(f"      {DIM}{gist}{RESET}")
        if not visible:
            msg = "(no matches)" if self.filter else "no failed or running builds"
            lines.append(f"  {DIM}{msg}{RESET}")
        if self.filter_input:
            lines.append(
                f" {YELLOW}/{self.filter}█{RESET}  "
                f"{DIM}[Enter] apply · [Esc] clear{RESET}"
            )
            return lines
        if (fl := self._flash_line()) is not None:
            lines.append(fl)
            return lines
        selected_build = filtered[self.cursor] if filtered else None
        action = self._action_label(selected_build)
        paging = " · [n/p] page" if pages > 1 else ""
        flt = f" · filter:{YELLOW}/{self.filter}{RESET}{DIM}" if self.filter else ""
        lines.append(
            f" {DIM}[Enter/1-{max(len(visible), 1)}] {action} · [j/k] move · "
            f"[d]→{'pager' if self.dump_action else 'dump'}"
            f"{paging} · [/] filter{flt} · [f/Esc] back{RESET}"
        )
        return lines

    # ── log viewing ──────────────────────────────────────────────────

    def _state_label(self, b: BuildOutput) -> tuple[str, bool]:
        running = b in self.running
        return (f"{b.phase or 'build'}, running" if running else "failed", running)

    def _dump_log(self, b: BuildOutput, state: str, running: bool) -> None:
        self.display.permanent(
            f"{DIM}────── log: {b.attr} ({state}, {fmt_duration(b.elapsed())}) ──────{RESET}",
            *(f"{DIM}{b.attr}>{RESET} {line}" for line in b.lines),
            f"{DIM}────── end ({len(b.lines)} lines"
            + (" so far" if running else "")
            + f") · re-fetch: nix log {b.drv_path} ──────{RESET}",
        )

    @staticmethod
    def _pager_cmd(b: BuildOutput, state: str, running: bool) -> list[str]:
        pager = shlex.split(os.environ.get("PAGER", "less"))
        if Path(pager[0]).name == "less":
            # Title in the prompt line so the user knows what they're reading.
            pager.append(f"-Ps{b.attr} ({state}) ?pB(%pB\\%).")
            # +F: follow live (Ctrl-C in less to scroll).
            # +G: failed log, the error is at the end.
            pager.append("+F" if running else "+G")
        return pager

    @staticmethod
    def _write_log_tmpfile(b: BuildOutput, state: str) -> Path:
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"-{b.attr.replace('/', '_')}.log", delete=False
        ) as tf:
            tf.write(f"# log: {b.attr} ({state}, {fmt_duration(b.elapsed())})\n")
            tf.writelines(f"{b.attr}> {line}\n" for line in b.lines)
            return Path(tf.name)

    async def _page_log(self, b: BuildOutput, state: str, running: bool) -> None:
        path = self._write_log_tmpfile(b, state)
        written = b.total_lines

        # For running builds, keep appending new log lines so `less +F`
        # follows live output. total_lines (not indexing) because the
        # ring buffer may rotate while we follow. The file is opened here
        # in sync context; writes are small appends to a local tmpfile.
        follow_file = path.open("a") if running else None

        async def follow(f: IO[str]) -> None:
            nonlocal written
            while True:
                new = b.total_lines - written
                if new > 0:
                    tail = list(b.lines)[-min(new, len(b.lines)) :]
                    f.writelines(f"{b.attr}> {line}\n" for line in tail)
                    f.flush()
                    written = b.total_lines
                await asyncio.sleep(0.2)

        # Hand the terminal to the pager: stop drawing (renderer paused via
        # pager_active, permanent lines queued by display.suspend), detach
        # the key reader so we don't steal the pager's keystrokes, and give
        # it a cooked tty. Builds keep running: the pager is awaited, not
        # blocking the event loop.
        env = os.environ.copy()
        # -R color, -X keep log on screen after quit, -i smartcase search.
        env.setdefault("LESS", "RXi")
        follower = (
            asyncio.create_task(follow(follow_file))
            if follow_file is not None
            else None
        )
        loop = asyncio.get_running_loop()
        fd = sys.stdin.fileno()
        self.display.suspend()
        loop.remove_reader(fd)
        # Ctrl-C must only reach the pager (exits less follow mode), not
        # abort all builds behind its back.
        loop.add_signal_handler(signal.SIGINT, lambda: None)
        termios.tcsetattr(fd, termios.TCSADRAIN, self.cooked_termios)
        self._write_ctl(SHOW_CURSOR)  # pager expects a visible cursor
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._pager_cmd(b, state, running), path, env=env
            )
            await proc.wait()
        finally:
            if proc is not None and proc.returncode is None:
                # Cancelled (shutdown) while the pager was open: don't
                # leave an orphaned less owning the terminal. The await
                # may itself be cancelled again; less got SIGTERM either
                # way and the process is reaped at loop shutdown.
                proc.terminate()
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=2)
            self._write_ctl(HIDE_CURSOR)
            loop.remove_signal_handler(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            if follower is not None:
                follower.cancel()
                await asyncio.gather(follower, return_exceptions=True)
            if follow_file is not None:
                follow_file.close()
            if not self._stopping:
                tty.setcbreak(fd)
                loop.add_reader(fd, self.on_stdin_readable)
            path.unlink()
            self.pager_active = False
            missed = self.display.resume()
            if missed:
                self.flash(f"while paging: {missed} new lines in scrollback ↑")

    # ── key handling ─────────────────────────────────────────────────

    def on_key(self, key: str) -> None:
        if self.mode is Mode.NORMAL:
            self._key_normal(key)
        elif self.filter_input:
            self._key_filter(key)
        else:
            self._key_list(key)

    def _key_normal(self, key: str) -> None:
        if key == "f":
            self._enter_list()
        elif key == "o" and self.last_viewed is not None:
            self._open(self.last_viewed)

    def _key_filter(self, key: str) -> None:
        if key in ("\r", "\n"):
            self.filter_input = False
            matches = self._filtered()
            if len(matches) == 1:
                # Single match: open it directly, fzf-style.
                self._open(matches[0])
        elif key == "\x1b":
            self.filter = ""
            self.filter_input = False
        elif key in ("\x7f", "\x08"):
            self.filter = self.filter[:-1]
        elif key.isprintable():
            self.filter += key
            self.cursor = 0

    def _key_list(self, key: str) -> None:
        if key == "\x1b" and self.filter:
            self.filter = ""
        elif key in ("f", "\x1b", "q"):
            self.mode = Mode.NORMAL
            self.filter = ""
            self.filter_input = False
            self._signal_if_idle()
        elif key == "/":
            self.filter_input = True
        elif key == "d":
            self.dump_action = not self.dump_action
        elif key in ("j", "k"):
            self._move_cursor(1 if key == "j" else -1)
        elif key in ("n", "p"):
            self._move_page(1 if key == "n" else -1)
        elif key in ("\r", "\n"):
            filtered = self._filtered()
            if filtered:
                self._open(filtered[min(self.cursor, len(filtered) - 1)])
            else:
                self.flash("nothing to open")
        elif key.isdigit():
            visible = self._page_slice()
            if 1 <= int(key) <= len(visible):
                self._open(visible[int(key) - 1])
            else:
                self.flash(f"no entry {key} on this page")
        elif key.isprintable():
            self.flash(f"unknown key {key!r} — [/] to filter")

    def _move_cursor(self, delta: int) -> None:
        """Clamped, with edge feedback (navigation never wraps)."""
        target = self.cursor + delta
        if 0 <= target < len(self._filtered()):
            self.cursor = target
        else:
            self.flash("already at bottom" if delta > 0 else "already at top")

    def _move_page(self, delta: int) -> None:
        target = self.page + delta
        if 0 <= target < self._pages():
            self.page = target
            self.cursor = target * self._page_size()
        else:
            self.flash("already at last page" if delta > 0 else "already at first page")

    def _open(self, b: BuildOutput) -> None:
        state, running = self._state_label(b)
        self.last_viewed = b
        if self.dump_action:
            self._dump_log(b, state, running)
            self.flash(f"dumped {len(b.lines)} lines ↑")
            return
        if self.pager_active:
            # Two opens can race within one stdin batch (the key reader is
            # only detached once _page_log runs): never spawn two pagers.
            return
        # Claim the pager synchronously, before _page_log gets scheduled.
        self.pager_active = True
        task = asyncio.create_task(self._page_log(b, state, running))
        self.bg_tasks.add(task)

        def on_done(t: asyncio.Task[None]) -> None:
            self.bg_tasks.discard(t)
            self._signal_if_idle()

        task.add_done_callback(on_done)

    def on_stdin_readable(self) -> None:
        # Drain everything available: sys.stdin.read(1) would buffer pasted
        # bytes inside the TextIO object where the selector can't see them.
        try:
            data = os.read(sys.stdin.fileno(), 1024)
        except BlockingIOError:
            return  # spurious wakeup
        if not data:
            # EOF (terminal hangup): stop listening or the selector would
            # fire this callback in a busy loop forever.
            asyncio.get_running_loop().remove_reader(sys.stdin.fileno())
            return
        for i, byte in enumerate(data):
            if byte == 0x1B and i + 1 < len(data):
                # Swallow escape sequences (arrow keys etc.) so their tail
                # bytes aren't misread as commands or filter input.
                self.on_key("\x1b")
                break
            self.on_key(chr(byte) if byte < 0x80 else "\ufffd")

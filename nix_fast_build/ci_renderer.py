"""Non-interactive (CI) renderer for nix build logs.

Replaces `nix-build -L` output. Each build's log is buffered and
emitted as one contiguous block when the build finishes, so logs of
concurrent builds never interleave. On Actions-style CI, successful
builds are folded with ::group:: markers; failed builds are never
folded. A periodic heartbeat names the running builds so the CI log
shows liveness, and builds that produce no output for stall_timeout
seconds are escalated to live streaming.
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from typing import IO

from .renderer import BuildOutput, fmt_duration

# Unfolded tail of a failure log; earlier output gets folded.
FAILURE_TAIL_LINES = 200

GREEN = "\x1b[32m"
RED = "\x1b[31m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"

STALL_TAIL_LINES = 5


class CIRenderer:
    def __init__(
        self,
        out: IO[str],
        *,
        color: bool,
        fold: bool,
        heartbeat_interval: float = 30.0,
        stall_timeout: float = 300.0,
        buffer_lines: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.out = out
        self.color = color
        self.fold = fold
        self.heartbeat_interval = heartbeat_interval
        self.stall_timeout = stall_timeout
        self.buffer_lines = buffer_lines
        self.clock = clock
        self.running: set[BuildOutput] = set()
        self._heartbeat_task: asyncio.Task[None] | None = None

    def _sgr(self, code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if self.color else s

    def _print(self, *lines: str) -> None:
        self.out.write("".join(f"{line}\n" for line in lines))
        self.out.flush()

    # ── build lifecycle ──────────────────────────────────────────────

    def start_build(self, attr: str, drv_path: str) -> BuildOutput:
        now = self.clock()
        build = BuildOutput(
            attr=attr,
            drv_path=drv_path,
            renderer=self,
            started_at=now,
            last_output_at=now,
            lines=deque(maxlen=self.buffer_lines),
        )
        self.running.add(build)
        self._print(f"▶ {build.attr} started")
        return build

    def abort_build(self, build: BuildOutput) -> None:
        """Drop a build without a verdict (cancellation/shutdown). A leaked
        entry would keep showing up in the heartbeat and trip the stall
        detector."""
        self.running.discard(build)

    def finish_build(self, build: BuildOutput, rc: int) -> None:
        self.running.discard(build)
        duration = fmt_duration(build.elapsed())
        if build.streaming:
            # Log already on screen; just print the verdict.
            self._print(self._verdict_line(build, rc, duration))
            return
        if rc == 0:
            self._emit_success(build, duration)
        else:
            self._emit_failure(build, rc, duration)

    def _verdict_line(self, build: BuildOutput, rc: int, duration: str) -> str:
        if rc == 0:
            return self._sgr(GREEN, f"✔  {build.attr} ({duration})")
        return self._sgr(RED, f"✘  {build.attr} failed after {duration} (rc={rc})")

    def _emit_success(self, build: BuildOutput, duration: str) -> None:
        verdict = self._verdict_line(build, 0, duration)
        if self.fold:
            self._print(f"::group::{verdict}", *self._prefixed(build), "::endgroup::")
        else:
            self._print(verdict, *self._prefixed(build))

    def _emit_failure(self, build: BuildOutput, rc: int, duration: str) -> None:
        # The error is almost always at the end: keep the tail visible,
        # fold the rest so multi-failure CI pages stay scrollable.
        dropped = ""
        if len(build.lines) == build.lines.maxlen:
            dropped = f" (oldest lines dropped, buffer={build.lines.maxlen})"
        lines = self._prefixed(build)
        body = lines
        if self.fold and len(lines) > FAILURE_TAIL_LINES:
            head = lines[:-FAILURE_TAIL_LINES]
            body = [
                f"::group::earlier output ({len(head)} lines)",
                *head,
                "::endgroup::",
                *lines[-FAILURE_TAIL_LINES:],
            ]
        self._print(
            self._verdict_line(build, rc, duration) + dropped,
            *body,
            self._sgr(RED, f"✘ end of log for {build.attr} · nix log {build.drv_path}"),
        )

    def _prefix(self, build: BuildOutput) -> str:
        return self._sgr(DIM, f"{build.attr}>")

    def _prefixed(self, build: BuildOutput) -> list[str]:
        prefix = self._prefix(build)
        return [f"{prefix} {line}" for line in build.lines]

    def emit_live_line(self, build: BuildOutput, line: str) -> None:
        self._print(f"{self._prefix(build)} {line}")

    def log_line(self, line: str) -> None:
        self._print(line)

    # ── heartbeat / stall detection ──────────────────────────────────

    def start_heartbeat(self) -> None:
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="ci-heartbeat"
        )

    async def stop_heartbeat(self) -> None:
        if self._heartbeat_task is None:
            return
        self._heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._heartbeat_task
        self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            self.heartbeat()

    def heartbeat(self) -> None:
        self._check_stalls()
        if not self.running:
            return
        builds = sorted(self.running, key=lambda b: -b.elapsed())
        parts = []
        for b in builds:
            detail = fmt_duration(b.elapsed())
            if b.phase:
                detail += f", {b.phase}"
            flag = " ⚠stalled" if b.streaming else ""
            parts.append(f"{b.attr} ({detail}){flag}")
        self._print(self._sgr(DIM, f"⏵ {len(builds)} building: " + ", ".join(parts)))

    def _check_stalls(self) -> None:
        now = self.clock()
        for b in self.running:
            if b.streaming or now - b.last_output_at < self.stall_timeout:
                continue
            silent = fmt_duration(now - b.last_output_at)
            tail = list(b.lines)[-STALL_TAIL_LINES:]
            prefix = self._prefix(b)
            self._print(
                self._sgr(BOLD, f"⚠ {b.attr}: no output for {silent}, last lines:"),
                *(f"{prefix} {line}" for line in tail),
                self._sgr(BOLD, f"⚠ {b.attr}: streaming further output live"),
            )
            # From now on lines appear immediately; the final block is
            # skipped for this build (log already on screen).
            b.streaming = True

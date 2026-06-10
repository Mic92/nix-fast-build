"""Shared pieces of the build log renderers (CI and TTY)."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .log_format import (
    BuildLogLine,
    LogEvent,
    Message,
    PhaseChanged,
    PlainLine,
)
from .term import sanitize_line


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Host(Protocol):
    """What BuildOutput needs from the renderer that owns it."""

    clock: Callable[[], float]

    def emit_live_line(self, build: "BuildOutput", line: str) -> None: ...


@dataclass(eq=False)  # identity-hashed so renderers can keep sets
class BuildOutput:
    """Per-build log sink fed by the internal-json parser."""

    attr: str
    drv_path: str
    renderer: Host
    started_at: float
    last_output_at: float
    lines: deque[str]
    phase: str | None = None
    streaming: bool = False

    def on_event(self, event: LogEvent) -> None:
        match event:
            case BuildLogLine(line=line) | PlainLine(line=line):
                self._add_line(sanitize_line(line))
            case PhaseChanged(phase=phase):
                self.phase = phase
                self._add_line(f"@ phase {phase}")
            case Message(level=level, msg=msg) if level <= 2:
                # Warnings and errors; the final level-0 error message
                # carries the failure reason and the nix log command.
                for msg_line in msg.splitlines():
                    self._add_line(sanitize_line(msg_line))

    def _add_line(self, line: str) -> None:
        self.last_output_at = self.renderer.clock()
        self.lines.append(line)
        if self.streaming:
            self.renderer.emit_live_line(self, line)

    def elapsed(self) -> float:
        return self.renderer.clock() - self.started_at


class Renderer(Protocol):
    """Interface the build workers drive."""

    def start_build(self, attr: str, drv_path: str) -> BuildOutput: ...

    def finish_build(self, build: BuildOutput, rc: int) -> None: ...

    def abort_build(self, build: BuildOutput) -> None: ...

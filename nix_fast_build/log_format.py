"""Parser for nix's `--log-format internal-json` output.

Nix and Lix share this format (same logging.hh enums and JSON shape):
stderr lines of the form

    @nix {"action": "start"|"stop"|"result"|"msg", ...}

Lines without the `@nix ` prefix (remote builders, wrapper tools) are
passed through as `PlainLine`.

Each nix-build process gets its own `LogParser` instance, so activity
ids never collide between concurrent builds.
"""

import json
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class ActivityType(IntEnum):
    """Activity types shared by Nix and Lix (logging.hh)."""

    UNKNOWN = 0
    COPY_PATH = 100
    FILE_TRANSFER = 101
    REALISE = 102
    COPY_PATHS = 103
    BUILDS = 104
    BUILD = 105
    OPTIMISE_STORE = 106
    VERIFY_PATHS = 107
    SUBSTITUTE = 108
    QUERY_PATH_INFO = 109
    POST_BUILD_HOOK = 110
    BUILD_WAITING = 111
    FETCH_TREE = 112  # nix only; lix does not emit it


class ResultType(IntEnum):
    """Result types shared by Nix and Lix (logging.hh); only the ones
    this parser turns into events."""

    BUILD_LOG_LINE = 101
    SET_PHASE = 104
    PROGRESS = 105
    POST_BUILD_LOG_LINE = 107


@dataclass
class Activity:
    id: int
    # int, not ActivityType: nix may add types this parser doesn't know.
    type: int
    text: str
    parent: int
    level: int
    fields: list[Any]


@dataclass
class ActivityStarted:
    activity: Activity


@dataclass
class ActivityStopped:
    activity: Activity | None
    id: int


@dataclass
class BuildLogLine:
    """A line of builder output (resBuildLogLine / resPostBuildLogLine)."""

    line: str
    activity: Activity | None


@dataclass
class PhaseChanged:
    """stdenv phase change (resSetPhase), e.g. "buildPhase"."""

    phase: str
    activity: Activity | None


@dataclass
class Progress:
    """resProgress: [done, expected, running, failed]."""

    done: int
    expected: int
    running: int
    failed: int
    activity: Activity | None


@dataclass
class Message:
    """Free-form log message; level 0 messages carry build errors."""

    level: int
    msg: str
    raw_msg: str | None


@dataclass
class PlainLine:
    """stderr line that is not internal-json (passed through verbatim)."""

    line: str


LogEvent = (
    ActivityStarted
    | ActivityStopped
    | BuildLogLine
    | PhaseChanged
    | Progress
    | Message
    | PlainLine
)

JSON_PREFIX = "@nix "


class LogParser:
    """Stateful per-process parser tracking the activity tree."""

    def __init__(self) -> None:
        self.activities: dict[int, Activity] = {}

    def parse_line(self, raw: bytes) -> LogEvent | None:
        """Parse one stderr line. Returns None for ignorable records."""
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text.startswith(JSON_PREFIX):
            return PlainLine(text)
        try:
            record = json.loads(text[len(JSON_PREFIX) :])
        except json.JSONDecodeError:
            # Truncated/corrupted record: show it rather than lose it.
            return PlainLine(text)
        if not isinstance(record, dict):
            return PlainLine(text)
        try:
            return self._dispatch(record)
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logger.debug("malformed internal-json record %r: %s", record, e)
            return None

    @staticmethod
    def _fields(record: dict[str, Any]) -> list[Any]:
        # null or non-list becomes []: a string must not reach indexing
        # code, "abc"[0] would misparse instead of failing.
        fields = record.get("fields")
        return fields if isinstance(fields, list) else []

    def _dispatch(self, record: dict[str, Any]) -> LogEvent | None:
        match record.get("action"):
            case "start":
                activity = Activity(
                    id=record["id"],
                    type=record.get("type", 0),
                    text=record.get("text", ""),
                    parent=record.get("parent", 0),
                    level=record.get("level", 0),
                    fields=self._fields(record),
                )
                self.activities[activity.id] = activity
                return ActivityStarted(activity)
            case "stop":
                act_id = record["id"]
                return ActivityStopped(self.activities.pop(act_id, None), act_id)
            case "result":
                return self._result(record)
            case "msg":
                return Message(
                    level=record.get("level", 0),
                    msg=record.get("msg", ""),
                    raw_msg=record.get("raw_msg"),
                )
            case _:
                return None

    def _result(self, record: dict[str, Any]) -> LogEvent | None:
        activity = self.activities.get(record["id"])
        fields = self._fields(record)
        match record.get("type"):
            case ResultType.BUILD_LOG_LINE | ResultType.POST_BUILD_LOG_LINE:
                return BuildLogLine(line=str(fields[0]), activity=activity)
            case ResultType.SET_PHASE:
                return PhaseChanged(phase=str(fields[0]), activity=activity)
            case ResultType.PROGRESS:
                done, expected, running, failed = (int(f) for f in fields[:4])
                return Progress(done, expected, running, failed, activity)
            case _:
                return None

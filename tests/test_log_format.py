"""Tests for the internal-json log parser against real nix/lix output."""

from pathlib import Path

from nix_fast_build.log_format import (
    ActivityStarted,
    ActivityStopped,
    ActivityType,
    BuildLogLine,
    LogEvent,
    LogParser,
    Message,
    PhaseChanged,
    PlainLine,
    Progress,
)

FIXTURES = Path(__file__).parent / "fixtures" / "internal_json"


def parse_fixture(name: str) -> list[LogEvent]:
    parser = LogParser()
    events = []
    with (FIXTURES / name).open("rb") as f:
        for line in f:
            event = parser.parse_line(line)
            if event is not None:
                events.append(event)
    return events


def test_nix_success_fixture() -> None:
    events = parse_fixture("nix-success.jsonl")
    log_lines = [e.line for e in events if isinstance(e, BuildLogLine)]
    assert "hello-line-1" in log_lines
    assert "phase output" in log_lines
    assert "ünïcode 漢字" in log_lines
    builds = [
        e
        for e in events
        if isinstance(e, ActivityStarted) and e.activity.type == ActivityType.BUILD
    ]
    # Remote builds emit two actBuild activities for the same drv (an
    # outer "building X on 'machine'" plus a nested one), so consumers
    # must key on the drv path in fields[0], not count activities.
    assert builds
    drvs = {b.activity.fields[0] for b in builds}
    assert len(drvs) == 1
    assert next(iter(drvs)).endswith("nfb-sample.drv")
    assert "nfb-sample.drv" in builds[0].activity.text


def test_nix_fail_fixture() -> None:
    events = parse_fixture("nix-fail.jsonl")
    errors = [e for e in events if isinstance(e, Message) and e.level == 0]
    assert errors, "expected at least one level-0 error message"
    assert any("builder failed with exit code 1" in e.msg for e in errors)
    # raw_msg present on the structured error
    assert any(e.raw_msg for e in errors)
    log_lines = [e.line for e in events if isinstance(e, BuildLogLine)]
    assert "starting" in log_lines
    assert "some error detail" in log_lines


def test_lix_fail_fixture() -> None:
    """Lix output parses identically (plus passthrough of plain lines)."""
    events = parse_fixture("lix-fail.jsonl")
    errors = [e for e in events if isinstance(e, Message) and e.level == 0]
    assert any("builder failed with exit code 1" in e.msg for e in errors)
    plain = [e for e in events if isinstance(e, PlainLine)]
    assert any("paths will be fetched" in e.line for e in plain)


def test_activity_lifecycle() -> None:
    parser = LogParser()
    started = parser.parse_line(
        b'@nix {"action":"start","id":7,"level":3,"parent":0,'
        b'"text":"building","type":105,"fields":["/nix/store/x.drv","local",1,1]}\n'
    )
    assert isinstance(started, ActivityStarted)
    assert parser.activities[7] is started.activity

    line = parser.parse_line(
        b'@nix {"action":"result","id":7,"type":101,"fields":["out"]}'
    )
    assert isinstance(line, BuildLogLine)
    assert line.line == "out"
    assert line.activity is started.activity

    stopped = parser.parse_line(b'@nix {"action":"stop","id":7}')
    assert isinstance(stopped, ActivityStopped)
    assert stopped.activity is started.activity
    assert not parser.activities


def test_set_phase() -> None:
    # Synthetic: emitted by stdenv builds (format from nix/lix logging.hh).
    parser = LogParser()
    parser.parse_line(
        b'@nix {"action":"start","id":1,"level":3,"parent":0,"text":"","type":105}'
    )
    event = parser.parse_line(
        b'@nix {"action":"result","id":1,"type":104,"fields":["buildPhase"]}'
    )
    assert isinstance(event, PhaseChanged)
    assert event.phase == "buildPhase"


def test_progress() -> None:
    parser = LogParser()
    event = parser.parse_line(
        b'@nix {"action":"result","id":1,"type":105,"fields":[2,5,1,0]}'
    )
    assert isinstance(event, Progress)
    assert (event.done, event.expected, event.running, event.failed) == (2, 5, 1, 0)
    assert event.activity is None  # unknown activity id tolerated


def test_robustness() -> None:
    parser = LogParser()
    # Plain stderr passthrough
    assert parser.parse_line(b"plain builder output\n") == PlainLine(
        "plain builder output"
    )
    # Truncated json degrades to passthrough
    truncated = parser.parse_line(b'@nix {"action":"start","id":')
    assert isinstance(truncated, PlainLine)
    # Non-dict json
    assert isinstance(parser.parse_line(b"@nix [1,2]"), PlainLine)
    # Unknown action / result types ignored
    assert parser.parse_line(b'@nix {"action":"frobnicate"}') is None
    assert (
        parser.parse_line(b'@nix {"action":"result","id":1,"type":999,"fields":[]}')
        is None
    )
    # Missing required field ignored, not crashing
    assert parser.parse_line(b'@nix {"action":"result","type":101}') is None
    # Empty fields array ignored, not crashing (IndexError path)
    assert (
        parser.parse_line(b'@nix {"action":"result","id":1,"type":101,"fields":[]}')
        is None
    )
    assert (
        parser.parse_line(b'@nix {"action":"result","id":1,"type":104,"fields":[]}')
        is None
    )
    # Too few progress fields ignored
    assert (
        parser.parse_line(b'@nix {"action":"result","id":1,"type":105,"fields":[1]}')
        is None
    )
    # Invalid utf-8 replaced
    event = parser.parse_line(b"bad \xff utf8")
    assert isinstance(event, PlainLine)
    assert "\ufffd" in event.line
    # stop for unknown activity
    stopped = parser.parse_line(b'@nix {"action":"stop","id":99}')
    assert isinstance(stopped, ActivityStopped)
    assert stopped.activity is None
    # null fields normalized to empty list, not stored as None
    started = parser.parse_line(
        b'@nix {"action":"start","id":5,"type":105,"fields":null}'
    )
    assert isinstance(started, ActivityStarted)
    assert started.activity.fields == []
    assert (
        parser.parse_line(b'@nix {"action":"result","id":5,"type":101,"fields":null}')
        is None
    )
    # non-list fields rejected instead of misparsed ("abc"[0] == "a")
    assert (
        parser.parse_line(b'@nix {"action":"result","id":5,"type":101,"fields":"abc"}')
        is None
    )
    # stop without id ignored, not crashing
    assert parser.parse_line(b'@nix {"action":"stop"}') is None

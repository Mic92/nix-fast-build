import io

from nix_fast_build.ci_renderer import CIRenderer
from nix_fast_build.log_format import (
    BuildLogLine,
    Message,
    PhaseChanged,
    PlainLine,
)
from nix_fast_build.renderer import BuildOutput, fmt_duration

DRV = "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-x.drv"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_renderer(**kwargs: object) -> tuple[CIRenderer, io.StringIO, FakeClock]:
    out = io.StringIO()
    clock = FakeClock()
    defaults: dict = {"color": False, "fold": False, "clock": clock}
    defaults.update(kwargs)
    renderer = CIRenderer(out, **defaults)
    return renderer, out, clock


def feed(build: BuildOutput, *lines: str) -> None:
    for line in lines:
        build.on_event(BuildLogLine(line=line, activity=None))


def test_fmt_duration() -> None:
    assert fmt_duration(5) == "5s"
    assert fmt_duration(65) == "1m05s"
    assert fmt_duration(3700) == "1h01m40s"


def test_success_block_contiguous() -> None:
    renderer, out, clock = make_renderer()
    a = renderer.start_build("pkg-a", DRV)
    b = renderer.start_build("pkg-b", DRV)
    feed(a, "a1")
    feed(b, "b1")
    feed(a, "a2")
    clock.now += 65
    renderer.finish_build(a, 0)
    renderer.finish_build(b, 0)
    text = out.getvalue()
    # Logs of concurrent builds don't interleave.
    assert "pkg-a> a1\npkg-a> a2\n" in text
    assert "✔  pkg-a (1m05s)" in text
    assert not renderer.running


def test_success_folded_failure_not() -> None:
    renderer, out, _clock = make_renderer(fold=True)
    good = renderer.start_build("good", DRV)
    feed(good, "fine")
    renderer.finish_build(good, 0)
    bad = renderer.start_build("bad", DRV)
    feed(bad, "boom")
    renderer.finish_build(bad, 1)
    text = out.getvalue()
    assert "::group::✔  good (0s)\ngood> fine\n::endgroup::\n" in text
    # Failure block has no fold markers around it.
    fail_block = text.split("✘  bad failed")[1]
    assert "::group::" not in fail_block
    assert "✘ end of log for bad · nix log " + DRV in text


def test_failure_buffer_cap_noted() -> None:
    renderer, out, _clock = make_renderer(buffer_lines=3)
    b = renderer.start_build("big", DRV)
    feed(b, "l1", "l2", "l3", "l4", "l5")
    renderer.finish_build(b, 1)
    text = out.getvalue()
    assert "big> l1" not in text  # rotated out
    assert "big> l5" in text
    assert "oldest lines dropped" in text


def test_event_handling() -> None:
    renderer, out, _clock = make_renderer()
    b = renderer.start_build("pkg", DRV)
    b.on_event(PhaseChanged(phase="buildPhase", activity=None))
    assert b.phase == "buildPhase"
    b.on_event(PlainLine(line="plain"))
    b.on_event(Message(level=0, msg="error: it broke\nsecond line", raw_msg=None))
    b.on_event(Message(level=5, msg="debug chatter", raw_msg=None))
    renderer.finish_build(b, 1)
    text = out.getvalue()
    assert "pkg> @ phase buildPhase" in text
    assert "pkg> plain" in text
    assert "pkg> error: it broke" in text
    assert "pkg> second line" in text
    assert "debug chatter" not in text


def test_sanitize_at_ingestion() -> None:
    renderer, out, _clock = make_renderer(fold=True)
    b = renderer.start_build("evil", DRV)
    feed(b, "::endgroup::", "10%\r100%", "a\x1b[2Jb")
    renderer.finish_build(b, 0)
    text = out.getvalue()
    # Prefix neutralizes CI command injection.
    assert "evil> ::endgroup::" in text
    assert "\n::endgroup::\nevil>" not in text
    assert "evil> 100%" in text
    assert "evil> ab" in text


def test_heartbeat_lists_running_longest_first() -> None:
    renderer, out, clock = make_renderer()
    a = renderer.start_build("old", DRV)
    clock.now += 100
    renderer.start_build("new", DRV)
    a.on_event(PhaseChanged(phase="buildPhase", activity=None))
    clock.now += 20
    renderer.heartbeat()
    line = out.getvalue().splitlines()[-1]
    assert "2 building" in line
    assert line.index("old (2m00s, buildPhase)") < line.index("new (20s)")


def test_heartbeat_silent_when_idle() -> None:
    renderer, out, _clock = make_renderer()
    renderer.heartbeat()
    assert out.getvalue() == ""


def test_stall_escalates_to_streaming() -> None:
    renderer, out, clock = make_renderer(stall_timeout=300)
    b = renderer.start_build("slow", DRV)
    feed(b, "t1", "t2", "t3", "t4", "t5", "t6")
    clock.now += 301
    renderer.heartbeat()
    text = out.getvalue()
    assert "⚠ slow: no output for 5m01s" in text
    assert "slow> t2" in text  # last 5 lines tail
    assert "slow> t1" not in text
    assert "streaming further output live" in text
    assert b.streaming
    # Subsequent lines appear immediately.
    feed(b, "after-stall")
    assert "slow> after-stall" in out.getvalue()
    # Stall reported only once.
    renderer.heartbeat()
    assert out.getvalue().count("no output for") == 1
    # Final block not re-emitted; only verdict.
    before = out.getvalue()
    renderer.finish_build(b, 1)
    delta = out.getvalue()[len(before) :]
    assert "slow> t6" not in delta
    assert "✘  slow failed" in delta


def test_color_applied() -> None:
    renderer, out, _clock = make_renderer(color=True)
    b = renderer.start_build("pkg", DRV)
    renderer.finish_build(b, 0)
    assert "\x1b[32m✔  pkg" in out.getvalue()

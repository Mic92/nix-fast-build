"""Regression test: --retries must re-run the build, not re-read the old exit code."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

from nix_fast_build import Options
from nix_fast_build.build import Build, BuildResult


def run_build(tmp_path: Path, retries: int) -> tuple[BuildResult, int]:
    """Build with a nix stand-in that fails the first attempt, succeeds after.

    Returns the build result and the number of nix build invocations.
    """
    attempts = tmp_path / "attempts"
    fake_nix = tmp_path / "nix"
    fake_nix.write_text(
        f"""#!/usr/bin/env bash
case " $* " in
  *" build "*) echo x >> {attempts}
               [ "$(wc -l < {attempts})" -ge 2 ] || exit 1 ;;
  *" log "*) echo "fake build log" ;;
esac
"""
    )
    fake_nix.chmod(0o755)

    async def go() -> BuildResult:
        build = Build("hello", "/nix/store/fake.drv", {"out": "/nix/store/fake"})
        async with AsyncExitStack() as stack:
            return await build.build(
                stack,
                Options(nix_bin=[str(fake_nix)], retries=retries, no_link=True),
            )
        # Unreachable; mypy can't tell AsyncExitStack never suppresses here.
        raise AssertionError

    result = asyncio.run(go())
    return result, len(attempts.read_text().splitlines())


def test_retry_reruns_build(tmp_path: Path) -> None:
    result, attempts = run_build(tmp_path, retries=1)
    assert attempts == 2, "retry must spawn a second nix build"
    assert result.return_code == 0


def test_no_retry_single_attempt(tmp_path: Path) -> None:
    result, attempts = run_build(tmp_path, retries=0)
    assert attempts == 1
    assert result.return_code != 0
    assert "fake build log" in result.log_output

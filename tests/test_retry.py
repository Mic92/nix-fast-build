"""Regression test: --retries must re-run the build, not re-read the old exit code."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from nix_fast_build import parse_args
from nix_fast_build.build import Build, BuildResult


def run_build(
    tmp_path: Path, retries: int | None, success_after: int = 2
) -> tuple[BuildResult, int]:
    """Build with a nix stand-in that fails the first attempt, succeeds after.

    Returns the build result and the number of nix build invocations.
    """
    attempts = tmp_path / "attempts"
    fake_nix = tmp_path / "nix"
    fake_nix.write_text(
        f"""#!/usr/bin/env bash
case " $* " in
  *" config show "*) echo '{{}}' ;;
  *" build "*) echo x >> {attempts}
               [ "$(wc -l < {attempts})" -ge {success_after} ] || exit 1 ;;
  *" log "*) echo "fake build log" ;;
esac
"""
    )
    fake_nix.chmod(0o755)

    async def go() -> BuildResult:
        args = ["--nix", str(fake_nix)]
        if retries is not None:
            args.extend(["--retries", str(retries)])
        opts = await parse_args(args)
        opts.build_gcroot_dir = tmp_path
        build = Build("hello", "/nix/store/fake.drv", {"out": "/nix/store/fake"})
        async with AsyncExitStack() as stack:
            return await build.build(stack, opts)
        # Unreachable; mypy can't tell AsyncExitStack never suppresses here.
        raise AssertionError

    result = asyncio.run(go())
    return result, len(attempts.read_text().splitlines())


@pytest.mark.parametrize("retries", [1, 2])
def test_retry_reruns_build(tmp_path: Path, retries: int) -> None:
    result, attempts = run_build(tmp_path, retries, success_after=retries + 1)
    assert attempts == retries + 1, "each retry must spawn another nix build"
    assert result.return_code == 0


@pytest.mark.parametrize("retries", [None, 0])
def test_no_retry_single_attempt(tmp_path: Path, retries: int | None) -> None:
    result, attempts = run_build(tmp_path, retries)
    assert attempts == 1
    assert result.return_code != 0
    assert "fake build log" in result.log_output


def test_retry_exhausted(tmp_path: Path) -> None:
    result, attempts = run_build(tmp_path, retries=2, success_after=4)
    assert attempts == 3
    assert result.return_code != 0
    assert "fake build log" in result.log_output


def test_negative_retries_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        asyncio.run(parse_args(["--retries", "-1"]))
    assert exc.value.code == 2

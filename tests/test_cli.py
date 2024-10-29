import asyncio
import json
import os
import pwd
import xml.etree
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from nix_fast_build import async_main

from .sshd import Sshd

TEST_ROOT = Path(__file__).parent.resolve()
FIXTURES = TEST_ROOT / "fixtures"


def cli(args: list[str]) -> int:
    return asyncio.run(async_main(args))


def test_help() -> None:
    with pytest.raises(SystemExit) as e:
        cli(["--help"])
    assert e.value.code == 0


def test_build() -> None:
    rc = cli(["--option", "builders", ""])
    assert rc == 0


def test_build_junit() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "test.xml"
        rc = cli(
            [
                "--option",
                "builders",
                "",
                "--result-format",
                "junit",
                "--result-file",
                str(path),
            ]
        )
        data = xml.etree.ElementTree.parse(path)  # noqa: S314
        assert data.getroot().tag == "testsuites"
        assert rc == 0


def test_build_json() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "test.json"
        rc = cli(["--option", "builders", "", "--result-file", str(path)])
        data = json.loads(path.read_text())
        assert len(data["results"]) > 0
        build_results = [r for r in data["results"] if r["type"] == "BUILD"]
        assert build_results
        for result in build_results:
            assert "outputs" in result
            assert result["outputs"]
        assert rc == 0


def test_reference_lock_file() -> None:
    """--reference-lock-file is forwarded to nix-eval-jobs."""
    # Pointing at the project's own lock file must behave like a normal build.
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--reference-lock-file",
            str(TEST_ROOT.parent / "flake.lock"),
        ]
    )
    assert rc == 0


def test_eval_error() -> None:
    rc = cli(["--option", "builders", "", "--flake", ".#legacyPackages"])
    assert rc == 1


def test_select_flake() -> None:
    """--select can filter out failing attributes so the build succeeds."""
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--flake",
            ".#legacyPackages",
            "--select",
            # legacyPackages.<system> only contains intentionally broken
            # packages; filtering them all out must yield a successful run.
            "systems: builtins.mapAttrs (_: _: { }) systems",
        ]
    )
    assert rc == 0


def test_select_expr() -> None:
    """--select is forwarded to nix-eval-jobs in non-flake mode."""
    rc = cli(
        [
            "--file",
            str(FIXTURES / "simple.nix"),
            "--select",
            "root: { inherit (root) hello; }",
            "--option",
            "builders",
            "",
        ]
    )
    assert rc == 0


def test_expr_build() -> None:
    """Non-flake mode: evaluate and build a simple Nix expression."""
    rc = cli(
        [
            "--file",
            str(FIXTURES / "simple.nix"),
            "--option",
            "builders",
            "",
        ]
    )
    assert rc == 0


def test_expr_build_with_attr() -> None:
    """Non-flake mode: evaluate a specific attribute with -A."""
    rc = cli(
        [
            "--file",
            str(FIXTURES / "simple.nix"),
            "-A",
            "hello",
            "--option",
            "builders",
            "",
        ]
    )
    assert rc == 0


def test_expr_remote_rejected() -> None:
    """--remote is not supported in non-flake mode."""
    with pytest.raises(SystemExit) as e:
        cli(
            [
                "--file",
                str(FIXTURES / "simple.nix"),
                "--remote",
                "somehost",
            ]
        )
    assert e.value.code == 2


def test_flake_mode_rejects_expr_flags() -> None:
    """Expr-specific flags like -A should be rejected in flake mode."""
    with pytest.raises(SystemExit) as e:
        cli(["-A", "hello"])
    assert e.value.code == 2


def test_flake_and_expr_mutually_exclusive() -> None:
    """--flake and --file cannot be used together."""
    with pytest.raises(SystemExit) as e:
        cli(["--flake", ".#checks", "--file", "default.nix"])
    assert e.value.code == 2


def test_remote(sshd: Sshd) -> None:
    login = pwd.getpwuid(os.getuid()).pw_name
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--remote",
            f"{login}@127.0.0.1",
            "--remote-ssh-option",
            "Port",
            str(sshd.port),
            "--remote-ssh-option",
            "IdentityFile",
            sshd.key,
            "--remote-ssh-option",
            "StrictHostKeyChecking",
            "no",
            "--remote-ssh-option",
            "UserKnownHostsFile",
            "/dev/null",
        ]
    )
    assert rc == 0

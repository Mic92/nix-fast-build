import asyncio
import json
import os
import pwd
import xml.etree
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from nix_fast_build import async_main

from .sshd import Sshd


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
        assert rc == 0


def test_eval_error() -> None:
    rc = cli(["--option", "builders", "", "--flake", ".#legacyPackages"])
    assert rc == 1


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


def test_github_summary_with_env() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "summary.md"
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(path)}):
            rc = cli(["--option", "builders", "", "--github-summary"])
            assert rc == 0
            assert path.exists()
            content = path.read_text()
            assert "| Target | Result |" in content
            assert "| --- | --- |" in content


def test_github_summary_without_env() -> None:
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
    with patch.dict(os.environ, env, clear=True):
        rc = cli(["--option", "builders", "", "--github-summary"])
        assert rc == 1

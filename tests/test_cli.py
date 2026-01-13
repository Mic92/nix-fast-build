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

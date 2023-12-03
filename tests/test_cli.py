import asyncio
import os
import pwd

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

import asyncio
import os
import pwd

from sshd import Sshd

from nix_fast_build import async_main


def cli(args: list[str]) -> None:
    asyncio.run(async_main(args))


def test_help() -> None:
    try:
        cli(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_build() -> None:
    try:
        cli(["--option", "builders", ""])
    except SystemExit as e:
        assert e.code == 0


def test_eval_error() -> None:
    try:
        cli(["--option", "builders", "", "--flake", ".#legacyPackages"])
    except SystemExit as e:
        assert e.code == 1


def test_remote(sshd: Sshd) -> None:
    login = pwd.getpwuid(os.getuid()).pw_name
    try:
        cli(
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
    except SystemExit as e:
        assert e.code == 0

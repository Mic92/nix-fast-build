import contextlib
import os
import signal
import subprocess
from collections.abc import Iterator
from typing import IO, Any

import pytest

_FILE = None | int | IO[Any]


class Command:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen[str]] = []

    def run(
        self,
        command: list[str],
        extra_env: dict[str, str] | None = None,
        stdin: _FILE = None,
        stdout: _FILE = None,
        stderr: _FILE = None,
    ) -> subprocess.Popen[str]:
        if extra_env is None:
            extra_env = {}
        env = os.environ.copy()
        env.update(extra_env)
        # We start a new session here so that we can than more reliably kill all childs as well
        p = subprocess.Popen(
            command,
            env=env,
            start_new_session=True,
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
            text=True,
        )
        self.processes.append(p)
        return p

    def terminate(self) -> None:
        # Stop in reverse order in case there are dependencies.
        # We just kill all processes as quickly as possible because we don't
        # care about corrupted state and want to make tests fasts.
        for p in reversed(self.processes):
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)


@pytest.fixture
def command() -> Iterator[Command]:
    """
    Starts a background command. The process is automatically terminated in the end.
    >>> p = command.run(["some", "daemon"])
    >>> print(p.pid)
    """
    c = Command()
    try:
        yield c
    finally:
        c.terminate()

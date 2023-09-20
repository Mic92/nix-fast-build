import os
import signal
import subprocess
from typing import IO, Any, Dict, Iterator, List, Union

import pytest

_FILE = Union[None, int, IO[Any]]


class Command:
    def __init__(self) -> None:
        self.processes: List[subprocess.Popen[str]] = []

    def run(
        self,
        command: List[str],
        extra_env: Dict[str, str] = {},
        stdin: _FILE = None,
        stdout: _FILE = None,
        stderr: _FILE = None,
    ) -> subprocess.Popen[str]:
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
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                pass


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

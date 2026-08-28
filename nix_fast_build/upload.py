import asyncio
import logging
import os
import shlex
import sys
import timeit
from asyncio import Queue
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .build import QueueWithContext, StopTask
from .options import Options, maybe_remote, nix_shell
from .processes import ensure_stop
from .results import Result, ResultType

logger = logging.getLogger(__name__)


@dataclass
class UploadItem:
    attr: str
    # output paths or .drv paths (resolved to outputs by the worker)
    paths: list[str]


UploadQueue = QueueWithContext[UploadItem | StopTask]


async def _run_with_stdin(
    cmd: list[str],
    lines: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> tuple[int, str]:
    logger.debug("run %s (%d paths on stdin)", shlex.join(cmd), len(lines))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE if capture else sys.stderr.fileno(),
    )
    async with ensure_stop(proc, cmd):
        stdout, _ = await proc.communicate(("\n".join(lines) + "\n").encode())
    assert proc.returncode is not None
    return proc.returncode, (stdout or b"").decode()


@dataclass
class Uploader:
    name: str
    result_type: ResultType
    opts: Options
    pushed: set[str] = field(default_factory=set)

    def command(self) -> tuple[list[str], dict[str, str] | None]:
        """Command reading newline-separated store paths on stdin (ARG_MAX)."""
        raise NotImplementedError

    async def send(self, paths: list[str]) -> int:
        cmd, env = self.command()
        rc, _ = await _run_with_stdin(maybe_remote(cmd, self.opts), paths, env=env)
        return rc

    async def push(self, raw: set[str]) -> int:
        paths = await resolve_valid_outputs(raw, self.opts)
        if not paths:
            return 0
        return await self.send(sorted(paths))


@dataclass
class NixCopyUploader(Uploader):
    def command(self) -> tuple[list[str], dict[str, str] | None]:
        assert self.opts.copy_to is not None
        return [
            "xargs",
            *self.opts.nix_command(
                ["copy", "--log-format", "raw", "--to", self.opts.copy_to]
            ),
        ], None


@dataclass
class CachixUploader(Uploader):
    socket_path: Path = Path()

    def command(self) -> tuple[list[str], dict[str, str] | None]:
        return [
            "xargs",
            *nix_shell("nixpkgs#cachix", "cachix"),
            "daemon",
            "push",
            "--socket",
            str(self.socket_path),
        ], None


@dataclass
class AtticUploader(Uploader):
    def command(self) -> tuple[list[str], dict[str, str] | None]:
        assert self.opts.attic_cache is not None
        args = [*nix_shell("nixpkgs#attic-client", "attic"), "push", "--stdin"]
        if self.opts.attic_ignore_upstream_cache_filter:
            args.append("--ignore-upstream-cache-filter")
        args.append(self.opts.attic_cache)
        return args, None


@dataclass
class Niks3Uploader(Uploader):
    def command(self) -> tuple[list[str], dict[str, str] | None]:
        assert self.opts.niks3_server is not None
        env = os.environ.copy()
        env["NIKS3_SERVER_URL"] = self.opts.niks3_server
        return ["xargs", *nix_shell("github:Mic92/niks3", "niks3"), "push"], env


async def resolve_valid_outputs(paths: set[str], opts: Options) -> set[str]:
    """Resolve .drv paths to outputs and drop invalid ones (failed builds)."""
    drvs = sorted(p for p in paths if p.endswith(".drv"))
    outs = {p for p in paths if not p.endswith(".drv")}
    if drvs:
        rc, stdout = await _run_with_stdin(
            maybe_remote(["xargs", "nix-store", "--query", "--outputs"], opts),
            drvs,
            capture=True,
        )
        if rc != 0:
            logger.warning("nix-store --query --outputs failed (rc=%d)", rc)
        outs.update(stdout.split())
    if not outs:
        return outs
    rc, stdout = await _run_with_stdin(
        maybe_remote(
            ["xargs", "nix-store", "--check-validity", "--print-invalid"], opts
        ),
        sorted(outs),
        capture=True,
    )
    if rc != 0:
        logger.warning("nix-store --check-validity failed (rc=%d)", rc)
        return outs
    return outs - set(stdout.split())


def _drain(queue: UploadQueue, first: UploadItem) -> list[UploadItem]:
    items = [first]
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return items
        queue.task_done()
        if isinstance(item, StopTask):
            queue.put_nowait(item)
            return items
        items.append(item)


async def run_upload_worker(
    queue: UploadQueue,
    result_queue: "Queue[Result | None]",
    uploader: Uploader,
) -> int:
    while True:
        async with queue.get_context() as first:
            if isinstance(first, StopTask):
                logger.debug("finish %s task", uploader.name)
                return 0
            items = _drain(queue, first)

            by_attr: dict[str, set[str]] = {}
            for item in items:
                new = set(item.paths) - uploader.pushed
                if new:
                    by_attr.setdefault(item.attr, set()).update(new)
            if not by_attr:
                continue

            start = timeit.default_timer()
            batch = set().union(*by_attr.values())
            rc = await uploader.push(batch)
            rcs = dict.fromkeys(by_attr, rc)
            if rc != 0 and len(by_attr) > 1:
                logger.warning(
                    "%s: batch of %d paths failed (rc=%d), retrying per attribute",
                    uploader.name,
                    len(batch),
                    rc,
                )
                for attr, paths in by_attr.items():
                    rcs[attr] = await uploader.push(paths)
            duration = (timeit.default_timer() - start) / len(by_attr)
            for attr, attr_rc in rcs.items():
                if attr_rc == 0:
                    uploader.pushed.update(by_attr[attr])
                await result_queue.put(
                    Result(
                        result_type=uploader.result_type,
                        attr=attr,
                        success=attr_rc == 0,
                        duration=duration,
                        error=f"{uploader.name} exited with {attr_rc}"
                        if attr_rc != 0
                        else None,
                    )
                )

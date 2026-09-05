import asyncio
import json
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
    # toplevel outputs always yield a Result, intermediates only on failure
    final: bool = True


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


# Conservative: with --remote the whole command line becomes a single argv
# entry for the remote shell, which Linux caps at MAX_ARG_STRLEN (128 KiB).
MAX_ARGS_BYTES = 96 * 1024


def _chunk_args(args: Sequence[str], limit: int = MAX_ARGS_BYTES) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for arg in args:
        n = len(arg.encode()) + 1
        if current and size + n > limit:
            chunks.append(current)
            current, size = [], 0
        current.append(arg)
        size += n
    if current:
        chunks.append(current)
    return chunks


async def _run_with_args(
    cmd: list[str],
    args: Sequence[str],
    opts: Options,
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> tuple[int, list[str]]:
    """Run cmd with args appended, chunked below ARG_MAX; stdout per chunk."""
    rc = 0
    out: list[str] = []
    for chunk in _chunk_args(args):
        full = maybe_remote([*cmd, *chunk], opts)
        logger.debug("run %s (+%d paths)", shlex.join(cmd), len(chunk))
        proc = await asyncio.create_subprocess_exec(
            *full,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture else sys.stderr.fileno(),
        )
        async with ensure_stop(proc, full):
            stdout, _ = await proc.communicate()
        assert proc.returncode is not None
        if proc.returncode != 0:
            rc = proc.returncode
        out.append((stdout or b"").decode())
    return rc, out


@dataclass
class Uploader:
    name: str
    result_type: ResultType
    opts: Options
    pushed: set[str] = field(default_factory=set)
    # whether command() reads newline-separated paths on stdin instead of argv
    reads_stdin = False

    def command(self) -> tuple[list[str], dict[str, str] | None]:
        raise NotImplementedError

    async def send(self, paths: list[str]) -> int:
        cmd, env = self.command()
        if self.reads_stdin:
            rc, _ = await _run_with_stdin(maybe_remote(cmd, self.opts), paths, env=env)
        else:
            rc, _ = await _run_with_args(cmd, paths, self.opts, env=env)
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
        return self.opts.nix_command(
            ["copy", "--log-format", "raw", "--to", self.opts.copy_to]
        ), None


@dataclass
class CachixUploader(Uploader):
    socket_path: Path = Path()

    def command(self) -> tuple[list[str], dict[str, str] | None]:
        return [
            *nix_shell("nixpkgs#cachix", "cachix"),
            "daemon",
            "push",
            "--socket",
            str(self.socket_path),
        ], None


@dataclass
class AtticUploader(Uploader):
    reads_stdin = True

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
        return [*nix_shell("github:Mic92/niks3", "niks3"), "push"], env


def parse_path_info(stdout: str) -> set[str]:
    """Valid paths from `nix path-info --json`: Nix >=2.19 emits an object
    with null for invalid paths, older Nix and Lix a list with `valid`."""
    data = json.loads(stdout)
    if isinstance(data, dict):
        return {p for p, info in data.items() if info is not None}
    return {e["path"] for e in data if e.get("valid", True)}


async def resolve_valid_outputs(paths: set[str], opts: Options) -> set[str]:
    """Resolve .drv paths to outputs and drop invalid ones (failed builds)."""
    args = sorted(f"{p}^*" if p.endswith(".drv") else p for p in paths)
    cmd = opts.nix_command(
        ["path-info", "--option", "substitute", "false", "--json", *opts.store_args]
    )
    rc, outputs = await _run_with_args(cmd, args, opts, capture=True)
    if rc != 0:
        logger.warning("nix path-info failed (rc=%d)", rc)
    return set().union(*(parse_path_info(o) for o in outputs if o))


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
            final: set[str] = set()
            for item in items:
                by_attr.setdefault(item.attr, set()).update(
                    set(item.paths) - uploader.pushed
                )
                if item.final:
                    final.add(item.attr)

            start = timeit.default_timer()
            batch = set().union(*by_attr.values())
            rc = await uploader.push(batch) if batch else 0
            rcs = dict.fromkeys(by_attr, rc)
            if rc != 0 and len(by_attr) > 1:
                logger.warning(
                    "%s: batch of %d paths failed (rc=%d), retrying per attribute",
                    uploader.name,
                    len(batch),
                    rc,
                )
                for attr, paths in by_attr.items():
                    rcs[attr] = await uploader.push(paths) if paths else 0
            duration = (timeit.default_timer() - start) / len(by_attr)
            for attr, attr_rc in rcs.items():
                if attr_rc == 0:
                    uploader.pushed.update(by_attr[attr])
                if attr_rc == 0 and attr not in final:
                    continue
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

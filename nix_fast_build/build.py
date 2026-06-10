import asyncio
import contextlib
import logging
import os
import shlex
from asyncio import Queue
from asyncio.subprocess import Process
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, TypeVar

from .options import Options, maybe_remote, nix_shell
from .processes import ensure_stop

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of a build operation."""

    return_code: int
    log_output: str


@dataclass
class Build:
    attr: str
    drv_path: str
    outputs: dict[str, str]

    async def build(
        self,
        stack: AsyncExitStack,
        build_output: IO[bytes],
        opts: Options,
        nom_pipe: IO[bytes] | None = None,
    ) -> BuildResult:
        """Build and return BuildResult."""
        rc = 0
        for attempt in range(opts.retries + 1):
            proc = await stack.enter_async_context(
                nix_build(
                    self.attr, self.drv_path, build_output, opts, nom_pipe=nom_pipe
                )
            )
            rc = await proc.wait()
            if rc == 0:
                logger.debug(f"build {self.attr} succeeded")
                return BuildResult(return_code=rc, log_output="")
            logger.warning(
                f"build {self.attr} exited with {rc} "
                f"(attempt {attempt + 1}/{opts.retries + 1})"
            )

        # If build failed, get the log using nix log
        log_output = await self.get_build_log(opts)
        return BuildResult(return_code=rc, log_output=log_output)

    async def get_build_log(self, opts: Options) -> str:
        """Get build log using nix log command."""
        cmd = maybe_remote(
            opts.nix_command(["log", self.drv_path, *opts.store_args]), opts
        )
        logger.debug("run %s", shlex.join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0 and stdout:
                return stdout.decode("utf-8", errors="replace")
            # If nix log fails, return stderr or empty
            if stderr:
                return stderr.decode("utf-8", errors="replace")
        except OSError as e:
            logger.debug(f"Failed to get build log: {e}")
        return ""

    async def upload(self, exit_stack: AsyncExitStack, opts: Options) -> int:
        if not opts.copy_to or not self.outputs:
            return 0
        cmd = opts.nix_command(
            [
                "copy",
                "--log-format",
                "raw",
                "--to",
                opts.copy_to,
                *list(self.outputs.values()),
            ]
        )
        cmd = maybe_remote(cmd, opts)
        logger.debug("run %s", shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()

    async def upload_cachix(self, cachix_socket_path: Path, opts: Options) -> int:
        if not self.outputs:
            return 0
        cmd = maybe_remote(
            [
                *nix_shell("nixpkgs#cachix", "cachix"),
                "daemon",
                "push",
                "--socket",
                str(cachix_socket_path),
                *list(self.outputs.values()),
            ],
            opts,
        )
        logger.debug("run %s", shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        return await proc.wait()

    async def _query_build_closure(self, opts: Options) -> list[str]:
        """Query all realised store paths in the build closure of this derivation.

        Returns output paths of all build-time requisites that exist in
        the store.  After a successful build these are guaranteed to be
        present because nix had to build or substitute every dependency.
        """
        query_cmd = maybe_remote(
            [
                "nix-store",
                "--query",
                "--requisites",
                "--include-outputs",
                self.drv_path,
            ],
            opts,
        )
        proc = await asyncio.create_subprocess_exec(
            *query_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "nix-store -qR --include-outputs failed for %s (rc=%d), "
                "falling back to output paths only",
                self.drv_path,
                proc.returncode,
            )
            return list(self.outputs.values())

        paths: list[str] = []
        for line in stdout.decode().splitlines():
            path = line.strip()
            if path and not path.endswith(".drv"):
                paths.append(path)
        return paths or list(self.outputs.values())

    async def upload_attic(self, opts: Options) -> int:
        if opts.attic_cache is None or not self.outputs:
            return 0
        push_args = ["push"]
        if opts.attic_ignore_upstream_cache_filter:
            push_args.append("--ignore-upstream-cache-filter")
        push_args.append(opts.attic_cache)
        if opts.attic_push_build_closure:
            paths = await self._query_build_closure(opts)
            push_args.append("--no-closure")
            push_args.extend(paths)
            logger.debug(
                "attic push: %d paths (build closure) for %s",
                len(paths),
                self.attr,
            )
        else:
            push_args.extend(self.outputs.values())
        cmd = maybe_remote(
            [
                *nix_shell("nixpkgs#attic-client", "attic"),
                *push_args,
            ],
            opts,
        )
        logger.debug("run %s", shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        return await proc.wait()

    async def download(self, exit_stack: AsyncExitStack, opts: Options) -> int:
        if not opts.remote_url or not opts.download or not self.outputs:
            return 0
        cmd = opts.nix_command(
            [
                "copy",
                "--log-format",
                "raw",
                "--no-check-sigs",
                "--from",
                opts.remote_url,
                *list(self.outputs.values()),
            ]
        )
        logger.debug("run %s", shlex.join(cmd))
        env = os.environ.copy()
        env["NIX_SSHOPTS"] = " ".join(opts.remote_ssh_options)
        proc = await asyncio.create_subprocess_exec(*cmd, env=env)
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()


T = TypeVar("T")


@dataclass
class OptionalQueue:
    """Post-build queue with the workers that drain it, for proper shutdown."""

    queue: "BuildQueue"
    worker_count: int
    name: str
    make_worker: Callable[[], Coroutine[Any, Any, int]]


class QueueWithContext(Queue[T]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.running_tasks: int = 0

    @asynccontextmanager
    async def get_context(self) -> AsyncIterator[T]:
        el = await super().get()
        try:
            self.running_tasks += 1
            yield el
        finally:
            self.running_tasks -= 1
            self.task_done()


@asynccontextmanager
async def nix_build(
    attr: str,
    installable: str,
    stderr: IO[Any] | None,
    opts: Options,
    nom_pipe: IO[bytes] | None = None,
) -> AsyncIterator[Process]:
    args = opts.nix_command(
        ["build", f"{installable}^*", "--keep-going", *opts.options, *opts.store_args]
    )
    if nom_pipe is not None:
        args += ["--log-format", "internal-json", "-v"]
    else:
        # Without nom, stream build logs directly to stderr so the user
        # can see what builders are doing (especially useful in CI).
        args += ["-L"]
    if opts.no_link:
        args += ["--no-link"]
    else:
        args += [
            "--out-link",
            opts.out_link + "-" + attr,
        ]

    args = maybe_remote(args, opts)
    logger.debug("run %s", shlex.join(args))

    if nom_pipe is not None:
        # Capture stderr per-process so we can forward complete lines to nom,
        # avoiding interleaved writes from concurrent builds on the shared pipe.
        proc = await asyncio.create_subprocess_exec(
            *args, stderr=asyncio.subprocess.PIPE
        )

        async def _forward_lines() -> None:
            assert proc.stderr is not None
            async for line in proc.stderr:
                nom_pipe.write(line)
                nom_pipe.flush()

        fwd_task = asyncio.create_task(_forward_lines())
        try:
            yield proc
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await fwd_task
    else:
        proc = await asyncio.create_subprocess_exec(*args, stderr=stderr)
        try:
            yield proc
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()


@dataclass
class Job:
    attr: str
    drv_path: str
    outputs: dict[str, str]


class StopTask:
    pass


JobQueue = QueueWithContext[Job | StopTask]
BuildQueue = QueueWithContext[Build | StopTask]

import asyncio
import contextlib
import logging
import os
import shlex
import sys
from asyncio import Queue
from asyncio.subprocess import Process
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from .log_format import ActivityStopped, ActivityType, LogParser
from .options import Options, maybe_remote
from .processes import ensure_stop
from .renderer import BuildOutput, Renderer

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
        opts: Options,
        renderer: Renderer | None = None,
        on_built: Callable[[str], None] | None = None,
    ) -> BuildResult:
        """on_built receives each intermediate .drv nix ran a builder for."""
        rc = 0
        sink: BuildOutput | None = None
        for attempt in range(opts.retries + 1):
            if renderer is not None:
                sink = renderer.start_build(self.attr, self.drv_path)
            try:
                proc = await stack.enter_async_context(
                    nix_build(
                        self.attr,
                        self.drv_path,
                        opts,
                        sink=sink,
                        on_built=on_built,
                    )
                )
                rc = await proc.wait()
            except BaseException:
                # Cancellation/shutdown: drop silently, no verdict.
                if renderer is not None and sink is not None:
                    renderer.abort_build(sink)
                raise
            if renderer is not None and sink is not None:
                renderer.finish_build(sink, rc)
            if rc == 0:
                logger.debug(f"build {self.attr} succeeded")
                return BuildResult(return_code=rc, log_output="")
            logger.warning(
                f"build {self.attr} exited with {rc} "
                f"(attempt {attempt + 1}/{opts.retries + 1})"
            )

        # For the result file: prefer the log captured from the failed
        # build; fall back to nix log (e.g. all lines rotated out).
        if sink is not None and sink.lines:
            log_output = "\n".join(sink.lines)
        else:
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

    def out_link_args(self, opts: Options) -> list[str]:
        if opts.out_link is not None:
            return ["--out-link", opts.out_link + "-" + self.attr]
        if opts.download_gcroot_dir is not None:
            return [
                "--out-link",
                str(opts.download_gcroot_dir / f"result-{self.attr}"),
            ]
        return []

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
                *self.out_link_args(opts),
                *list(self.outputs.values()),
            ]
        )
        logger.debug("run %s", shlex.join(cmd))
        env = os.environ.copy()
        env["NIX_SSHOPTS"] = " ".join(opts.remote_ssh_options)
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env, stdout=sys.stderr.fileno()
        )
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()


T = TypeVar("T")


@dataclass
class OptionalQueue:
    """Post-build queue with the workers that drain it, for proper shutdown."""

    queue: "QueueWithContext[Any]"
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
    opts: Options,
    sink: BuildOutput | None = None,
    on_built: Callable[[str], None] | None = None,
) -> AsyncIterator[Process]:
    args = opts.nix_command(
        ["build", f"{installable}^*", "--keep-going", *opts.options, *opts.store_args]
    )
    args += ["--log-format", "internal-json", "-v"]
    if opts.store is not None:
        # outputs live in a remote store, a local out-link would dangle
        args += ["--no-link"]
    elif opts.out_link is not None and opts.remote is None:
        # with --remote the persistent link is created locally on download
        args += ["--out-link", opts.out_link + "-" + attr]
    else:
        assert opts.build_gcroot_dir is not None
        args += ["--out-link", str(opts.build_gcroot_dir / f"result-{attr}")]

    args = maybe_remote(args, opts)
    logger.debug("run %s", shlex.join(args))

    # Capture stderr per-process: complete lines go to the renderer's
    # per-build sink, so concurrent builds never interleave mid-line.
    proc = await asyncio.create_subprocess_exec(
        *args,
        stderr=asyncio.subprocess.PIPE,
        # 10MB buffer to accommodate for large lines
        limit=10485760,
    )

    async def _forward_lines() -> None:
        assert proc.stderr is not None
        parser = LogParser()
        try:
            async for line in proc.stderr:
                event = parser.parse_line(line)
                if event is None:
                    continue
                if (
                    on_built is not None
                    and isinstance(event, ActivityStopped)
                    and event.activity is not None
                    and event.activity.type == ActivityType.BUILD
                    and event.activity.fields
                    and str(event.activity.fields[0]) != installable
                ):
                    on_built(str(event.activity.fields[0]))
                if sink is not None:
                    sink.on_event(event)
        except ValueError:
            # Line exceeded the stream limit. Stop forwarding but don't
            # let the exception escape the cleanup that awaits this task.
            logger.warning("build %s: log line exceeded buffer limit, dropped", attr)

    fwd_task = asyncio.create_task(_forward_lines())
    try:
        yield proc
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await fwd_task


@dataclass
class Job:
    attr: str
    drv_path: str
    outputs: dict[str, str]


class StopTask:
    pass


JobQueue = QueueWithContext[Job | StopTask]
BuildQueue = QueueWithContext[Build | StopTask]

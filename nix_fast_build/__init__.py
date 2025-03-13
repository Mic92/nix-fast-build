import argparse
import asyncio
import contextlib
import enum
import json
import logging
import multiprocessing
import os
import shlex
import shutil
import signal
import subprocess
import sys
import timeit
import xml.etree.ElementTree as ET
from asyncio import Queue, TaskGroup
from asyncio.subprocess import Process
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import IO, Any, TypeVar

logger = logging.getLogger(__name__)


class Error(Exception):
    pass


class Pipe:
    def __init__(self) -> None:
        fds = os.pipe()
        self.read_file = os.fdopen(fds[0], "rb")
        self.write_file = os.fdopen(fds[1], "wb")

    def __enter__(self) -> "Pipe":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.read_file.close()
        self.write_file.close()


def nix_command(args: list[str]) -> list[str]:
    return ["nix", "--experimental-features", "nix-command flakes", *args]


class ResultFormat(enum.Enum):
    JSON = enum.auto()
    JUNIT = enum.auto()


@dataclass
class Options:
    flake_url: str = ""
    flake_fragment: str = ""
    options: list[str] = field(default_factory=list)
    remote: str | None = None
    remote_ssh_options: list[str] = field(default_factory=list)
    always_upload_source: bool = False
    systems: set[str] = field(default_factory=set)
    eval_max_memory_size: int = 4096
    skip_cached: bool = False
    eval_workers: int = field(default_factory=multiprocessing.cpu_count)
    max_jobs: int = 0
    retries: int = 0
    debug: bool = False
    copy_to: str | None = None
    nom: bool = True
    download: bool = True
    no_link: bool = False
    out_link: str = "result"
    result_format: ResultFormat = ResultFormat.JSON
    result_file: Path | None = None

    cachix_cache: str | None = None

    attic_cache: str | None = None

    @property
    def remote_url(self) -> None | str:
        if self.remote is None:
            return None
        return f"ssh://{self.remote}"


class ResultType(enum.Enum):
    EVAL = enum.auto()
    BUILD = enum.auto()
    UPLOAD = enum.auto()
    DOWNLOAD = enum.auto()
    CACHIX = enum.auto()
    ATTIC = enum.auto()


@dataclass
class Result:
    result_type: ResultType
    attr: str
    success: bool
    duration: float
    error: str | None


def _maybe_remote(
    cmd: list[str], remote: str | None, remote_ssh_options: list[str]
) -> list[str]:
    if remote:
        return ["ssh", remote, *remote_ssh_options, "--", shlex.join(cmd)]
    return cmd


def maybe_remote(cmd: list[str], opts: Options) -> list[str]:
    return _maybe_remote(cmd, opts.remote, opts.remote_ssh_options)


async def get_nix_config(
    remote: str | None, remote_ssh_options: list[str]
) -> dict[str, str]:
    args = _maybe_remote(
        nix_command(["config", "show", "--json"]), remote, remote_ssh_options
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as e:
        msg = f"nix not found in PATH, try to run {shlex.join(args)}"
        raise Error(msg) from e

    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        msg = f"Failed to get nix config, {shlex.join(args)} exited with {proc.returncode}"
        raise Error(msg)
    data = json.loads(stdout)

    config = {}
    for key, value in data.items():
        config[key] = value["value"]
    return config


async def parse_args(args: list[str]) -> Options:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--flake",
        default=".#checks",
        help="Flake url to evaluate/build (default: .#checks",
    )
    parser.add_argument(
        "-j",
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum number of build jobs to run in parallel (0 for unlimited)",
    )
    parser.add_argument(
        "--option",
        help="Nix option to set",
        action="append",
        nargs=2,
        metavar=("name", "value"),
        default=[],
    )
    parser.add_argument(
        "--remote-ssh-option",
        help="ssh option when accessing remote",
        action="append",
        nargs=2,
        metavar=("name", "value"),
        default=[],
    )
    parser.add_argument(
        "--cachix-cache",
        help="Cachix cache to upload to",
        default=None,
    )
    parser.add_argument(
        "--attic-cache",
        help="Attic cache to upload to",
        default=None,
    )
    parser.add_argument(
        "--no-nom",
        help="Don't use nix-output-monitor to print build output (default: false)",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--systems",
        help="Space-separated list of systems to build for (default: current system)",
        default=None,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of times to retry failed builds",
    )
    parser.add_argument(
        "--no-link",
        help="Do not create an out-link for builds (default: false)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--out-link",
        help="Name of the out-link for builds (default: result)",
        default="result",
    )
    parser.add_argument(
        "--remote",
        type=str,
        help="Remote machine to build on",
    )
    parser.add_argument(
        "--always-upload-source",
        help="Always upload sources to remote machine. This is needed if the remote machine cannot access all sources (default: false)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-download",
        help="Do not download build results from remote machine",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip-cached",
        help="Skip builds that are already present in the binary cache (default: false)",
        action="store_true",
    )
    parser.add_argument(
        "--copy-to",
        help="Copy build results to the given path (passed to nix copy, i.e. file:///tmp/cache?compression=none)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug logging output",
    )
    parser.add_argument(
        "--eval-max-memory-size",
        type=int,
        default=4096,
        help="Maximum memory size for nix-eval-jobs (in MiB) per worker. After the limit is reached, the worker is restarted.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of evaluation threads spawned",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="File to write build results to",
    )
    parser.add_argument(
        "--result-format",
        choices=["json", "junit"],
        default="json",
        help="Format of the build result file",
    )

    a = parser.parse_args(args)

    flake_parts = a.flake.split("#")
    flake_url = flake_parts[0]
    flake_fragment = ""
    if len(flake_parts) == 2:
        flake_fragment = flake_parts[1]

    options = []
    for name, value in a.option:
        options.extend(["--option", name, value])
    remote_ssh_options = []
    for name, value in a.remote_ssh_option:
        remote_ssh_options.extend(["-o", f"{name}={value}"])

    nix_config = await get_nix_config(a.remote, remote_ssh_options)
    if a.max_jobs is None:
        a.max_jobs = int(nix_config.get("max-jobs", 0))
    if a.no_nom is None:
        if a.remote:
            # only if we have an official binary cache, otherwise we need to build ghc...
            a.no_nom = nix_config.get("system", "") not in [
                "aarch64-darwin",
                "x86_64-darwin",
                "aarch64-linux",
                "x86_64-linux",
            ]
        else:
            a.no_nom = shutil.which("nom") is None
    if a.systems is None:
        systems = {nix_config.get("system", "")}
    else:
        systems = set(a.systems.split(" "))

    return Options(
        flake_url=flake_url,
        flake_fragment=flake_fragment,
        always_upload_source=a.always_upload_source,
        remote=a.remote,
        skip_cached=a.skip_cached,
        options=options,
        remote_ssh_options=remote_ssh_options,
        max_jobs=a.max_jobs,
        nom=not a.no_nom,
        download=not a.no_download,
        debug=a.debug,
        systems=systems,
        eval_max_memory_size=a.eval_max_memory_size,
        eval_workers=a.eval_workers,
        copy_to=a.copy_to,
        cachix_cache=a.cachix_cache,
        attic_cache=a.attic_cache,
        no_link=a.no_link,
        out_link=a.out_link,
        result_format=ResultFormat[a.result_format.upper()],
        result_file=a.result_file,
    )


def nix_flake_metadata(flake_url: str) -> dict[str, Any]:
    cmd = nix_command(
        [
            "flake",
            "metadata",
            "--json",
            flake_url,
        ]
    )
    logger.info(f"run {shlex.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    if proc.returncode != 0:
        msg = (
            f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
        )
        raise Error(msg)

    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, OSError) as e:
        msg = f"failed to parse output of {shlex.join(cmd)}: {e}\nGot: {proc.stdout.decode('utf-8', 'replace')}"
        raise Error(msg) from e
    return data


def is_path_input(node: dict[str, dict[str, str]]) -> bool:
    locked = node.get("locked")
    if not locked:
        return False
    return locked["type"] == "path" or locked.get("url", "").startswith("file://")


def check_for_path_inputs(data: dict[str, Any]) -> bool:
    return any(is_path_input(node) for node in data["locks"]["nodes"].values())


def upload_sources(opts: Options) -> str:
    if not opts.always_upload_source:
        flake_data = nix_flake_metadata(opts.flake_url)
        url = flake_data["resolvedUrl"]
        has_path_inputs = check_for_path_inputs(flake_data)
        if not has_path_inputs and not is_path_input(flake_data):
            # No need to upload sources, we can just build the flake url directly
            # FIXME: this might fail for private repositories?
            return url
        if not has_path_inputs:
            # Just copy the flake to the remote machine, we can substitute other inputs there.
            path = flake_data["path"]
            env = os.environ.copy()
            env["NIX_SSHOPTS"] = " ".join(opts.remote_ssh_options)
            assert opts.remote_url
            cmd = nix_command(
                [
                    "copy",
                    "--to",
                    opts.remote_url,
                    "--no-check-sigs",
                    path,
                ]
            )
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, env=env, check=False)
            if proc.returncode != 0:
                msg = f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
                raise Error(msg)
            return path

    # Slow path: we need to upload all sources to the remote machine
    assert opts.remote_url
    cmd = nix_command(
        [
            "flake",
            "archive",
            "--to",
            opts.remote_url,
            "--json",
            opts.flake_url,
        ]
    )
    print("run " + shlex.join(cmd))
    logger.info("run %s", shlex.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        msg = (
            f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
        )
        raise Error(msg)
    try:
        return json.loads(proc.stdout)["path"]
    except (json.JSONDecodeError, OSError) as e:
        msg = f"failed to parse output of {shlex.join(cmd)}: {e}\nGot: {proc.stdout.decode('utf-8', 'replace')}"
        raise Error(msg) from e


def nix_shell(fallback_package: str, wanted_command: str) -> list[str]:
    bash_cmd = 'pkg=$1; shift; cmd=("$@"); if command -v "${cmd[0]}" >/dev/null; then exec "${cmd[@]}"; else exec nix --experimental-features "nix-command flakes" shell "$pkg" -c "${cmd[@]}"; fi'
    return ["bash", "-c", bash_cmd, "bash", fallback_package, wanted_command]


@asynccontextmanager
async def ensure_stop(
    proc: Process,
    cmd: list[str],
    wait_timeout: float = 3.0,
    signal_no: int = signal.SIGTERM,
) -> AsyncIterator[Process]:
    try:
        yield proc
    finally:
        if proc.returncode is not None:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal_no)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=wait_timeout)
                except TimeoutError:
                    print(f"Failed to stop process {shlex.join(cmd)}. Killing it.")
                    proc.kill()
                    await proc.wait()


@asynccontextmanager
async def remote_temp_dir(opts: Options) -> AsyncIterator[Path]:
    assert opts.remote
    ssh_cmd = ["ssh", opts.remote, *opts.remote_ssh_options, "--"]
    cmd = [*ssh_cmd, "mktemp", "-d"]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    tempdir = line.decode().strip()
    rc = await proc.wait()
    if rc != 0:
        msg = f"Failed to create temporary directory on remote machine {opts.remote}: {rc}"
        raise Error(msg)
    try:
        yield Path(tempdir)
    finally:
        cmd = [*ssh_cmd, "rm", "-rf", tempdir]
        logger.info("run %s", shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()


@asynccontextmanager
async def nix_eval_jobs(tmp_dir: Path, opts: Options) -> AsyncIterator[Process]:
    args = [
        "--gc-roots-dir",
        str(tmp_dir / "gcroots"),
        "--force-recurse",
        "--max-memory-size",
        str(opts.eval_max_memory_size),
        "--workers",
        str(opts.eval_workers),
        "--flake",
        f"{opts.flake_url}#{opts.flake_fragment}",
        *opts.options,
    ]
    if opts.skip_cached:
        args.append("--check-cache-status")
    if opts.remote:
        args = nix_shell("nixpkgs#nix-eval-jobs", "nix-eval-jobs") + args
    else:
        args = ["nix-eval-jobs", *args]
    args = maybe_remote(args, opts)
    logger.info("run %s", shlex.join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        # 128 KiB buffer to accommodate for large lines
        limit=1024 * 128,
    )
    async with ensure_stop(proc, args) as proc:
        yield proc


@asynccontextmanager
async def nix_output_monitor(pipe: Pipe, opts: Options) -> AsyncIterator[Process]:
    cmd = maybe_remote([*nix_shell("nixpkgs#nix-output-monitor", "nom")], opts)
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=pipe.read_file)
    try:
        yield proc
    finally:
        # FIXME: show cursor again after nom messing it up (nom doesn't handle signals properly)
        try:
            pipe.write_file.close()
            pipe.read_file.close()
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        finally:
            print("\033[?25h")


@asynccontextmanager
async def run_cachix_daemon(
    exit_stack: AsyncExitStack, tmp_dir: Path, cachix_cache: str, opts: Options
) -> AsyncIterator[Path]:
    sock_path = tmp_dir / "cachix.sock"
    cmd = maybe_remote(
        [
            *nix_shell("nixpkgs#cachix", "cachix"),
            "daemon",
            "run",
            "--socket",
            str(sock_path),
            cachix_cache,
        ],
        opts,
    )
    proc = await asyncio.create_subprocess_exec(*cmd)
    try:
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        while True:
            if sock_path.exists():
                break
            await asyncio.sleep(0.1)
        yield sock_path
    finally:
        await run_cachix_daemon_stop(exit_stack, sock_path, opts)


async def run_cachix_daemon_stop(
    exit_stack: AsyncExitStack, sock_path: Path | None, opts: Options
) -> int:
    if sock_path is None:
        return 0
    cmd = maybe_remote(
        [
            *nix_shell("nixpkgs#cachix", "cachix"),
            "daemon",
            "stop",
            "--socket",
            str(sock_path),
        ],
        opts,
    )
    proc = await asyncio.create_subprocess_exec(*cmd)
    await exit_stack.enter_async_context(ensure_stop(proc, cmd))
    return await proc.wait()


@dataclass
class Build:
    attr: str
    drv_path: str
    outputs: dict[str, str]

    async def build(
        self, stack: AsyncExitStack, build_output: IO[str], opts: Options
    ) -> int:
        proc = await stack.enter_async_context(
            nix_build(self.attr, self.drv_path, build_output, opts)
        )
        rc = 0
        for _ in range(opts.retries + 1):
            rc = await proc.wait()
            if rc == 0:
                logger.debug(f"build {self.attr} succeeded")
                return rc
            logger.warning(f"build {self.attr} exited with {rc}")
        return rc

    async def nix_copy(
        self, args: list[str], exit_stack: AsyncExitStack, opts: Options
    ) -> int:
        cmd = maybe_remote(nix_command(["copy", "--log-format", "raw", *args]), opts)
        logger.debug("run %s", shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()

    async def upload(self, exit_stack: AsyncExitStack, opts: Options) -> int:
        if not opts.copy_to:
            return 0
        cmd = nix_command(
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

    async def upload_cachix(
        self, cachix_socket_path: Path | None, opts: Options
    ) -> int:
        if cachix_socket_path is None:
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

    async def upload_attic(self, opts: Options) -> int:
        if opts.attic_cache is None:
            return 0
        cmd = maybe_remote(
            [
                *nix_shell("nixpkgs#attic-client", "attic"),
                "push",
                opts.attic_cache,
                self.outputs["out"],
            ],
            opts,
        )
        logger.debug("run %s", shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        return await proc.wait()

    async def download(self, exit_stack: AsyncExitStack, opts: Options) -> int:
        if not opts.remote_url or not opts.download:
            return 0
        cmd = nix_command(
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
    attr: str, installable: str, stderr: IO[Any] | None, opts: Options
) -> AsyncIterator[Process]:
    args = ["nix-build", installable, "--keep-going", *opts.options]
    if opts.no_link:
        args += ["--no-link"]
    else:
        args += [
            "--out-link",
            opts.out_link + "-" + attr,
        ]

    args = maybe_remote(args, opts)
    logger.debug("run %s", shlex.join(args))
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


async def run_evaluation(
    eval_proc: Process,
    build_queue: Queue[Job | StopTask],
    upload_queue: Queue[Build | StopTask],
    result: list[Result],
    opts: Options,
) -> int:
    assert eval_proc.stdout
    async for line in eval_proc.stdout:
        logger.debug(line.decode())
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            msg = f"Failed to parse line of nix-eval-jobs output: {line.decode()}"
            raise Error(msg) from e
        error = job.get("error")
        attr = job.get("attr", "unknown-flake-attribute")
        result.append(
            Result(
                result_type=ResultType.EVAL,
                attr=attr,
                success=error is None,
                # TODO: maybe add this to nix-eval-jobs?
                duration=0.0,
                error=error,
            )
        )
        if error:
            continue
        cache_status = job.get("cacheStatus")
        if cache_status is None:
            # Legacy attribute
            if job.get("isCached", False):
                continue
        # Skip remotely cached jobs, but still consider
        # them for pushing if they are cached locally
        elif cache_status == "cached":
            continue
        elif cache_status == "local":
            upload_queue.put_nowait(Build(attr, job["drvPath"], job.get("outputs", {})))
        system = job.get("system")
        if system and system not in opts.systems:
            continue
        drv_path = job.get("drvPath")
        if not drv_path:
            msg = f"nix-eval-jobs did not return a drvPath: {line.decode()}"
            raise Error(msg)
        outputs = job.get("outputs", {})
        build_queue.put_nowait(Job(attr, drv_path, outputs))
    return await eval_proc.wait()


async def run_builds(
    stack: AsyncExitStack,
    build_output: IO,
    build_queue: QueueWithContext[Job | StopTask],
    upload_queue: QueueWithContext[Build | StopTask],
    cachix_queue: QueueWithContext[Build | StopTask],
    attic_queue: QueueWithContext[Build | StopTask],
    download_queue: QueueWithContext[Build | StopTask],
    results: list[Result],
    opts: Options,
) -> int:
    drv_paths: set[Any] = set()

    while True:
        async with build_queue.get_context() as next_job:
            if isinstance(next_job, StopTask):
                logger.debug("finish build task")
                return 0
            job = next_job
            print(f"  building {job.attr}")
            if job.drv_path in drv_paths:
                continue
            drv_paths.add(job.drv_path)
            build = Build(job.attr, job.drv_path, job.outputs)
            start_time = timeit.default_timer()
            rc = await build.build(stack, build_output, opts)
            results.append(
                Result(
                    result_type=ResultType.BUILD,
                    attr=job.attr,
                    success=rc == 0,
                    duration=start_time - timeit.default_timer(),
                    # TODO: add log output here
                    error=f"build exited with {rc}" if rc != 0 else None,
                )
            )
            if rc != 0:
                continue
            upload_queue.put_nowait(build)
            download_queue.put_nowait(build)
            cachix_queue.put_nowait(build)
            attic_queue.put_nowait(build)


async def run_uploads(
    stack: AsyncExitStack,
    upload_queue: QueueWithContext[Build | StopTask],
    results: list[Result],
    opts: Options,
) -> int:
    while True:
        async with upload_queue.get_context() as build:
            if isinstance(build, StopTask):
                logger.debug("finish upload task")
                return 0
            start_time = timeit.default_timer()
            rc = await build.upload(stack, opts)
            results.append(
                Result(
                    result_type=ResultType.UPLOAD,
                    attr=build.attr,
                    success=rc == 0,
                    duration=start_time - timeit.default_timer(),
                    # TODO: add log output here
                    error=f"upload exited with {rc}" if rc != 0 else None,
                )
            )


async def run_cachix_upload(
    cachix_queue: QueueWithContext[Build | StopTask],
    cachix_socket_path: Path | None,
    results: list[Result],
    opts: Options,
) -> int:
    while True:
        async with cachix_queue.get_context() as build:
            if isinstance(build, StopTask):
                logger.debug("finish cachix upload task")
                return 0
            start_time = timeit.default_timer()
            rc = await build.upload_cachix(cachix_socket_path, opts)
            results.append(
                Result(
                    result_type=ResultType.CACHIX,
                    attr=build.attr,
                    success=rc == 0,
                    duration=start_time - timeit.default_timer(),
                    error=f"cachix upload exited with {rc}" if rc != 0 else None,
                )
            )


async def run_attic_upload(
    attic_queue: QueueWithContext[Build | StopTask],
    results: list[Result],
    opts: Options,
) -> int:
    while True:
        async with attic_queue.get_context() as build:
            if isinstance(build, StopTask):
                logger.debug("finish attic upload task")
                return 0
            start_time = timeit.default_timer()
            rc = await build.upload_attic(opts)
            results.append(
                Result(
                    result_type=ResultType.ATTIC,
                    attr=build.attr,
                    success=rc == 0,
                    duration=start_time - timeit.default_timer(),
                    error=f"attic upload exited with {rc}" if rc != 0 else None,
                )
            )


async def run_downloads(
    stack: AsyncExitStack,
    download_queue: QueueWithContext[Build | StopTask],
    results: list[Result],
    opts: Options,
) -> int:
    while True:
        async with download_queue.get_context() as build:
            if isinstance(build, StopTask):
                logger.debug("finish download task")
                return 0
            start_time = timeit.default_timer()
            rc = await build.download(stack, opts)
            results.append(
                Result(
                    result_type=ResultType.DOWNLOAD,
                    attr=build.attr,
                    success=rc == 0,
                    duration=start_time - timeit.default_timer(),
                    error=f"download exited with {rc}" if rc != 0 else None,
                )
            )


async def report_progress(
    build_queue: QueueWithContext,
    upload_queue: QueueWithContext,
    download_queue: QueueWithContext,
) -> int:
    old_status = ""
    try:
        while True:
            builds = build_queue.qsize() + build_queue.running_tasks
            uploads = upload_queue.qsize() + upload_queue.running_tasks
            downloads = download_queue.qsize() + download_queue.running_tasks
            new_status = f"builds: {builds}, uploads: {uploads}, downloads: {downloads}"
            if new_status != old_status:
                logger.info(new_status)
                old_status = new_status
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    return 0


@dataclass
class Summary:
    successes: int = 0
    failures: int = 0
    failed_attrs: list[str] = field(default_factory=list)


async def run(stack: AsyncExitStack, opts: Options) -> int:
    if opts.remote:
        tmp_dir = await stack.enter_async_context(remote_temp_dir(opts))
    else:
        tmp_dir = Path(stack.enter_context(TemporaryDirectory()))

    eval_proc = await stack.enter_async_context(nix_eval_jobs(tmp_dir, opts))
    pipe: Pipe | None = None
    output_monitor: Process | None = None
    if opts.nom:
        pipe = stack.enter_context(Pipe())
        output_monitor = await stack.enter_async_context(nix_output_monitor(pipe, opts))

    cachix_socket_path: Path | None = None
    if opts.cachix_cache:
        cachix_socket_path = await stack.enter_async_context(
            run_cachix_daemon(stack, tmp_dir, opts.cachix_cache, opts)
        )
    results: list[Result] = []
    build_queue: QueueWithContext[Job | StopTask] = QueueWithContext()
    cachix_queue: QueueWithContext[Build | StopTask] = QueueWithContext()
    attic_queue: QueueWithContext[Build | StopTask] = QueueWithContext()
    upload_queue: QueueWithContext[Build | StopTask] = QueueWithContext()
    download_queue: QueueWithContext[Build | StopTask] = QueueWithContext()

    async with TaskGroup() as tg:
        tasks = []
        tasks.append(
            tg.create_task(
                run_evaluation(eval_proc, build_queue, upload_queue, results, opts)
            )
        )
        evaluation = tasks[0]
        build_output = sys.stdout.buffer
        if pipe:
            build_output = pipe.write_file
        logger.debug("Starting %d build tasks", opts.max_jobs)
        for i in range(opts.max_jobs):
            tasks.append(
                tg.create_task(
                    run_builds(
                        stack,
                        build_output,
                        build_queue,
                        upload_queue,
                        cachix_queue,
                        attic_queue,
                        download_queue,
                        results,
                        opts,
                    ),
                    name=f"build-{i}",
                )
            )
            tasks.append(
                tg.create_task(
                    run_uploads(stack, upload_queue, results, opts),
                    name=f"upload-{i}",
                )
            )
            tasks.append(
                tg.create_task(
                    run_cachix_upload(
                        cachix_queue,
                        cachix_socket_path,
                        results,
                        opts,
                    ),
                    name=f"cachix-{i}",
                )
            )
            tasks.append(
                tg.create_task(
                    run_attic_upload(
                        attic_queue,
                        results,
                        opts,
                    ),
                    name=f"attic-{i}",
                )
            )
            tasks.append(
                tg.create_task(
                    run_downloads(stack, download_queue, results, opts),
                    name=f"download-{i}",
                )
            )
        if not opts.nom:
            logger.debug("Starting progress reporter")
            tasks.append(
                tg.create_task(
                    report_progress(build_queue, upload_queue, download_queue),
                    name="progress",
                )
            )
        logger.debug("Waiting for evaluation to finish...")
        eval_rc = await evaluation

        logger.debug("Evaluation finished, waiting for builds to finish...")
        for _ in range(opts.max_jobs):
            build_queue.put_nowait(StopTask())
        await build_queue.join()

        logger.debug("Builds finished, waiting for uploads to finish...")
        for _ in range(opts.max_jobs):
            upload_queue.put_nowait(StopTask())
        await upload_queue.join()

        logger.debug("Uploads finished, waiting for cachix uploads to finish...")
        for _ in range(opts.max_jobs):
            cachix_queue.put_nowait(StopTask())
        await cachix_queue.join()

        logger.debug("Uploads finished, waiting for attic uploads to finish...")
        for _ in range(opts.max_jobs):
            attic_queue.put_nowait(StopTask())
        await attic_queue.join()

        logger.debug("Uploads finished, waiting for downloads to finish...")
        for _ in range(opts.max_jobs):
            download_queue.put_nowait(StopTask())
        await download_queue.join()

        if not opts.nom:
            logger.debug("Stopping progress reporter")
            tasks[-1].cancel()
            await tasks[-1]

        for task in tasks:
            assert task.done(), f"Task {task.get_name()} is not done"

    rc = 0
    stats_by_type = defaultdict(Summary)
    for r in results:
        stats = stats_by_type[r.result_type]
        stats.successes += 1 if r.success else 0
        stats.failures += 1 if not r.success else 0
        stats.failed_attrs.append(r.attr)
        if not r.success:
            rc = 1
    for result_type, summary in stats_by_type.items():
        if summary.failures == 0:
            continue
        logger.error(
            f"{result_type.name}: {summary.successes} successes, {summary.failures} failures"
        )
        failed_attrs = [
            f"{opts.flake_url}#{opts.flake_fragment}.{attr}"
            for attr in summary.failed_attrs
        ]
        logger.error(f"Failed attributes: {' '.join(failed_attrs)}")
    if eval_rc != 0:
        logger.error(f"nix-eval-jobs exited with {eval_proc.returncode}")
        rc = 1
    if (
        output_monitor
        and output_monitor.returncode != 0
        and output_monitor.returncode is not None
    ):
        logger.error(f"nix-output-monitor exited with {output_monitor.returncode}")
        rc = 1

    if opts.result_file:
        with opts.result_file.open("w") as f:
            if opts.result_format == ResultFormat.JSON:
                dump_json(f, results)
            elif opts.result_format == ResultFormat.JUNIT:
                dump_junit_xml(f, opts.flake_url, opts.flake_fragment, results)

    return rc


def capitalize_first_letter(s: str) -> str:
    return s[0].upper() + s[1:].lower()


def dump_json(file: IO[str], results: list[Result]) -> None:
    json.dump(
        {
            "results": [
                {
                    "type": r.result_type.name,
                    "attr": r.attr,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error,
                }
                for r in results
            ]
        },
        file,
        indent=2,
        sort_keys=True,
    )


def dump_junit_xml(
    file: IO[str], flake_url: str, flake_fragment: str, build_results: list[Result]
) -> None:
    """
    Generates a JUnit XML report based on the results of Nix builds.

    Args:
        build_results (List[BuildResult]): A list of BuildResult instances containing build result data.
        output_file (str): The name of the output file where the XML report will be written.
    """
    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(
        testsuites,
        "testsuite",
        {
            "name": f"{flake_url}#{flake_fragment}",
            "tests": str(len(build_results)),
            "failures": str(sum(1 for r in build_results if not r.success)),
        },
    )

    for result in build_results:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": capitalize_first_letter(result.result_type.name),
                "name": result.attr,
                "time": str(result.duration),
            },
        )

        if not result.success:
            failure = ET.SubElement(
                testcase,
                "failure",
                {
                    "message": result.error or "<no message>",
                    "type": "BuildFailure",
                },
            )
            failure.text = result.error

    ET.ElementTree(testsuites).write(file, encoding="unicode")


async def async_main(args: list[str]) -> int:
    opts = await parse_args(args)
    if opts.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    stack = AsyncExitStack()
    # using async wait here seems to make the return value skipped in the non-execptional case
    try:
        if opts.remote_url:
            opts.flake_url = upload_sources(opts)
        return await run(stack, opts)
    finally:
        await stack.aclose()


def main() -> None:
    try:
        sys.exit(asyncio.run(async_main(sys.argv[1:])))
    except KeyboardInterrupt as e:
        logger.info(f"nix-fast-build was canceled by the user ({e})")
        sys.exit(1)
    except Error:
        logger.exception("nix-fast-build failed")
        sys.exit(1)

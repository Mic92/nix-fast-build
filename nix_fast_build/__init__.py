import argparse
import asyncio
import json
import multiprocessing
import os
import shlex
import shutil
import signal
import subprocess
import sys
from abc import ABC
from asyncio import Queue, TaskGroup
from asyncio.subprocess import Process
from collections import defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import IO, Any, AsyncIterator, Coroutine, DefaultDict, NoReturn


def die(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    sys.exit(1)


class Pipe:
    def __init__(self) -> None:
        fds = os.pipe()
        self.read_file = os.fdopen(fds[0], "rb")
        self.write_file = os.fdopen(fds[1], "wb")

    def __enter__(self) -> "Pipe":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.read_file.close()
        self.write_file.close()


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
    eval_workers: int = multiprocessing.cpu_count()
    max_jobs: int = 0
    retries: int = 0
    verbose: bool = False
    copy_to: str | None = None
    nom: bool = True
    download: bool = True

    @property
    def remote_url(self) -> None | str:
        if self.remote is None:
            return None
        return f"ssh://{self.remote}"


def _maybe_remote(
    cmd: list[str], remote: str | None, remote_ssh_options: list[str]
) -> list[str]:
    if remote:
        return ["ssh", remote] + remote_ssh_options + ["--", shlex.join(cmd)]
    else:
        return cmd


def maybe_remote(cmd: list[str], opts: Options) -> list[str]:
    return _maybe_remote(cmd, opts.remote, opts.remote_ssh_options)


async def get_nix_config(
    remote: str | None, remote_ssh_options: list[str]
) -> dict[str, str]:
    args = _maybe_remote(["nix", "show-config", "--json"], remote, remote_ssh_options)
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        die(f"nix not found in PATH, try to run {shlex.join(args)}")

    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        die(
            f"Failed to get nix config, {shlex.join(args)} exited with {proc.returncode}"
        )
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
        "--no-nom",
        help="Use nix-output-monitor to print build output (default: false)",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--systems",
        help="Comma-separated list of systems to build for (default: current system)",
        default=None,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of times to retry failed builds",
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
        "--verbose",
        action="store_true",
        help="Print verbose output",
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
        systems = set([nix_config.get("system", "")])
    else:
        systems = set(a.systems.split(","))

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
        verbose=a.verbose,
        systems=systems,
        eval_max_memory_size=a.eval_max_memory_size,
        eval_workers=a.eval_workers,
        copy_to=a.copy_to,
    )


def nix_flake_metadata(flake_url: str) -> dict[str, Any]:
    cmd = [
        "nix",
        "flake",
        "metadata",
        "--json",
        flake_url,
    ]
    print("$ " + shlex.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        die(
            f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
        )

    try:
        data = json.loads(proc.stdout)
    except Exception as e:
        die(
            f"failed to parse output of {shlex.join(cmd)}: {e}\nGot: {proc.stdout.decode('utf-8', 'replace')}"
        )
    return data


def is_path_input(node: dict[str, dict[str, str]]) -> bool:
    locked = node.get("locked")
    if not locked:
        return False
    return locked["type"] == "path" or locked.get("url", "").startswith("file://")


def check_for_path_inputs(data: dict[str, Any]) -> bool:
    for node in data["locks"]["nodes"].values():
        if is_path_input(node):
            return True
    return False


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
            cmd = ["nix", "copy", "--to", opts.remote_url, "--no-check-sigs", path]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, env=env)
            if proc.returncode != 0:
                die(
                    f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
                )
            return path

    # Slow path: we need to upload all sources to the remote machine
    cmd = [
        "nix",
        "flake",
        "archive",
        "--to",
        opts.remote_url,
        "--json",
        opts.flake_url,
    ]
    print("$ " + shlex.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        die(
            f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
        )
    try:
        return json.loads(proc.stdout)["path"]
    except Exception as e:
        die(
            f"failed to parse output of {shlex.join(cmd)}: {e}\nGot: {proc.stdout.decode('utf-8', 'replace')}"
        )


def nix_shell(packages: list[str]) -> list[str]:
    return (
        [
            "nix",
            "shell",
            "--extra-experimental-features",
            "nix-command",
            "--extra-experimental-features",
            "flakes",
        ]
        + packages
        + ["-c"]
    )


@asynccontextmanager
async def ensure_stop(
    proc: Process, cmd: list[str], timeout: float = 3.0, signal_no: int = signal.SIGTERM
) -> AsyncIterator[Process]:
    try:
        yield proc
    finally:
        if proc.returncode is not None:
            return
        proc.send_signal(signal_no)
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Failed to stop process {shlex.join(cmd)}. Killing it.")
            proc.kill()
            await proc.wait()


@asynccontextmanager
async def remote_temp_dir(opts: Options) -> AsyncIterator[str]:
    assert opts.remote
    ssh_cmd = ["ssh", opts.remote] + opts.remote_ssh_options + ["--"]
    cmd = ssh_cmd + ["mktemp", "-d"]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    tempdir = line.decode().strip()
    rc = await proc.wait()
    if rc != 0:
        die(
            f"Failed to create temporary directory on remote machine {opts.remote}: {rc}"
        )
    try:
        yield tempdir
    finally:
        cmd = ssh_cmd + ["rm", "-rf", tempdir]
        print("$ " + shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()


@asynccontextmanager
async def nix_eval_jobs(stack: AsyncExitStack, opts: Options) -> AsyncIterator[Process]:
    if opts.remote:
        gc_root_dir = await stack.enter_async_context(remote_temp_dir(opts))
    else:
        gc_root_dir = stack.enter_context(TemporaryDirectory())

    args = [
        "nix-eval-jobs",
        "--gc-roots-dir",
        gc_root_dir,
        "--force-recurse",
        "--max-memory-size",
        str(opts.eval_max_memory_size),
        "--workers",
        str(opts.eval_workers),
        "--flake",
        f"{opts.flake_url}#{opts.flake_fragment}",
    ] + opts.options
    if opts.skip_cached:
        args.append("--check-cache-status")
    if opts.remote:
        args = nix_shell(["nixpkgs#nix-eval-jobs"]) + args
    args = maybe_remote(args, opts)
    print("$ " + shlex.join(args))
    proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.PIPE)
    async with ensure_stop(proc, args) as proc:
        yield proc


@asynccontextmanager
async def nix_output_monitor(pipe: Pipe, opts: Options) -> AsyncIterator[Process]:
    cmd = maybe_remote(nix_shell(["nixpkgs#nix-output-monitor"]) + ["nom"], opts)
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=pipe.read_file)
    try:
        yield proc
    finally:
        # nom doesn't properly handle signals, so we have to close its stdin to stop it
        pipe.write_file.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            print(f"Failed to stop process {shlex.join(cmd)}. Killing it.")
            proc.kill()
            await proc.wait()


@dataclass
class Build:
    attr: str
    drv_path: str
    outputs: dict[str, str]

    async def build(
        self, stack: AsyncExitStack, build_output: IO[str], opts: Options
    ) -> int:
        proc = await stack.enter_async_context(
            nix_build(self.drv_path + "^*", build_output, opts)
        )
        rc = 0
        for _ in range(opts.retries + 1):
            rc = await proc.wait()
            if rc == 0:
                if opts.verbose:
                    print(f"build {self.attr} succeeded")
                return rc
            print(f"build {self.attr} exited with {rc}", file=sys.stderr)
        return rc

    async def nix_copy(
        self, args: list[str], exit_stack: AsyncExitStack, opts: Options
    ) -> int:
        cmd = maybe_remote(["nix", "copy", "--log-format", "raw"] + args, opts)
        if opts.verbose:
            print("$ " + shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()

    async def upload(self, exit_stack: AsyncExitStack, opts: Options) -> int:
        if not opts.copy_to:
            return 0
        cmd = ["nix", "copy", "--log-format", "raw", "--to", opts.copy_to] + list(
            self.outputs.values()
        )
        cmd = maybe_remote(cmd, opts)
        if opts.verbose:
            print("$ " + shlex.join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd)
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()

    async def download(self, exit_stack: AsyncExitStack, opts: Options) -> int:
        if not opts.remote_url or not opts.download:
            return 0
        cmd = [
            "nix",
            "copy",
            "--log-format",
            "raw",
            "--no-check-sigs",
            "--from",
            opts.remote_url,
        ] + list(self.outputs.values())
        if opts.verbose:
            print("$ " + shlex.join(cmd))
        env = os.environ.copy()
        env["NIX_SSHOPTS"] = " ".join(opts.remote_ssh_options)
        proc = await asyncio.create_subprocess_exec(*cmd, env=env)
        await exit_stack.enter_async_context(ensure_stop(proc, cmd))
        return await proc.wait()


@dataclass
class Failure(ABC):
    attr: str
    error_message: str


class EvalFailure(Failure):
    pass


class BuildFailure(Failure):
    pass


class UploadFailure(Failure):
    pass


class DownloadFailure(Failure):
    pass


@asynccontextmanager
async def nix_build(
    installable: str, stderr: IO[Any] | None, opts: Options
) -> AsyncIterator[Process]:
    args = [
        "nix",
        "build",
        installable,
        "--log-format",
        "raw",
        "--keep-going",
    ] + opts.options
    args = maybe_remote(args, opts)
    if opts.verbose:
        print("$ " + shlex.join(args))
    proc = await asyncio.create_subprocess_exec(*args, stderr=stderr)
    try:
        yield proc
    finally:
        proc.kill()


async def run_evaluation(
    eval_proc: Process,
    build_queue: Queue[tuple[str, str, str]],
    failures: list[Failure],
    opts: Options,
) -> None:
    assert eval_proc.stdout
    async for line in eval_proc.stdout:
        if opts.verbose:
            print(line, end="")
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            die(f"Failed to parse line of nix-eval-jobs output: {line.decode()}")
        error = job.get("error")
        attr = job.get("attr", "unknown-flake-attribute")
        if error:
            failures.append(EvalFailure(attr, error))
            continue
        is_cached = job.get("isCached", False)
        if is_cached:
            continue
        system = job.get("system")
        if system and system not in opts.systems:
            continue
        drv_path = job.get("drvPath")
        if not drv_path:
            die(f"nix-eval-jobs did not return a drvPath: {line.decode()}")
        outputs = job.get("outputs", {})
        build_queue.put_nowait((attr, drv_path, outputs))


async def run_builds(
    stack: AsyncExitStack,
    build_output: IO,
    build_queue: Queue,
    upload_queue: Queue,
    download_queue: Queue,
    failures: list[Failure],
    opts: Options,
) -> NoReturn:
    drv_paths: set[Any] = set()

    while True:
        attr, drv_path, outputs = await build_queue.get()
        print(f"  building {attr}")
        if drv_path in drv_paths:
            continue
        drv_paths.add(drv_path)
        build = Build(attr, drv_path, outputs)
        rc = await build.build(stack, build_output, opts)
        build_queue.task_done()
        if rc == 0:
            upload_queue.put_nowait(build)
            download_queue.put_nowait(build)
        else:
            failures.append(BuildFailure(build.attr, f"build exited with {rc}"))


async def run_uploads(
    stack: AsyncExitStack,
    upload_queue: Queue[Build],
    failures: list[Failure],
    opts: Options,
) -> NoReturn:
    while True:
        build = await upload_queue.get()
        rc = await build.upload(stack, opts)
        if rc != 0:
            failures.append(UploadFailure(build.attr, f"upload exited with {rc}"))
        upload_queue.task_done()


async def run_downloads(
    stack: AsyncExitStack,
    download_queue: Queue[Build],
    failures: list[Failure],
    opts: Options,
) -> NoReturn:
    while True:
        build = await download_queue.get()
        rc = await build.download(stack, opts)
        if rc != 0:
            failures.append(DownloadFailure(build.attr, f"download exited with {rc}"))
        download_queue.task_done()


async def report_progress(
    build_queue: Queue,
    upload_queue: Queue,
    download_queue: Queue,
) -> NoReturn:
    old_status = ""
    while True:
        new_status = f"builds: {build_queue.qsize()}, uploads: {upload_queue.qsize()}, downloads: {download_queue.qsize()}"
        if new_status != old_status:
            print(new_status)
            old_status = new_status
        await asyncio.sleep(0.5)


async def run(stack: AsyncExitStack, opts: Options) -> int:
    eval_proc_future = stack.enter_async_context(nix_eval_jobs(stack, opts))
    pipe: Pipe | None = None
    output_monitor_future: Coroutine[None, None, Process] | None = None
    if opts.nom:
        pipe = stack.enter_context(Pipe())
        output_monitor_future = stack.enter_async_context(
            nix_output_monitor(pipe, opts)
        )
    eval_proc = await eval_proc_future
    output_monitor: Process | None = None
    if output_monitor_future:
        output_monitor = await output_monitor_future
    failures: DefaultDict[type, list[Failure]] = defaultdict(list)
    build_queue: Queue[tuple[str, str, str]] = Queue()
    upload_queue: Queue[Build] = Queue()
    download_queue: Queue[Build] = Queue()

    evaluation = run_evaluation(eval_proc, build_queue, failures[EvalFailure], opts)
    async with TaskGroup() as tg:
        build_output = sys.stdout.buffer
        if pipe:
            build_output = pipe.write_file
        tasks = []
        for _ in range(opts.max_jobs):
            tasks.append(
                tg.create_task(
                    run_builds(
                        stack,
                        build_output,
                        build_queue,
                        upload_queue,
                        download_queue,
                        failures[BuildFailure],
                        opts,
                    )
                )
            )
        tasks.append(
            tg.create_task(
                run_uploads(stack, upload_queue, failures[UploadFailure], opts)
            )
        )
        tasks.append(
            tg.create_task(
                run_downloads(stack, download_queue, failures[DownloadFailure], opts)
            )
        )
        if not opts.nom:
            tasks.append(
                tg.create_task(
                    report_progress(build_queue, upload_queue, download_queue)
                )
            )
        await evaluation
        await build_queue.join()
        await upload_queue.join()
        await download_queue.join()
        for task in tasks:
            task.cancel()

    for failure_type in [EvalFailure, BuildFailure, UploadFailure, DownloadFailure]:
        for failure in failures[failure_type]:
            print(
                f"{failure_type.__name__} for {failure.attr}: {failure.error_message}"
            )
    if eval_proc.returncode != 0 and eval_proc.returncode is not None:
        print(f"nix-eval-jobs exited with {eval_proc.returncode}")
    if (
        output_monitor
        and output_monitor.returncode != 0
        and output_monitor.returncode is not None
    ):
        print(f"nix-output-monitor exited with {output_monitor.returncode}")

    return 0


async def async_main(args: list[str]) -> None:
    opts = await parse_args(args)

    rc = 0
    async with AsyncExitStack() as stack:
        if opts.remote_url:
            opts.flake_url = upload_sources(opts)
        rc = await run(stack, opts)
    sys.exit(rc)


def main() -> None:
    try:
        asyncio.run(async_main(sys.argv[1:]))
    except KeyboardInterrupt:
        pass  # don't print a stack trace on Ctrl-C

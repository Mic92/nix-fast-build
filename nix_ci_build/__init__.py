import argparse
import json
import multiprocessing
import os
import select
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import IO, Any, Iterator, NoReturn


def die(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    sys.exit(1)


@dataclass
class Options:
    flake: str = ""
    options: list[str] = field(default_factory=list)
    systems: set[str] = field(default_factory=set)
    eval_max_memory_size: int = 4096
    skip_cached: bool = False
    eval_workers: int = multiprocessing.cpu_count()
    max_jobs: int = 0
    retries: int = 0
    verbose: bool = False
    copy_to: str | None = None


def run_nix(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(["nix"] + args, text=True, capture_output=True)
    except FileNotFoundError:
        die("nix not found in PATH")
    return proc


def current_system() -> str:
    proc = run_nix(["eval", "--impure", "--raw", "--expr", "builtins.currentSystem"])
    if proc.returncode != 0:
        die(f"Failed to determine current system: {proc.stderr}")
    return proc.stdout.strip()


def max_jobs() -> int:
    proc = run_nix(["show-config", "max-jobs"])
    if proc.returncode != 0:
        die(f"Failed to determine number of CPUs: {proc.stderr}")
    return int(proc.stdout.strip())


def parse_args(args: list[str]) -> Options:
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
        default=max_jobs(),
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
        "--systems",
        help="Comma-separated list of systems to build for (default: current system)",
        default=current_system(),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of times to retry failed builds",
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
    systems = set(a.systems.split(","))

    options = []
    for name, value in a.option:
        options.extend(["--option", name, value])
    return Options(
        flake=a.flake,
        skip_cached=a.skip_cached,
        options=options,
        max_jobs=a.max_jobs,
        verbose=a.verbose,
        systems=systems,
        eval_max_memory_size=a.eval_max_memory_size,
        eval_workers=a.eval_workers,
        copy_to=a.copy_to,
    )


@contextmanager
def nix_eval_jobs(opts: Options) -> Iterator[subprocess.Popen[str]]:
    with TemporaryDirectory() as d:
        args = [
            "nix-eval-jobs",
            "--gc-roots-dir",
            d,
            "--force-recurse",
            "--max-memory-size",
            str(opts.eval_max_memory_size),
            "--workers",
            str(opts.eval_workers),
            "--flake",
            opts.flake,
        ] + opts.options
        if opts.skip_cached:
            args.append("--check-cache-status")
        print("$ " + " ".join(args))
        with subprocess.Popen(args, text=True, stdout=subprocess.PIPE) as proc:
            try:
                yield proc
            finally:
                proc.kill()


@contextmanager
def nix_build(
    installable: str, stdout: IO[Any] | None, opts: Options
) -> Iterator[subprocess.Popen]:
    log_format = "raw"
    args = [
        "nix",
        "build",
        installable,
        "--log-format",
        log_format,
        "--keep-going",
    ] + opts.options
    if opts.verbose:
        print("$ " + " ".join(args))
    with subprocess.Popen(args, text=True, stderr=stdout) as proc:
        try:
            yield proc
        finally:
            proc.kill()


@dataclass
class Build:
    attr: str
    drv_path: str
    outputs: dict[str, str]
    proc: subprocess.Popen[str]
    retries: int
    rc: int | None = None


@dataclass
class EvalError:
    attr: str
    error: str


def wait_for_any_build(builds: list[Build]) -> Build:
    while True:
        for i, build in enumerate(builds):
            rc = build.proc.poll()
            if rc is not None:
                del builds[i]
                build.rc = rc
                return build
        time.sleep(0.05)


def drain_builds(
    builds: list[Build], stdout: IO[Any] | None, stack: ExitStack, opts: Options
) -> list[Build]:
    build = wait_for_any_build(builds)
    if build.rc != 0:
        print(f"build {build.attr} exited with {build.rc}", file=sys.stderr)
        if build.retries < opts.retries:
            print(f"retrying build {build.attr} [{build.retries + 1}/{opts.retries}]")
            builds.append(
                create_build(
                    build.attr,
                    build.drv_path,
                    build.outputs,
                    stdout,
                    stack,
                    opts,
                    build.retries + 1,
                )
            )
        else:
            return [build]
    return []


def create_build(
    attr: str,
    drv_path: str,
    outputs: dict[str, str],
    stdout: IO[Any] | None,
    exit_stack: ExitStack,
    opts: Options,
    retries: int = 0,
) -> Build:
    nix_build_proc = exit_stack.enter_context(nix_build(drv_path + "^*", stdout, opts))
    if opts.copy_to:
        if opts.verbose:
            print(f"copying {attr} to {opts.copy_to}")
        exit_stack.enter_context(
            subprocess.Popen(
                [
                    "nix",
                    "copy",
                    "--to",
                    opts.copy_to,
                ]
                + list(outputs.values()),
            )
        )
    return Build(attr, drv_path, outputs, nix_build_proc, retries=retries)


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


def stop_gracefully(proc: subprocess.Popen, timeout: int = 1) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def nix_output_monitor(fd: int) -> Iterator[subprocess.Popen]:
    proc = subprocess.Popen(["nom"], stdin=fd)
    try:
        yield proc
    finally:
        stop_gracefully(proc)


def run_builds(stack: ExitStack, opts: Options) -> int:
    eval_error = []
    build_failures = []
    drv_paths = set()
    proc = stack.enter_context(nix_eval_jobs(opts))
    assert proc.stdout
    pipe = stack.enter_context(Pipe())
    nom_proc: subprocess.Popen | None = None
    stdout = pipe.write_file
    builds: list[Build] = []
    for line in proc.stdout:
        if opts.verbose:
            print(line, end="")
        if nom_proc is None:
            nom_proc = stack.enter_context(nix_output_monitor(pipe.read_file.fileno()))
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            die(f"Failed to parse line of nix-eval-jobs output: {line}")
        error = job.get("error")
        attr = job.get("attr", "unknown-flake-attribute")
        if error:
            eval_error.append(EvalError(attr, error))
            continue
        is_cached = job.get("isCached", False)
        if is_cached:
            continue
        system = job.get("system")
        if system and system not in opts.systems:
            continue
        drv_path = job.get("drvPath")
        if not drv_path:
            die(f"nix-eval-jobs did not return a drvPath: {line}")
        while len(builds) >= opts.max_jobs and opts.max_jobs != 0:
            build_failures += drain_builds(builds, stdout, stack, opts)
        print(f"  building {attr}")
        if drv_path in drv_paths:
            continue
        drv_paths.add(drv_path)
        outputs = job.get("outputs", {})
        builds.append(create_build(attr, drv_path, outputs, stdout, stack, opts))

    while builds:
        build_failures += drain_builds(builds, stdout, stack, opts)

    if nom_proc is not None:
        stop_gracefully(nom_proc)

    eval_rc = proc.wait()
    if eval_rc != 0:
        print(
            f"nix-eval-jobs exited with {eval_rc}, check logs for details",
            file=sys.stderr,
        )

    for error in eval_error:
        print(f"{error.attr}: {error.error}", file=sys.stderr)

    for build in build_failures:
        print(f"{build.attr}: build failed with {build.rc}", file=sys.stderr)

    if len(build_failures) > 0 or len(eval_error) > 0 or eval_rc != 0:
        return 1
    else:
        return 0


def main() -> None:
    opts = parse_args(sys.argv[1:])
    rc = 0
    with ExitStack() as stack:
        rc = run_builds(stack, opts)
    sys.exit(rc)


if __name__ == "__main__":
    main()

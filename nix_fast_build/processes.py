import asyncio
import contextlib
import logging
import shlex
import signal
import subprocess
import sys
from asyncio.subprocess import Process
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from .errors import Error
from .options import EvalMode, Options, maybe_remote, nix_shell

logger = logging.getLogger(__name__)


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
                    print(
                        f"Failed to stop process {shlex.join(cmd)}. Killing it.",
                        file=sys.stderr,
                    )
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
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=sys.stderr.fileno())
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
        *opts.options,
    ]
    if opts.impure:
        args.append("--impure")
    if opts.eval_mode == EvalMode.FLAKE:
        args.extend(
            [
                "--flake",
                f"{opts.flake_url}#{opts.flake_fragment}",
            ]
        )
        if opts.select_expr is not None:
            args.extend(["--select", opts.select_expr])
        for input_path, flake_url in opts.override_inputs:
            args.extend(["--override-input", input_path, flake_url])
        if opts.reference_lock_file:
            args.extend(["--reference-lock-file", opts.reference_lock_file])
    else:
        # Non-flake mode: pass expression file as positional arg
        args.extend(opts.expr_args)
        # nix-eval-jobs only accepts a single --select, so compose -A
        # navigation with any user-supplied select function.
        if opts.expr_attr and opts.select_expr is not None:
            args.extend(
                ["--select", f"root: ({opts.select_expr}) (root.{opts.expr_attr})"]
            )
        elif opts.expr_attr:
            args.extend(["--select", f"root: root.{opts.expr_attr}"])
        elif opts.select_expr is not None:
            args.extend(["--select", opts.select_expr])
        args.append(opts.expr_file)
    if opts.skip_cached:
        args.append("--check-cache-status")
    if opts.remote:
        args = nix_shell("nixpkgs#nix-eval-jobs", "nix-eval-jobs") + args
    else:
        args = [*opts.nix_eval_jobs_bin, *args]
    args = maybe_remote(args, opts)
    logger.info("run %s", shlex.join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        # 10MB buffer to accommodate for large lines
        limit=10485760,
    )
    async with ensure_stop(proc, args) as proc:
        yield proc


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
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=sys.stderr.fileno())
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
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=sys.stderr.fileno())
    await exit_stack.enter_async_context(ensure_stop(proc, cmd))
    return await proc.wait()

import asyncio
import json
import logging
import os
import shlex
import sys
import timeit
from asyncio import Queue
from asyncio.subprocess import Process
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

from .build import Build, BuildQueue, Job, JobQueue, OptionalQueue, StopTask
from .errors import Error
from .options import Options, maybe_remote, nix_shell
from .renderer import Renderer
from .results import Result, ResultType

logger = logging.getLogger(__name__)


def _job_outputs(job: dict[str, Any]) -> dict[str, str]:
    return {k: v for k, v in job.get("outputs", {}).items() if v is not None}


async def run_evaluation(
    eval_proc: Process,
    build_queue: JobQueue,
    upload_queue: BuildQueue | None,
    result_queue: "Queue[Result | None]",
    opts: Options,
) -> int:
    assert eval_proc.stdout
    async for line in eval_proc.stdout:
        if opts.should_stop:
            logger.debug("fail-fast: stopping evaluation")
            eval_proc.terminate()
            break
        logger.debug(line.decode())
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            msg = f"Failed to parse line of nix-eval-jobs output: {line.decode()}"
            raise Error(msg) from e
        error = job.get("error")
        attr = job.get("attr", "unknown-attribute")
        await result_queue.put(
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
            opts.signal_stop()
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
        elif cache_status == "local" and upload_queue is not None:
            upload_queue.put_nowait(Build(attr, job["drvPath"], _job_outputs(job)))
        system = job.get("system")
        if system and system not in opts.systems:
            continue
        drv_path = job.get("drvPath")
        if not drv_path:
            msg = f"nix-eval-jobs did not return a drvPath: {line.decode()}"
            raise Error(msg)
        build_queue.put_nowait(Job(attr, drv_path, _job_outputs(job)))
    return await eval_proc.wait()


async def run_builds(
    stack: AsyncExitStack,
    build_queue: JobQueue,
    optional_queues: list[BuildQueue],
    result_queue: "Queue[Result | None]",
    opts: Options,
    renderer: Renderer | None = None,
) -> int:
    drv_paths: set[Any] = set()

    while True:
        async with build_queue.get_context() as next_job:
            if isinstance(next_job, StopTask):
                logger.debug("finish build task")
                return 0
            if opts.should_stop:
                logger.debug("fail-fast: skipping build of %s", next_job.attr)
                continue
            job = next_job
            if job.drv_path in drv_paths:
                continue
            drv_paths.add(job.drv_path)
            build = Build(job.attr, job.drv_path, job.outputs)
            start_time = timeit.default_timer()
            build_result = await build.build(stack, opts, renderer=renderer)
            await result_queue.put(
                Result(
                    result_type=ResultType.BUILD,
                    attr=job.attr,
                    success=build_result.return_code == 0,
                    duration=timeit.default_timer() - start_time,
                    error=f"build exited with {build_result.return_code}"
                    if build_result.return_code != 0
                    else None,
                    log_output=build_result.log_output
                    if build_result.return_code != 0
                    else None,
                    outputs=job.outputs or None,
                )
            )
            if build_result.return_code != 0:
                opts.signal_stop()
                continue
            for queue in optional_queues:
                queue.put_nowait(build)


async def run_queue_worker(
    queue: BuildQueue,
    result_queue: "Queue[Result | None]",
    result_type: ResultType,
    label: str,
    push: Callable[[Build], Awaitable[int]],
) -> int:
    """Apply push to each queued build, recording one Result per build."""
    while True:
        async with queue.get_context() as build:
            if isinstance(build, StopTask):
                logger.debug("finish %s task", label)
                return 0
            start_time = timeit.default_timer()
            rc = await push(build)
            await result_queue.put(
                Result(
                    result_type=result_type,
                    attr=build.attr,
                    success=rc == 0,
                    duration=timeit.default_timer() - start_time,
                    error=f"{label} exited with {rc}" if rc != 0 else None,
                )
            )


async def run_niks3_upload(
    niks3_queue: BuildQueue,
    result_queue: "Queue[Result | None]",
    opts: Options,
) -> int:
    while True:
        # Wait for at least one item
        async with niks3_queue.get_context() as first_item:
            if isinstance(first_item, StopTask):
                logger.debug("finish niks3 upload task")
                return 0

            # Collect this build plus any others currently queued
            builds: list[Build] = [first_item]
            while not niks3_queue.empty():
                try:
                    item = niks3_queue.get_nowait()
                    niks3_queue.task_done()
                    if isinstance(item, StopTask):
                        # Put it back for proper shutdown
                        niks3_queue.put_nowait(item)
                        break
                    builds.append(item)
                except asyncio.QueueEmpty:
                    break

            # Collect all output paths
            all_outputs: list[str] = []
            for build in builds:
                all_outputs.extend(build.outputs.values())

            start_time = timeit.default_timer()
            env = os.environ.copy()
            # niks3_server is guaranteed non-None since this worker is only created when configured
            assert opts.niks3_server is not None
            env["NIKS3_SERVER_URL"] = opts.niks3_server
            cmd = maybe_remote(
                [
                    *nix_shell("github:Mic92/niks3", "niks3"),
                    "push",
                    *all_outputs,
                ],
                opts,
            )
            logger.debug("run %s", shlex.join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd, env=env, stdout=sys.stderr.fileno()
            )
            rc = await proc.wait()
            duration = timeit.default_timer() - start_time

            # Record result for each build
            for build in builds:
                await result_queue.put(
                    Result(
                        result_type=ResultType.NIKS3,
                        attr=build.attr,
                        success=rc == 0,
                        duration=duration / len(builds),
                        error=f"niks3 upload exited with {rc}" if rc != 0 else None,
                    )
                )


async def report_progress(
    build_queue: JobQueue,
    optional_queues: list[OptionalQueue],
) -> int:
    old_status = ""
    queues = [("builds", build_queue)] + [(oq.name, oq.queue) for oq in optional_queues]
    try:
        while True:
            new_status = ", ".join(
                f"{name}: {queue.qsize() + queue.running_tasks}"
                for name, queue in queues
            )
            if new_status != old_status:
                logger.info(new_status)
                old_status = new_status
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    return 0

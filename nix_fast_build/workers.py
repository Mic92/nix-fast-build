import asyncio
import json
import logging
import timeit
from asyncio import Queue
from asyncio.subprocess import Process
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

from .build import Build, BuildQueue, Job, JobQueue, OptionalQueue, StopTask
from .errors import Error
from .options import Options
from .renderer import Renderer
from .results import Result, ResultType
from .upload import UploadItem, UploadQueue

logger = logging.getLogger(__name__)


def _job_outputs(job: dict[str, Any]) -> dict[str, str]:
    return {k: v for k, v in job.get("outputs", {}).items() if v is not None}


async def run_evaluation(
    eval_proc: Process,
    build_queue: JobQueue,
    upload_queues: list[UploadQueue],
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
        cache_status = job.get("cacheStatus")
        if cache_status is None and job.get("isCached", False):
            cache_status = "cached"
        outputs = _job_outputs(job)
        await result_queue.put(
            Result(
                result_type=ResultType.EVAL,
                attr=attr,
                success=error is None,
                # TODO: maybe add this to nix-eval-jobs?
                duration=0.0,
                error=error,
                outputs=outputs or None,
                drv_path=job.get("drvPath"),
                cache_status=cache_status,
            )
        )
        if error:
            opts.signal_stop()
            continue
        # Skip remotely cached jobs, but still consider
        # them for pushing if they are cached locally
        if cache_status == "cached":
            continue
        if cache_status == "local":
            for uq in upload_queues:
                uq.put_nowait(UploadItem(attr, list(outputs.values())))
            # already valid locally: build only if a result symlink is wanted
            if opts.out_link is None:
                continue
        system = job.get("system")
        if system and system not in opts.systems:
            continue
        drv_path = job.get("drvPath")
        if not drv_path:
            msg = f"nix-eval-jobs did not return a drvPath: {line.decode()}"
            raise Error(msg)
        build_queue.put_nowait(Job(attr, drv_path, outputs))
    return await eval_proc.wait()


async def run_builds(
    stack: AsyncExitStack,
    build_queue: JobQueue,
    upload_queues: list[UploadQueue],
    build_queues: list[BuildQueue],
    result_queue: "Queue[Result | None]",
    *,
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
            on_built = None
            if opts.push_build_closure and upload_queues:

                def on_built(drv: str, attr: str = job.attr) -> None:
                    for uq in upload_queues:
                        uq.put_nowait(UploadItem(attr, [drv], final=False))

            start_time = timeit.default_timer()
            build_result = await build.build(
                stack, opts, renderer=renderer, on_built=on_built
            )
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
            if job.outputs:
                for uq in upload_queues:
                    uq.put_nowait(UploadItem(job.attr, list(job.outputs.values())))
            for bq in build_queues:
                bq.put_nowait(build)


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

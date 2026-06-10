import asyncio
import logging
import sys
from asyncio import TaskGroup
from collections import defaultdict
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio.subprocess import Process

from .build import Build, Job, OptionalQueue, QueueWithContext, StopTask
from .errors import Error
from .options import EvalMode, Options, ResultFormat, parse_args
from .pipe import Pipe
from .processes import (
    nix_eval_jobs,
    nix_output_monitor,
    remote_temp_dir,
    run_cachix_daemon,
    stop_nom,
)
from .report import (
    dump_json,
    dump_junit_xml,
    get_ci_summary_file,
    write_ci_summary,
)
from .results import Result, ResultType, Summary
from .sources import upload_sources
from .workers import (
    report_progress,
    run_attic_upload,
    run_builds,
    run_cachix_upload,
    run_downloads,
    run_evaluation,
    run_niks3_upload,
    run_uploads,
)

__all__ = [
    "Error",
    "EvalMode",
    "Options",
    "Result",
    "ResultFormat",
    "ResultType",
    "async_main",
    "get_ci_summary_file",
    "main",
    "parse_args",
    "write_ci_summary",
]

logger = logging.getLogger(__name__)


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

    # Build list of optional queues that are actually needed
    optional_queues: list[OptionalQueue] = []

    upload_queue: QueueWithContext[Build | StopTask] | None = None
    if opts.copy_to:
        upload_queue = QueueWithContext()
        optional_queues.append(OptionalQueue(upload_queue, opts.max_jobs, "upload"))

    cachix_queue: QueueWithContext[Build | StopTask] | None = None
    if cachix_socket_path:
        cachix_queue = QueueWithContext()
        optional_queues.append(OptionalQueue(cachix_queue, opts.max_jobs, "cachix"))

    attic_queue: QueueWithContext[Build | StopTask] | None = None
    if opts.attic_cache:
        attic_queue = QueueWithContext()
        optional_queues.append(OptionalQueue(attic_queue, opts.max_jobs, "attic"))

    niks3_queue: QueueWithContext[Build | StopTask] | None = None
    if opts.niks3_server:
        niks3_queue = QueueWithContext()
        optional_queues.append(OptionalQueue(niks3_queue, 1, "niks3"))

    download_queue: QueueWithContext[Build | StopTask] | None = None
    if opts.remote_url and opts.download:
        download_queue = QueueWithContext()
        optional_queues.append(OptionalQueue(download_queue, opts.max_jobs, "download"))

    async with TaskGroup() as tg:
        tasks = []
        tasks.append(
            tg.create_task(
                run_evaluation(eval_proc, build_queue, upload_queue, results, opts)
            )
        )
        evaluation = tasks[0]
        # When nom is enabled, each nix-build captures its own stderr and
        # forwards complete lines to the shared nom pipe, avoiding
        # interleaved writes from concurrent builds.  build_output is only
        # used as the nix-build stderr fd when nom is *not* active.
        nom_pipe: IO[bytes] | None = pipe.write_file if pipe else None
        build_output = sys.stdout.buffer
        logger.debug("Starting %d build tasks", opts.max_jobs)
        for i in range(opts.max_jobs):
            tasks.append(
                tg.create_task(
                    run_builds(
                        stack,
                        build_output,
                        build_queue,
                        [oq.queue for oq in optional_queues],
                        results,
                        opts,
                        nom_pipe=nom_pipe,
                    ),
                    name=f"build-{i}",
                )
            )
            if upload_queue is not None:
                tasks.append(
                    tg.create_task(
                        run_uploads(stack, upload_queue, results, opts),
                        name=f"upload-{i}",
                    )
                )
            if cachix_queue is not None and cachix_socket_path is not None:
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
            if attic_queue is not None:
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
            if download_queue is not None:
                tasks.append(
                    tg.create_task(
                        run_downloads(stack, download_queue, results, opts),
                        name=f"download-{i}",
                    )
                )
        # Single niks3 worker since it batches uploads internally
        if niks3_queue is not None:
            tasks.append(
                tg.create_task(
                    run_niks3_upload(
                        niks3_queue,
                        results,
                        opts,
                    ),
                    name="niks3",
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

        for oq in optional_queues:
            logger.debug("Waiting for %s to finish...", oq.name)
            for _ in range(oq.worker_count):
                oq.queue.put_nowait(StopTask())
            await oq.queue.join()

        if not opts.nom:
            logger.debug("Stopping progress reporter")
            tasks[-1].cancel()
            await tasks[-1]

        for task in tasks:
            assert task.done(), f"Task {task.get_name()} is not done"

    # Stop nom before logging the summary so its final redraw doesn't clobber
    # the output; the teardown in the stack is a no-op afterwards.
    if pipe is not None and output_monitor is not None:
        await stop_nom(output_monitor, pipe)

    rc = 0
    stats_by_type: dict[ResultType, Summary] = defaultdict(Summary)
    for r in results:
        stats = stats_by_type[r.result_type]
        stats.successes += 1 if r.success else 0
        stats.failures += 1 if not r.success else 0
        if not r.success:
            stats.failed_attrs.append(r.attr)
            rc = 1
    for result_type, summary in stats_by_type.items():
        if summary.failures == 0:
            continue
        logger.error(
            f"{result_type.name}: {summary.successes} successes, {summary.failures} failures"
        )
        if opts.eval_mode == EvalMode.FLAKE:
            failed_attrs = [
                f"{opts.flake_url}#{opts.flake_fragment}.{attr}"
                for attr in summary.failed_attrs
            ]
        else:
            failed_attrs = [
                f"{opts.expr_file}.{attr}" if opts.expr_attr else attr
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
                dump_junit_xml(f, opts.display_name, results)

    # Write CI summary if configured (GitHub Actions, Gitea Actions, or Forgejo Actions)
    ci_summary_file = get_ci_summary_file()
    if ci_summary_file:
        write_ci_summary(ci_summary_file, opts, results, rc)

    return rc


async def async_main(args: list[str]) -> int:
    opts = await parse_args(args)
    if opts.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    stack = AsyncExitStack()
    # using async wait here seems to make the return value skipped in the non-execptional case
    try:
        if opts.remote_url and opts.eval_mode == EvalMode.FLAKE:
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

import asyncio
import json
import logging
import os
import sys
from asyncio import Queue, TaskGroup
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .build import Build, BuildQueue, JobQueue, OptionalQueue, StopTask
from .ci_renderer import CIRenderer
from .errors import Error
from .options import EvalMode, Options, ResultFormat, parse_args
from .processes import (
    nix_eval_jobs,
    read_eval_stderr_lines,
    remote_temp_dir,
    run_cachix_daemon,
)
from .report import (
    dump_json,
    dump_junit_xml,
    get_ci_summary_file,
    write_ci_summary,
)
from .results import Result, ResultType, Summary
from .sources import upload_sources
from .term import fold_markers, is_ignored_eval_line, sanitize_line, want_color
from .tty_renderer import TTYRenderer
from .workers import (
    report_progress,
    run_builds,
    run_evaluation,
    run_niks3_upload,
    run_queue_worker,
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


def start_renderer(stack: AsyncExitStack, opts: Options) -> CIRenderer | TTYRenderer:
    """Pick and start the build log renderer.

    Teardown is registered on the exit stack so the render/heartbeat task
    doesn't leak when the build TaskGroup raises; the normal path stops
    the renderer explicitly before the summary (stop is idempotent).
    """
    if (
        opts.interactive
        and sys.stderr.isatty()
        and sys.stdin.isatty()
        and os.environ.get("TERM", "") not in ("", "dumb")
    ):
        tty_renderer = TTYRenderer(sys.stderr)
        tty_renderer.start()
        stack.push_async_callback(tty_renderer.stop)
        return tty_renderer
    ci_renderer = CIRenderer(
        sys.stderr,
        color=want_color(sys.stderr.isatty()),
        fold=fold_markers() and not opts.no_fold,
        stall_timeout=opts.stall_timeout,
    )
    ci_renderer.start_heartbeat()
    stack.push_async_callback(ci_renderer.stop_heartbeat)
    return ci_renderer


async def run(stack: AsyncExitStack, opts: Options) -> int:
    if opts.remote:
        tmp_dir = await stack.enter_async_context(remote_temp_dir(opts))
    else:
        tmp_dir = Path(stack.enter_context(TemporaryDirectory()))

    # Renderer first: from here on all terminal output (including log
    # records and eval stderr) must go through it.
    renderer = start_renderer(stack, opts)
    eval_jobs = await stack.enter_async_context(
        nix_eval_jobs(tmp_dir, opts, color_tty=isinstance(renderer, TTYRenderer))
    )
    eval_proc = eval_jobs.proc

    async def forward_eval_stderr() -> None:
        """Route nix-eval-jobs stderr (eval warnings/errors) through the
        renderer; written directly it would corrupt the TTY region."""
        async for raw in read_eval_stderr_lines(eval_jobs.stderr):
            line = sanitize_line(raw.decode(errors="replace").rstrip("\r\n"))
            if not line:
                continue
            if is_ignored_eval_line(line):
                continue
            renderer.log_line(line)

    cachix_socket_path: Path | None = None
    if opts.cachix_cache:
        cachix_socket_path = await stack.enter_async_context(
            run_cachix_daemon(stack, tmp_dir, opts.cachix_cache, opts)
        )
    results: list[Result] = []
    result_queue: Queue[Result | None] = Queue()
    build_queue = JobQueue()

    # Build list of optional queues that are actually needed
    optional_queues: list[OptionalQueue] = []

    def add_queue(
        name: str,
        result_type: ResultType,
        push: Callable[[Build], Awaitable[int]],
    ) -> BuildQueue:
        queue = BuildQueue()
        optional_queues.append(
            OptionalQueue(
                queue,
                opts.max_jobs,
                name,
                lambda: run_queue_worker(queue, result_queue, result_type, name, push),
            )
        )
        return queue

    upload_queue: BuildQueue | None = None
    if opts.copy_to:
        upload_queue = add_queue(
            "upload", ResultType.UPLOAD, lambda b: b.upload(stack, opts)
        )

    if cachix_socket_path is not None:
        # Local alias so mypy sees a non-None capture in the lambda.
        socket_path = cachix_socket_path
        add_queue(
            "cachix", ResultType.CACHIX, lambda b: b.upload_cachix(socket_path, opts)
        )

    if opts.attic_cache:
        add_queue("attic", ResultType.ATTIC, lambda b: b.upload_attic(opts))

    if opts.niks3_server:
        # Single niks3 worker since it batches uploads internally
        niks3_queue = BuildQueue()
        optional_queues.append(
            OptionalQueue(
                niks3_queue,
                1,
                "niks3",
                lambda: run_niks3_upload(niks3_queue, result_queue, opts),
            )
        )

    if opts.remote_url and opts.download:
        add_queue("download", ResultType.DOWNLOAD, lambda b: b.download(stack, opts))

    async def dequeue_results() -> None:
        # Stream results as they arrive and collect them for the final
        # summary and result file. A None sentinel stops the task.
        while True:
            result = await result_queue.get()
            if result is None:
                return
            if opts.stream_json_lines:
                print(json.dumps(result.as_dict()), flush=True)
            results.append(result)

    async with TaskGroup() as tg:
        tasks: list[asyncio.Task[object]] = []
        dequeue_task = tg.create_task(dequeue_results(), name="dequeue-results")
        tasks.append(dequeue_task)
        # Ends on its own at eval stderr EOF (process exit).
        tasks.append(tg.create_task(forward_eval_stderr(), name="eval-stderr"))
        evaluation = tg.create_task(
            run_evaluation(eval_proc, build_queue, upload_queue, result_queue, opts)
        )
        tasks.append(evaluation)
        logger.debug("Starting %d build tasks", opts.max_jobs)
        tasks.extend(
            tg.create_task(
                run_builds(
                    stack,
                    build_queue,
                    [oq.queue for oq in optional_queues],
                    result_queue,
                    opts,
                    renderer=renderer,
                ),
                name=f"build-{i}",
            )
            for i in range(opts.max_jobs)
        )
        for oq in optional_queues:
            tasks.extend(
                tg.create_task(oq.make_worker(), name=f"{oq.name}-{i}")
                for i in range(oq.worker_count)
            )
        progress_task = None
        if isinstance(renderer, CIRenderer):
            # Queue backlog reporter; the TTY renderer shows liveness
            # itself, and logger output would corrupt its region.
            logger.debug("Starting progress reporter")
            progress_task = tg.create_task(
                report_progress(build_queue, optional_queues),
                name="progress",
            )
            tasks.append(progress_task)
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

        if progress_task is not None:
            logger.debug("Stopping progress reporter")
            progress_task.cancel()
            await progress_task

        # All workers are done; stop the result consumer.
        result_queue.put_nowait(None)
        await dequeue_task

        for task in tasks:
            assert task.done(), f"Task {task.get_name()} is not done"

    # Stop the renderer before logging the summary so the TTY region or a
    # heartbeat doesn't clobber it. Idempotent; the stack callbacks handle
    # exceptional paths.
    if isinstance(renderer, TTYRenderer):
        # Don't yank the UI away while the user is reading logs in the
        # browser or a pager; wait until they leave.
        await renderer.wait_until_idle()
        await renderer.stop()
    else:
        await renderer.stop_heartbeat()

    rc = 0
    stats_by_type: dict[ResultType, Summary] = defaultdict(Summary)
    for r in results:
        stats = stats_by_type[r.result_type]
        if r.success:
            stats.successes += 1
        else:
            stats.failures += 1
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
        logger.error(f"nix-eval-jobs exited with {eval_rc}")
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
        write_ci_summary(ci_summary_file, results, rc)

    return rc


async def async_main(args: list[str]) -> int:
    opts = await parse_args(args)
    if opts.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

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

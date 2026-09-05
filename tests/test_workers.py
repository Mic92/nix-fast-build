import asyncio
import json
from typing import TYPE_CHECKING, Any

from nix_fast_build.build import JobQueue
from nix_fast_build.options import Options
from nix_fast_build.upload import UploadQueue
from nix_fast_build.workers import run_evaluation

if TYPE_CHECKING:
    from nix_fast_build.results import Result


class FakeEvalProc:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.stdout = self._lines(jobs)

    @staticmethod
    async def _lines(jobs: list[dict[str, Any]]) -> Any:
        for job in jobs:
            yield (json.dumps(job) + "\n").encode()

    async def wait(self) -> int:
        return 0


JOB = {
    "attr": "foo",
    "drvPath": "/nix/store/aaa-foo.drv",
    "outputs": {"out": "/nix/store/aaa-foo"},
    "system": "x86_64-linux",
    "cacheStatus": "local",
}


def _eval(opts: Options) -> tuple[JobQueue, UploadQueue]:
    build_queue: JobQueue = JobQueue()
    upload_queue: UploadQueue = UploadQueue()
    result_queue: asyncio.Queue[Result | None] = asyncio.Queue()
    asyncio.run(
        run_evaluation(
            FakeEvalProc([JOB]),  # type: ignore[arg-type]
            build_queue,
            [upload_queue],
            result_queue,
            opts,
        )
    )
    return build_queue, upload_queue


def test_local_cache_status_skips_build() -> None:
    build_queue, upload_queue = _eval(Options(systems={"x86_64-linux"}))
    assert build_queue.qsize() == 0
    assert upload_queue.qsize() == 1


def test_local_cache_status_builds_when_out_link_requested() -> None:
    build_queue, _ = _eval(Options(systems={"x86_64-linux"}, out_link="result"))
    assert build_queue.qsize() == 1


def test_eval_result_reports_outputs_and_drv_path() -> None:
    result_queue: asyncio.Queue[Result | None] = asyncio.Queue()
    asyncio.run(
        run_evaluation(
            FakeEvalProc([{**JOB, "cacheStatus": "cached"}]),  # type: ignore[arg-type]
            JobQueue(),
            [],
            result_queue,
            Options(systems={"x86_64-linux"}),
        )
    )
    result = result_queue.get_nowait()
    assert result is not None
    data = result.as_dict()
    assert data["type"] == "EVAL"
    assert data["outputs"] == {"out": "/nix/store/aaa-foo"}
    assert data["drvPath"] == "/nix/store/aaa-foo.drv"
    assert data["cacheStatus"] == "cached"


UNSUPPORTED_JOB = {
    "attr": "bar",
    "error": "error: Package bar-1.0 in /x is not available on the requested "
    'hostPlatform:\n  hostPlatform.system = "x86_64-linux"',
}


def _eval_error(opts: Options) -> "Result":
    result_queue: asyncio.Queue[Result | None] = asyncio.Queue()
    asyncio.run(
        run_evaluation(
            FakeEvalProc([UNSUPPORTED_JOB]),  # type: ignore[arg-type]
            JobQueue(),
            [],
            result_queue,
            opts,
        )
    )
    result = result_queue.get_nowait()
    assert result is not None
    return result


def test_unsupported_platform_fails_by_default() -> None:
    opts = Options(fail_fast=True)
    result = _eval_error(opts)
    assert not result.success
    assert not result.skipped
    assert opts.should_stop


def test_skip_unsupported_reports_skipped() -> None:
    opts = Options(fail_fast=True, skip_unsupported=True)
    result = _eval_error(opts)
    assert result.success
    assert result.as_dict()["skipped"] is True
    assert not opts.should_stop

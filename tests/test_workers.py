import asyncio
import json
from typing import TYPE_CHECKING, Any

from nix_fast_build.build import BuildQueue, JobQueue
from nix_fast_build.options import Options
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


def _eval(opts: Options) -> tuple[JobQueue, BuildQueue]:
    build_queue: JobQueue = JobQueue()
    upload_queue: BuildQueue = BuildQueue()
    result_queue: asyncio.Queue[Result | None] = asyncio.Queue()
    asyncio.run(
        run_evaluation(
            FakeEvalProc([JOB]),  # type: ignore[arg-type]
            build_queue,
            upload_queue,
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

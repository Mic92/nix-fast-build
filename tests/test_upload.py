import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nix_fast_build import Options
from nix_fast_build.build import StopTask
from nix_fast_build.results import Result, ResultType
from nix_fast_build.upload import (
    Uploader,
    UploadItem,
    UploadQueue,
    run_upload_worker,
)

# --query --outputs maps X.drv -> X. --print-invalid reports "broken" paths.
FAKE_NIX_STORE = """#!/usr/bin/env bash
mode=$*
for p in "$@"; do
  [[ $p == /nix/store/* ]] || continue
  case "$mode" in
    *--outputs*) echo "${p%.drv}" ;;
    *--print-invalid*) [[ $p == *broken* ]] && echo "$p" ;;
  esac
done
exit 0
"""


@dataclass
class RecordingUploader(Uploader):
    calls: list[list[str]] = field(default_factory=list)
    fail_if: str = ""

    async def send(self, paths: list[str]) -> int:
        self.calls.append(paths)
        return 1 if self.fail_if and self.fail_if in paths else 0


@pytest.fixture
def fake_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "nix-store").write_text(FAKE_NIX_STORE)
    (tmp_path / "nix-store").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")


def run_worker(uploader: Uploader, batches: list[list[UploadItem]]) -> list[Result]:

    async def go() -> list[Result]:
        queue = UploadQueue()
        results: asyncio.Queue[Result | None] = asyncio.Queue()
        task = asyncio.create_task(run_upload_worker(queue, results, uploader))
        for batch in batches:
            for item in batch:
                queue.put_nowait(item)
            await queue.join()
        queue.put_nowait(StopTask())
        await task
        out: list[Result] = []
        while not results.empty():
            r = results.get_nowait()
            assert r is not None
            out.append(r)
        return out

    return asyncio.run(go())


@pytest.mark.usefixtures("fake_path")
def test_batches_resolves_drvs_filters_invalid_and_dedups() -> None:
    up = RecordingUploader("test", ResultType.ATTIC, Options())
    results = run_worker(
        up,
        [
            [
                UploadItem("a", ["/nix/store/x-lib.drv"]),
                UploadItem("b", ["/nix/store/x-lib.drv", "/nix/store/y-broken.drv"]),
                UploadItem("a", ["/nix/store/a-out"]),
            ],
            [UploadItem("b", ["/nix/store/x-lib.drv", "/nix/store/b-out"])],
        ],
    )
    assert up.calls == [
        ["/nix/store/a-out", "/nix/store/x-lib"],
        ["/nix/store/b-out"],
    ]
    assert [(r.attr, r.success) for r in results] == [
        ("a", True),
        ("b", True),
        ("b", True),
    ]


@pytest.mark.usefixtures("fake_path")
def test_failed_batch_falls_back_per_attr() -> None:
    up = RecordingUploader(
        "test", ResultType.ATTIC, Options(), fail_if="/nix/store/b-bad"
    )
    results = run_worker(
        up,
        [
            [
                UploadItem("a", ["/nix/store/a-out"]),
                UploadItem("b", ["/nix/store/b-bad"]),
            ]
        ],
    )
    assert up.calls == [
        ["/nix/store/a-out", "/nix/store/b-bad"],
        ["/nix/store/a-out"],
        ["/nix/store/b-bad"],
    ]
    assert {(r.attr, r.success) for r in results} == {("a", True), ("b", False)}
    assert up.pushed == {"/nix/store/a-out"}

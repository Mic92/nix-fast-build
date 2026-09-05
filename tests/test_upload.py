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
    _chunk_args,
    parse_path_info,
    run_upload_worker,
)


def test_parse_path_info_nix_and_lix_formats() -> None:
    new = '{"/nix/store/a":{"narSize":1},"/nix/store/b":null}'
    lix = '[{"path":"/nix/store/a","valid":true},{"path":"/nix/store/b","valid":false}]'
    assert parse_path_info(new) == {"/nix/store/a"}
    assert parse_path_info(lix) == {"/nix/store/a"}


# `nix path-info --json`: X.drv^* -> X, "broken" paths are invalid (null).
FAKE_NIX = """#!/usr/bin/env bash
sep="{"
for p in "$@"; do
  [[ $p == /nix/store/* ]] || continue
  p=${p%^*}; p=${p%.drv}
  if [[ $p == *broken* ]]; then v=null; else v='{"narSize":1}'; fi
  printf '%s"%s":%s' "$sep" "$p" "$v"; sep=","
done
echo "}"
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
    (tmp_path / "nix").write_text(FAKE_NIX)
    (tmp_path / "nix").chmod(0o755)
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
                UploadItem("a", ["/nix/store/x-lib.drv"], final=False),
                UploadItem(
                    "b",
                    ["/nix/store/x-lib.drv", "/nix/store/y-broken.drv"],
                    final=False,
                ),
                UploadItem("a", ["/nix/store/a-out"]),
            ],
            [UploadItem("b", ["/nix/store/x-lib.drv", "/nix/store/b-out"])],
            # alias attr with only already-pushed paths still gets a result
            [UploadItem("a-alias", ["/nix/store/a-out"])],
        ],
    )
    assert up.calls == [
        ["/nix/store/a-out", "/nix/store/x-lib"],
        ["/nix/store/b-out"],
    ]
    assert [(r.attr, r.success) for r in results] == [
        ("a", True),
        ("b", True),
        ("a-alias", True),
    ]


def test_chunk_args_respects_limit() -> None:
    args = [f"/nix/store/{i:04}-p" for i in range(10)]
    chunks = _chunk_args(args, limit=50)
    assert [a for c in chunks for a in c] == args
    assert all(sum(len(a) + 1 for a in c) <= 50 for c in chunks)
    assert len(chunks) > 1
    assert _chunk_args(["x" * 100], limit=50) == [["x" * 100]]
    assert _chunk_args([], limit=50) == []


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

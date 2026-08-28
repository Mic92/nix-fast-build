import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

from nix_fast_build import Options
from nix_fast_build.build import Build

# substituted dep (108), built intermediate (105), built toplevel (105)
LOG = r"""
@nix {"action":"start","id":1,"level":4,"parent":0,"text":"querying info","type":109,"fields":["/nix/store/aaa-dep","https://cache.nixos.org"]}
@nix {"action":"stop","id":1}
@nix {"action":"start","id":2,"level":3,"parent":0,"text":"fetching dep","type":108,"fields":["/nix/store/aaa-dep","https://cache.nixos.org"]}
@nix {"action":"stop","id":2}
@nix {"action":"start","id":3,"level":3,"parent":0,"text":"building '/nix/store/bbb-wheel.drv'","type":105,"fields":["/nix/store/bbb-wheel.drv","",1,1]}
@nix {"action":"result","id":3,"type":101,"fields":["compiling"]}
@nix {"action":"stop","id":3}
@nix {"action":"start","id":4,"level":3,"parent":0,"text":"building '/nix/store/ccc-hello.drv'","type":105,"fields":["/nix/store/ccc-hello.drv","",1,1]}
@nix {"action":"stop","id":4}
""".strip()


def test_reports_built_intermediates(tmp_path: Path) -> None:
    log = tmp_path / "log"
    log.write_text(LOG + "\n")
    fake_nix = tmp_path / "nix"
    fake_nix.write_text(f"#!/usr/bin/env bash\ncat {log} >&2\n")
    fake_nix.chmod(0o755)
    built: list[str] = []

    async def go() -> int:
        build = Build(
            "hello", "/nix/store/ccc-hello.drv", {"out": "/nix/store/ccc-hello"}
        )
        async with AsyncExitStack() as stack:
            res = await build.build(
                stack,
                Options(nix_bin=[str(fake_nix)], build_gcroot_dir=tmp_path),
                on_built=built.append,
            )
            return res.return_code
        raise AssertionError

    assert asyncio.run(go()) == 0
    assert built == ["/nix/store/bbb-wheel.drv"]

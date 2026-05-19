import asyncio
import json
import os
import pwd
import xml.etree
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from nix_fast_build import Build, Options, async_main, parse_args

from .sshd import Sshd

TEST_ROOT = Path(__file__).parent.resolve()
FIXTURES = TEST_ROOT / "fixtures"


def cli(args: list[str]) -> int:
    return asyncio.run(async_main(args))


def test_help() -> None:
    with pytest.raises(SystemExit) as e:
        cli(["--help"])
    assert e.value.code == 0


def test_build() -> None:
    rc = cli(["--option", "builders", ""])
    assert rc == 0


def test_build_junit() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "test.xml"
        rc = cli(
            [
                "--option",
                "builders",
                "",
                "--result-format",
                "junit",
                "--result-file",
                str(path),
            ]
        )
        data = xml.etree.ElementTree.parse(path)  # noqa: S314
        assert data.getroot().tag == "testsuites"
        assert rc == 0


def test_build_json() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "test.json"
        rc = cli(["--option", "builders", "", "--result-file", str(path)])
        data = json.loads(path.read_text())
        assert len(data["results"]) > 0
        build_results = [r for r in data["results"] if r["type"] == "BUILD"]
        assert build_results
        for result in build_results:
            assert "outputs" in result
            assert result["outputs"]
        assert rc == 0


def test_reference_lock_file() -> None:
    """--reference-lock-file is forwarded to nix-eval-jobs."""
    # Pointing at the project's own lock file must behave like a normal build.
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--reference-lock-file",
            str(TEST_ROOT.parent / "flake.lock"),
        ]
    )
    assert rc == 0


def test_eval_error() -> None:
    rc = cli(["--option", "builders", "", "--flake", ".#legacyPackages"])
    assert rc == 1


def test_select_flake() -> None:
    """--select can filter out failing attributes so the build succeeds."""
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--flake",
            ".#legacyPackages",
            "--select",
            # legacyPackages.<system> only contains intentionally broken
            # packages; filtering them all out must yield a successful run.
            "systems: builtins.mapAttrs (_: _: { }) systems",
        ]
    )
    assert rc == 0


def test_select_expr() -> None:
    """--select is forwarded to nix-eval-jobs in non-flake mode."""
    rc = cli(
        [
            "--file",
            str(FIXTURES / "simple.nix"),
            "--select",
            "root: { inherit (root) hello; }",
            "--option",
            "builders",
            "",
        ]
    )
    assert rc == 0


def test_expr_build() -> None:
    """Non-flake mode: evaluate and build a simple Nix expression."""
    rc = cli(
        [
            "--file",
            str(FIXTURES / "simple.nix"),
            "--option",
            "builders",
            "",
        ]
    )
    assert rc == 0


def test_expr_build_with_attr() -> None:
    """Non-flake mode: evaluate a specific attribute with -A."""
    rc = cli(
        [
            "--file",
            str(FIXTURES / "simple.nix"),
            "-A",
            "hello",
            "--option",
            "builders",
            "",
        ]
    )
    assert rc == 0


def test_expr_remote_rejected() -> None:
    """--remote is not supported in non-flake mode."""
    with pytest.raises(SystemExit) as e:
        cli(
            [
                "--file",
                str(FIXTURES / "simple.nix"),
                "--remote",
                "somehost",
            ]
        )
    assert e.value.code == 2


def test_flake_mode_rejects_expr_flags() -> None:
    """Expr-specific flags like -A should be rejected in flake mode."""
    with pytest.raises(SystemExit) as e:
        cli(["-A", "hello"])
    assert e.value.code == 2


def test_flake_and_expr_mutually_exclusive() -> None:
    """--flake and --file cannot be used together."""
    with pytest.raises(SystemExit) as e:
        cli(["--flake", ".#checks", "--file", "default.nix"])
    assert e.value.code == 2


def test_remote(sshd: Sshd) -> None:
    login = pwd.getpwuid(os.getuid()).pw_name
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--remote",
            f"{login}@127.0.0.1",
            "--remote-ssh-option",
            "Port",
            str(sshd.port),
            "--remote-ssh-option",
            "IdentityFile",
            sshd.key,
            "--remote-ssh-option",
            "StrictHostKeyChecking",
            "no",
            "--remote-ssh-option",
            "UserKnownHostsFile",
            "/dev/null",
        ]
    )
    assert rc == 0


def test_store_args_property() -> None:
    """store_args must always pin --eval-store auto so nix build finds the
    locally-evaluated .drv instead of re-evaluating against the remote store."""
    opts = Options(store="ssh-ng://example.org")
    assert opts.store_args == [
        "--eval-store",
        "auto",
        "--store",
        "ssh-ng://example.org",
    ]


def test_store_args_property_none() -> None:
    """No store flag set means no extra args."""
    opts = Options(store=None)
    assert opts.store_args == []


def test_store_implies_no_link() -> None:
    """--store implies --no-link: outputs land in the remote store, so a
    local out-link symlink would dangle."""
    opts = asyncio.run(parse_args(["--store", "ssh-ng://example.org"]))
    assert opts.store == "ssh-ng://example.org"
    assert opts.no_link is True


def test_store_remote_rejected() -> None:
    """--store and --remote are different remote-build mechanisms."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--remote", "y"])
    assert e.value.code == 2


def test_store_copy_to_rejected() -> None:
    """nix copy reads from the local store; --store outputs aren't there."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--copy-to", "file:///tmp/cache"])
    assert e.value.code == 2


def test_store_cachix_rejected() -> None:
    """cachix reads from the local store; --store outputs aren't there."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--cachix-cache", "mycache"])
    assert e.value.code == 2


def test_store_attic_rejected() -> None:
    """attic reads from the local store; --store outputs aren't there."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--attic-cache", "mycache"])
    assert e.value.code == 2


def test_store_niks3_rejected() -> None:
    """niks3 reads from the local store; --store outputs aren't there."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--niks3-server", "https://x"])
    assert e.value.code == 2


def test_store_out_link_rejected() -> None:
    """A result symlink would dangle: outputs live in the remote store."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--out-link", "build-result"])
    assert e.value.code == 2


def test_store_option_store_rejected() -> None:
    """--option store and --store are ambiguous about which value wins."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--option", "store", "ssh-ng://y"])
    assert e.value.code == 2


def test_store_option_eval_store_rejected() -> None:
    """--option eval-store conflicts with the auto eval-store --store implies."""
    with pytest.raises(SystemExit) as e:
        cli(["--store", "ssh-ng://x", "--option", "eval-store", "local"])
    assert e.value.code == 2


def test_store_unusable_fails_build() -> None:
    """An unusable --store must cause builds to fail.

    Proves --store actually reaches nix build, not just argparse: nix
    rejects an unknown store scheme synchronously when opening the store,
    so this fails fast and without a real remote.
    """
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--store",
            "totally-bogus-scheme://x",
        ]
    )
    assert rc != 0


def test_store_args_reach_get_build_log() -> None:
    """The build log lives in the build store; nix log must be pointed there.

    nix log against an unknown store scheme errors with a message naming the
    scheme, while nix log against the local store for a missing path errors
    with "build log ... is not available". The scheme name in the returned
    log proves nix log was invoked with --store.
    """
    opts = Options(store="totally-bogus-scheme://x")
    b = Build(
        attr="x",
        drv_path="/nix/store/" + "a" * 32 + "-fake.drv",
        outputs={},
    )
    log = asyncio.run(b.get_build_log(opts))
    assert "totally-bogus-scheme" in log


def test_store_ssh_ng(sshd: Sshd, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end build against an ssh-ng:// store.

    Unlike --remote (ssh shell exec), --store ssh-ng:// streams the daemon
    protocol over an SSH channel. ssh-ng:// has no per-store SSH option
    flags, so port/key go through NIX_SSHOPTS like nix itself documents.
    """
    login = pwd.getpwuid(os.getuid()).pw_name
    monkeypatch.setenv(
        "NIX_SSHOPTS",
        f"-p {sshd.port} -i {sshd.key} "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
    )
    rc = cli(
        [
            "--option",
            "builders",
            "",
            "--store",
            f"ssh-ng://{login}@127.0.0.1",
        ]
    )
    assert rc == 0

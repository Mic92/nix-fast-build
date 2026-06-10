import argparse
import asyncio
import enum
import json
import multiprocessing
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .errors import Error


def _nix_command(nix_bin: list[str], args: list[str]) -> list[str]:
    return [*nix_bin, "--experimental-features", "nix-command flakes", *args]


class ResultFormat(enum.Enum):
    JSON = enum.auto()
    JUNIT = enum.auto()


class EvalMode(enum.Enum):
    FLAKE = enum.auto()
    EXPR = enum.auto()


@dataclass
class Options:
    nix_bin: list[str] = field(default_factory=lambda: ["nix"])
    nix_eval_jobs_bin: list[str] = field(default_factory=lambda: ["nix-eval-jobs"])
    nix_build_bin: list[str] = field(default_factory=lambda: ["nix-build"])
    eval_mode: EvalMode = EvalMode.FLAKE
    flake_url: str = ""
    flake_fragment: str = ""
    expr_file: str = ""
    expr_attr: str = ""
    expr_args: list[str] = field(default_factory=list)
    impure: bool = False
    options: list[str] = field(default_factory=list)
    store: str | None = None
    remote: str | None = None
    remote_ssh_options: list[str] = field(default_factory=list)
    always_upload_source: bool = False
    systems: set[str] = field(default_factory=set)
    eval_max_memory_size: int = 4096
    skip_cached: bool = False
    eval_workers: int = field(default_factory=multiprocessing.cpu_count)
    max_jobs: int = 0
    retries: int = 0
    debug: bool = False
    copy_to: str | None = None
    nom: bool = True
    download: bool = True
    no_link: bool = False
    out_link: str = "result"
    result_format: ResultFormat = ResultFormat.JSON
    result_file: Path | None = None
    override_inputs: list[list[str]] = field(default_factory=list)
    select_expr: str | None = None
    fail_fast: bool = False

    reference_lock_file: str | None = None

    _stop_event: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )

    def signal_stop(self) -> None:
        """Signal all tasks to stop (used by --fail-fast)."""
        if self.fail_fast:
            self._stop_event.set()

    @property
    def should_stop(self) -> bool:
        return self.fail_fast and self._stop_event.is_set()

    cachix_cache: str | None = None

    attic_cache: str | None = None
    attic_ignore_upstream_cache_filter: bool = False
    attic_push_build_closure: bool = False

    niks3_server: str | None = None

    def nix_command(self, args: list[str]) -> list[str]:
        return _nix_command(self.nix_bin, args)

    @property
    def remote_url(self) -> None | str:
        if self.remote is None:
            return None
        return f"ssh://{self.remote}"

    @property
    def store_args(self) -> list[str]:
        """Extra args to point nix build/log at the build store."""
        if self.store is None:
            return []
        # --eval-store auto ensures nix finds locally-evaluated .drvs
        return ["--eval-store", "auto", "--store", self.store]

    @property
    def display_name(self) -> str:
        """Human-readable name for the evaluation target, used in reports."""
        if self.eval_mode == EvalMode.FLAKE:
            return f"{self.flake_url}#{self.flake_fragment}"
        name = self.expr_file
        if self.expr_attr:
            name += f".{self.expr_attr}"
        return name


class ResultType(enum.Enum):
    EVAL = enum.auto()
    BUILD = enum.auto()
    UPLOAD = enum.auto()
    DOWNLOAD = enum.auto()
    CACHIX = enum.auto()
    ATTIC = enum.auto()
    NIKS3 = enum.auto()


@dataclass
class Result:
    result_type: ResultType
    attr: str
    success: bool
    duration: float
    error: str | None
    log_output: str | None = None
    outputs: dict[str, str] | None = None


def _maybe_remote(
    cmd: list[str], remote: str | None, remote_ssh_options: list[str]
) -> list[str]:
    if remote:
        return ["ssh", remote, *remote_ssh_options, "--", shlex.join(cmd)]
    return cmd


def maybe_remote(cmd: list[str], opts: Options) -> list[str]:
    return _maybe_remote(cmd, opts.remote, opts.remote_ssh_options)


def nix_shell(fallback_package: str, wanted_command: str) -> list[str]:
    bash_cmd = 'pkg=$1; shift; cmd=("$@"); if command -v "${cmd[0]}" >/dev/null; then exec "${cmd[@]}"; else exec nix --experimental-features "nix-command flakes" shell "$pkg" -c "${cmd[@]}"; fi'
    return ["bash", "-c", bash_cmd, "bash", fallback_package, wanted_command]


async def get_nix_config(
    nix_bin: list[str], remote: str | None, remote_ssh_options: list[str]
) -> dict[str, str]:
    args = _maybe_remote(
        _nix_command(nix_bin, ["config", "show", "--json"]), remote, remote_ssh_options
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as e:
        msg = f"nix not found in PATH, try to run {shlex.join(args)}"
        raise Error(msg) from e

    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        msg = f"Failed to get nix config, {shlex.join(args)} exited with {proc.returncode}"
        raise Error(msg)
    data = json.loads(stdout)

    config = {}
    for key, value in data.items():
        config[key] = value["value"]
    return config


async def parse_args(args: list[str]) -> Options:
    parser = argparse.ArgumentParser()

    # Binary paths (env vars as fallback for backward compatibility)
    parser.add_argument(
        "--nix",
        default=os.environ.get("NIX_FAST_BUILD_NIX", "nix"),
        help="Path to the nix binary (default: $NIX_FAST_BUILD_NIX or 'nix')",
    )
    parser.add_argument(
        "--nix-eval-jobs",
        default=os.environ.get("NIX_FAST_BUILD_EVAL_JOBS", "nix-eval-jobs"),
        help="Path to the nix-eval-jobs binary (default: $NIX_FAST_BUILD_EVAL_JOBS or 'nix-eval-jobs')",
    )
    parser.add_argument(
        "--nix-build",
        default=os.environ.get("NIX_FAST_BUILD_NIX_BUILD", "nix-build"),
        help="Path to the nix-build binary (default: $NIX_FAST_BUILD_NIX_BUILD or 'nix-build')",
    )

    # Evaluation mode: --flake (default) or --file (non-flake)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "-f",
        "--flake",
        default=None,
        help="Flake url to evaluate/build (default: .#checks)",
    )
    eval_group.add_argument(
        "--file",
        nargs="?",
        const="default.nix",
        default=None,
        help="Nix expression file to evaluate (default: default.nix). Mutually exclusive with --flake.",
    )

    # Non-flake specific options
    parser.add_argument(
        "-A",
        "--attr",
        default="",
        help="Attribute path to evaluate in non-flake mode (like nix-build -A)",
    )
    parser.add_argument(
        "--arg",
        action="append",
        nargs=2,
        metavar=("name", "value"),
        default=[],
        help="Pass the value expr as the argument name to Nix functions (non-flake mode)",
    )
    parser.add_argument(
        "--argstr",
        action="append",
        nargs=2,
        metavar=("name", "value"),
        default=[],
        help="Pass the string as the argument name to Nix functions (non-flake mode)",
    )
    parser.add_argument(
        "-I",
        "--include",
        action="append",
        default=[],
        help="Add path to the Nix search path (non-flake mode)",
    )
    parser.add_argument(
        "--impure",
        action="store_true",
        default=False,
        help="Allow impure expressions (default in --file mode)",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        default=False,
        help="Enforce pure evaluation in --file mode (overrides the default impure behavior)",
    )

    parser.add_argument(
        "-j",
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum number of build jobs to run in parallel (0 for unlimited)",
    )
    parser.add_argument(
        "--option",
        help="Nix option to set",
        action="append",
        nargs=2,
        metavar=("name", "value"),
        default=[],
    )
    parser.add_argument(
        "--remote-ssh-option",
        help="ssh option when accessing remote",
        action="append",
        nargs=2,
        metavar=("name", "value"),
        default=[],
    )
    parser.add_argument(
        "--cachix-cache",
        help="Cachix cache to upload to",
        default=None,
    )
    parser.add_argument(
        "--attic-cache",
        help="Attic cache to upload to",
        default=None,
    )
    parser.add_argument(
        "--attic-ignore-upstream-cache-filter",
        help="Pass --ignore-upstream-cache-filter to attic push, uploading all paths even if attic thinks they exist in an upstream cache (default: false)",
        action="store_true",
    )
    parser.add_argument(
        "--attic-push-build-closure",
        help="Also push the build-time closure to attic, not just the runtime closure. Useful for caching intermediate build products in ephemeral CI environments (default: false)",
        action="store_true",
    )
    parser.add_argument(
        "--niks3-server",
        help="Niks3 server URL to upload to (auth from ~/.config/niks3/auth-token or NIKS3_AUTH_TOKEN_FILE)",
        default=None,
    )
    parser.add_argument(
        "--no-nom",
        help="Don't use nix-output-monitor to print build output (default: false)",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--systems",
        help="Space-separated list of systems to build for (default: current system)",
        default=None,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of times to retry failed builds",
    )
    parser.add_argument(
        "--no-link",
        help="Do not create an out-link for builds (default: false)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--out-link",
        help="Name of the out-link for builds (default: result)",
        default="result",
    )
    parser.add_argument(
        "--store",
        type=str,
        help="Nix store URL to build against (e.g. ssh-ng://host). "
        "Evaluation stays local; only builds are dispatched. Implies --no-link.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        help="Remote machine to build on",
    )
    parser.add_argument(
        "--always-upload-source",
        help="Always upload sources to remote machine. This is needed if the remote machine cannot access all sources (default: false)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-download",
        help="Do not download build results from remote machine",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip-cached",
        help="Skip builds that are already present in the binary cache (default: false)",
        action="store_true",
    )
    parser.add_argument(
        "--copy-to",
        help="Copy build results to the given path (passed to nix copy, i.e. file:///tmp/cache?compression=none)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug logging output",
    )
    parser.add_argument(
        "--eval-max-memory-size",
        type=int,
        default=4096,
        help="Maximum memory size for nix-eval-jobs (in MiB) per worker. After the limit is reached, the worker is restarted.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of evaluation threads spawned",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="File to write build results to",
    )
    parser.add_argument(
        "--result-format",
        choices=["json", "junit"],
        default="json",
        help="Format of the build result file",
    )
    parser.add_argument(
        "--override-input",
        action="append",
        nargs=2,
        metavar=("input_path", "flake_url"),
        help="Override a specific flake input (e.g. `dwarffs/nixpkgs`).",
    )
    parser.add_argument(
        "--select",
        metavar="NIX_FUNCTION",
        default=None,
        help=(
            "Nix function applied to the evaluation root to filter or "
            "transform the set of attributes to build (passed through to "
            "nix-eval-jobs --select). The function receives the attribute "
            "selected by --flake/-A. "
            "Example: 'checks: builtins.removeAttrs checks [\"slow-test\"]'"
        ),
    )
    parser.add_argument(
        "--reference-lock-file",
        type=str,
        default=None,
        help="Read the given lock file instead of `flake.lock` within the top-level flake.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop as soon as any build or evaluation fails, instead of continuing with remaining builds.",
    )

    a = parser.parse_args(args)

    # Determine evaluation mode
    eval_mode = EvalMode.EXPR if a.file is not None else EvalMode.FLAKE

    # Validate: --store conflicts — outputs stay in the remote store,
    # so local-store features and --remote cannot work alongside it.
    if a.store:
        conflicts = [
            (a.remote, "--remote"),
            (a.copy_to, "--copy-to"),
            (a.cachix_cache, "--cachix-cache"),
            (a.attic_cache, "--attic-cache"),
            (a.niks3_server, "--niks3-server"),
            (a.out_link != "result", "--out-link"),
        ]
        for value, flag in conflicts:
            if value:
                parser.error(f"{flag} cannot be used with --store")
        if any(name in ("store", "eval-store") for name, _ in a.option):
            parser.error("--option store/eval-store conflicts with --store")

    # Validate: --remote is not supported in non-flake mode
    if eval_mode == EvalMode.EXPR and a.remote:
        parser.error("--remote is not supported in non-flake (--file) mode")

    # Validate: --override-input only makes sense for flakes
    if eval_mode == EvalMode.EXPR and a.override_input:
        parser.error("--override-input is not supported in non-flake (--file) mode")

    # Validate: --always-upload-source only makes sense for flakes
    if eval_mode == EvalMode.EXPR and a.always_upload_source:
        parser.error(
            "--always-upload-source is not supported in non-flake (--file) mode"
        )

    # Validate: non-flake-specific flags should not be used with --flake
    if eval_mode == EvalMode.FLAKE:
        if a.attr:
            parser.error("-A/--attr is only supported in non-flake (--file) mode")
        if a.arg:
            parser.error("--arg is only supported in non-flake (--file) mode")
        if a.argstr:
            parser.error("--argstr is only supported in non-flake (--file) mode")
        if a.include:
            parser.error("-I/--include is only supported in non-flake (--file) mode")
        if a.pure:
            parser.error("--pure is only supported in non-flake (--file) mode")

    # In --file mode, default to impure unless --pure is explicitly given
    impure = not a.pure if eval_mode == EvalMode.EXPR else a.impure

    # Parse flake mode settings
    flake_url = ""
    flake_fragment = ""
    if eval_mode == EvalMode.FLAKE:
        flake_spec = a.flake if a.flake is not None else ".#checks"
        flake_parts = flake_spec.split("#")
        flake_url = flake_parts[0]
        if len(flake_parts) == 2:
            flake_fragment = flake_parts[1]

    # Parse expr mode settings
    expr_file = ""
    expr_attr = ""
    expr_args: list[str] = []
    if eval_mode == EvalMode.EXPR:
        expr_file = a.file
        expr_attr = a.attr
        for name, value in a.arg:
            expr_args.extend(["--arg", name, value])
        for name, value in a.argstr:
            expr_args.extend(["--argstr", name, value])
        for path in a.include:
            expr_args.extend(["--include", path])

    options = []
    for name, value in a.option:
        options.extend(["--option", name, value])
    remote_ssh_options = []
    for name, value in a.remote_ssh_option:
        remote_ssh_options.extend(["-o", f"{name}={value}"])

    nix_bin = shlex.split(a.nix)
    nix_eval_jobs_bin = shlex.split(a.nix_eval_jobs)
    nix_build_bin = shlex.split(a.nix_build)

    nix_config = await get_nix_config(nix_bin, a.remote, remote_ssh_options)
    if a.max_jobs is None:
        a.max_jobs = int(nix_config.get("max-jobs", 0))
    if a.no_nom is None:
        if a.remote:
            # only if we have an official binary cache, otherwise we need to build ghc...
            a.no_nom = nix_config.get("system", "") not in [
                "aarch64-darwin",
                "x86_64-darwin",
                "aarch64-linux",
                "x86_64-linux",
            ]
        else:
            a.no_nom = shutil.which("nom") is None
    if a.systems is None:
        systems = {nix_config.get("system", "")}
    else:
        systems = set(a.systems.split(" "))

    return Options(
        nix_bin=nix_bin,
        nix_eval_jobs_bin=nix_eval_jobs_bin,
        nix_build_bin=nix_build_bin,
        eval_mode=eval_mode,
        flake_url=flake_url,
        flake_fragment=flake_fragment,
        expr_file=expr_file,
        expr_attr=expr_attr,
        expr_args=expr_args,
        impure=impure,
        always_upload_source=a.always_upload_source,
        remote=a.remote,
        skip_cached=a.skip_cached,
        options=options,
        remote_ssh_options=remote_ssh_options,
        max_jobs=a.max_jobs,
        nom=not a.no_nom,
        download=not a.no_download,
        debug=a.debug,
        systems=systems,
        eval_max_memory_size=a.eval_max_memory_size,
        eval_workers=a.eval_workers,
        copy_to=a.copy_to,
        cachix_cache=a.cachix_cache,
        attic_cache=a.attic_cache,
        attic_ignore_upstream_cache_filter=a.attic_ignore_upstream_cache_filter,
        attic_push_build_closure=a.attic_push_build_closure,
        niks3_server=a.niks3_server,
        store=a.store,
        no_link=a.no_link or bool(a.store),
        out_link=a.out_link,
        result_format=ResultFormat[a.result_format.upper()],
        result_file=a.result_file,
        override_inputs=a.override_input or [],
        select_expr=a.select,
        reference_lock_file=a.reference_lock_file,
        fail_fast=a.fail_fast,
    )

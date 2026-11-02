import re
import shutil
import subprocess

from nix_fast_build import get_argument_parser

# Explicit inventory of nix build flags intentionally unsupported by nix-fast-build,
# along with clear justifications.
EXPLICITLY_UNSUPPORTED_NIX_BUILD_FLAGS: dict[str, str] = {
    "--stdin": "Installables are evaluated via flake or expression, not read from stdin.",
    "--profile": "nix-fast-build roots outputs temporarily during the run, optionally creating symlinks via --out-link.",
    "--json": "nix-fast-build has its own JSON streaming and result reporting (--result-format json, --stream-json-lines).",
    "--pretty": "nix-fast-build handles its own output rendering.",
    "--no-pretty": "nix-fast-build handles its own output rendering.",
    "--print-out-paths": "nix-fast-build manages and reports build output paths directly.",
    "--debugger": "Interactive debugger is incompatible with concurrent background worker pools.",
    "--print-build-logs": "Build logs are captured and rendered via nix-fast-build's TTY / CI renderers.",
    "--commit-lock-file": "Flake lockfile operations are handled during evaluation or managed via --option.",
    "--no-update-lock-file": "Can be passed via --option or managed separately.",
    "--no-write-lock-file": "Can be passed via --option or managed separately.",
    "--output-lock-file": "Managed via --reference-lock-file or nix options.",
    "--update-input": "Deprecated in upstream nix in favor of 'nix flake update'.",
    "--inputs-from": "Flake registry entries are configured via flake files or nix options.",
    "--no-use-registries": "Can be set via --option.",
    "--no-registries": "Deprecated upstream in favor of --no-use-registries.",
    "--recreate-lock-file": "Deprecated upstream in favor of 'nix flake update'.",
    "--override-flake": "Handled via nix configuration or lockfile references.",
    "--arg-from-file": "Specific nix build helper, can be passed using Nix expressions or file import.",
    "--arg-from-stdin": "Interactive/stdin evaluation helpers are not used in batch builds.",
    "--eval-store": "Handled automatically or mutually exclusive with --store.",
    "--expr": "nix-fast-build evaluates files/flakes, or expressions via --select.",
    "--version": "nix-fast-build is a separate program with its own versioning.",
    "--quiet": "Verbosity is managed by nix-fast-build renderers.",
    "--verbose": "Verbosity is managed by nix-fast-build renderers.",
    "--log-format": "nix-fast-build manages log formats for its subprocesses internally.",
}


def _extract_nix_build_flags() -> set[str]:
    """Extract flags from nix build --help."""
    nix_bin = shutil.which("nix")
    if not nix_bin:
        return set()
    out = subprocess.check_output([nix_bin, "build", "--help"], text=True)
    return set(re.findall(r"(?:^|\s)(--[a-z0-9-]+)", out))


def _extract_nix_fast_build_flags() -> set[str]:
    """Extract all long options registered in nix-fast-build argument parser."""
    parser = get_argument_parser()
    flags = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt)
    return flags


def test_exhaustive_nix_build_flags() -> None:
    """Verify that all flags from `nix build` are either supported by nix-fast-build
    or explicitly listed in EXPLICITLY_UNSUPPORTED_NIX_BUILD_FLAGS with an explanation.
    """
    nix_flags = _extract_nix_build_flags()
    assert nix_flags, "Failed to extract flags from 'nix build --help'"

    fast_build_flags = _extract_nix_fast_build_flags()
    unsupported_flags = set(EXPLICITLY_UNSUPPORTED_NIX_BUILD_FLAGS.keys())

    accounted_for = fast_build_flags | unsupported_flags
    unhandled = nix_flags - accounted_for

    assert not unhandled, (
        f"Found unhandled flags from `nix build`: {sorted(unhandled)}. "
        "Please either implement them in nix_fast_build/options.py or "
        "document why they are unsupported in EXPLICITLY_UNSUPPORTED_NIX_BUILD_FLAGS."
    )

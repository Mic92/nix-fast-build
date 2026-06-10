import json
import logging
import os
import shlex
import subprocess
from typing import Any

from .errors import Error
from .options import Options

logger = logging.getLogger(__name__)


def nix_flake_metadata(opts: Options) -> dict[str, Any]:
    cmd = opts.nix_command(
        [
            "flake",
            "metadata",
            "--json",
            opts.flake_url,
        ]
    )
    logger.info(f"run {shlex.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    if proc.returncode != 0:
        msg = (
            f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
        )
        raise Error(msg)

    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, OSError) as e:
        msg = f"failed to parse output of {shlex.join(cmd)}: {e}\nGot: {proc.stdout.decode('utf-8', 'replace')}"
        raise Error(msg) from e
    return data


def is_path_input(node: dict[str, dict[str, str]]) -> bool:
    locked = node.get("locked")
    if not locked:
        return False
    return locked["type"] == "path" or locked.get("url", "").startswith("file://")


def check_for_path_inputs(data: dict[str, Any]) -> bool:
    return any(is_path_input(node) for node in data["locks"]["nodes"].values())


def upload_sources(opts: Options) -> str:
    if not opts.always_upload_source:
        flake_data = nix_flake_metadata(opts)
        url = flake_data["resolvedUrl"]
        has_path_inputs = check_for_path_inputs(flake_data)
        if not has_path_inputs and not is_path_input(flake_data):
            # No need to upload sources, we can just build the flake url directly
            # FIXME: this might fail for private repositories?
            return url
        if not has_path_inputs:
            # Just copy the flake to the remote machine, we can substitute other inputs there.
            path = flake_data["path"]
            env = os.environ.copy()
            env["NIX_SSHOPTS"] = " ".join(opts.remote_ssh_options)
            assert opts.remote_url
            cmd = opts.nix_command(
                [
                    "copy",
                    "--to",
                    opts.remote_url,
                    "--no-check-sigs",
                    path,
                ]
            )
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, env=env, check=False)
            if proc.returncode != 0:
                msg = f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
                raise Error(msg)
            return path

    # Slow path: we need to upload all sources to the remote machine
    assert opts.remote_url
    cmd = opts.nix_command(
        [
            "flake",
            "archive",
            "--to",
            opts.remote_url,
            "--json",
            opts.flake_url,
        ]
    )
    print("run " + shlex.join(cmd))
    logger.info("run %s", shlex.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        msg = (
            f"failed to upload sources: {shlex.join(cmd)} failed with {proc.returncode}"
        )
        raise Error(msg)
    try:
        return json.loads(proc.stdout)["path"]
    except (json.JSONDecodeError, OSError) as e:
        msg = f"failed to parse output of {shlex.join(cmd)}: {e}\nGot: {proc.stdout.decode('utf-8', 'replace')}"
        raise Error(msg) from e

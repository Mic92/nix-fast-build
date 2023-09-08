# nix-ci-build

nix-eval-jobs + nix-output-monitor = ♥

Evaluate and build nix packages in parallel.
Under the hood uses output of nix-eval-jobs to flake attributes evaluate in parallel
and spawns a nix-build for each flake attribute.
Finally it uses the nix-output-monitor to monitor the output.

Usage:

```
$ nix-ci-build
```

This will in parallel evaluate and build `.#checks.$currentSystem`

## Reference

```
$ nix-ci-build --help
usage: nix-ci-build [-h] [-f FLAKE] [-j MAX_JOBS] [--option OPTION] [--systems SYSTEMS] [--retries RETRIES]
                    [--verbose]

options:
  -h, --help            show this help message and exit
  -f FLAKE, --flake FLAKE
                        Flake url to evaluate/build (default: .#checks
  -j MAX_JOBS, --max-jobs MAX_JOBS
                        Maximum number of build jobs to run in parallel (0 for unlimited)
  --option OPTION       Nix option to set
  --systems SYSTEMS     Comma-separated list of systems to build for (default: current system)
  --retries RETRIES     Number of times to retry failed builds
  --verbose             Print verbose output
```

## Installation

```
$ nix run github:Mic92/nix-ci-build
```

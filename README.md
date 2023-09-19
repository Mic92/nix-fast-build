# nix-fast-build

Combine the power of `nix-eval-jobs` with `nix-output-monitor` to speed-up your ci evaluation and building process.
(formally known as nix-ci-build)

## Why `nix-fast-build`?

**Problem**: Evaluating and building big flakes i.e. with numerous NixOS
machines can be painfully slow.

**Our Solution**: `nix-fast-build` offers a seamless experience by evaluating and
building your nix packages concurrently, drastically reducing the overall time.

## How Does It Work?

Under the hood:

1. It leverages the output from `nix-eval-jobs` to evaluate flake attributes in
   parallel.
2. For each flake attribute, a separate `nix-build` process is spawned.
3. Lastly, `nix-output-monitor` to show the build progress nicely.

## Usage

To get started, simply run:

```console
$ nix-fast-build
```

or:

```
$ nix run github:Mic92/nix-fast-build
```

This command will concurrently evaluate and build the attributes
`.#checks.$currentSystem`.

---

Enjoy faster and more efficient NixOS builds with `nix-fast-build`!

## Reference

```console
usage: nix-fast-build [-h] [-f FLAKE] [-j MAX_JOBS] [--option name value] [--no-nom] [--systems SYSTEMS]
                    [--retries RETRIES] [--remote REMOTE] [--always-upload-source] [--no-download]
                    [--skip-cached] [--copy-to COPY_TO] [--verbose]
                    [--eval-max-memory-size EVAL_MAX_MEMORY_SIZE] [--eval-workers EVAL_WORKERS]

options:
  -h, --help            show this help message and exit
  -f FLAKE, --flake FLAKE
                        Flake url to evaluate/build (default: .#checks
  -j MAX_JOBS, --max-jobs MAX_JOBS
                        Maximum number of build jobs to run in parallel (0 for unlimited)
  --option name value   Nix option to set
  --no-nom              Use nix-output-monitor to print build output (default: false)
  --systems SYSTEMS     Comma-separated list of systems to build for (default: current system)
  --retries RETRIES     Number of times to retry failed builds
  --remote REMOTE       Remote machine to build on
  --always-upload-source
                        Always upload sources to remote machine. This is needed if the remote machine cannot
                        access all sources (default: false)
  --no-download         Do not download build results from remote machine
  --skip-cached         Skip builds that are already present in the binary cache (default: false)
  --copy-to COPY_TO     Copy build results to the given path (passed to nix copy, i.e.
                        file:///tmp/cache?compression=none)
  --verbose             Print verbose output
  --eval-max-memory-size EVAL_MAX_MEMORY_SIZE
                        Maximum memory size for nix-eval-jobs (in MiB) per worker. After the limit is
                        reached, the worker is restarted.
  --eval-workers EVAL_WORKERS
                        Number of evaluation threads spawned
```

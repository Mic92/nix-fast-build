# nix-fast-build 🚀 (previously known as nix-ci-build)

Combine the power of `nix-eval-jobs` with `nix-output-monitor` to speed-up your
evaluation and building process. `nix-fast-build` an also integrates with remote
machines by uploading the current flake, performing the evaluation/build
remotely, and then transferring the resultant store paths back to you.

## Why `nix-fast-build`?

**Problem**: Evaluating and building big flakes i.e. with numerous NixOS
machines can be painfully slow. For instance, rebuilding the already-compiled
[disko integration test suite](https://github.com/nix-community/disko) demands
1:50 minutes on an AMD Ryzen 9 7950X3D. But, it only takes a
[10 seconds](https://github.com/Mic92/nix-fast-build/issues/1) with
`nix-fast-build`.

**Solution**: `nix-fast-build` makes builds faster by evaluating and building
building your nix packages concurrently, reducing the overall time.

## How Does It Work?

Under the hood:

1. It leverages the output from `nix-eval-jobs` to evaluate flake attributes in
   parallel.
2. As soon as attributes complete evaluation, `nix-fast-build` initiates their
   build, even if the overall evaluation is ongoing.
3. Lastly, `nix-output-monitor` to show the build progress nicely.
4. (Optional) Once a build finishes, `nix-fast-build` can initiate its upload to
   a designated remote binary cache.

## Usage

To get started, run:

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

## Remote building

When leveraging the remote-builder protocol, uploading pre-built paths or
sources from the local machine can often turn into a bottleneck.
`nix-fast-build` does not use the remote-builder protocol. Instead it uploads
only the flake and executes all evaluation/build operations on the remote end.
At the end `nix-fast-build` will download the finished builds to the local
machine while not having to download all build dependencies in between.

Here is how to use it:

```
nix run github:Mic92/nix-ci-build -- --remote youruser@yoursshhostname
```

Replace `youruser@yoursshhostname` with your SSH login credentials for the
target machine. Please note that as of now, you must be recognized as a trusted
user on the remote endpoint to access this feature.

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

# nix-fast-build 🚀

Combine the power of `nix-eval-jobs` with `nix-output-monitor` to speed-up your
evaluation and building process. `nix-fast-build` works with both flakes and
plain Nix expressions, and can integrate with remote machines by uploading the
current flake, performing the evaluation/build remotely, and then transferring
the resultant store paths back to you.

## Why `nix-fast-build`?

**Problem**: Evaluating and building big flakes i.e. with numerous NixOS
machines can be painfully slow. For instance, rebuilding the already-compiled
[disko integration test suite](https://github.com/nix-community/disko) demands
1:50 minutes on an AMD Ryzen 9 7950X3D. But, it only takes a
[10 seconds](https://github.com/Mic92/nix-fast-build/issues/1) with
`nix-fast-build`.

**Solution**: `nix-fast-build` makes builds faster by evaluating and building
your nix packages concurrently, reducing the overall time.

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

This command will concurrently evaluate all systems in `.#checks` and build the
attributes in `.#checks.$currentSystem`.

---

Enjoy faster and more efficient NixOS builds with `nix-fast-build`!

## Building non-flake Nix expressions

`nix-fast-build` also works with plain Nix expressions (no flake required) via
the `--file` flag:

```console
$ nix-fast-build --file ./release.nix
```

If `--file` is passed without a file argument, it defaults to `default.nix`.

Use `-A` to select a specific attribute, just like `nix-build -A`:

```console
$ nix-fast-build --file ./release.nix -A mypackage
```

You can pass arguments to the Nix expression with `--arg` and `--argstr`, and
extend the Nix search path with `-I`:

```console
$ nix-fast-build --file ./release.nix --argstr system x86_64-linux
$ nix-fast-build --file ./ci.nix -I nixpkgs=flake:nixpkgs
```

Evaluation in `--file` mode is impure by default (allowing `builtins.getFlake`,
`<nixpkgs>`, etc.). Use `--pure` to enforce strict pure evaluation.

**Note:** Remote building (`--remote`) is not supported in `--file` mode, since
there is no flake archive mechanism to upload arbitrary Nix expressions to a
remote machine.

## Remote building

When leveraging the remote-builder protocol, uploading pre-built paths or
sources from the local machine can often turn into a bottleneck.
`nix-fast-build` does not use the remote-builder protocol. Instead it uploads
only the flake and executes all evaluation/build operations on the remote end.
At the end `nix-fast-build` will download the finished builds to the local
machine while not having to download all build dependencies in between.

Here is how to use it:

```
nix run github:Mic92/nix-fast-build -- --remote youruser@yoursshhostname
```

Replace `youruser@yoursshhostname` with your SSH login credentials for the
target machine. Please note that as of now, you must be recognized as a trusted
user on the remote endpoint to access this feature.

## CI-Friendly Output

By default, `Nix-output-monitor` (abbreviated as `nom`) updates its output every
0.5 seconds. In standard terminal environments, this frequent update is
unnoticeable, as `nom` erases the previous output before displaying the new one.
However, in Continuous Integration (CI) systems, each update appears as a
separate line of output.

To make output more concise for CI environments, use the `--no-nom` flag. This
replaces `nom` with a streamlined status reporter, which updates only when
there's a change in the number of pending builds, uploads, or downloads.

## CI Job Summaries

When running in GitHub Actions, Gitea Actions, or Forgejo Actions,
nix-fast-build automatically generates job summaries that appear directly in the
CI UI. The summary includes:

- Overall build status with success/failure counts
- Detailed sections for each failed build with build logs
- Collapsible sections showing successful operations (builds, evaluations,
  uploads, etc.)

This feature is automatically enabled when one of the following environment
variables is available:

- `GITHUB_STEP_SUMMARY` (GitHub Actions)
- `GITEA_STEP_SUMMARY` (Gitea Actions)
- `FORGEJO_STEP_SUMMARY` (Forgejo Actions)

**Note:** Job summaries are fully supported in GitHub Actions. Gitea and Forgejo
set the environment variables but do not yet display the summary content in the
UI (see [gitea#23721](https://github.com/go-gitea/gitea/issues/23721)). The
summary file is still written, so it will work once support is added upstream.

Example workflow:

```yaml
- name: Build with nix-fast-build
  run: nix-fast-build --no-nom --skip-cached
```

Build logs for failed packages are retrieved using `nix log` and displayed in
collapsible sections within the summary. Very long logs are automatically
truncated to the last 100 lines.

## Avoiding Redundant Package Downloads

By default, `nix build` will download pre-built packages, leading to needless
downloads even when there are no changes to any package. This can be especially
burdensome for CI environments without a persistent Nix store, such as GitHub
Actions.

To optimize this, use the `--skip-cached` flag with `nix-fast-build`. This
ensures that only those packages missing from the binary caches will be built.

## Specifying Build Systems

By default, `nix-fast-build` evaluates all architectures but only initiates
builds for the current system. You can modify this behavior with the `--systems`
flag. For instance, using `--systems "aarch64-linux x86_64-linux"` will prompt
builds for both `aarch64-linux` and `x86_64-linux` architectures. Ensure that
your system is capable of building for the specified architectures, either
locally or through the remote builder protocol.

## Building different flake attributes

`nix-fast-build` by default builds `.#checks.$currentSystem`, which refers to
all checks for the current flake. You can modify this default behavior by using
the `--flake` flag to specify a different attribute path.

Example:

```console
$ nix run github:Mic92/nix-fast-build -- --flake github:NixOS/nixpkgs#legacyPackages.x86_64-linux.hello
```

**Note:** Always provide the complete flake path. Unlike `nix build`,
`nix-fast-build` does not iterate over different attributes; the full path must
be explicitly stated.

## Excluding or filtering attributes

Use `--select` to pass a Nix function to `nix-eval-jobs` that transforms the
evaluation root before traversal. The function receives the attribute selected
by `--flake` (or `-A` in `--file` mode), so you can drop expensive jobs or pick
a subset without changing your flake:

```console
$ nix-fast-build --flake .#checks.x86_64-linux \
    --select 'checks: builtins.removeAttrs checks ["slow-integration-test"]'
```

## Only evaluate the current system

By default nix-fast-build will evaluate all systems in `.#checks`, you can limit
it to the current system by using this command:

```console
$ nix run github:Mic92/nix-fast-build -- --skip-cached --no-nom --flake ".#checks.$(nix eval --raw --impure --file builtins.currentSystem)"
```

## Cachix support

nix-fast-build can upload to cachix like this:

```console
$ nix-fast-build --cachix-cache mic92
```

nix-fast-build assumes that your current machine is either logged in to cachix
or has the environment variables `CACHIX_SIGNING_KEY` or `CACHIX_AUTH_TOKEN`
set. These environment variables are currently not propagated to ssh when using
the `--remote` flag, instead the user is expected that cachix credentials are
configured on the remote machine.

## Attic support

nix-fast-build can upload to attic like this:

```console
$ nix-fast-build --attic-cache mic92
```

nix-fast-build assumes that your current machine is either logged in to attic.
Authentication is not propagated to ssh when using the `--remote` flag, instead
the user is expected that attic credentials are configured on the remote
machine.

## Niks3 support

nix-fast-build can upload to a [niks3](https://github.com/Mic92/niks3) server
like this:

```console
$ nix-fast-build --niks3-server https://niks3.example.com
```

niks3 reads authentication from `~/.config/niks3/auth-token` by default, or from
a custom path via the `NIKS3_AUTH_TOKEN_FILE` environment variable.

When using the `--remote` flag, ensure that the auth token file exists on the
remote machine (either at the default XDG path or via `NIKS3_AUTH_TOKEN_FILE`).

## Machine-readable builds results

nix-fast-build supports both its own json format and junit:

Example for json output:

```console
nix-fast-build --result-file result.json
cat ./result.json
{
   "results": [
     {
       "attr": "riscv64-linux.package-default",
       "duration": 0.0,
       "error": null,
       "success": true,
       "type": "EVAL"
     },
# ...
```

Example for junit result output:

```console
nix-fast-build --result-format junit --result-file result.xml
```

```console
nix-shell -p python3Packages.junit2html --run 'junit2html result.xml result.html'
```

## Reference

```console
usage: nix-fast-build [-h] [--nix NIX] [--nix-eval-jobs NIX_EVAL_JOBS]
                      [--nix-build NIX_BUILD] [-f FLAKE | --file [FILE]]
                      [-A ATTR] [--arg name value] [--argstr name value]
                      [-I INCLUDE] [--impure] [--pure] [-j MAX_JOBS]
                      [--option name value] [--remote-ssh-option name value]
                      [--cachix-cache CACHIX_CACHE]
                      [--attic-cache ATTIC_CACHE]
                      [--attic-ignore-upstream-cache-filter]
                      [--attic-push-build-closure]
                      [--niks3-server NIKS3_SERVER] [--no-nom]
                      [--systems SYSTEMS] [--retries RETRIES] [--no-link]
                      [--out-link OUT_LINK] [--remote REMOTE]
                      [--always-upload-source] [--no-download] [--skip-cached]
                      [--copy-to COPY_TO] [--debug]
                      [--eval-max-memory-size EVAL_MAX_MEMORY_SIZE]
                      [--eval-workers EVAL_WORKERS]
                      [--result-file RESULT_FILE]
                      [--result-format {json,junit}]
                      [--override-input input_path flake_url]
                      [--select NIX_FUNCTION]
                      [--reference-lock-file REFERENCE_LOCK_FILE]

options:
  -h, --help            show this help message and exit
  --nix NIX             Path to the nix binary (default: $NIX_FAST_BUILD_NIX
                        or 'nix')
  --nix-eval-jobs NIX_EVAL_JOBS
                        Path to the nix-eval-jobs binary (default:
                        $NIX_FAST_BUILD_EVAL_JOBS or 'nix-eval-jobs')
  --nix-build NIX_BUILD
                        Path to the nix-build binary (default:
                        $NIX_FAST_BUILD_NIX_BUILD or 'nix-build')
  -f, --flake FLAKE     Flake url to evaluate/build (default: .#checks)
  --file [FILE]         Nix expression file to evaluate (default:
                        default.nix). Mutually exclusive with --flake.
  -A, --attr ATTR       Attribute path to evaluate in non-flake mode (like
                        nix-build -A)
  --arg name value      Pass the value expr as the argument name to Nix
                        functions (non-flake mode)
  --argstr name value   Pass the string as the argument name to Nix functions
                        (non-flake mode)
  -I, --include INCLUDE
                        Add path to the Nix search path (non-flake mode)
  --impure              Allow impure expressions (default in --file mode)
  --pure                Enforce pure evaluation in --file mode (overrides the
                        default impure behavior)
  -j, --max-jobs MAX_JOBS
                        Maximum number of build jobs to run in parallel (0 for
                        unlimited)
  --option name value   Nix option to set
  --remote-ssh-option name value
                        ssh option when accessing remote
  --cachix-cache CACHIX_CACHE
                        Cachix cache to upload to
  --attic-cache ATTIC_CACHE
                        Attic cache to upload to
  --attic-ignore-upstream-cache-filter
                        Pass --ignore-upstream-cache-filter to attic push,
                        uploading all paths even if attic thinks they exist in
                        an upstream cache (default: false)
  --attic-push-build-closure
                        Also push the build-time closure to attic, not just
                        the runtime closure. Useful for caching intermediate
                        build products in ephemeral CI environments (default:
                        false)
  --niks3-server NIKS3_SERVER
                        Niks3 server URL to upload to (auth from
                        ~/.config/niks3/auth-token or NIKS3_AUTH_TOKEN_FILE)
  --no-nom              Don't use nix-output-monitor to print build output
                        (default: false)
  --systems SYSTEMS     Space-separated list of systems to build for (default:
                        current system)
  --retries RETRIES     Number of times to retry failed builds
  --no-link             Do not create an out-link for builds (default: false)
  --out-link OUT_LINK   Name of the out-link for builds (default: result)
  --remote REMOTE       Remote machine to build on
  --always-upload-source
                        Always upload sources to remote machine. This is
                        needed if the remote machine cannot access all sources
                        (default: false)
  --no-download         Do not download build results from remote machine
  --skip-cached         Skip builds that are already present in the binary
                        cache (default: false)
  --copy-to COPY_TO     Copy build results to the given path (passed to nix
                        copy, i.e. file:///tmp/cache?compression=none)
  --debug               debug logging output
  --eval-max-memory-size EVAL_MAX_MEMORY_SIZE
                        Maximum memory size for nix-eval-jobs (in MiB) per
                        worker. After the limit is reached, the worker is
                        restarted.
  --eval-workers EVAL_WORKERS
                        Number of evaluation threads spawned
  --result-file RESULT_FILE
                        File to write build results to
  --result-format {json,junit}
                        Format of the build result file
  --override-input input_path flake_url
                        Override a specific flake input (e.g.
                        `dwarffs/nixpkgs`).
  --select NIX_FUNCTION
                        Nix function applied to the evaluation root to filter
                        or transform the set of attributes to build (passed
                        through to nix-eval-jobs --select). The function
                        receives the attribute selected by --flake/-A.
                        Example: 'checks: builtins.removeAttrs checks ["slow-
                        test"]'
  --reference-lock-file REFERENCE_LOCK_FILE
                        Read the given lock file instead of `flake.lock`
                        within the top-level flake.
```

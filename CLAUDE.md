# Development Workflow

## Build

```bash
nix build
```

## Test

```bash
nix develop --command pytest --verbose
```

NOTE: run formating command before running tests.

## Format

```bash
nix fmt
```

## Linting

```bash
nix flake check
```

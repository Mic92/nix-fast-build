#!/usr/bin/env bash
# Re-record the TUI demo (demo.gif/demo.webm) with vhs.
set -euo pipefail
cd "$(dirname "$0")"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
system=$(nix eval --impure --raw --expr builtins.currentSystem)

# A fresh nonce makes every check rebuild instead of hitting the store.
NFB_DEMO="$tmp/demo"
export NFB_DEMO
mkdir -p "$NFB_DEMO"
sed -e "s/@system@/$system/g" -e "s/@nonce@/$(date +%s)/" flake.nix.in > "$NFB_DEMO/flake.nix"
git -C "$NFB_DEMO" init -q
git -C "$NFB_DEMO" add flake.nix
nix flake lock "$NFB_DEMO"
git -C "$NFB_DEMO" add flake.lock

nix build --out-link "$tmp/bin" ../..
PATH="$tmp/bin/bin:$PATH" nix run nixpkgs#vhs -- demo.tape

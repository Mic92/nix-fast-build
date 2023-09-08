{
  description = "Evaluate and build in parallel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } ({ lib, ... }: {
      systems = [
        "aarch64-linux"
        "x86_64-linux"
        "riscv64-linux"

        "x86_64-darwin"
        "aarch64-darwin"
      ];
      perSystem = { pkgs, self', ... }: {
        packages.nix-ci-builds = pkgs.callPackage ./default.nix {};
        packages.default = self'.packages.nix-ci-builds;
      };
    });
}

{
  description = "Evaluate and build in parallel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } ({ lib, ... }:
      let
        officialPlatforms = [
          "aarch64-linux"
          "x86_64-linux"

          "x86_64-darwin"
          "aarch64-darwin"
        ];
      in
      {
        imports = [ ./treefmt.nix ];
        systems = officialPlatforms ++ [ "riscv64-linux" ];
        perSystem = { pkgs, self', ... }: {
          packages.nix-fast-build = pkgs.callPackage ./default.nix {
            # we don't want to compile ghc otherwise
            nix-output-monitor =
              if lib.elem pkgs.hostPlatform.system officialPlatforms then
                pkgs.nix-output-monitor
              else
                null;
          };
          legacyPackages = {
            hello-broken = pkgs.hello.overrideAttrs (_old: {
              meta.broken = true;
            });
          };
          packages.default = self'.packages.nix-fast-build;

          checks =
            let
              packages = lib.mapAttrs' (n: lib.nameValuePair "package-${n}") self'.packages;
              devShells = lib.mapAttrs' (n: lib.nameValuePair "devShell-${n}") self'.devShells;
            in
            packages // devShells;
        };
      });
}

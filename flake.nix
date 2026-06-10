{
  description = "Evaluate and build in parallel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
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
        perSystem =
          {
            pkgs,
            self',
            config,
            ...
          }:
          {
            packages.nix-fast-build = pkgs.callPackage ./default.nix {
              # we don't want to compile ghc otherwise
              nix-output-monitor =
                if lib.elem pkgs.stdenv.hostPlatform.system officialPlatforms then
                  pkgs.nix-output-monitor
                else
                  null;
            };
            legacyPackages = {
              hello-broken = pkgs.hello.overrideAttrs (_old: {
                meta.broken = true;
              });
              hello-build-fails = pkgs.runCommand "hello-build-fails" { } ''
                echo "This build will fail"
                exit 1
              '';
              # Test package where a dependency fails
              hello-dep-fails =
                let
                  failing-dep = pkgs.runCommand "failing-dep" { } ''
                    echo "Dependency build failure"
                    exit 1
                  '';
                in
                pkgs.runCommand "hello-dep-fails" { buildInputs = [ failing-dep ]; } ''
                  echo "This should not run because dependency failed"
                  mkdir -p $out
                '';
            };
            packages.default = self'.packages.nix-fast-build;

            checks =
              let
                packages = lib.mapAttrs' (n: lib.nameValuePair "package-${n}") self'.packages;
                devShells = lib.mapAttrs' (n: lib.nameValuePair "devShell-${n}") self'.devShells;
                # Building inputDerivation turns build-time deps into runtime
                # references, so CI binary caches retain them.
                closures = lib.mapAttrs' (n: drv: lib.nameValuePair "closure-${n}" drv.inputDerivation) (
                  packages // devShells // { treefmt = config.treefmt.build.check inputs.self; }
                );
                # Flake inputs are fetched, not built, so caches miss them too.
                # Same for bashInteractive, which nix develop fetches (all
                # outputs) for its shell.
                flake-inputs = pkgs.runCommand "flake-inputs" { } ''
                  printf '%s\n' ${
                    lib.concatMapStringsSep " " toString (
                      lib.attrValues (lib.filterAttrs (_: lib.isAttrs) inputs)
                      ++ map (o: pkgs.bashInteractive.${o}) pkgs.bashInteractive.outputs
                    )
                  } > $out
                '';
              in
              packages // devShells // closures // { closure-flake-inputs = flake-inputs; };
          };
      }
    );
}

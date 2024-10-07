{ lib, inputs, ... }:
{
  imports = [ inputs.treefmt-nix.flakeModule ];

  perSystem =
    { pkgs, ... }:
    {
      treefmt = {
        # Used to find the project root
        projectRootFile = ".git/config";

        programs.deno.enable = pkgs.lib.meta.availableOn pkgs.stdenv.hostPlatform pkgs.deno;
        programs.mypy.enable = true;
        programs.mypy.directories."root".directory = ".";
        programs.deadnix.enable = true;
        programs.nixfmt.enable = pkgs.lib.meta.availableOn pkgs.stdenv.buildPlatform pkgs.nixfmt-rfc-style.compiler;
        programs.ruff.format = true;
        programs.ruff.check = true;
      };
    };
}

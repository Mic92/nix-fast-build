{ lib, inputs, ... }: {
  imports = [
    inputs.treefmt-nix.flakeModule
  ];

  perSystem = { pkgs, ... }: {
    treefmt = {
      # Used to find the project root
      projectRootFile = "flake.lock";

      programs.mypy.enable = true;
      programs.deno.enable = lib.elem pkgs.hostPlatform.system [
        "aarch64-linux"
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      settings.formatter = {
        nix = {
          command = "sh";
          options = [
            "-eucx"
            ''
              # First deadnix
              ${lib.getExe pkgs.deadnix} --edit "$@"
              # Then nixpkgs-fmt
              ${lib.getExe pkgs.nixpkgs-fmt} "$@"
            ''
            "--"
          ];
          includes = [ "*.nix" ];
        };

        python = {
          command = "sh";
          options = [
            "-eucx"
            ''
              ${lib.getExe pkgs.ruff} --fix "$@"
              ${lib.getExe pkgs.black} "$@"
            ''
            "--" # this argument is ignored by bash
          ];
          includes = [ "*.py" ];
        };
      };
    };
  };
}

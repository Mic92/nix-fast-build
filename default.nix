{ python311, makeWrapper, nix, nix-eval-jobs, nix-output-monitor, lib, bashInteractive }:
let
  path = lib.makeBinPath [ nix nix-eval-jobs nix-output-monitor ];
in
python311.pkgs.buildPythonApplication {
  pname = "nix-ci-build";
  version = "0.1.0";
  format = "pyproject";
  src = ./.;
  buildInputs = with python311.pkgs; [ setuptools bashInteractive ];
  nativeBuildInputs = [ makeWrapper ];
  preFixup = ''
    makeWrapperArgs+=(--prefix PATH : ${path})
  '';
  shellHook = ''
    export PATH=${path}:$PATH
  '';
}

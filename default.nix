{ python3, makeWrapper, nix, nix-eval-jobs, nix-output-monitor, lib }:
let
  path = lib.makeBinPath [ nix nix-eval-jobs nix-output-monitor ];
in
python3.pkgs.buildPythonApplication {
  pname = "nix-ci-build";
  version = "0.1.0";
  format = "pyproject";
  src = ./.;
  buildInputs = with python3.pkgs; [ setuptools ];
  nativeBuildInputs = [ makeWrapper ];
  preFixup = ''
    makeWrapperArgs+=(--prefix PATH : ${path})
  '';
  shellHook = ''
    export PATH=${path}:$PATH
  '';
}

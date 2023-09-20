{ python311, makeWrapper, nix, nix-eval-jobs, nix-output-monitor, lib, bashInteractive }:
let
  path = lib.makeBinPath [ nix nix-eval-jobs nix-output-monitor ];
in
python311.pkgs.buildPythonApplication {
  pname = "nix-fast-build";
  version = "0.1.0";
  format = "pyproject";
  src = ./.;
  buildInputs = with python311.pkgs; [
    setuptools
    bashInteractive
  ];
  nativeBuildInputs = [
    makeWrapper
    python311.pkgs.pytest
  ];
  preFixup = ''
    makeWrapperArgs+=(--prefix PATH : ${path})
  '';
  shellHook = ''
    export PATH=${path}:$PATH
  '';
  meta = {
    description = "Combine the power of nix-eval-jobs with nix-output-monitor to speed-up your evaluation and building process.";
    homepage = "https://github.com/Mic92/nix-fast-build";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ mic92 ];
    mainProgram = "nix-fast-build";
  };
}

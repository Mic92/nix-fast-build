{
  python3Packages,
  makeWrapper,
  nix-eval-jobs,
  lib,
  bashInteractive,
}:
let
  path = lib.makeBinPath [
    nix-eval-jobs
    nix-eval-jobs.nix
  ];
in
python3Packages.buildPythonApplication {
  pname = "nix-fast-build";
  version = (lib.trivial.importTOML ./pyproject.toml).project.version;
  format = "pyproject";
  src = ./.;
  buildInputs = with python3Packages; [
    setuptools
    bashInteractive
  ];
  nativeBuildInputs = [
    makeWrapper
    python3Packages.pytest
  ];
  preFixup = ''
    makeWrapperArgs+=(--prefix PATH : ${path})
  '';
  postFixup = ''
    # don't leak python into devshell
    rm $out/nix-support/propagated-build-inputs
  '';
  shellHook = ''
    export PATH=${path}:$PATH
  '';
  meta = {
    description = "Speed up your Nix evaluation and building process with parallel evaluation and building";
    homepage = "https://github.com/Mic92/nix-fast-build";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ mic92 ];
    mainProgram = "nix-fast-build";
  };
}

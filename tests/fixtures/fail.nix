let
  flake = builtins.getFlake (toString ./../..);
  pkgs = import flake.inputs.nixpkgs { };
in
{
  fail = pkgs.runCommand "fail" { } "exit 1";
  hello = pkgs.hello;
}

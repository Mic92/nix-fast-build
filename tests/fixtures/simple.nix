let
  flake = builtins.getFlake (toString ./../..);
  pkgs = import flake.inputs.nixpkgs { };
in
{
  hello = pkgs.hello;
}

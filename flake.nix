{
  description = "CS229 solutions";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    jupyterWith.url = "github:tweag/jupyenv";
    flake-utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };

    mach-nix.url = "github:davhau/mach-nix";
  };

  outputs = { self, nixpkgs, mach-nix, jupyterWith, flake-utils, ... }:
    let
      pythonVersion = "python39";
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          system = system;
        };
        my-python-packages = ps: with ps; [
          pandas
          requests
          numpy
          matplotlib
          jupyter
          ipykernel
        ];
        my-python = pkgs.python311.withPackages my-python-packages;
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = [ pkgs.yapf pkgs.quarto my-python ];
          shellHook = ''
            export PYTHONPATH="${my-python}/bin/python"
            export QUARTO_PYTHON="${my-python}/bin/python"
            export PATH="${my-python}/bin:$PATH"
            export YAPF_PATH="${pkgs.yapf}/bin/yapf"
          '';
        };
      }
    );
}

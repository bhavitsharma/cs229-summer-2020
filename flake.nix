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
        inherit (jupyterWith.lib.${system}) mkJupyterlabNew;
        mach = mach-nix.lib.${system};
        # call the pkgs.quarto package derivation with extraPythonPackages attribute
        quarto_new = pkgs.quarto.override {
          extraPythonPackages = ps: with ps; [
            numpy
            matplotlib
            pandas
          ];
        };

        pythonEnv = mach.mkPython {
          python = pythonVersion;
          requirements = builtins.readFile ./requirements.txt;
        };
        jupyterlab = mkJupyterlabNew ({ ... }: {
          imports = [ (import ./kernel.nix) ];
        });
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = [ pythonEnv pkgs.yapf quarto_new pkgs.python310Packages.ipykernel jupyterlab ];

          shellHook = ''
            export PYTHONPATH="${pythonEnv}/bin/python"
            export YAPF_PATH="${pkgs.yapf}/bin/yapf"
            export QUARTO_PATH="${quarto_new}/bin/quarto"
          '';
        };
      }
    );
}

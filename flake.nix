{
  description = "CS229 solutions";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };

    mach-nix.url = "github:davhau/mach-nix";
  };

  outputs = { self, nixpkgs, mach-nix, flake-utils, ... }:
    let
      pythonVersion = "python39";
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        mach = mach-nix.lib.${system};

        pythonEnv = mach.mkPython {
          python = pythonVersion;
          requirements = builtins.readFile ./requirements.txt;
        };
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = [ pythonEnv pkgs.yapf ];

          shellHook = ''
            export PYTHONPATH="${pythonEnv}/bin/python"
            export YAPF_PATH="${pkgs.yapf}/bin/yapf"
          '';
        };
      }
    );
}

{
  description = "A Python environment with pandas and numpy";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3.withPackages (ps: with ps; [
          pandas
          numpy
        ]);
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [ python ];
        };
      }
    );
}

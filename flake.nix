{
  description = "circuit-steering: orchestrates tiny-aya-circuits and nothing-else-broke on Tiny Aya";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      tools = pkgs: [ pkgs.uv pkgs.python312 pkgs.git pkgs.just pkgs.bash ];
      # Every app runs the checkout's own recipe from the caller's working directory (runs/,
      # .cache/ and the sibling repos live there), with the tools above on PATH.
      just = pkgs: name: recipe: {
        type = "app";
        program = toString (pkgs.writeShellScript "circuit-steering-${name}" ''
          export PATH=${pkgs.lib.makeBinPath (tools pkgs)}:$PATH
          exec just ${recipe} "$@"
        '');
      };
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell { packages = tools pkgs; };
      });
      apps = forAll (pkgs: {
        default = just pkgs "check" "check";
        check = just pkgs "check" "check";
        full = just pkgs "full" "chain full";
        tail = just pkgs "tail" "chain tail";
        circuits = just pkgs "circuits" "chain circuits";
        safetune = just pkgs "safetune" "chain safetune";
      });
    };
}

{
  pkgs ? import <nixpkgs> { },
}:

let
  nixpkgs-26-05 = fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/nixos-26.05.tar.gz";
  };

  pkgs-26-05 = import nixpkgs-26-05 {
    inherit (pkgs) system;
  };

in
pkgs.mkShell {
  packages = [
    pkgs-26-05.zig_0_16
    pkgs.libx11
    pkgs.vulkan-loader
    pkgs.vulkan-validation-layers
    pkgs.vulkan-tools
    pkgs.glslang
    pkgs.git
    pkgs.shader-slang
  ];

  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
    pkgs.kdePackages.wayland
    pkgs.libxkbcommon
    pkgs.libGL
  ]}";
}

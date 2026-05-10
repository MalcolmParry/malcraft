{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = [
    pkgs.zig_0_15
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

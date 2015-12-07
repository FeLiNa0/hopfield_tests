with import <nixpkgs> {}; {
  pyEnv = stdenv.mkDerivation {
    name = "py";
    buildInputs = [ stdenv python3 python34Packages.matplotlib python34Packages.numpy python34Packages.ipython ];
  };
}

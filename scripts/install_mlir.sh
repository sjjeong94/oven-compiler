sudo apt update
sudo apt install -y git cmake ninja-build python3 python3-pip \
  python3-setuptools python3-dev clang lld llvm-dev libncurses5
  
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

git checkout llvmorg-20.1.8

mkdir build
cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=../install

cmake --build . --target install

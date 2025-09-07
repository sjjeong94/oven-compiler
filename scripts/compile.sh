input_file=$1

if [ ! -d compiled ]; then
	mkdir compiled
fi

./build-ninja/tools/oven-opt $input_file -o compiled/kernel.mlir --oven-to-llvm
./llvm-project/build/bin/mlir-translate compiled/kernel.mlir --mlir-to-llvmir -o compiled/kernel.ll
./llvm-project/build/bin/llc compiled/kernel.ll -o compiled/kernel.ptx -mtriple nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx80

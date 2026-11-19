#!/bin/bash
# One-time setup on a Brev H100 instance (Ubuntu 22.04/24.04 image with the NVIDIA driver and CUDA
# under /usr/local/cuda). Run from the repo root: bash cluster/brev_setup.sh
set -euo pipefail

sudo apt-get update
sudo apt-get install -y cmake ninja-build clang-18 lld-18 libc++-18-dev libc++abi-18-dev libomp-18-dev \
  || { echo "clang-18 not in this image's apt sources; add apt.llvm.org (see docs/running.md)"; exit 1; }

CUDA_ROOT=${CUDA_HOME:-/usr/local/cuda}
nvidia-smi
"$CUDA_ROOT/bin/nvcc" --version | tail -1

cmake --preset clang-release \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_CUDA_COMPILER=clang++-18 \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_FLAGS="--cuda-path=$CUDA_ROOT"
cmake --build --preset clang-release
ctest --preset clang-release
build/clang-release/lettuce-bench backend=cuda,cpu n=1e4,1e5,1e6 steps=3

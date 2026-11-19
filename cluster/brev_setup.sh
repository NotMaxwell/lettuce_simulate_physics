#!/bin/bash
# Set up, build and test on a Brev GPU instance (Ubuntu 22.04 or 24.04 with the NVIDIA driver and
# CUDA, normally in /usr/local/cuda). Run from the repo root:
#   bash cluster/brev_setup.sh
# Everything is also written to out/brev-report.txt; paste that file back when asking for help.
#
# It picks the toolchain itself: Clang 18 (Clang 20 when CUDA is newer than 12.3, which Clang 18
# does not know), installed from apt.llvm.org when the distro lacks it; CMake from pip when the
# distro's is older than 3.24; and if the Clang build fails anyway, the GCC + nvcc fallback.
set -uo pipefail
mkdir -p out
REPORT=out/brev-report.txt
exec > >(tee "$REPORT") 2>&1

step() { echo; echo "=== $*"; }
die() { echo "FATAL: $*"; exit 1; }

step "machine"
. /etc/os-release && echo "$PRETTY_NAME"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || die "nvidia-smi failed: no driver?"
CUDA_ROOT=${CUDA_HOME:-/usr/local/cuda}
[ -x "$CUDA_ROOT/bin/nvcc" ] || CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc || echo /x/x/nvcc)")")
[ -x "$CUDA_ROOT/bin/nvcc" ] || die "no CUDA toolkit found (set CUDA_HOME)"
CUDA_VER=$("$CUDA_ROOT/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "CUDA $CUDA_VER at $CUDA_ROOT"
NGPU=$(nvidia-smi -L | wc -l)

LLVM=${LLVM:-18}
if [ "$(printf '%s\n' "$CUDA_VER" 12.3 | sort -V | tail -1)" != "12.3" ]; then LLVM=${LLVM_NEW:-20}; fi
echo "using LLVM $LLVM"

step "packages"
sudo apt-get update -qq
sudo apt-get install -y -qq ninja-build libopenmpi-dev openmpi-bin g++-12 python3-pip wget gnupg lsb-release \
  software-properties-common || die "apt-get install failed"
if ! sudo apt-get install -y -qq "clang-$LLVM" "lld-$LLVM" "libc++-$LLVM-dev" "libc++abi-$LLVM-dev" "libomp-$LLVM-dev"; then
  echo "clang-$LLVM not in this distro's apt; adding apt.llvm.org"
  wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh || die "cannot download llvm.sh"
  sudo bash /tmp/llvm.sh "$LLVM" all || die "llvm.sh failed"
fi
CMV=$(cmake --version 2>/dev/null | awk 'NR==1{print $3}')
if [ -z "$CMV" ] || [ "$(printf '%s\n' "$CMV" 3.24 | sort -V | head -1)" != "3.24" ]; then
  echo "CMake ${CMV:-missing} is older than 3.24; installing from pip"
  python3 -m pip install --user --quiet "cmake>=3.24" || die "pip install cmake failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
cmake --version | head -1

step "build (clang-$LLVM, CUDA via clang)"
B=build/clang-release
if cmake --preset clang-release -DCMAKE_C_COMPILER="clang-$LLVM" -DCMAKE_CXX_COMPILER="clang++-$LLVM" \
     -DCMAKE_CUDA_COMPILER="clang++-$LLVM" -DLETTUCE_CUDA=ON \
     -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_FLAGS="--cuda-path=$CUDA_ROOT" &&
   cmake --build --preset clang-release; then
  echo "clang build OK"
else
  step "clang build failed; falling back to GCC 12 + nvcc"
  B=build/nvcc-release
  cmake --preset nvcc-release -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=g++-12 || die "nvcc configure failed"
  cmake --build --preset nvcc-release || die "nvcc build failed too"
fi

step "unit tests (the CUDA ones must say run, not skipped)"
"$B/lettuce-tests"; TESTS_RC=$?
"$B/lettuce-tests" cuda 2>&1 | tail -1

if [ "$NGPU" -gt 1 ]; then
  step "multi-GPU tests, $NGPU ranks"
  mpirun -n "$NGPU" --bind-to none "$B/lettuce-mpi-tests"; MPI_RC=$?
else
  MPI_RC=0
fi

step "bench (one GPU, plus the CPU for comparison)"
"$B/lettuce-bench" backend=cuda,cpu n=1e4,1e5,1e6 method=direct,tree steps=3 direct_max=100000
"$B/lettuce-bench" backend=cuda method=tree n=1e7 steps=3
if [ "$NGPU" -gt 1 ]; then
  step "bench, 1 vs $NGPU GPUs"
  for np in 1 "$NGPU"; do
    mpirun -n "$np" --bind-to none "$B/lettuce-bench" backend=cuda method=tree n=1e6,1e7 steps=3
  done
fi

step "short physics run on the GPU"
"$B/lettuce" configs/plummer.ini n=1000000 method=tree steps=100 diag_every=50 backend=cuda out_dir=out/brev-plummer

step "summary"
echo "build: $B   unit tests rc=$TESTS_RC   mpi tests rc=$MPI_RC   report: $REPORT"
[ "$TESTS_RC" -eq 0 ] && [ "$MPI_RC" -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED: see above"

#!/bin/bash
# Set up, build and test on a Brev GPU instance (Ubuntu 22.04 or 24.04 with the NVIDIA driver and
# CUDA, normally in /usr/local/cuda). Two ways to run it:
#   bash cluster/brev_setup.sh        from the repo root
#   paste the whole file into Brev's "Setup Script" box (runs as root on a fresh VM, no repo yet)
# Outside a repo root it clones/updates the fork into $LETTUCE_DIR. As root with a sudo-capable
# login user it re-runs itself as that user so the clone, build and out/ belong to them.
# The report is <repo>/out/brev-report.txt; paste that file back when asking for help.
#
# Env: LETTUCE_DIR (default ~/lettuce_simulate_physics), LETTUCE_BRANCH (default main),
#      CUDA_HOME, LLVM, LLVM_NEW.
#
# It picks the toolchain itself: Clang 18 (Clang 20 when CUDA is newer than 12.3, which Clang 18
# does not know), installed from apt.llvm.org when the distro lacks it; CMake from pip when the
# distro's is older than 3.24; and if the Clang build fails anyway, the GCC + nvcc fallback.
set -uo pipefail

die() { echo "FATAL: $*"; exit 1; }
export HOME=${HOME:-/root}

if [ "$(id -u)" -eq 0 ]; then SUDO=(); else SUDO=(sudo); fi
# First boot: unattended-upgrades may hold the dpkg lock for minutes.
aptg() { "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 "$@"; }

if [ "$(id -u)" -eq 0 ] && [ -z "${LETTUCE_REEXEC:-}" ]; then
  SELF_SRC=$(readlink -f "${BASH_SOURCE[0]:-}" 2>/dev/null)
  [ -n "${LETTUCE_DIR:-}" ] || { [ -f CMakePresets.json ] && [ -f cluster/brev_setup.sh ] && LETTUCE_DIR=$PWD; }
  cd /
  RUSER=${SUDO_USER:-}
  if [ -z "$RUSER" ] || [ "$RUSER" = root ]; then RUSER=$(getent passwd 1000 | cut -d: -f1); fi
  if [ -n "$RUSER" ] && [ "$RUSER" != root ] && command -v sudo >/dev/null && sudo -n -u "$RUSER" sudo -n true 2>/dev/null; then
    ENVV=(LETTUCE_REEXEC=1)
    for v in CUDA_HOME LLVM LLVM_NEW LETTUCE_DIR LETTUCE_BRANCH; do
      if [ -n "${!v:-}" ]; then ENVV+=("$v=${!v}"); fi
    done
    echo "running as root; re-running as $RUSER"
    # The script may live somewhere that user cannot read: hand over a copy.
    if [ -f "$SELF_SRC" ] && [ -r "$SELF_SRC" ] &&
       SELF=$(mktemp /tmp/brev_setup.XXXXXX.sh) && cp "$SELF_SRC" "$SELF" && chmod 755 "$SELF"; then
      sudo -u "$RUSER" -H env "${ENVV[@]}" bash "$SELF" "$@"; rc=$?
      rm -f "$SELF"
      exit $rc
    fi
    # Pasted via bash -c or stdin: no file to copy, so clone as the user and run the repo's copy.
    DIR=${LETTUCE_DIR:-$(getent passwd "$RUSER" | cut -d: -f6)/lettuce_simulate_physics}
    command -v git >/dev/null || { aptg update -qq && aptg install -y -qq git; } || die "cannot install git"
    if [ -d "$DIR/.git" ]; then
      sudo -u "$RUSER" -H git -C "$DIR" pull --ff-only || echo "WARNING: git pull failed; building the existing checkout"
    else
      sudo -u "$RUSER" -H git clone --branch "${LETTUCE_BRANCH:-main}" \
        https://github.com/NotMaxwell/lettuce_simulate_physics.git "$DIR" || die "git clone failed"
    fi
    cd "$DIR" || die "cannot cd to $DIR"
    sudo -u "$RUSER" -H env "${ENVV[@]}" LETTUCE_DIR="$DIR" bash "$DIR/cluster/brev_setup.sh" "$@"
    exit $?
  fi
  echo "no sudo-capable login user (or script not readable); staying root"
fi
if [ "$(id -u)" -eq 0 ]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi

LETTUCE_DIR=${LETTUCE_DIR:-${HOME:-/root}/lettuce_simulate_physics}
if [ -f CMakePresets.json ] && [ -f cluster/brev_setup.sh ]; then REPO=$PWD; else REPO=$LETTUCE_DIR; fi
echo "report will be at $REPO/out/brev-report.txt"

if [ "$REPO" != "$PWD" ]; then
  command -v git >/dev/null || { aptg update -qq && aptg install -y -qq git; } || die "cannot install git"
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only || echo "WARNING: git pull failed; building the existing checkout"
  else
    git clone --branch "${LETTUCE_BRANCH:-main}" https://github.com/NotMaxwell/lettuce_simulate_physics.git "$REPO" ||
      die "git clone failed"
  fi
  cd "$REPO" || die "cannot cd to $REPO"
fi

mkdir -p out
REPORT=$PWD/out/brev-report.txt
exec > >(tee "$REPORT") 2>&1

step() { echo; echo "=== $*"; }

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
aptg update -qq
aptg install -y -qq ninja-build libopenmpi-dev openmpi-bin g++-12 python3-pip wget gnupg lsb-release \
  software-properties-common || die "apt-get install failed"
if ! aptg install -y -qq "clang-$LLVM" "lld-$LLVM" "libc++-$LLVM-dev" "libc++abi-$LLVM-dev" "libomp-$LLVM-dev"; then
  echo "clang-$LLVM not in this distro's apt; adding apt.llvm.org"
  wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh || die "cannot download llvm.sh"
  "${SUDO[@]}" bash /tmp/llvm.sh "$LLVM" all || die "llvm.sh failed"
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

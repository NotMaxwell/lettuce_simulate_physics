#!/bin/bash
# Set up, build and test on a Brev GPU instance (Ubuntu with the NVIDIA driver; CUDA on the host
# is only needed in host mode). Two ways to run it:
#   bash cluster/brev_setup.sh        from the repo root
#   paste the whole file into Brev's "Setup Script" box (runs as root on a fresh VM, no repo yet)
# Outside a repo root it clones/updates the fork into $LETTUCE_DIR. As root with a sudo-capable
# login user it re-runs itself as that user so the clone, build and out/ belong to them.
# The report is <repo>/out/brev-report.txt; paste that file back when asking for help.
#
# LETTUCE_MODE=auto|container|host (default auto). Brev images often pin the universe packages we
# need (ninja, openmpi, g++-12, clang) to apt priority -1, so host builds can be impossible; auto
# builds inside a persistent Docker container ($LETTUCE_CONTAINER, repo mounted at /work) when
# `docker run --gpus all` works, else on the host. Run binaries in it with
# `docker exec -it lettuce-build bash` (they need the container's libc++/libomp).
#
# Env: LETTUCE_DIR (default ~/lettuce_simulate_physics), LETTUCE_BRANCH (default main),
#      LETTUCE_IMAGE (default nvidia/cuda:12.8.1-devel-ubuntu24.04: clang-20 builds against CUDA 12.8
#      with a "partially supported" warning; a CUDA 13.0 image failed CMake's CUDA compiler check), LETTUCE_CONTAINER (default lettuce-build),
#      CUDA_HOME, LLVM, LLVM_NEW.
#
# Host toolchain: Clang 18 (Clang 20 when CUDA is newer than 12.3, which Clang 18 does not know),
# installed from apt.llvm.org when the distro lacks it; CMake from apt, else pip, when older than
# 3.24; and if the Clang build fails anyway, the GCC + nvcc fallback.
set -uo pipefail

die() { echo "FATAL: $*"; exit 1; }
export HOME=${HOME:-/root}

if [ "$(id -u)" -eq 0 ]; then SUDO=(); else SUDO=(sudo); fi
# First boot: unattended-upgrades may hold the dpkg lock for minutes.
aptg() { "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 "$@"; }

if [ "$(id -u)" -eq 0 ] && [ -z "${LETTUCE_REEXEC:-}" ] && [ -z "${LETTUCE_IN_CONTAINER:-}" ]; then
  SELF_SRC=$(readlink -f "${BASH_SOURCE[0]:-}" 2>/dev/null)
  [ -n "${LETTUCE_DIR:-}" ] || { [ -f CMakePresets.json ] && [ -f cluster/brev_setup.sh ] && LETTUCE_DIR=$PWD; }
  cd /
  RUSER=${SUDO_USER:-}
  if [ -z "$RUSER" ] || [ "$RUSER" = root ]; then RUSER=$(getent passwd 1000 | cut -d: -f1); fi
  if [ -n "$RUSER" ] && [ "$RUSER" != root ] && command -v sudo >/dev/null && sudo -n -u "$RUSER" sudo -n true 2>/dev/null; then
    ENVV=(LETTUCE_REEXEC=1)
    for v in CUDA_HOME LLVM LLVM_NEW LETTUCE_DIR LETTUCE_BRANCH LETTUCE_MODE LETTUCE_IMAGE LETTUCE_CONTAINER; do
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
MODE=${LETTUCE_MODE:-auto}
IMAGE=${LETTUCE_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}
CNAME=${LETTUCE_CONTAINER:-lettuce-build}
MODE_NOTE=""
DOCKER=()
detect_docker() {
  command -v docker >/dev/null || return 1
  if docker info >/dev/null 2>&1; then DOCKER=(docker)
  elif command -v sudo >/dev/null && sudo -n docker info >/dev/null 2>&1; then DOCKER=(sudo docker)
  else return 1; fi
}
if [ -n "${LETTUCE_IN_CONTAINER:-}" ]; then
  MODE=host
elif [ "$MODE" = auto ]; then
  if ! detect_docker; then MODE_NOTE="auto mode: docker not usable; building on the host"; MODE=host
  elif ! "${DOCKER[@]}" run --rm --gpus all "$IMAGE" nvidia-smi -L >/dev/null 2>&1; then
    MODE_NOTE="auto mode: 'docker run --gpus all $IMAGE nvidia-smi -L' failed; building on the host"; MODE=host
  else MODE=container; fi
elif [ "$MODE" = container ]; then
  detect_docker || die "LETTUCE_MODE=container but docker is not usable (docker info failed, also via sudo)"
elif [ "$MODE" != host ]; then
  die "LETTUCE_MODE must be auto, container or host (got $MODE)"
fi

if [ "$MODE" = container ]; then
  exec > >(tee "$PWD/out/brev-host.txt") 2>&1
  echo "mode: container ($IMAGE, container $CNAME); host log out/brev-host.txt"
  # Cap the container below the VM's RAM so a runaway allocation is OOM-killed inside it. Uncapped,
  # a VM with no swap can stop responding (sshd included) before the kernel's OOM killer acts.
  MEM_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  RESERVE_KB=$(( MEM_KB / 10 > 8388608 ? MEM_KB / 10 : 8388608 ))
  MEMCAP=$(( MEM_KB > RESERVE_KB ? (MEM_KB - RESERVE_KB) * 1024 : 0 ))  # 0: no cap (tiny VM)
  MEMCAP_ARGS=()
  [ "$MEMCAP" -gt 0 ] && MEMCAP_ARGS=(--memory "$MEMCAP" --memory-swap "$MEMCAP")
  if "${DOCKER[@]}" container inspect "$CNAME" >/dev/null 2>&1; then
    HAVE_IMG=$("${DOCKER[@]}" inspect -f '{{.Config.Image}}' "$CNAME")
    HAVE_SRC=$("${DOCKER[@]}" inspect -f '{{range .Mounts}}{{if eq .Destination "/work"}}{{.Source}}{{end}}{{end}}' "$CNAME")
    HAVE_MEM=$("${DOCKER[@]}" inspect -f '{{.HostConfig.Memory}}' "$CNAME")
    if [ "$HAVE_IMG" != "$IMAGE" ] || [ "$HAVE_SRC" != "$REPO" ] || [ "$HAVE_MEM" != "$MEMCAP" ]; then
      echo "container $CNAME has image $HAVE_IMG, /work=$HAVE_SRC, memory cap $HAVE_MEM; recreating"
      "${DOCKER[@]}" rm -f "$CNAME" >/dev/null || die "cannot remove $CNAME"
    fi
  fi
  if "${DOCKER[@]}" container inspect "$CNAME" >/dev/null 2>&1; then
    echo "reusing container $CNAME"
    "${DOCKER[@]}" start "$CNAME" >/dev/null || die "docker start $CNAME failed"
  else
    "${DOCKER[@]}" run -d --name "$CNAME" --gpus all --ipc=host --ulimit memlock=-1 \
      ${MEMCAP_ARGS[@]+"${MEMCAP_ARGS[@]}"} -v "$REPO":/work -w /work "$IMAGE" sleep infinity >/dev/null ||
      die "docker run failed"
  fi
  EXE=()
  for v in CUDA_HOME LLVM LLVM_NEW; do
    if [ -n "${!v:-}" ]; then EXE+=(-e "$v=${!v}"); fi
  done
  "${DOCKER[@]}" exec -w /work -e LETTUCE_MODE=host -e LETTUCE_IN_CONTAINER=1 -e "LETTUCE_CONTAINER=$CNAME" -e "LETTUCE_DOCKER=${DOCKER[*]}" \
    -e HOME=/root "${EXE[@]}" "$CNAME" bash cluster/brev_setup.sh; rc=$?
  "${DOCKER[@]}" exec "$CNAME" chown -R "$(stat -c %u:%g "$REPO")" /work/build /work/out 2>/dev/null
  echo
  echo "container build finished (rc=$rc); report: $REPO/out/brev-report.txt"
  echo "shell for running binaries: ${DOCKER[*]} exec -it $CNAME bash   (cd /work)"
  exit $rc
fi

REPORT=$PWD/out/brev-report.txt
exec > >(tee "$REPORT") 2>&1

step() { echo; echo "=== $*"; }

step "machine"
[ -z "$MODE_NOTE" ] || echo "$MODE_NOTE"
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
aptg install -y -qq tzdata || die "cannot install tzdata"
PKGS=(ninja-build libopenmpi-dev openmpi-bin g++-12 python3-pip wget gnupg lsb-release software-properties-common)
nocand() { for p in "$@"; do
  [ "$(apt-cache policy "$p" 2>/dev/null | awk '/Candidate:/{print $2}')" != "(none)" ] || echo "$p"; done; }
if [ -n "$(nocand "${PKGS[@]}")" ]; then
  if [ "${ID:-}" = ubuntu ]; then
    echo "packages without apt candidate: $(nocand "${PKGS[@]}" | tr '\n' ' '); enabling universe"
    command -v add-apt-repository >/dev/null || aptg install -y -qq software-properties-common
    "${SUDO[@]}" add-apt-repository -y universe && aptg update -qq
  fi
  MISSING=$(nocand "${PKGS[@]}" | tr '\n' ' ')
  [ -z "$MISSING" ] || die "no apt candidate for: $MISSING. The image probably pins them to priority -1 in /etc/apt/preferences.d (check: apt-cache policy ninja-build). Re-run with LETTUCE_MODE=container."
fi
aptg install -y -qq "${PKGS[@]}" || die "apt-get install failed"
if ! aptg install -y -qq "clang-$LLVM" "lld-$LLVM" "libc++-$LLVM-dev" "libc++abi-$LLVM-dev" "libomp-$LLVM-dev"; then
  echo "clang-$LLVM not in this distro's apt; adding apt.llvm.org"
  wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh || die "cannot download llvm.sh"
  "${SUDO[@]}" bash /tmp/llvm.sh "$LLVM" all || die "llvm.sh failed"
fi
cmake_old() {
  CMV=$(cmake --version 2>/dev/null | awk 'NR==1{print $3}')
  [ -z "$CMV" ] || [ "$(printf '%s\n' "$CMV" 3.24 | sort -V | head -1)" != "3.24" ]
}
if cmake_old; then
  echo "CMake ${CMV:-missing} is missing or older than 3.24; trying apt"
  aptg install -y -qq cmake || echo "apt cmake install failed"
fi
if cmake_old; then
  echo "CMake ${CMV:-missing} still too old; installing from pip"
  if ! PIPOUT=$(python3 -m pip install --user --quiet "cmake>=3.24" 2>&1); then
    if grep -q externally-managed <<<"$PIPOUT"; then
      python3 -m pip install --user --quiet --break-system-packages "cmake>=3.24" || die "pip install cmake failed"
    else echo "$PIPOUT"; die "pip install cmake failed"; fi
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
cmake --version | head -1

step "build (clang-$LLVM, CUDA via clang)"
for d in build/clang-release build/nvcc-release; do
  HD=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$d/CMakeCache.txt" 2>/dev/null)
  if [ -n "$HD" ] && [ "$HD" != "$PWD" ]; then echo "removing stale $d (configured for $HD)"; rm -rf "$d"; fi
done
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
CUDA_LINE=$("$B/lettuce-tests" cuda 2>&1 | tail -1); echo "$CUDA_LINE"
CUDA_SKIPPED=$(sed -n 's/.* \([0-9][0-9]*\) skipped.*/\1/p' <<<"$CUDA_LINE")
CUDA_FAILED=$(sed -n 's/.* \([0-9][0-9]*\) failed.*/\1/p' <<<"$CUDA_LINE")

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
[ -z "${LETTUCE_IN_CONTAINER:-}" ] || echo "run binaries with: ${LETTUCE_DOCKER:-docker} exec -it ${LETTUCE_CONTAINER:-lettuce-build} bash (cd /work)"
echo "build: $B   unit tests rc=$TESTS_RC   cuda failed=${CUDA_FAILED:-?} skipped=${CUDA_SKIPPED:-?}   mpi tests rc=$MPI_RC   report: $REPORT"
SKIP_RC=0
if [ "$NGPU" -gt 0 ] && [ "${CUDA_SKIPPED:-0}" -gt 0 ]; then
  echo "CUDA tests skipped on a GPU machine: GPU not visible"; SKIP_RC=1
fi
if [ "$TESTS_RC" -eq 0 ] && [ "$MPI_RC" -eq 0 ] && [ "$SKIP_RC" -eq 0 ]; then echo "ALL TESTS PASSED"; exit 0; fi
echo "SOME TESTS FAILED: see above"; exit 1

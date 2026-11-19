#!/bin/bash
# Performance run on one ASA-X H100 node. Submit from the repo root:
#   sbatch cluster/asax_h100.slurm [config.ini] [key=value ...]
# e.g. sbatch cluster/asax_h100.slurm configs/plummer.ini n=10000000 method=tree steps=200
#
# EDIT ME: partition, account and module names below are placeholders. Check them with
#   sinfo -s ; sacctmgr show assoc user=$USER ; module avail cuda llvm clang cmake
#SBATCH --job-name=lettuce
#SBATCH --partition=EDIT_ME_h100_partition
#SBATCH --account=EDIT_ME_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=out/slurm-%j.log

set -euo pipefail

module purge
module load EDIT_ME_cuda/12      # CUDA >= 12.0 (sm_90)
module load EDIT_ME_llvm/18      # Clang >= 17 for sm_90 via clang CUDA; see docs/running.md for the nvcc fallback
module load EDIT_ME_cmake/3.28

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
CUDA_ROOT=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}

# One build directory per node type, reused across jobs.
BUILD=build/asax-h100
if [ ! -x "$BUILD/lettuce" ]; then
  cmake --preset clang-release -B "$BUILD" -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_FLAGS="--cuda-path=$CUDA_ROOT"
fi
cmake --build "$BUILD" -j "$OMP_NUM_THREADS"

nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
"$BUILD/lettuce-tests"

if [ $# -eq 0 ]; then
  # No arguments: the scaling table.
  "$BUILD/lettuce-bench" backend=cuda n=1e5,1e6,1e7,1e8 method=direct,tree steps=5 direct_max=1000000 \
    csv="out/bench-${SLURM_JOB_ID}.csv"
else
  "$BUILD/lettuce" "$@" out_dir="out/run-${SLURM_JOB_ID}"
fi

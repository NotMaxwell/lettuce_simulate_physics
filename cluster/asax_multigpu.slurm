#!/bin/bash
# Multi-GPU run on one ASA-X node: one MPI rank per H100. Submit from the repo root:
#   sbatch cluster/asax_multigpu.slurm [config.ini] [key=value ...]
#   sbatch cluster/asax_multigpu.slurm configs/plummer.ini n=1e8 method=tree steps=200
#
# EDIT ME: partition, account, module names and the GPU count per node (check with
#   sinfo -o "%P %G %N" ; scontrol show node <node> | grep Gres). Confirm your account may request
#   more than one GPU per job: sacctmgr show assoc user=$USER format=account,qos,maxtres
#SBATCH --job-name=lettuce-mgpu
#SBATCH --partition=EDIT_ME_h100_partition
#SBATCH --account=EDIT_ME_account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # = GPUs per node
#SBATCH --gpus-per-task=1            # each rank sees one GPU as device 0
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=out/slurm-%j.log

set -euo pipefail

module purge
module load EDIT_ME_cuda/12
module load EDIT_ME_llvm/18
module load EDIT_ME_openmpi        # any MPI; CUDA-aware is not needed (exchange is host-staged)
module load EDIT_ME_cmake/3.28

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
CUDA_ROOT=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}

BUILD=build/asax-h100
if [ ! -x "$BUILD/lettuce" ] || [ ! -x "$BUILD/lettuce-mpi-tests" ]; then
  cmake --preset clang-release -B "$BUILD" -DLETTUCE_MPI=ON \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_FLAGS="--cuda-path=$CUDA_ROOT"
fi
cmake --build "$BUILD" -j "$OMP_NUM_THREADS"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
# srun launches one rank per task. If this cluster's Open MPI was built without Slurm/PMIx support,
# replace "srun" with "mpirun -n $SLURM_NTASKS --bind-to none".
LAUNCH="srun --cpu-bind=cores"

$LAUNCH "$BUILD/lettuce-mpi-tests"

if [ $# -eq 0 ]; then
  # Strong scaling: the same n on 1, 2 and all GPUs of the node.
  for np in 1 2 "$SLURM_NTASKS"; do
    srun -n "$np" --gpus-per-task=1 --cpu-bind=cores "$BUILD/lettuce-bench" backend=cuda method=tree \
      n=1e7,1e8 steps=5 csv="out/scaling-${SLURM_JOB_ID}-np${np}.csv"
  done
else
  $LAUNCH "$BUILD/lettuce" "$@" out_dir="out/run-${SLURM_JOB_ID}"
fi

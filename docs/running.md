# Running on Brev and ASA-X

## Brev H100 (functional testing)

1. Create an H100 instance with an Ubuntu image that has the NVIDIA driver and CUDA
   (`nvidia-smi` works, CUDA in `/usr/local/cuda`). Clone the repo there.
2. `bash cluster/brev_setup.sh`: installs Clang 18 + libc++ + libomp + lld + Ninja, configures
   `clang-release` against `/usr/local/cuda`, builds, runs `ctest`, then a short bench.
3. Check the CUDA tests actually **ran**: `build/clang-release/lettuce-tests cuda` must report
   `17 run, 0 failed, 0 skipped`. Skipped means no device was found. On a multi-GPU instance the
   script also runs `lettuce-mpi-tests` with one rank per GPU and a 1-vs-all-GPU bench.

Clang 18 officially knows CUDA up to 12.3. With a newer toolkit it warns and usually still works.
If it fails, either install a newer Clang from apt.llvm.org (`wget https://apt.llvm.org/llvm.sh &&
sudo bash llvm.sh 20 all`, then pass `-DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_CUDA_COMPILER=clang++-20`),
or use the fallback: `cmake --preset nvcc-release && cmake --build --preset nvcc-release`.

First things to look at on the GPU:

```bash
B=build/clang-release
$B/lettuce-tests cuda                                  # correctness vs CPU and brute force
$B/lettuce-bench backend=cuda,cpu n=1e4,1e5,1e6 steps=3
$B/lettuce configs/plummer.ini n=1000000 method=tree steps=200 diag_every=50
$B/lettuce configs/collision.ini n=2000000 method=tree steps=3000 render_every=50
```

## ASA-X cluster (performance runs)

`cluster/asax_h100.slurm` is a single-node, single-GPU job. Before the first submit, edit
the lines marked `EDIT_ME`: partition, account, and the CUDA / LLVM / CMake module names.
Find them with `sinfo -s`, `sacctmgr show assoc user=$USER` and `module avail`.

```bash
sbatch cluster/asax_h100.slurm                                   # scaling bench, 1e5..1e8 bodies -> out/bench-<job>.csv
sbatch cluster/asax_h100.slurm configs/plummer.ini n=1e8 method=tree steps=100 diag_every=20
```

The job builds into `build/asax-h100` once and reuses it, runs the test suite, then the bench or
the run you passed. Output: `out/slurm-<job>.log` and `out/run-<job>/`.

If the cluster's LLVM module is older than its CUDA, swap the `cmake --preset clang-release ...`
line for `cmake --preset nvcc-release -B "$BUILD"` (GCC ≥ 12, verified, for the C++23 host code).

For several GPUs in one job, use `cluster/asax_multigpu.slurm` (one rank per GPU; see
[multigpu.md](multigpu.md)). First confirm your account may request more than one GPU per job.

Memory: about 105–117 bytes per body on the GPU in tree mode, so an 80 GB H100 holds roughly 650M
bodies. This is the CUDA backend author's estimate, not yet measured. Direct summation past ~10⁶
bodies is only useful as a check.

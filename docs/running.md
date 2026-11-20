# Running on Brev and ASA-X

## Brev (functional testing)

### Pick the instance
- GPU: H200 (141 GB, `sm_90`, runs the default build unchanged, fits ~1B bodies in tree mode by
  the unmeasured ~110 B/body estimate). H100 is enough for plain functional tests. Take 2+ GPUs
  only to run the multi-GPU tests. Other families need a rebuild with
  `-DCMAKE_CUDA_ARCHITECTURES` (80 A100, 89 L40S/Ada, 100 B200, 120 RTX PRO 6000 Blackwell; the
  last two need CUDA ≥ 12.8); the binary refuses to start on a GPU it was not built for.
- Check "Instance Attributes" before deploying: some providers' instances cannot be stopped and
  bill until deleted. Prefer one that says it can be stopped and restarted.
- Software Configuration → Edit → VM Mode (comes with the driver, CUDA, Python, Docker). Jupyter
  on is fine and gives a browser terminal on port 8888 as a fallback.

### Set up: either way works
- **Setup Script box:** turn on "Run a Setup Script" and paste the whole of
  `cluster/brev_setup.sh`. It clones the fork into the login user's home
  (`~/lettuce_simulate_physics`, override with `LETTUCE_DIR` / `LETTUCE_BRANCH`), re-runs itself
  as that user when Brev starts it as root, and waits for the first-boot apt lock.
- **By hand:** after connecting, `tmux new -s lettuce`, then
  `git clone https://github.com/NotMaxwell/lettuce_simulate_physics.git && cd lettuce_simulate_physics && bash cluster/brev_setup.sh`.

Either way it installs the toolchain, builds, and runs the unit tests, the multi-GPU tests (on
multi-GPU instances), the bench and a 1M-body run. It chooses:
- Clang 18, or Clang 20 when CUDA is newer than 12.3, from apt.llvm.org if the distro lacks it;
- CMake from pip if the distro's is older than 3.24 (Ubuntu 22.04 ships 3.22);
- GCC 12 + nvcc if the Clang build fails.

Overrides, all optional: `CUDA_HOME` (toolkit path if not `/usr/local/cuda` or on `PATH`), `LLVM`
(Clang version for CUDA ≤ 12.3), `LLVM_NEW` (Clang version for CUDA > 12.3).

### Connect and check
```bash
brev login                         # once, on the laptop
brev refresh                       # adds the instance as an ssh host
brev shell <name>                  # or: ssh <name>
tail -f ~/lettuce_simulate_physics/out/brev-report.txt
```
The report should end with `ALL TESTS PASSED`, and the CUDA line should read
`17 run, 0 failed, 0 skipped`. Skipped means the GPU was not found, which is not a pass. The
`build:` line says whether it used `build/clang-release` or the `build/nvcc-release` fallback.

If `nvcc` is missing (driver only), install `cuda-toolkit-12-8` (no newer than the "CUDA Version"
`nvidia-smi` shows) from NVIDIA's apt repo (`cuda-keyring`), then rerun the script.

### Large runs (H200)
```bash
B=build/clang-release
$B/lettuce-bench backend=cuda method=tree n=1e6,1e7,1e8 steps=3
$B/lettuce-bench backend=cuda method=tree n=5e8 steps=2
$B/lettuce-bench backend=cuda method=tree n=1e9 steps=2     # tests the ~110 B/body estimate
```
On several GPUs prefix with `mpirun -n <gpus> --bind-to none`.

Live view from your laptop: start a run with `live=8080`, then
`ssh -L 8080:localhost:8080 <name>` and open `http://localhost:8080/`. The tunnel works even on
instances with fixed ports.

### Finish
```bash
scp -r <name>:lettuce_simulate_physics/out .scratch/brev-out   # on the laptop
brev stop <name>                                                # storage still bills; brev delete when done
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

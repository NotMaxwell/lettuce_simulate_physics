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
  (`~/lettuce_simulate_physics`, override with `LETTUCE_DIR` / `LETTUCE_BRANCH`) and re-runs itself
  as that user when Brev starts it as root.
- **By hand:** after connecting, `tmux new -A -s lettuce`, then
  `git clone https://github.com/NotMaxwell/lettuce_simulate_physics.git && cd lettuce_simulate_physics && bash cluster/brev_setup.sh`.

Where it builds (`LETTUCE_MODE=auto|container|host`, default auto):
- **container** (auto picks it when `docker run --gpus all` works, as on Brev VMs): a persistent
  container `lettuce-build` from `nvidia/cuda:12.8.1-devel-ubuntu24.04` (`LETTUCE_IMAGE`,
  `LETTUCE_CONTAINER`) with the repo mounted at `/work`. Brev images can pin the Ubuntu packages we
  need to apt priority -1 (seen on Nebius), which makes host builds impossible; the container's apt
  is clean, and its CUDA 12.8 suits Clang 20 while the host driver runs it. Binaries need the
  container's libc++/libomp, so run them inside: `docker exec -it lettuce-build bash`, `cd /work`.
- **host**: installs Clang 18 (Clang 20 for CUDA > 12.3, from apt.llvm.org if needed), enables
  `universe`, gets CMake from apt (pip if still < 3.24), and stops with a clear message if a
  package is pinned away. Overrides: `CUDA_HOME`, `LLVM`, `LLVM_NEW`.

Either way: Clang build, GCC + nvcc if that fails, then the unit tests, the multi-GPU tests (on
multi-GPU instances), the bench and a 1M-body run.

### Connect and check
```bash
brev login                         # once, on the laptop
brev refresh                       # adds the instance as an ssh host
brev shell <name>                  # or: ssh <name>
export TERM=xterm-256color         # if your terminal (e.g. Ghostty) is unknown to the VM; tmux needs it
tail -f ~/lettuce_simulate_physics/out/brev-report.txt
```
The report should end with `ALL TESTS PASSED` and the script exits 0. The CUDA line should read
`17 run, 0 failed, 0 skipped`; on a GPU machine the script counts any skipped CUDA test as a
failure. The `build:` line says whether it used `build/clang-release` or the `build/nvcc-release`
fallback. In container mode the outer steps are logged to `out/brev-host.txt`.

### Large runs (H200)
In container mode, first `docker exec -it lettuce-build bash` and `cd /work`.
```bash
B=build/clang-release      # or build/nvcc-release, per the report's "build:" line
$B/lettuce-bench backend=cuda method=tree n=1e6,1e7,1e8 steps=3
$B/lettuce-bench backend=cuda method=tree n=5e8 steps=2
```
Do not run `n=1e9` until the hang below is understood. On several GPUs prefix with
`mpirun -n <gpus> --bind-to none`.

### Measured: Brev H200 SXM, 2026-11-20
Nebius, 1× H200 141 GB, 16 vCPU, driver 580, built with GCC 13 + nvcc 12.8 (`nvcc-release`, the
fallback) inside `nvidia/cuda:12.8.1-devel-ubuntu24.04`. Plummer, eps 0.01, theta 0.6, group walk 64.

| method | n | force ms/step | Ginter/s | CPU, 16 threads, ms/step |
|---|---|---|---|---|
| direct | 1e5 | 5.68 | 1761 | 547.5 |
| tree | 1e6 | 5.40 | 472 | 379.3 |
| tree | 1e7 | 50.9 | 532 | |
| tree | 1e8 | 532 | 538 | |
| tree | 5e8 | 2771 | 532 | |

1M-body Plummer, 100 steps of dt 1/128: dE/E0 = −1.5e-6, 5.24 ms/step.

Open issues from that run:
- `n=1e9` printed the bench header, then the VM stopped responding (sshd unreachable; the instance
  had to be deleted). 5e8 ran fine. Suspect host memory (200 GB RAM) during setup or upload, not
  GPU memory; not yet investigated.
- Two CUDA tests fail on the GPU (both duplicate-Morton-key cases): "tree with coincident bodies"
  (theta 0 rel rms 5.1e-5 > 1e-5) and "group walk at theta 0 is exact, with groups split by
  duplicate keys" (interaction count ≠ 3000·2999, pe off by 8.4e-5). The group walk had never run on
  a GPU before this.
- The Clang CUDA build failed: CUDA 12.8's `crt/host_defines.h` rejects libc++ on x86.

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

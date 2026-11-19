# Running on Brev and ASA-X

## Brev H100 (functional testing)

1. Create an instance with one H100 (2 or more to test multi-GPU) and an Ubuntu 22.04/24.04 image
   that has the NVIDIA driver and CUDA. Connect with `brev shell <name>` or plain ssh.
2. `git clone` the fork, then `git checkout claude/hopeful-faraday-w1o2ca`.
3. `bash cluster/brev_setup.sh`. It installs the toolchain, builds, and runs the unit tests, the
   multi-GPU tests (on multi-GPU instances), the bench and a 1M-body run. It chooses:
   - Clang 18, or Clang 20 when CUDA is newer than 12.3, from apt.llvm.org if the distro lacks it;
   - CMake from pip if the distro's is older than 3.24 (Ubuntu 22.04 ships 3.22);
   - GCC 12 + nvcc if the Clang build fails.

   Everything is logged to `out/brev-report.txt`.
4. Check the end of the report for `ALL TESTS PASSED`, and that the CUDA line reads
   `17 run, 0 failed, 0 skipped`. Skipped means the GPU was not found, which is not a pass.

Live view from your laptop: start a run with `live=8080`, forward the port, then open
`http://localhost:8080/`.

```bash
ssh -L 8080:localhost:8080 <instance>
```

Brev also offers its own port forwarding; either works.

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

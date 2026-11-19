# lettuce: celestial-body gravity simulation

A 3D Newtonian N-body simulation in **C++23 + CUDA**, built with **Clang/LLVM**, aimed at NVIDIA
H100 GPUs. Bodies interact only through softened gravity (no collisions or destruction) and are
integrated with a kick-drift-kick leapfrog. Forces come from exact direct summation or a
Barnes-Hut tree with quadrupole moments, on the GPU (CUDA) or the CPU (OpenMP). Runs are
headless: they print conservation diagnostics and write a CSV, binary snapshots (restartable) and
PNG density renders.

## Quick start

```bash
cmake --preset clang-release          # or clang-cpu without the CUDA toolkit
cmake --build --preset clang-release
ctest --preset clang-release
build/clang-release/lettuce configs/plummer.ini
```

Needs CMake ≥ 3.24, Ninja, Clang ≥ 17 with libc++ and libomp, lld, and for the GPU build CUDA
≥ 12.0. MPI (any implementation) is optional and enables multi-GPU runs. If CUDA is not in `/usr/local/cuda`, add
`-DCUDAToolkit_ROOT=<cuda> -DCMAKE_CUDA_FLAGS=--cuda-path=<cuda>`. The `nvcc-release` preset
(GCC + nvcc) is the fallback where Clang is older than the installed CUDA. Running on Brev and the
ASA-X cluster: [docs/running.md](docs/running.md).

## Starting configurations

Each file in [`configs/`](configs/) is a ready-to-run setup, and any key can be overridden on the
command line:

```bash
build/clang-release/lettuce configs/disk.ini n=1000000 method=tree disk.thickness=0.1 render_every=50
build/clang-release/lettuce scenario=kepler kepler.e=0.9 steps=50000 dt=0.0005 eps=0
build/clang-release/lettuce --list                     # scenarios
build/clang-release/lettuce configs/solar.ini --print  # effective configuration
```

| Scenario | What it is |
|---|---|
| `plummer` | star cluster in virial equilibrium |
| `kepler` | two-body orbit (integrator accuracy check) |
| `uniform` | uniform sphere/cube, cold collapse, optional spin |
| `disk` | rotating exponential disk galaxy around a central mass |
| `collision` | two inclined disk galaxies colliding |
| `solar` | the Sun, eight planets (J2000 elements) and an asteroid belt; AU, years, solar masses |
| `systems` | star cluster in which every star has its own planets |

Keys, units and per-scenario parameters: [configs/README.md](configs/README.md).

## Watch it live

```bash
build/clang-release/lettuce configs/disk.ini steps=0 live=8080   # open http://localhost:8080/
```

Drag to rotate, scroll to zoom, colour by density or speed; pause, single-step or stop the run
from the page. On a remote GPU machine, forward the port with `ssh -L 8080:localhost:8080 <host>`.
Details, including ASA-X compute nodes: [docs/live.md](docs/live.md).

## Multiple GPUs

```bash
mpirun -n 4 build/clang-release/lettuce configs/plummer.ini n=1e8 method=tree   # one GPU per rank
```

Bodies are split across GPUs by a cost-balanced space-filling curve, and the ranks exchange
locally essential trees. Accuracy matches a single GPU. Design, measurements and limits:
[docs/multigpu.md](docs/multigpu.md). On ASA-X: `cluster/asax_multigpu.slurm`.

## Output

Everything goes to `out_dir` (default `out/`):

- stdout and `diag.csv`: energy, relative energy error, kinetic/potential energy, virial ratio
  2K/|W|, momentum, angular-momentum error and ms/step, every `diag_every` steps
- `config.ini`: the effective configuration (rerun it to reproduce the run)
- `snap_<step>.snap` every `snap_every` steps; resume with `restart=<file>`
- `frame_<step>.png` every `render_every` steps; `lettuce-render <snap> <png> view=30,60`
  renders a snapshot afterwards

## Benchmark

```bash
build/clang-release/lettuce-bench backend=cuda,cpu n=1e4,1e5,1e6 method=direct,tree steps=5
```

Prints force time split into sort/build/walk, ms per full step, and interactions per second.
Measured on a 4-core 2.1 GHz Xeon, CPU backend only (no GPU numbers yet):

| method | n | ms / force | Ginteractions/s |
|---|---|---|---|
| direct | 1e4 | 13.2 | 7.6 |
| direct | 1e5 | 1166 | 8.6 |
| tree θ=0.6 | 1e5 | 98 | 3.4 |
| tree θ=0.6 | 1e6 | 929 | 3.7 |

## Layout

- `src/` — `bodies.hpp`, `config.*`, `scenarios.*`, `backend.hpp`, `cpu_backend.cpp`,
  `cuda_backend.cu`, `simulation.*`, `snapshot.*`, `render.*`, `live.*`, `parallel.*` (MPI), `main.cpp`, `bench.cpp`, `render_main.cpp`
- `tests/` — unit and physics tests (`build/<preset>/lettuce-tests [filter]`); CUDA tests skip without a GPU
- `configs/` — starting configurations · `cluster/` — Brev setup and ASA-X Slurm job
- `docs/design.md` — physics, algorithms, precision · `vendor/` — stb_image_write

## CLion

Open the repository folder. CLion reads `CMakePresets.json` and offers the presets as CMake
profiles (`clang-release`, `clang-debug`, `clang-cpu`, ...). Enable the ones you want under
Settings → Build, Execution, Deployment → CMake. Without a local CUDA toolkit, use
`clang-cpu`. For the GPU build, set `CUDAToolkit_ROOT` in the profile's CMake options if
CUDA is not in `/usr/local/cuda`. To build and run on a Brev instance from your machine, add it
as a Remote Development host or a remote toolchain over SSH.

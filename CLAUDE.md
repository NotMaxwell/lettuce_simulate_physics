# Rules for AI agents working in this repo

## Project
A celestial-body gravity simulation (3D Newtonian N-body) in C++23 + CUDA. Headless: it writes
diagnostics, snapshots and PNG renders. Targets: NVIDIA H100 (sm_90). Functional testing on a
Brev H100 instance, performance runs on the ASA-X cluster's H100 nodes. The CPU backend (OpenMP)
is the fallback and the reference the CUDA backend is checked against.

## Hard rules
- All work goes to the fork `NotMaxwell/lettuce_simulate_physics` (remote `origin`).
  Never add, commit to, push to, or open PRs against `commonLuke/lettuce_simulate_physics`.
- Languages: C++23 for host code (no std::print, GCC 13 lacks it), CUDA for kernels (clang CUDA,
  C++23; the nvcc fallback tops out at C++20, so .cu files and headers they include stay C++20-clean),
  CMake for the build. Toolchain is Clang/LLVM (libc++, lld, libomp). Vendored single headers
  live in `vendor/`. Job scripts for clusters live in `cluster/` (Slurm/bash is fine there only).
- Measure performance with `lettuce-bench` before and after a change; report real numbers, never
  guesses. Numbers from this container (no GPU) are CPU-only; say so.
- Comment only what the code does not make obvious: why a choice was made, a non-obvious
  invariant, a physics or numerics subtlety. No narration comments.
- Report honestly: failing builds/tests are stated with output; skipped steps are named
  (CUDA tests skip without a GPU — that is "not run", not "passed").
- Ask the user when a decision is genuinely theirs (physics scope, deleting things, pushing
  anywhere new). Otherwise pick the sensible default, say so, and keep going.

## Build and verify
- Presets (also what CLion imports): `cmake --preset clang-release` (Clang for host + CUDA),
  `clang-cpu` (no CUDA), `clang-debug`, `nvcc-release` (GCC + nvcc fallback). Build dirs are
  `build/<preset>`; `cmake --build --preset <p>` then `ctest --preset <p>`.
- In this dev container: no GPU, CUDA code is compile-checked only. Clang CUDA needs
  `-DCUDAToolkit_ROOT=$PWD/.scratch/cuda-root -DCMAKE_CUDA_FLAGS=--cuda-path=$PWD/.scratch/cuda-root`
  (symlinked standard layout over Ubuntu's split CUDA 12.0 package); nvcc 12.0 needs
  `-DCMAKE_CUDA_HOST_COMPILER=g++-12`.
- Tests: `build/<preset>/lettuce-tests [name-filter]`. Run: `build/<preset>/lettuce configs/plummer.ini
  [key=value ...]`. Bench: `build/<preset>/lettuce-bench`.
- Scratch files and extra build dirs go in `.scratch/` (git-ignored), never `/tmp`.

## Architecture
- `bodies.hpp` host struct-of-arrays · `config.*` .ini + `key=value` overrides
- `scenarios.*` initial conditions (each reads its own `<scenario>.<key>` params)
- `backend.hpp` interface: state stays on the device between steps
- `cpu_backend.cpp` OpenMP direct + Barnes-Hut · `cuda_backend.cu` CUDA direct + Barnes-Hut
- `simulation.*` KDK leapfrog driver, diagnostics, output cadence
- `snapshot.*` binary snapshots/restart · `render.*` PNG density projection
- `main.cpp` (`lettuce`), `bench.cpp` (`lettuce-bench`), `render_main.cpp` (`lettuce-render`)
- Design notes: `docs/design.md`. Running on Brev / ASA-X: `docs/running.md`.

## Subagents and cost
- The lead agent manages: defines interfaces, delegates modules, integrates, verifies.
- Use the cheapest model that can do the job: Haiku for boilerplate, Sonnet for ordinary
  module work, Opus for hard algorithmic/performance work, Fable only for high-reasoning tasks.
- Give each subagent one owned file set and a way to verify alone; do not have two agents
  edit the same file. Keep prompts and reports short; do not re-read what is already known.

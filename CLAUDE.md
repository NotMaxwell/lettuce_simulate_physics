# Rules for AI agents working in this repo

## Project
A celestial-body gravity simulation (3D Newtonian N-body) in C++23 + CUDA. Headless: it writes
diagnostics, snapshots and PNG renders. Targets: NVIDIA H100 (sm_90). Functional testing on a
Brev H100 instance, performance runs on the ASA-X cluster's H100 nodes. The CPU backend (OpenMP)
is the fallback and the reference the CUDA backend is checked against.
`python/` is the original pygame toy this project grew out of; it is history, do not edit it.

## Hard rules
- All work goes to the fork `NotMaxwell/lettuce_simulate_physics` (remote `origin`).
  Never add, commit to, push to, or open PRs against `commonLuke/lettuce_simulate_physics`.
- Languages: C++23 for host code (GCC >= 13; no std::print, GCC 13 lacks it), CUDA (C++17 dialect, nvcc 12 has no C++23) for kernels; headers included by .cu files stay C++17-clean, CMake for the build. Vendored single headers
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
- Configure: `cmake -S . -B build` (Release by default; CUDA is used when `nvcc` is found,
  `-DLETTUCE_CUDA=OFF|ON` to force). `-DCMAKE_CUDA_ARCHITECTURES="80;90"` to add A100.
- In this dev container CUDA 12.0 needs `-DCMAKE_CUDA_HOST_COMPILER=g++-12`. There is no GPU here:
  CUDA code is compile-checked only.
- `cmake --build build -j` then `ctest --test-dir build --output-on-failure`
  (or `build/lettuce-tests [name-filter]`).
- Run: `build/lettuce configs/plummer.ini [key=value ...]`. Bench: `build/lettuce-bench`.
- Scratch files and build dirs go in `.scratch/` (git-ignored), never `/tmp`.

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

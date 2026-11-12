# Rules for AI agents working in this repo

## Project
A merge-sandbox physics game, being rewritten from Python/pygame into Zig 0.17 + SDL3.
Goal: 1000+ balls at a locked 60 fps with headroom, and the game should visibly show off
its optimizations (perf overlay, stress mode). The Python files at the repo root are the
behavioural reference for the port; do not edit them.

## Hard rules
- All work goes to the fork `NotMaxwell/lettuce_simulate_physics` (remote `origin`).
  Never add, commit to, push to, or open PRs against `commonLuke/lettuce_simulate_physics`.
- If it can be Zig, it is Zig. C is allowed only for the platform layer (SDL3) and the
  vendored single-header decoders in `vendor/`. No Python, shell or other-language tooling
  in the shipped game or build.
- CPU and GPU optimizations are both welcome (SIMD, threads, batching, spatial structures).
  Measure before and after with `zig build bench`; report real numbers, never guesses.
- Comment only what the code does not make obvious: why a choice was made, a non-obvious
  invariant, a deliberate deviation from the Python original. No narration comments.
- Report honestly: failing builds/tests are stated with output; skipped steps are named.
- Ask the user when a decision is genuinely theirs (game design, deleting things, pushing
  anywhere new). Otherwise pick the sensible default, say so, and keep going.

## Zig 0.17 specifics
- Zig changes between releases. Do not write std-lib calls from memory: check
  `/usr/lib/zig/std` (grep it) and build often.
- C is imported through the `c` module (`@import("c")`, generated from `src/c.h` by
  `addTranslateC` in `build.zig`), not `@cImport`.
- Use SDL for timing (`SDL_GetPerformanceCounter`) and file loading (`SDL_LoadFile`);
  allocator is `std.heap.smp_allocator`; containers are unmanaged (`.empty`, pass `gpa`).

## Build and verify
- `zig build` / `zig build run` — the game (optimized by default; `-Doptimize=debug` for bounds/overflow checks).
- `zig build test` — unit tests. `zig build bench -- <balls> <ticks>` — headless physics timing.
- Headless runs: `SDL_VIDEODRIVER=offscreen SDL_AUDIODRIVER=dummy` (gives a real OpenGL renderer).
- Scratch files go in `.scratch/` (git-ignored), never `/tmp`.

## Architecture (see `src/world.zig` for the data contract)
- `config.zig` constants · `world.zig` data (balls are a `MultiArrayList`, struct-of-arrays)
- `physics.zig` fixed 60 Hz simulation tick · `effects.zig` fire particles, ash, explosions
- `render.zig` batched GPU drawing, camera, HUD · `audio.zig` sound · `main.zig` loop + input
- Simulation never touches SDL; rendering never mutates the world.

## Subagents and cost
- The lead agent manages: defines interfaces, delegates modules, integrates, verifies.
- Use the cheapest model that can do the job: Haiku for boilerplate, Sonnet for ordinary
  module work, Opus for hard algorithmic/performance work, Fable only for high-reasoning tasks.
- Give each subagent one owned file set and a way to verify alone; do not have two agents
  edit the same file. Keep prompts and reports short; do not re-read what is already known.

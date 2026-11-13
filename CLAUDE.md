# Rules for AI agents working in this repo

## Project
A merge-sandbox physics game in Zig 0.17 + SDL3, rewritten from a Python/pygame original.
Goal: 1000+ balls at a locked 60 fps with headroom, and the game should visibly show off
its optimizations (perf overlay, stress mode). The original Python version lives in `python/`
as the behavioural reference; do not edit it.

## Hard rules
- All work goes to the fork `NotMaxwell/lettuce_simulate_physics` (remote `origin`).
  Never add, commit to, push to, or open PRs against `commonLuke/lettuce_simulate_physics`.
- If it can be Zig, it is Zig. C is allowed only for the platform layer (SDL3) and the
  vendored single-header decoders in `vendor/`. No Python, shell or other-language tooling
  in the shipped game or build. (`.devcontainer/` is dev-only tooling, not part of either.)
- CPU and GPU optimizations are both welcome (SIMD, threads, batching, spatial structures).
  Measure before and after with `zig build bench`; report real numbers, never guesses.
- Comment only what the code does not make obvious: why a choice was made, a non-obvious
  invariant, a deliberate deviation from the Python original. No narration comments.
- Report honestly: failing builds/tests are stated with output; skipped steps are named.
- Ask the user when a decision is genuinely theirs (game design, deleting things, pushing
  anywhere new). Otherwise pick the sensible default, say so, and keep going.

## Zig 0.17 specifics
- Zig changes between releases. Do not write std-lib calls from memory: check
  `/usr/lib/zig/std` (grep it; `/opt/zig/lib/std` in the dev container) and build often.
- C is imported through the `c` module (`@import("c")`, generated from `src/c.h` by
  `addTranslateC` in `build.zig`), not `@cImport`.
- Use SDL for timing (`SDL_GetPerformanceCounter`) and file loading (`SDL_LoadFile`);
  allocator is `std.heap.smp_allocator`; containers are unmanaged (`.empty`, pass `gpa`).

## Build and verify
- `zig build` / `zig build run` — the game (optimized by default; `-Doptimize=debug` for bounds/overflow checks).
- `zig build test` — unit tests (one needs a Vulkan device). `zig build bench -- <balls> <ticks>` — headless physics timing.
- Headless runs: `SDL_VIDEODRIVER=offscreen SDL_AUDIODRIVER=dummy`; the SDL_GPU (Vulkan) device
  still works, only presenting is skipped. The dev container sets both by default.
- Scratch files go in `.scratch/` (git-ignored), never `/tmp`.

## Architecture (see `src/world.zig` and `src/gpu_layout.zig` for the data contracts)
- `config.zig` constants · `world.zig` CPU data (balls are a `MultiArrayList`, struct-of-arrays)
- `physics.zig` / `effects.zig` CPU simulation, now only the `zig build bench` baseline (see TODO.md)
- `gpu_world.zig` GPU balls: owns the solver, gravity and game kernels, records one command buffer per tick
- `gpu_sim/` solver and gravity compute kernels · `gpu_game/` merge, fire, spawn, action kernels
- `gpu_layout.zig` host/shader struct layouts · `gfx.zig` SDL_GPU device and batched drawing
- `render.zig` camera, HUD · `audio.zig` sound · `main.zig` loop + input
- Simulation never touches SDL; rendering never mutates the world.

## Dev container
Optional. Gives a pinned Zig 0.17, Mesa Vulkan (RADV/ANV with `/dev/dri`, lavapipe CPU
fallback without it) and Claude Code, isolated from the host. Use it for builds, tests, bench
and sandboxed agent sessions; play the game windowed on the host.
Verified 2026-11-12 (Arch base, Mesa 26.2.7): tests, autoplay and bench match the host. The first bench
run in a fresh cache can be far slower; repeat it.

One-time host setup: `sudo usermod -aG docker $USER`, then log out and back in.

- **VS Code / devcontainer CLI:** "Dev Containers: Reopen in Container", or
  `devcontainer up --workspace-folder . && devcontainer exec --workspace-folder . bash`.
- **Plain Docker** (repo mounted at the same absolute path so git worktrees still resolve):
  ```bash
  docker build -t lettuce-dev .devcontainer      # add --build-arg USER_UID=$(id -u) if not 1000
  docker run --rm -it --device /dev/dri \
    -v "$PWD:$PWD" -w "$PWD" \
    -v lettuce-zig-cache:/home/dev/.cache/zig \
    -v lettuce-claude-config:/home/dev/.claude \
    lettuce-dev bash
  ```
- Inside: `zig build test`, `zig build bench -- 1000 600`, or `claude` (log in once; the
  `lettuce-claude-config` volume keeps the login for later containers). `vulkaninfo --summary`
  shows whether you got the real GPU or lavapipe; bench numbers from lavapipe are not comparable.
- **Parallel agent sessions:** one worktree + one container each. The main repo must be mounted
  too, because a worktree's `.git` file points into it by absolute path:
  ```bash
  git worktree add ../lettuce-gravity -b gravity && WT=$(realpath ../lettuce-gravity)
  docker run --rm -it --device /dev/dri -v "$PWD:$PWD" -v "$WT:$WT" -w "$WT" \
    -v lettuce-zig-cache:/home/dev/.cache/zig -v lettuce-claude-config:/home/dev/.claude \
    lettuce-dev claude
  ```
  `claude --dangerously-skip-permissions` is acceptable only inside a container; it still has
  network access and the repo mounted read-write.
- **Headless container agent** (how the lead delegates): write the task prompt to
  `.scratch/prompts/<name>.md`, create the worktree from the current branch, then run in the background
  ```bash
  docker run --rm --device /dev/dri -v "$PWD:$PWD" -v "$WT:$WT" -w "$WT" \
    -v lettuce-zig-cache:/home/dev/.cache/zig -v lettuce-claude-config:/home/dev/.claude \
    lettuce-dev claude -p --agent builder --dangerously-skip-permissions \
    "$(cat .scratch/prompts/<name>.md)" > .scratch/agents/<name>.log 2>&1
  ```
  The agent commits on its worktree branch; a second container agent (`--agent architect` or `builder`)
  reviews the branch diff before the lead merges.
- Reset: `docker volume rm lettuce-zig-cache lettuce-claude-config` (the second logs you out).

## Subagents and cost
@.claude/multi-agent.md

Repo-specific ownership:
- Lead-only (shared contracts): `build.zig`, `build.zig.zon`, `src/world.zig`, `src/gpu_layout.zig`,
  `src/config.zig`, `CLAUDE.md`, `.claude/`, `.devcontainer/`. Agents propose changes to these in
  their report instead of editing them.
- Natural owned sets: `src/gpu_sim/` · `src/gpu_game/` (+ `gpu_world.zig`) · `src/gfx.zig` +
  `src/shaders/` · `src/render.zig` · `src/audio.zig` · `src/main.zig`.
- Kernel modules cannot import outside their own directory, so `gpu_sim/common.zig` and
  `gpu_game/common.zig` are duplicated on purpose; keep them in step when one changes.
- Perf work goes to `optimizer` and is verified with `zig build bench` on the real GPU, before and after.
- `isolation: "worktree"` branches from `main`, not the current branch. For parallel agents on a feature
  branch, create the worktree yourself (`git worktree add -b <name> ../lettuce-<name> HEAD`) and give its path.

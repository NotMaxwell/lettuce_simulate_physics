# TODO

## Dead-code cleanup (left over from the CPU → GPU move)
- `src/gpu_sim`: remove the Jacobi solver (`Solver.jacobi`, `k_contacts.zig`, `k_collide.zig`, the
  Jacobi branches in `collide.zig`/`k_finish.zig` and its tuning fields). It is not the default and
  fails the settling check (piles collapse).
- `src/gpu_sim/collide.zig`: `Sim.gridView()` has no callers; the fire kernels read `counts`,
  `sorted_id` and `ball_cell` directly. Switch them to `gridView()` (or delete it) so the grid
  layout lives in one place.
- `src/gpu_sim/collide.zig`: the `.bin` timing part records nothing since binning moved into integrate.
- `src/render.zig`: the CPU ball path (`drawTrails`, `drawBalls`, `drawParticles`) is unreachable from
  the game now that main always draws GPU balls. Keep only what `render_demo.zig` still needs, or move
  the demo to GPU balls and delete it.
- `src/world.zig` / `src/physics.zig` / `src/effects.zig`: CPU ball simulation is now only used by
  `zig build bench` (CPU baseline) and for bounds helpers. Decide whether the CPU baseline stays;
  if not, strip it down to the helpers.

## Bugs
- After every ball has died (e.g. all 1M burnt out), the gravity stage sampled ~100 ms and the tick
  stalled for hundreds of ms. Suspect the FMM with all masses zero (`src/gpu_sim/gravity.zig`).
- Merges are capped at ~16k per tick by the event buffer (`max_events`).

## Not yet ported to the GPU
- Impact heat (the solver does not expose approach speed); ash-ash collisions.
- Trails (off in GPU mode).
- Extra impulse on release after a grab; click pick/spawn arrives 2–3 ticks late.
- HUD max level never decreases.

## Unverified
- Windowed play with vsync, resize, high-DPI mouse scaling, and audio (all runs so far are headless).
- `Sim.elastic = false` (retention-based ball restitution) at scale.

# TODO

## Dead-code cleanup (left over from the CPU → GPU move)
- `src/render.zig`: the CPU ball path (`drawTrails`, `drawBalls`, `drawParticles`) is unreachable from
  the game now that main always draws GPU balls. Keep only what `render_demo.zig` still needs, or move
  the demo to GPU balls and delete it.
- The CPU ball simulation (`world.zig` / `physics.zig` / `effects.zig`) stays as the `zig build bench`
  baseline for comparison (decided 2026-11-12).

## Bugs
- Reported: the window closes instantly during merging (user, windowed, release build). On the RX 7800 XT
  (RADV) every start crashed in `vkCreateComputePipelines` until fa8009a (invalid SPIR-V from
  `blockSum`), which may have been this. Needs a windowed retest on that machine.
- Merges are capped at ~16k per tick by the event buffer (`max_events`).
- Zig's SPIR-V backend emits one validation error in every kernel (VUID-StandaloneSpirv-OpTypeRuntimeArray-04680,
  missing `Block` decoration). RADV accepts it today; a stricter driver may not.

## Not yet ported to the GPU
- Impact heat (the solver does not expose approach speed); ash-ash collisions.
- Trails (off in GPU mode).
- Extra impulse on release after a grab; click pick/spawn arrives 2–3 ticks late.
- HUD max level never decreases.

## Unverified
- Windowed play with vsync, resize, high-DPI mouse scaling, and audio (all runs so far are headless).
- `Sim.elastic = false` (retention-based ball restitution) at scale.

## Galaxy mode (feature chain)
Balls orbiting each other in open space and forming little systems. Gravity stays 2D (log potential:
pull g·m/r), so circular speed is sqrt(g·M_enclosed) at any radius; disks have flat rotation curves.
1. [x] Zoom/pan/follow camera (wheel, middle-drag, Home, C).
2. [x] Scenario builders, `src/galaxy.zig`: disk, colliding disks, cold cloud, cluster of star systems.
3. [ ] Galaxy mode in the GPU world: explicit g and softening, fixed open bounds, collision modes
       solid / accrete / ghost, keys G, 1-4, X, [ ], click spawns a star system; `--galaxy`, `--collide`.
4. [ ] GPU trails so orbits are visible (key R).
5. [ ] Diagnostics on the HUD: energy and angular-momentum drift.

## Scale: features at 1M balls, optimize toward 1B (standing goal, 2026-11-12)
End goal: simulate small-universe matter on the Alabama supercomputer (student account). Every feature
must work at 1M balls on one GPU; every design choice should keep a path open to 1B.
Today (RX 7800 XT): 1M balls 2.39 ms/tick with gravity, 1.48 ms without; 100k 0.32-0.45 ms.
1. [ ] Profile per pass at 1M/4M/16M (GPU timestamps, not blocking samples) and publish the table
       in the perf overlay and `zig build bench`.
2. [ ] Raise `max_balls` (1 << 20 today) to what VRAM allows; find the real per-ball byte cost.
3. [ ] Lean "matter" layout for galaxy/universe mode: pos + vel + mass (~20 B/body) without the
       merge-game fields (heat, state, body.friction...), so 1B bodies is ~20 GB, i.e. multi-GPU.
4. [ ] Kernel work: fuse passes, Morton/Hilbert ordering for locality, indirect dispatch sized on
       the GPU, no per-tick CPU readback that scales with N.
5. [ ] Rendering at scale: point sprites / density splatting for sub-pixel bodies, LOD, culling.
6. [ ] Precision: f32 positions break down over large extents; cell-relative or fixed-point coords.
7. [ ] Headless "sim only" path for clusters: no window, periodic snapshots to disk, offline rendering.
8. [ ] Past one GPU: domain decomposition, distributed tree/FMM or TreePM, multi-node communication.
Open questions for the user (block 3, 6, 8): 2D log-potential gravity or 3D 1/r^2 for the universe
sim; what GPUs, drivers (Vulkan available?) and scheduler the cluster offers.

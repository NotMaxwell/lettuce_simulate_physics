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
- Zig's SPIR-V backend omits the `Block` decoration (VUID-04680); `src/tools/spirv_fixup.zig` patches
  it at build time (9d83558). Every shader passes `spirv-val --target-env vulkan1.2`
  (gravity_kernels fixed on the sim branch).
- The game clamps escaped bodies into a fixed, loose gravity box (pull from the box edge). The sim's
  GPU-computed tight box made 1M disk 6.8 ms/tick vs the game's 16 ms: try it in galaxy mode.

## Not yet ported to the GPU
- Impact heat (the solver does not expose approach speed); ash-ash collisions.
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
3. [x] Galaxy mode in the GPU world: explicit g and softening, fixed open bounds, collision modes
       solid / accrete / ghost (galaxy-only), keys O, 1-4, X, [ ], click spawns a star system;
       `--galaxy`, `--collide`. Body radius capped at 10 (`galaxy.max_body_radius`) so solid mode is
       playable; this costs ~20% per tick (disk 100k 1.57 -> 1.95 ms, 1M 13.9 -> 16.4 ms; cap 40 is
       baseline). Worth revisiting: why a radius cap slows the gravity pass.
4. [x] GPU trails so orbits are visible (key R); cleared whenever the world is cleared. Only checked
       headless: look at the head segment and R toggle in a window.
5. [ ] Diagnostics on the HUD: energy and angular-momentum drift.

## Scale (game): features at 1M balls, optimize toward 1B (standing goal, 2026-11-12)
Every game feature must work at 1M balls on one GPU.
Today (RX 7800 XT): 1M balls 2.39 ms/tick with gravity, 1.48 ms without; 100k 0.32-0.45 ms;
galaxy disk 1M 13.8 ms/tick (gravity 12.7).
1. [ ] Profile per pass at 1M/4M/16M (GPU timestamps, not blocking samples) and publish the table
       in the perf overlay and `zig build bench`.
2. [ ] Raise `max_balls` (1 << 20 today) to what VRAM allows; find the real per-ball byte cost.
3. [ ] Kernel work: fuse passes, Morton/Hilbert ordering for locality, indirect dispatch sized on
       the GPU, no per-tick CPU readback that scales with N.
4. [ ] Rendering at scale: point sprites / density splatting for sub-pixel bodies, LOD, culling.

## Sim version: headless matter simulation (`zig build sim`)
End goal: simulate small-universe matter, ~1-2B bodies, on the Alabama supercomputer (student
account). A separate headless executable that shares the gravity kernels but none of the merge-game
state. Staging: 100M bodies on the local RX 7800 XT (16 GB), then multi-GPU on one cluster node,
then multi-node to 2B.
Rough sizing for 2B (estimate, to be measured): lean layout ~20 B/body in 2D (40 GB) or ~28 B in 3D
(56 GB), about 2.5x that with tree/sort scratch, so 2-8 datacenter GPUs. Linear extrapolation from
1M gives 5-30 s/tick on one 7800-class GPU: batch jobs of hours, not real time.

Decisions for the user (block the phases marked):
- D1 Gravity: keep 2D log potential, or 3D 1/r^2 (real matter; needs a 3D tree, more memory). Blocks 2+.
- D2 Cluster: which system, which GPUs, is Vulkan usable on compute nodes (else a CUDA/OpenCL backend),
     scheduler (Slurm?), GPUs per job, walltime, storage quota. On a compute node:
     `nvidia-smi`, `vulkaninfo --summary`, `sinfo`. Blocks 10+.
- D3 Multi-node communication: MPI is C, an exception to "if it can be Zig, it is Zig"; the
     alternative is a Zig-native transport. Job scripts are shell: allowed as ops tooling like
     `.devcontainer/`? Blocks 12+.
- D4 Initial conditions: scaled-up toy scenarios (disk, cloud, collisions) or cosmological (periodic
     box, comoving coordinates, Zel'dovich ICs from a power spectrum). Blocks 15.

Phase 0, groundwork (no decision needed, 2D for now):
1. [x] `src/sim/` + `zig build sim`: headless SDL_GPU device, no window or audio. CLI:
       `--n --ticks --scenario --dt --out <dir> --every <k> --bench`.
2. [~] Lean matter buffers: struct-of-arrays pos, vel, mass, nothing else; layout contract in
       `gpu_layout.zig`. Dimension behind one comptime switch so D1 is a flag, not a rewrite.
       Today 56 B/body (pos 8, vel 8, BallBody 16 of which only mass is read, keys 8, sorted 16) plus
       a fixed ~2.2 GB gravity pyramid at level 12. Next: separate f32 mass buffer (-12 B), `sorted`
       without id (-4 B). Dimension switch not done yet.
3. [x] Gravity-only tick on the existing FMM with a symplectic kick-drift-kick (leapfrog)
       integrator; no collisions or merging. Tree box computed on the GPU every tick (no readback).
       1M disk: |dE/E0| ~3e-5, dL/L0 ~5e-7 over 1000 ticks. At default dt, 200k `systems` and
       `collision` drift dE/K0 2-8% in 400 ticks (close pairs): needs item 13 or a smaller dt.
4. [x] Diagnostics by GPU reduction every k ticks: total energy, momentum, angular momentum,
       f64 accumulation; printed. PE is exact up to ~46k bodies, a hashed sample above (drift only).
5. [x] Snapshot format: small header + raw struct-of-arrays chunks; written asynchronously from a
       staging buffer; restart from a snapshot (checkpoints, for walltime limits).
6. [x] Offline renderer: snapshot -> density image / frame sequence by GPU splatting
       (`zig build render-snap`), so cluster output can be viewed locally.
7. [~] Physics tests: two-body orbit period, equilibrium disk (2D) / Plummer sphere (3D) stays
       virialised, energy drift bound over 10k ticks, snapshot round-trip, restart is bit-identical.
       Done: two-body period, 2000-body disk drift over 2000 ticks, snapshot round-trip, truncated
       snapshot rejected, restart vs continuous within 2e-3 (not bit-identical: float atomics in
       the gravity solver make two identical runs differ; exact repeatability needs a fixed-order
       gravity sum), render smoke test. Saturating splat untested; snapshots/render untested at 100M+.
8. [x] `--bench` per-pass timings (blocking samples, not timestamps yet). RX 7800 XT, disk, ms/tick:
       1M 6.8 · 4M 24 · 16M 110 · 64M 505 · 160M 942 · 200M 1431 · 220M 1723 (largest; 240M loses the
       device after upload, ~15.7 GB needed vs ~15.1 free, VRAM overcommit suspected, unconfirmed).
       Ticks are split into several submits at n >= 8M to stay under the driver watchdog.

Snapshots: 256 B header + planar f32 pos/vel/mass (4096-aligned) + footer, written to .tmp then
renamed; ~1% overhead (async, chunked). `zig build render-snap -- <snaps> --out DIR --log`.

Phase 1, one GPU at scale (target 100M locally):
9.  [ ] Memory audit: real bytes/body including tree and sort scratch; buffers past the per-binding
        limit (maxStorageBufferRange, often 4 GB) split into chunks.
10. [ ] GPU radix sort into Morton/Hilbert order every few ticks for locality.
11. [ ] Indirect dispatch sized on the GPU; no CPU readback that scales with N.
12. [ ] Precision: f32 positions break down over large extents; cell-relative or fixed-point coords.
13. [ ] Hierarchical (block) time steps, so dense cores do not force a tiny global dt.
14. [ ] Run the cluster's GPU vendor locally if possible (NVIDIA headless Vulkan) to shake out driver
        differences before D2 hardware time is spent.

Phase 2, one cluster node, several GPUs (needs D1, D2):
15. [ ] Portable build for the cluster (static binary; bundle the Vulkan loader if the nodes lack it).
16. [ ] Domain decomposition across GPUs (space-filling-curve split), tree data exchanged between
        domains, load balancing by measured cost.
17. [ ] Initial-condition generator for the D4 choice at 100M+ bodies, generated on the GPU.

Phase 3, multi-node to 2B (needs D3):
18. [ ] Communication layer between nodes.
19. [ ] Parallel snapshot I/O (one file per rank); 2B bodies is ~40-56 GB per snapshot.
20. [ ] Strong- and weak-scaling tables, 100M -> 2B, reported with real numbers.
21. [ ] If 3D cosmology: TreePM (particle-mesh FFT for the long range, tree for the short range)
        in a periodic box.

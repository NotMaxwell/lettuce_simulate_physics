# 3D matter sim: design (2026-11-13)

The headless sim (`zig build sim`) goes 3D with Newtonian 1/r^2 gravity; the game and the 2D sim stay
as they are. Target: ≤200M bodies on one local GPU (RX 7800 XT, 16 GB), later 1-2B on the Alabama
Supercomputer Center's A100/H100 nodes (Vulkan there unknown; PTX via the CUDA driver API is the
fallback, see "Backends").

## Algorithm
Barnes-Hut on a binary radix tree (Karras 2012 topology) over Morton-sorted leaf buckets of ≤32
bodies, quadrupole moments, stackless rope traversal, no atomics. The same tree later becomes the
short-range half of TreePM for a periodic box, and Morton ranges give the multi-GPU split.
Direct summation is the test oracle and the method for N ≤ 100k.

Rejected: 3D dense-pyramid FMM (8^L cells: 5.4 GB at L8, 344 GB at L10 for order-4 Cartesian; the
central cell of a Plummer sphere holds ~1.1e-4·N bodies at L8, i.e. a quadratic near field).
TreePM is phase 3 (needs a periodic box and a GPU FFT; a 1024^3 f32 mesh is 4.3 GB).

Opening criterion: open a node if d < l/θ + |com − centre|. CPU prototype, 1M bodies, f64
(`.scratch/architect-3d/treecmp2.zig`, git-ignored, in the main checkout):

| tree | nodes/N | θ | rms force error uniform / Plummer | Plummer per body: visits / cells / pairs |
|---|---|---|---|---|
| radix, cap 32, quad | 0.09 | 0.6 | 8.6e-4 / 5.4e-4 | 1898 / 909 / 939 |
| | | 0.5 | 2.9e-4 / 2.7e-4 | 2846 / 1352 / 1652 |
| | | 0.4 | 1.2e-4 / 1.1e-4 | 4727 / 2217 / 3396 |

Monopole-only roughly doubles the error at θ=0.6; keep quadrupoles. Build-free implicit trees need
3-10x more interactions at equal error. Estimated (not measured) 25-75 ms/tick at 1M, 0.5-1.4 s at 16M.

Determinism: the sum order can be fixed. Today's 2D nondeterminism comes from a u32 atomic whose return
value becomes the in-cell rank (`gravity_kernels.zig` binBalls/scatter), not from float atomics. The 3D
pipeline is a stable sort, a tree that is a pure function of the keys, a fixed child order and a
sequential per-body walk, so reruns must be bit-identical.

## Positions, boundaries, units
- Positions are u32 fractions of the root cube per axis. Periodic wrap is integer overflow, minimum
  image is a wrapped subtract, Morton keys are the top 21 bits per axis, and near differences are exact.
- Open boundaries: fixed cube, non-wrapping difference, escapers clamped and counted; doubling the box
  is exact (x/2 + 2^30).
- Per-tick API: `Step{kick, drift}` factors (comoving later), explicit `G`, `eps` as a length,
  `box_size: f64`, `boundary` {open, periodic}, force kernel {newton, treepm_short(r_s, r_cut)}.

## Data contract (bytes)
```
Pos3     {x, y, z: u32}                       12
Vel3     {x, y, z: f32}                       12
mass     f32                                   4  (absent if Params.uniform_mass)
SortRec  {key_hi, key_lo, idx: u32}           12  x2 ping-pong; the idle copy is permute/build scratch
Node     {com: Pos3, mass, open_r2: f32,
          child, skip, info: u32}             32  child: left child, or bit31|first body of a leaf;
                                                  skip: rope (right child = skip(left)); info: count|depth
NodeQuad {xx, xy, xz, yy, yz, zz: f32}        24
```
Node capacity N/4 with an overflow flag (measured need 0.09N), about 14 B/body. Total about 66 B/body
(62 with uniform mass). About 228M bodies fit in 15.1 GB. The largest binding is 12 B/body, so SDL_GPU's
4 GiB binding limit hits at 357M.

## Integrator and diagnostics
Global KDK leapfrog, no hierarchical steps in phase 1. The walk also writes |a|. Diagnostics print a
histogram of sqrt(2η·eps/|a|) in power-of-two rungs, to decide whether block time steps are worth it.
Diagnostics: KE, P×3, L×3 and mass (fills `diag_stride` 8). PE comes from the walk on diagnostic ticks;
print the virial ratio 2K/|W|.

## Front end
- `zig build sim -- --dim 3 ...`, dispatched in `src/sim/main.zig` to `src/sim/sim3.zig`, with kernels
  in `src/sim/k3/`. Initial conditions in `src/ic3.zig`: plummer, kepler, uniform.
- Snapshot v2: dim=3, seven fields (u32 pos xyz, f32 vel xyz, f32 mass), stored in sorted order for
  bit-identical restart. Box, boundary and cosmology go in the 56 reserved header bytes.
- Renderer: orthographic projection, `--view xy|xz|yz|yaw,pitch`, `--slab <thickness>`; the
  integer-atomic splat is unchanged.

## Work packages
| # | Agent | Owned files | Work | Acceptance |
|---|---|---|---|---|
| 0+2 | builder | `gpu_layout.zig` (3D structs), `build.zig` (k3 objects), `src/ic3.zig`, `src/sim/k3/common.zig` (shim), `k3/direct.zig`, `k3/step.zig`, `sim3.zig`, `sim/main.zig` | contract, direct-sum kick, drift, diagnostics, `--dim 3` | GPU vs f64 CPU on 10k rms ≤ 1e-5; circular Kepler at 2000 steps/period returns within 1e-3·a; e=0.5 over 10 periods \|dE/E\| ≤ 1e-3; Plummer 10k over 10 crossing times 2K/\|W\| = 1 ± 0.05; 100k ≤ 60 ms/tick |
| 1 | builder | `src/sim/ref3.zig` | f64 CPU direct sum, plus radix tree and walk on the Node contract (port of the prototype) | 10k Plummer rms ≤ 1e-3 at θ=0.6, ≤ 2.5e-4 at θ=0.4 |
| 3 | optimizer | `k3/sort.zig`, `sort3.zig` | keys, stable radix sort, permute | sorted, bijective, reruns identical; ≤ 8 ms at 1M, ≤ 120 ms at 16M |
| 4 | optimizer | `k3/build.zig`, `tree3.zig` | leaves by a ±32 window, Karras nodes, ropes, depth-ordered moments | nodes match ref3; nodes/N ≤ 0.12 |
| 5 | optimizer | `k3/walk.zig` | tree kick | 10k vs direct rms ≤ 1.2e-3, p99 ≤ 5e-3 at θ=0.6; 1M Plummer over 1000 ticks \|dE/E0\| ≤ 1e-3, 2K/\|W\| = 1 ± 0.03; bit-identical reruns; ≤ 75 ms at 1M, ≤ 1.4 s at 16M |
| 6 | builder | `snapshot.zig`, `snapper.zig`, `render*.zig` | snapshot v2, projection | round trip, bit-identical restart, render smoke |
| 7 | builder | `cuda.zig`, PTX build step | CUDA host backend | PTX builds; cluster smoke test |
| 8 | verifier | none | scale table 1M-200M | ≤ 66 B/body |

Packages 3, 4 and 5 can run in parallel against trees built by ref3; 6 runs in parallel with them; 7
follows 2. Prototype first: a single GPU walk kernel over an uploaded ref3 tree, to measure node-ops/s.

## SPIR-V backend notes (Zig 0.17)
- `@clz` is not implemented ("TODO (SPIR-V): implement unary operation 'clz'"). A shift-based clz64
  compiles and validates (`.scratch/architect-3d/spv_probe.zig`). `@popCount` is untested.
- Signed ints get mistyped in some ops; use unsigned wrapped deltas.
- Known hazards: no comptime index 0 into workgroup arrays or buffer array fields (`x[l - l]`); no
  Function variables holding storage-buffer pointers; above 32768 groups, use
  `workgroup_id[0] + workgroup_id[1]*dispatch_x`; chunk long dispatches for the driver watchdog.
- Not needed: subgroup ops, 64-bit atomics, float atomics.

## Backends (SPIR-V and PTX)
`zig build-obj -target nvptx64-cuda -mcpu=sm_80` emits PTX `.entry` kernels with `callconv(.kernel)`,
`@workItemId`/`@workGroupId`/`@workGroupSize`, `addrspace(.global)` pointers, `addrspace(.shared)`
arrays, `@atomicRmw`, `@clz`, and `bar.sync` via inline asm (`.scratch/architect-3d/ptx_probe*.zig`).
Nothing was assembled or run; there is no NVIDIA GPU here.

k3 kernels go through a comptime shim (`k3/common.zig`): `gid`, `lid`, `groupId`, `barrier`,
`Shared(T, n)`, `atomicAdd`, `clz`, index-based buffer accessors, and per-backend entry wrappers.
`storage_buffer`/`uniform` address spaces are SPIR-V only. The host side gets a `Backend` interface
(alloc, copy, launch, sync) over SDL_GPU or a dlopened libcuda. CUDA has no 4 GiB binding cap, so one
80 GB GPU holds about 1.2B bodies.

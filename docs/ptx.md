# PTX backend (work package 7, first half)

Status: compile-only, all five k3 kernel files build. Nothing here was assembled (`ptxas` is not installed) or run; there is no
NVIDIA GPU on the build machine.

## What builds
`zig build ptx` (separate step, not part of `install`/`test`/`sim`/`bench`) compiles the k3 kernels
with `-target nvptx64-cuda -mcpu sm_80 -O ReleaseFast` and installs `zig-out/ptx/k3_<name>.ptx`.
nvptx64 can only emit assembly in Zig 0.17 (`-femit-asm`; `getEmittedAsm`), which is the PTX text.

| file | `.entry` points | status |
|---|---|---|
| `k3_direct.ptx` | kick | builds |
| `k3_step.ptx` | drift, diag, rungsClear, rungs | builds |
| `k3_tree.ptx` | leafCount, leafWrite, scanBlock, scanAdd, leafFinish, karras, depthLeaf, bigLeaves, depthArgs, moments | builds |
| `k3_sort.ptx` | keys, histogram, scanRows, scatter, gather12, gather4 | builds |
| `k3_walk.ptx` | walk64, walk128, walk256, walk512, walk1024, walkGroup64, walkStats, walkNoLeafMem, walkNoLeaf, walkGroupStats64, walkGroupNoLeafMem64, walkGroupNoLeaf64 (12) | builds |

All five are in the `ptx` step. Nothing was assembled or run (no `ptxas`, no NVIDIA GPU).

## Launch contract per entry point
`G` = `L.group_size` = 256 (src/gpu_layout.zig). Block is always (B, 1, 1). Grid is (min(g, 32768), ceil(g / 32768), 1)
for `g` groups, except the indirect entries. Verified against the `callconv` of each `export fn` and the host dispatches
in src/sim/{sim3,sort3,tree3,walk3}.zig.

| file | entry | block B | groups g |
|---|---|---|---|
| direct | kick | G | ceil(count / G) per chunk |
| step | drift | G | ceil(n / G) |
| step | rungsClear | G | 1 (16 bins) |
| step | rungs | G | ceil(n / G) |
| step | diag | G | ceil(ceil(n / diag_per_thread) / G); the kernel returns early past that count |
| sort | keys, gather12, gather4 | G | ceil(n / G) |
| sort | histogram, scatter | G | ceil(n / (G * 8)) (`sort.block`) |
| sort | scanRows | G | radix = 256 (one group per digit) |
| tree | leafCount, leafWrite | G | ceil(n / 2048) (`block` = G * 8) |
| tree | scanBlock | G | ceil(count / 2048) for that scan level |
| tree | scanAdd | G | ceil(count / G) for that scan level |
| tree | leafFinish, depthArgs | G | 1 |
| tree | karras, depthLeaf, bigLeaves, moments | G | indirect: x/y/z read by the host from the args buffer the kernels write (`arg_moments`, offsets in tree3.zig); a zero grid must be skipped |
| walk | walk64, walk128, walk256, walk512, walk1024 | 64, 128, 256, 512, 1024 as named | ceil(count / B) |
| walk | walkGroup64 | 64 | count (one group per unit; `walk3.group_bodies` = 64) |
| walk | walkGroupStats64, walkGroupNoLeafMem64, walkGroupNoLeaf64 | 64 | count, as walkGroup64 (profiling) |
| walk | walkStats, walkNoLeafMem, walkNoLeaf | 1024 | ceil(count / 1024) (profiling) |

**Warning: the block size is not a free parameter.** Kernels use the compile-time group size for strides, shared-array
sizes, scan widths and loop bounds (e.g. `radix == group_size`, `sh[f * G + l]`, walk `max_group` = 64 shared arrays and
the group walk's 64 bodies per group). Launching with a different block size silently corrupts results rather than
failing. Likewise the group walk's results depend on its 64.

**Open item: no `.maxntid`.** The PTX has no `.maxntid`/`.reqntid`, so ptxas may allocate more than 64 registers per thread.
A 1024-thread entry (`walk1024`, `walkStats`, `walkNoLeafMem`, `walkNoLeaf`) that needs more than 64 registers would then fail
at launch with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. Fix: emit `.maxntid` or post-process the PTX. Check with
`cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)` / `MAX_THREADS_PER_BLOCK` and fall back to a smaller `walkNNN`.

## Host contract
Backend is chosen in `src/sim/k3/common.zig` from `builtin.cpu.arch`; kernel bodies are untouched.
- Each `ro`/`rw(T, binding, name)` is one exported `.global` 8-byte variable `name` holding a device
  pointer; the kernel loads it (`ld.global.b64 rd, [name]`). The host writes it with
  `cuModuleGetGlobal` + `cuMemcpyHtoD` before launch. Binding numbers are ignored.
- `uniform(T, name)` is an exported `.global` variable `name` of `@sizeOf(T)` bytes (e.g. `params[48]`),
  written the same way. Symbol names are per module (one module per kernel file).
- Launch: block = (group size, 1, 1) (`L.group_size`, or the number in a `walkNNN` name);
  grid = (min(g, 32768), ceil(g / 32768), 1) for g groups. `groupId()` = ctaid.x + ctaid.y * 32768,
  `gid()` = groupId() * ntid.x + tid.x, `lid()` = tid.x. Same flat indexing as the SPIR-V dispatch.
- Shared arrays are static `.shared`; no dynamic shared memory. `barrier()` is `bar.sync 0`.
- Pointer writes are stream-ordered (`cuMemcpyHtoDAsync` on the launch stream) if kernels are in flight.

## Shim differences
SPIR-V half is unchanged in substance: the five `k3_*.spv` differ from before only in `OpName`
strings and `OpLine` numbers (`common.Arr` is now `common.spv.Arr`); instructions are identical.
New exports: `ccFor(x)`, `atomicIncShared(p)`, `leaf_cap_bodies`. PTX uses `@atomicRmw` and `@clz`.

## What did not compile and how it was handled
- Two Zig comptime pitfalls in the shim: an anonymous struct inside a generic function is shared
  across calls unless it captures every distinguishing argument (the buffer name, the `Shared` name),
  otherwise `mass` aliased `pot` and `sx`/`sy` aliased; and `addrspace(.constant)` is not supported
  on nvptx. Both fixed in common.zig (`const key = name;` capture, `.global` for the uniform).
- `export`ed buffer structs must be `extern struct`.
- sort.zig `sharedInc` used inline SPIR-V asm (`unsupported output constraint` on nvptx); it now calls
  `C.atomicIncShared`. The SPIR-V shim body is the same asm (OpAtomicIAdd, workgroup scope, semantics 0, value 1).
- walk.zig `cc` now returns `C.ccFor(x)`.
- walk.zig imported `tree.zig` only for `leaf_cap_bodies`, which pulled tree's kernels and a second `params`
  symbol into the walk module (SPIR-V renamed it `params_0`; walk.spv had 22 entry points). The assert now uses
  `C.leaf_cap_bodies`, and tree.zig re-exports it (`pub const leaf_cap_bodies: u32 = C.leaf_cap_bodies;`).

## Second half (not done)
A `Backend` interface (alloc, copy, launch, sync) with two implementations: SDL_GPU (today's path)
and a dlopened `libcuda.so` (cuInit, cuModuleLoadData on the PTX, cuModuleGetGlobal for the symbols
above, cuLaunchKernel, cuMemAlloc, no 4 GiB binding cap). Open questions: `.maxntid` (see above); the device-scope atomics are
relaxed, as on SPIR-V; shared memory above 48 KiB needs `cuFuncSetAttribute`.

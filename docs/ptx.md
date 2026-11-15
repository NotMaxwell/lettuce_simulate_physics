# PTX backend (work package 7, first half)

Status: compile-only. Nothing here was assembled (`ptxas` is not installed) or run; there is no
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
| `k3_sort.ptx` | keys, histogram, scanRows, scatter, gather12, gather4 | needs the sort.zig edit below (verified on a scratch copy) |
| `k3_walk.ptx` | walk64..walk1024, walkGroup64, walkStats, walkNoLeaf, walkNoLeafMem, walkGroup*64 (12) | needs the walk.zig edits below (verified on a scratch copy) |

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
- sort.zig lines 34-44: `sharedInc` uses inline SPIR-V asm (`unsupported output constraint` on nvptx).
  Replace the body with `C.atomicIncShared(p);` (keeps the same signature).
- walk.zig lines 49-51: `cc` returns `.spirv_kernel`. Replace with `return C.ccFor(x);`.
- walk.zig line 46 imports `tree.zig`, which on PTX pulls tree's kernels and a second `params` symbol
  in (an exported symbol collision; SPIR-V silently renames it `params_0`, and walk.spv carries tree's
  10 entry points as dead weight). Replace with `C.leaf_cap_bodies`, and in tree.zig line 28 write
  `pub const leaf_cap_bodies: u32 = C.leaf_cap_bodies;`. After that `k3_sort`/`k3_walk` build; add
  `"sort", "walk"` to the list in the `ptx` step of build.zig.

## Second half (not done)
A `Backend` interface (alloc, copy, launch, sync) with two implementations: SDL_GPU (today's path)
and a dlopened `libcuda.so` (cuInit, cuModuleLoadData on the PTX, cuModuleGetGlobal for the symbols
above, cuLaunchKernel, cuMemAlloc, no 4 GiB binding cap). Open questions: `.maxntid`/`__launch_bounds__`
is not emitted, so register allocation assumes up to 1024 threads; the device-scope atomics are
relaxed, as on SPIR-V; shared memory above 48 KiB needs `cuFuncSetAttribute`.

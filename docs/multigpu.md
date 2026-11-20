# Multi-GPU

One MPI rank per GPU. Any run becomes multi-GPU by starting it under `mpirun` or `srun`; the
config does not change:

```bash
mpirun -n 4 build/clang-release/lettuce configs/plummer.ini n=1e8 method=tree
mpirun -n 4 build/clang-release/lettuce-bench backend=cuda method=tree n=1e7,1e8
```

Ranks on a node take consecutive GPUs, or device 0 when Slurm gives each task its own GPU
(`--gpus-per-task=1`). On ASA-X use `cluster/asax_multigpu.slurm`.

## How it works
Code: `src/parallel.{hpp,cpp}`, which follows the Bonsai GPU tree code (Bédorf et al. 2012, 2014).

1. **Domains.** Bodies are split into contiguous ranges along a global Morton curve (a sample sort).
   Each range holds equal *cost*: the backends report per-body interaction counts. Rebalancing
   happens every `rebalance_every` force calls (default 10); the domains drift a little in between,
   which is still correct, just slightly less efficient. No rank takes more than 3/2 of an even
   share of the bodies (`par::max_rank_bodies`): equal cost puts more bodies on ranks over cheap
   outer regions (measured worst 1.24× an even share at 8 ranks, 1.45× at 16), and the cap is what
   the memory pre-flight plans for. A range over the cap is split at global positions in key order
   (a collective bisection on the key), so ranks still own contiguous key ranges and neighbours
   share at most the finest Morton cell at their boundary.
2. **Locally essential trees.** Each rank builds its own tree, then sends every other rank the part
   of that tree that any point in the receiver's bounding box could need:
   - distant regions as single multipole nodes;
   - nearby leaves as their bodies.

   The format (`LetTree` in `backend.hpp`) is the same for both backends, so CPU and CUDA ranks can
   be mixed.
3. **Forces.** Each rank walks its own tree plus every received LET with the same group walk, so
   accuracy matches a single GPU. Measured on 2 and 4 ranks: force error 2.5–2.9e-4 at θ=0.6 against
   direct summation, the same as one rank; energy error 1.1e-6 over 100 steps with rebalancing.
4. **Everything else.**
   - Diagnostics are summed over ranks.
   - Each rank writes its own snapshot piece (`snap_<step>_r<k>.snap`) and records the piece count
     in its header. Restart from any piece: `restart=out/snap_00000200_r0.snap` reads exactly the
     pieces piece 0 records, with any number of ranks; higher-numbered pieces left in the directory
     by an earlier run with more ranks are ignored with a warning (never deleted). Snapshots
     written before the count existed fall back to every piece found.
   - On restart, rank 0 drops the diag.csv rows at or after the restart step before appending.
   - Frames and the live view are drawn on rank 0 from a gathered subsample (at most 16M bodies).

The exchange is staged through host memory with plain MPI, so it needs no CUDA-aware MPI or
NVLink, and the same code runs across nodes over InfiniBand.

## Measured so far (CPU only, this 4-core dev box)
1M-body Plummer, tree θ=0.6, ms per force call, all on the same 4 cores, so this measures overhead,
not speed-up:

| layout | ms | vs 1 rank |
|---|---|---|
| 1 rank × 4 threads | 705 | – |
| 2 ranks × 2 threads | 742 | +5% |
| 4 ranks × 1 thread | 872 | +24% (17 ms of it exchange, the rest extra remote interactions) |

The overhead shrinks as the bodies per rank grow, since the boundary (the LET) grows like surface,
not volume. GPU scaling has not been measured; the first H100 runs will tell.

## Known limits (v1)
- **Every rank builds the full initial set**, then keeps its share. That costs N × 32 bytes of host
  memory per rank at start-up; distributed initial-condition generation is the fix before multi-node
  billions.
- **Host-staged exchange.** On CUDA the LET export downloads the tree's node array each force call
  and walks it on the CPU; rebalancing downloads and re-uploads the bodies. A device-side export and
  CUDA-aware MPI are the next optimisations once GPU timings show where the time goes.
- **Thread pinning.** With several CPU threads per rank, keep MPI from pinning a rank to one core
  (`mpirun --bind-to none`, or `srun --cpu-bind=cores` with `--cpus-per-task`). Open MPI pins by
  default with ≤ 2 ranks; measured here, that turned 742 ms into 1510 ms.
- **No overlap yet** of the exchange with the local walk.

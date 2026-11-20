# Design

## Physics
Newtonian gravity between point masses with Plummer softening:

    a_i   =  G Σ_j m_j (x_j − x_i) / (|x_j − x_i|² + ε²)^(3/2)
    φ_i   = −G Σ_{j≠i} m_j / sqrt(|x_j − x_i|² + ε²)
    W     =  ½ Σ_i m_i φ_i

No collisions, mergers, fragmentation or other destruction: bodies are conserved, and the only
interaction is gravity. Open boundaries (bodies can travel arbitrarily far).

Time integration is a global kick-drift-kick leapfrog (second order, symplectic, time reversible):

    v += a·dt/2;  x += v·dt;  a = F(x);  v += a·dt/2

Between steps velocities are synchronous with positions, so diagnostics, snapshots and restarts
need no correction. Energy error oscillates rather than drifts as long as dt resolves the
shortest dynamical time (pericentre passages, the softening scale). There are no adaptive or
per-body time steps yet.

## Force methods
- **direct**: exact O(N²) pair sum. The reference, and the fastest choice up to ~5·10⁴ bodies
  (`method = auto` switches at `auto_direct_max`).
- **tree**: Barnes-Hut, O(N log N), after the design in the `gpu-physics` branch's
  `docs/sim3d.md`:
  1. Bounding cube, 63-bit Morton keys (21 bits per axis). When the 21 levels run out with more
     than a leaf's worth of non-coincident bodies (a far escaper shrinks the cluster to a few
     cells), those bodies are re-keyed inside their own bounding cube and subdivided further
     (CPU: per node; CUDA: a secondary key per run of identical keys). Without this the walk
     degrades to O(N^2).
  2. Stable radix sort of the bodies by key. Bodies *stay* in sorted order between steps, so a
     backend's arrays are permuted and `Bodies::id` carries identity.
  3. Tree over the sorted keys, leaves of up to ~32 consecutive bodies.
  4. Monopole plus traceless quadrupole moments, bottom-up.
  5. Walk. CUDA: one thread per body in sorted order (neighbouring threads traverse
     near-identical paths, which keeps warps coherent). CPU: one interaction list per group of
     ≤128 sorted bodies, opening a node if it is too close to the group's bounding box (2.4x
     faster than per-body lists on the CPU). A node is opened if `d < l/θ + |com − centre|` (l = node side), the
     offset term guarding against lopsided cells. θ = 0.6 gives ~1e-3 rms relative force
     error at quadrupole order (the measured prototype numbers are in the `gpu-physics` branch).

## Backends
`backend.hpp` defines the interface. State is resident in the backend between steps: the driver
uploads once and downloads only for snapshots and frames, so a GPU run transfers nothing per step
except a few reduction results on diagnostic steps.

- **cuda** (`cuda_backend.cu`): tiled shared-memory direct kernel; Morton keys plus `cub::DeviceRadixSort`;
  radix tree with bottom-up moments; one thread per body for the walk. Diagnostics are reduced in
  double on the device.
- **cpu** (`cpu_backend.cpp`): the same algorithms with OpenMP and vectorised pair loops. It is
  the fallback and the oracle the CUDA tests compare against.

Precision: positions, velocities and accelerations are `float` (fast on every GPU, and
H100 FP32 throughput is 2x FP64). Reductions for diagnostics use `double`. `float` positions limit
dynamic range to ~1e-7 of the system size. Fine for clusters and galaxies; the Solar System scenario
resolves positions to ~300 km at Neptune (30 AU), and long integrations there will show it. `float` also sets an energy-error floor of a few 1e-6
per orbit (the Kepler test measures 5e-6; the same orbit in double drifts 4e-15).

## Toolchain
Clang/LLVM is the primary toolchain: clang++ compiles both host code (C++23, libc++, libomp, lld)
and CUDA (clang's native CUDA support, sm_90). nvcc is kept as a fallback (`nvcc-release` preset)
for sites where Clang is older than the installed CUDA, so `.cu` sources and the headers they
include must stay C++20-clean.

## Multiple GPUs
One MPI rank per GPU, with cost-balanced Morton domains and locally-essential-tree exchange; see
[multigpu.md](multigpu.md).

## Not done yet
- Block (hierarchical) time steps; periodic boxes / TreePM; cosmological integration.
- Drawing live-view frames on the GPU (today they are drawn on the CPU after a copy, fine up to ~10M bodies).

// Force/integration backends. State lives in the backend (on the GPU for CUDA) between steps;
// the host only uploads at start and downloads for diagnostics output, snapshots and renders.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bodies.hpp"

// Method is resolved before it reaches a backend: only direct or tree.
enum class ForceMethod { direct, tree };

struct ForceParams {
    double G = 1;
    double eps = 0.01;   // Plummer softening length: a = G m d / (|d|^2 + eps^2)^(3/2)
    double theta = 0.6;  // tree only
    ForceMethod method = ForceMethod::direct;
};

// Reductions over the current state, in double. Velocities are whatever the backend holds, so
// call this only when they are synchronous with positions (after the closing half kick).
struct Diag {
    double mass = 0;
    double ke = 0;
    double pe = 0;          // valid only if has_pe
    bool has_pe = false;
    double px = 0, py = 0, pz = 0;   // linear momentum
    double lx = 0, ly = 0, lz = 0;   // angular momentum about the origin
    double cx = 0, cy = 0, cz = 0;   // centre of mass
};

struct PhaseTimes {
    double force_ms = 0;     // whole force evaluation, including the parts below
    double sort_ms = 0;      // tree: Morton keys + sort + permute
    double build_ms = 0;     // tree: topology + moments
    double walk_ms = 0;      // tree: traversal; direct: the pair loop
    double step_ms = 0;      // last kick + drift
    uint64_t interactions = 0;  // body-body plus body-node evaluations in the last force call (0 if not counted)
    double comm_ms = 0;         // multi-rank only: LET export + exchange (host-staged)
    double balance_ms = 0;      // multi-rank only: domain rebalance in this call (0 when none)
};

// --- Distributed (multi-GPU) support -----------------------------------------------------------
// Each rank owns a subset of the bodies. Forces on a rank's bodies come from its own tree plus a
// "locally essential tree" (LET) from every other rank: the part of that rank's tree that any
// target inside the receiver's bounding box could open (Bonsai, Bedorf et al. 2012), as a pruned,
// self-contained tree. The format is backend-neutral and trivially copyable, so it travels over
// MPI as raw bytes and a LET exported by one backend type can be walked by another.

struct Box3 {
    float lo[3] = {0, 0, 0};
    float hi[3] = {0, 0, 0};
};

struct LetNode {
    float cx, cy, cz;                    // centre of mass
    float mass;
    float qxx, qxy, qxz, qyy, qyz, qzz;  // traceless quadrupole about the centre of mass
    // Open the node for a target at x if |x - com|^2 < rcrit2, where rcrit2 = (l/theta + |com - cell centre|)^2
    // as in the local walk. Use a large finite value (3.0e38f) for "always open": no infinities,
    // the CUDA build uses fast math.
    float rcrit2;
    int32_t first_child;  // index of the first child in LetTree::nodes, children contiguous; -1 if none
    int32_t child_count;
    int32_t body_first;   // index of the first body in LetTree::b*, bodies contiguous; -1 if none
    int32_t body_count;
};
static_assert(sizeof(LetNode) == 60, "LetNode is sent over MPI as raw bytes");

// Receiver walk, per target x, starting at nodes[0]:
//   if (first_child < 0 && body_count == 0) or |x - com|^2 >= rcrit2: add the node's multipole
//   else if body_count > 0: add each body directly (softened, no self term: remote bodies are never the target)
//   else: visit the children.
// A node the exporter accepted for the whole target box is "terminal" (no children, no bodies):
// any target in the box accepts it, so its rcrit2 is irrelevant. An empty LET has no nodes.
struct LetTree {
    std::vector<LetNode> nodes;
    std::vector<float> bx, by, bz, bm;
};

class Backend {
public:
    virtual ~Backend() = default;
    // Human-readable, e.g. "cuda: NVIDIA H100 80GB HBM3 (sm_90, 132 SMs)" or "cpu: 16 threads".
    virtual std::string name() const = 0;

    virtual void upload(const Bodies& b) = 0;
    // Bodies come back in the backend's current order; `id` identifies them.
    virtual void download(Bodies& b) = 0;

    // Fills the backend's acceleration buffer for the current positions. With `potential`, also
    // the per-body potential phi_i = -G sum_j m_j / sqrt(|d|^2 + eps^2) (self term excluded),
    // which the next diagnostics() call turns into pe = 1/2 sum m_i phi_i.
    virtual void compute_forces(const ForceParams& p, bool potential) = 0;
    virtual void kick(double dt) = 0;   // v += a dt
    virtual void drift(double dt) = 0;  // x += v dt

    virtual Diag diagnostics() = 0;
    // Exact O(N^2) potential energy, independent of the force method (test oracle, small N).
    virtual double exact_potential_energy(double G, double eps) = 0;

    virtual PhaseTimes times() const = 0;

    // --- Distributed support (used by the MPI layer; a single-rank run never calls these) ---

    // Bounding box of the current positions.
    virtual Box3 local_bounds() = 0;
    // Tree method: sort the bodies and build the local tree for the current positions, as the first
    // half of compute_forces would. Direct method: no-op.
    virtual void build_tree(const ForceParams& p) = 0;
    // After build_tree (tree method): the part of the local tree a target anywhere inside `target`
    // needs at opening angle p.theta. A node is terminal if its distance to the box satisfies the
    // opening criterion, i.e. min over x in box of |x - com|^2 >= rcrit2; a leaf that must be opened
    // exports its bodies. Direct method: one always-open root holding every local body.
    virtual LetTree export_let(const Box3& target, const ForceParams& p) = 0;
    // Same output as compute_forces (accelerations, optional potential, times()), from the local
    // bodies plus every remote LET. For the tree method it uses the tree from build_tree and must
    // not rebuild it. compute_forces(p, pot) is equivalent to build_tree(p) + compute_forces_with(p, pot, {}).
    virtual void compute_forces_with(const ForceParams& p, bool potential,
                                     const std::vector<const LetTree*>& remote) = 0;
    // Optional: per-body cost of the last force call (e.g. interactions), in download() order, for
    // load balancing. Returns false if not provided; the MPI layer then balances by body count.
    virtual bool body_costs(std::vector<float>& cost) { (void)cost; return false; }
};

std::unique_ptr<Backend> make_cpu_backend(int threads);
// Returns nullptr and sets `why` when CUDA support was not compiled in or no usable device exists.
std::unique_ptr<Backend> make_cuda_backend(int device, std::string& why);
bool cuda_compiled();
// CUDA devices visible to this process (0 without CUDA support or a driver).
int cuda_device_count();

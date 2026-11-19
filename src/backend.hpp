// Force/integration backends. State lives in the backend (on the GPU for CUDA) between steps;
// the host only uploads at start and downloads for diagnostics output, snapshots and renders.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

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
};

std::unique_ptr<Backend> make_cpu_backend(int threads);
// Returns nullptr and sets `why` when CUDA support was not compiled in or no usable device exists.
std::unique_ptr<Backend> make_cuda_backend(int device, std::string& why);
bool cuda_compiled();

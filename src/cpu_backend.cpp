// OpenMP CPU backend: blocked direct summation and a Barnes-Hut octree with quadrupoles.
// It is also the reference the CUDA backend is validated against.
#include <omp.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend.hpp"

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

constexpr int kLeafMax = 16;      // bodies per leaf bucket (multipole granularity)
// Walk groups are the largest subtrees with at most this many bodies. Groups bigger than a leaf
// amortise the list build over more bodies (measured at N=100k, theta 0.6: group 16 -> 128 cuts the walk 2.4x).
constexpr int kGroupMax = 128;
constexpr int kMaxLevel = 21;     // Morton bits per axis
constexpr int kSmallLeaf = 4;     // accepted leaves this small are summed directly: exact and no dearer
constexpr int kRadixBits = 11;
constexpr int kRadixBuckets = 1 << kRadixBits;
constexpr std::size_t kParMin = 1 << 16;  // key restarts on fewer bodies than this run serially

// One cache line. Children of a node are contiguous and have larger indices than the node,
// so a reverse index sweep is a valid bottom-up order.
struct alignas(64) Node {
    float cx, cy, cz, m;                    // centre of mass, mass
    float qxx, qxy, qxz, qyy, qyz, qzz;     // traceless quadrupole about the centre of mass
    float rcrit2;                           // open if (distance to group)^2 < rcrit2
    uint32_t first;                         // first child (internal) or first body (leaf)
    uint32_t nchild;                        // 0 for a leaf
    uint32_t lo, hi;                        // body range [lo, hi) in sorted order
};
static_assert(sizeof(Node) == 64);

struct Geo { float gx, gy, gz, half; };     // geometric cell centre and half side

// A walk group is a subtree of consecutive sorted bodies: its bodies share an interaction list, built against the group's
// bounding box, so any node that some body of the group would open is opened for all of them.
struct Group { float bx, by, bz, hx, hy, hz; uint32_t lo, hi; };

inline uint64_t spread21(uint64_t v) {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8) & 0x100f00f00f00f00fULL;
    v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}


// 8-wide kernels in GCC/Clang vector extensions. The point is rsqrt + one Newton step instead of
// sqrt + div (about 2x the pair rate on AVX2); relative error after the step is ~1e-7.
typedef float v8f __attribute__((vector_size(32)));
typedef int32_t v8i __attribute__((vector_size(32)));

inline v8f load8(const float* p) { v8f v; std::memcpy(&v, p, sizeof v); return v; }
inline v8i load8(const int32_t* p) { v8i v; std::memcpy(&v, p, sizeof v); return v; }
inline float hsum(v8f v) { return ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7])); }

// 1/sqrt(r2) where mask is set, exactly 0 elsewhere (also where r2 == 0, which gives inf/NaN
// before masking).
inline v8f rsqrt_masked(v8f r2, v8i mask) {
#if defined(__AVX__)
    v8f y = (v8f)_mm256_rsqrt_ps((__m256)r2);
    y = y * (1.5f - 0.5f * r2 * y * y);
#else
    v8f y;
    for (int l = 0; l < 8; ++l) y[l] = 1.f / std::sqrt(r2[l]);
#endif
    return (v8f)((v8i)y & mask);
}

// Body-body: sums m_j d / r^3 and m_j / r over j in [0, n), skipping j whose index equals self.
// Index of entry k is idx[k] if idx, else base + k. n must be a multiple of 8.
template <bool POT>
inline void p2p8(const float* bx, const float* by, const float* bz, const float* bm, const int32_t* idx, int32_t base,
                 long n, float xi, float yi, float zi, int32_t self, float e2, float& oax, float& oay, float& oaz,
                 float& op) {
    v8f ax = {}, ay = {}, az = {}, p = {};
    const v8i lane = {0, 1, 2, 3, 4, 5, 6, 7};
    for (long k = 0; k < n; k += 8) {
        v8f dx = load8(bx + k) - xi, dy = load8(by + k) - yi, dz = load8(bz + k) - zi;
        v8f r2 = dx * dx + dy * dy + dz * dz + e2;
        v8i j = idx ? load8(idx + k) : lane + int32_t(base + k);
        v8f inv = rsqrt_masked(r2, (v8i)(r2 > 0.f) & (j != self));
        v8f mi = load8(bm + k) * inv;
        v8f f = mi * inv * inv;
        ax += f * dx; ay += f * dy; az += f * dz;
        if constexpr (POT) p += mi;
    }
    oax += hsum(ax); oay += hsum(ay); oaz += hsum(az);
    if constexpr (POT) op += hsum(p);
}

template <class T>
void grow(std::vector<T>& v, std::size_t n) {
    if (v.size() < n) v.resize(n);
}

struct WalkScratch {
    std::vector<float> bx, by, bz, bm;
    std::vector<int32_t> bi;
    std::vector<float> nx, ny, nz, nm, qxx, qxy, qxz, qyy, qyz, qzz;
    std::vector<uint32_t> stack;
    std::vector<int32_t> let_stack;
    std::size_t nb = 0, nn = 0;
    void need_bodies(std::size_t n) {
        if (bx.size() >= n) return;
        std::size_t c = std::max<std::size_t>(n, 2 * bx.size());
        bx.resize(c); by.resize(c); bz.resize(c); bm.resize(c); bi.resize(c);
    }
    void need_nodes(std::size_t n) {
        if (nx.size() >= n) return;
        std::size_t c = std::max<std::size_t>(n, 2 * nx.size());
        for (auto* v : {&nx, &ny, &nz, &nm, &qxx, &qxy, &qxz, &qyy, &qyz, &qzz}) v->resize(c);
    }
};

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(int threads) : nt_(threads > 0 ? threads : omp_get_max_threads()) {
        scratch_.resize(nt_);
    }

    std::string name() const override { return "cpu: " + std::to_string(nt_) + " threads (OpenMP)"; }

    // Direct: 12 state arrays + costs (52 B/body). Tree adds keys, indices and scratch (32 B/body)
    // plus nodes; the 110 B/body total is lettuce-bench's measured peak RSS at 3e7 Plummer bodies
    // (138 B/body) minus the bench's own 32 B/body Bodies.
    MemoryNeed memory_need(uint64_t n, ForceMethod m) const override {
        return {0, n * (m == ForceMethod::tree ? 110u : 52u)};
    }

    void upload(const Bodies& b) override {
        n_ = b.size();
        x_ = b.x; y_ = b.y; z_ = b.z;
        vx_ = b.vx; vy_ = b.vy; vz_ = b.vz;
        m_ = b.m; id_ = b.id;
        ax_.assign(n_, 0.f); ay_.assign(n_, 0.f); az_.assign(n_, 0.f); pot_.assign(n_, 0.f);
        has_pot_ = false;
        tree_ok_ = false;
        cost_.clear();
    }

    void download(Bodies& b) override {
        b.x = x_; b.y = y_; b.z = z_;
        b.vx = vx_; b.vy = vy_; b.vz = vz_;
        b.m = m_; b.id = id_;
    }

    void compute_forces(const ForceParams& p, bool potential) override {
        build_tree(p);
        compute_forces_with(p, potential, {});
    }

    void build_tree(const ForceParams& p) override {
        pend_sort_ms_ = pend_build_ms_ = 0;
        if (p.method != ForceMethod::tree || n_ == 0) return;
        auto ts = clk::now();
        sort_bodies();
        pend_sort_ms_ = ms_since(ts);
        auto tb = clk::now();
        build_octree(p.theta);
        pend_build_ms_ = ms_since(tb);
        tree_ok_ = true;
    }

    void compute_forces_with(const ForceParams& p, bool potential, const std::vector<const LetTree*>& remote) override {
        auto t0 = clk::now();
        PhaseTimes t;
        t.step_ms = times_.step_ms;
        if (p.method == ForceMethod::tree && n_ > 0 && !tree_ok_) build_tree(p);
        t.sort_ms = pend_sort_ms_;
        t.build_ms = pend_build_ms_;
        pend_sort_ms_ = pend_build_ms_ = 0;
        times_ = t;
        has_pot_ = potential;
        cost_.resize(n_);
        if (n_ == 0) return;
        const float G = float(p.G), e2 = float(p.eps * p.eps);
        auto tw = clk::now();
        if (p.method == ForceMethod::direct) {
            if (potential) direct<true>(G, e2);
            else direct<false>(G, e2);
            times_.interactions = uint64_t(n_) * n_;
            times_.interactions += potential ? direct_remote<true>(G, e2, remote) : direct_remote<false>(G, e2, remote);
        } else {
            times_.interactions = potential ? walk<true>(G, e2, remote) : walk<false>(G, e2, remote);
        }
        times_.walk_ms = ms_since(tw);
        times_.force_ms = times_.sort_ms + times_.build_ms + ms_since(t0);
    }

    Box3 local_bounds() override {
        Box3 bx;
        if (n_ == 0) return bx;
        const long ln = long(n_);
        float mnx = x_[0], mny = y_[0], mnz = z_[0], mxx = mnx, mxy = mny, mxz = mnz;
#pragma omp parallel for num_threads(nt_) reduction(min : mnx, mny, mnz) reduction(max : mxx, mxy, mxz)
        for (long i = 0; i < ln; ++i) {
            mnx = std::min(mnx, x_[i]); mny = std::min(mny, y_[i]); mnz = std::min(mnz, z_[i]);
            mxx = std::max(mxx, x_[i]); mxy = std::max(mxy, y_[i]); mxz = std::max(mxz, z_[i]);
        }
        bx.lo[0] = mnx; bx.lo[1] = mny; bx.lo[2] = mnz;
        bx.hi[0] = mxx; bx.hi[1] = mxy; bx.hi[2] = mxz;
        return bx;
    }

    // Reads only the tree and body arrays, so concurrent calls for different targets are safe.
    LetTree export_let(const Box3& target, const ForceParams& p) override {
        LetTree out;
        if (n_ == 0) return out;
        if (p.method == ForceMethod::direct) return export_direct();
        if (!tree_ok_) throw std::logic_error("cpu export_let: call build_tree first");
        if (nodes_[0].m <= 0) return out;
        const float bcx = 0.5f * (target.lo[0] + target.hi[0]), bhx = 0.5f * (target.hi[0] - target.lo[0]);
        const float bcy = 0.5f * (target.lo[1] + target.hi[1]), bhy = 0.5f * (target.hi[1] - target.lo[1]);
        const float bcz = 0.5f * (target.lo[2] + target.hi[2]), bhz = 0.5f * (target.hi[2] - target.lo[2]);
        // The output array doubles as the breadth-first queue: src[k] is the local node behind
        // out.nodes[k]. Children are appended as one block when their parent is processed, which
        // keeps every sibling set contiguous.
        std::vector<uint32_t> src;
        src.reserve(256);
        out.nodes.reserve(256);
        auto emit = [&](uint32_t ni) {
            const Node& nd = nodes_[ni];
            LetNode l{};
            l.cx = nd.cx; l.cy = nd.cy; l.cz = nd.cz; l.mass = nd.m;
            l.qxx = nd.qxx; l.qxy = nd.qxy; l.qxz = nd.qxz; l.qyy = nd.qyy; l.qyz = nd.qyz; l.qzz = nd.qzz;
            l.rcrit2 = nd.rcrit2;
            l.first_child = -1; l.child_count = 0; l.body_first = -1; l.body_count = 0;
            out.nodes.push_back(l);
            src.push_back(ni);
        };
        emit(0);
        for (std::size_t k = 0; k < out.nodes.size(); ++k) {
            const Node& nd = nodes_[src[k]];
            float dx = std::max(std::fabs(nd.cx - bcx) - bhx, 0.f);
            float dy = std::max(std::fabs(nd.cy - bcy) - bhy, 0.f);
            float dz = std::max(std::fabs(nd.cz - bcz) - bhz, 0.f);
            if (dx * dx + dy * dy + dz * dz >= nd.rcrit2) continue;  // terminal: emitted as is
            if (nd.nchild == 0) {
                const std::size_t b0 = out.bx.size(), nb = nd.hi - nd.lo;
                out.bx.insert(out.bx.end(), x_.begin() + nd.lo, x_.begin() + nd.hi);
                out.by.insert(out.by.end(), y_.begin() + nd.lo, y_.begin() + nd.hi);
                out.bz.insert(out.bz.end(), z_.begin() + nd.lo, z_.begin() + nd.hi);
                out.bm.insert(out.bm.end(), m_.begin() + nd.lo, m_.begin() + nd.hi);
                out.nodes[k].body_first = int32_t(b0);
                out.nodes[k].body_count = int32_t(nb);
                continue;
            }
            const int32_t first = int32_t(out.nodes.size());
            for (uint32_t c = nd.first; c < nd.first + nd.nchild; ++c)
                if (nodes_[c].m > 0) emit(c);
            out.nodes[k].first_child = first;
            out.nodes[k].child_count = int32_t(out.nodes.size()) - first;
        }
        return out;
    }

    bool body_costs(std::vector<float>& cost) override {
        if (cost_.size() != n_ || n_ == 0) return false;
        cost = cost_;
        return true;
    }

    void kick(double dt) override {
        auto t0 = clk::now();
        const float h = float(dt);
        float *vx = vx_.data(), *vy = vy_.data(), *vz = vz_.data();
        const float *ax = ax_.data(), *ay = ay_.data(), *az = az_.data();
#pragma omp parallel for simd num_threads(nt_) schedule(static)
        for (std::size_t i = 0; i < n_; ++i) {
            vx[i] += ax[i] * h; vy[i] += ay[i] * h; vz[i] += az[i] * h;
        }
        kick_ms_ = ms_since(t0);
        times_.step_ms = kick_ms_ + drift_ms_;
    }

    void drift(double dt) override {
        auto t0 = clk::now();
        const float h = float(dt);
        float *x = x_.data(), *y = y_.data(), *z = z_.data();
        const float *vx = vx_.data(), *vy = vy_.data(), *vz = vz_.data();
#pragma omp parallel for simd num_threads(nt_) schedule(static)
        for (std::size_t i = 0; i < n_; ++i) {
            x[i] += vx[i] * h; y[i] += vy[i] * h; z[i] += vz[i] * h;
        }
        has_pot_ = false;
        tree_ok_ = false;
        drift_ms_ = ms_since(t0);
        times_.step_ms = kick_ms_ + drift_ms_;
    }

    Diag diagnostics() override {
        double mass = 0, ke = 0, pe = 0, px = 0, py = 0, pz = 0, lx = 0, ly = 0, lz = 0, cx = 0, cy = 0, cz = 0;
        const long n = long(n_);
#pragma omp parallel for num_threads(nt_) schedule(static) \
    reduction(+ : mass, ke, pe, px, py, pz, lx, ly, lz, cx, cy, cz)
        for (long i = 0; i < n; ++i) {
            double m = m_[i], x = x_[i], y = y_[i], z = z_[i], vx = vx_[i], vy = vy_[i], vz = vz_[i];
            mass += m;
            ke += 0.5 * m * (vx * vx + vy * vy + vz * vz);
            pe += 0.5 * m * double(pot_[i]);
            px += m * vx; py += m * vy; pz += m * vz;
            lx += m * (y * vz - z * vy); ly += m * (z * vx - x * vz); lz += m * (x * vy - y * vx);
            cx += m * x; cy += m * y; cz += m * z;
        }
        Diag d;
        d.mass = mass; d.ke = ke;
        d.has_pe = has_pot_;
        d.pe = has_pot_ ? pe : 0;
        d.px = px; d.py = py; d.pz = pz;
        d.lx = lx; d.ly = ly; d.lz = lz;
        if (mass > 0) { d.cx = cx / mass; d.cy = cy / mass; d.cz = cz / mass; }
        return d;
    }

    double exact_potential_energy(double G, double eps) override {
        const double e2 = eps * eps;
        const long n = long(n_);
        double pe = 0;
#pragma omp parallel for num_threads(nt_) schedule(dynamic, 64) reduction(+ : pe)
        for (long i = 0; i < n; ++i) {
            const double xi = x_[i], yi = y_[i], zi = z_[i];
            double s = 0;
            for (long j = i + 1; j < n; ++j) {
                double dx = x_[j] - xi, dy = y_[j] - yi, dz = z_[j] - zi;
                double r2 = dx * dx + dy * dy + dz * dz + e2;
                if (r2 > 0) s += double(m_[j]) / std::sqrt(r2);
            }
            pe -= double(m_[i]) * s;
        }
        return G * pe;
    }

    PhaseTimes times() const override { return times_; }

private:
    // ---- direct ----------------------------------------------------------------------------
    template <bool POT>
    void direct(float G, float e2) {
        constexpr std::size_t kIB = 128, kJT = 4096;  // j tile: 4096 * 16 B fits in L1+L2 comfortably
        const std::size_t n = n_;
        const float *X = x_.data(), *Y = y_.data(), *Z = z_.data(), *M = m_.data();
        const long nblk = long((n + kIB - 1) / kIB);
#pragma omp parallel for num_threads(nt_) schedule(dynamic, 1)
        for (long b = 0; b < nblk; ++b) {
            const std::size_t i0 = std::size_t(b) * kIB, i1 = std::min(n, i0 + kIB);
            float sax[kIB] = {}, say[kIB] = {}, saz[kIB] = {}, sp[kIB] = {};
            for (std::size_t j0 = 0; j0 < n; j0 += kJT) {
                const std::size_t j1 = std::min(n, j0 + kJT);
                for (std::size_t i = i0; i < i1; ++i) {
                    const float xi = X[i], yi = Y[i], zi = Z[i];
                    const int32_t ii = int32_t(i);
                    float ax = 0, ay = 0, az = 0, p = 0;
                    const std::size_t jv = j0 + ((j1 - j0) & ~std::size_t(7));
                    p2p8<POT>(X + j0, Y + j0, Z + j0, M + j0, nullptr, int32_t(j0), long(jv - j0), xi, yi, zi, ii, e2,
                              ax, ay, az, p);
                    for (std::size_t j = jv; j < j1; ++j) {
                        float dx = X[j] - xi, dy = Y[j] - yi, dz = Z[j] - zi;
                        float r2 = dx * dx + dy * dy + dz * dz + e2;
                        // Excluding by index keeps phi correct for distinct coincident bodies
                        // with eps > 0; r2 > 0 guards eps = 0.
                        float inv = (r2 > 0.f && int32_t(j) != ii) ? 1.f / std::sqrt(r2) : 0.f;
                        float mi = M[j] * inv;
                        float f = mi * inv * inv;
                        ax += f * dx; ay += f * dy; az += f * dz;
                        if constexpr (POT) p += mi;
                    }
                    sax[i - i0] += ax; say[i - i0] += ay; saz[i - i0] += az;
                    if constexpr (POT) sp[i - i0] += p;
                }
            }
            for (std::size_t i = i0; i < i1; ++i) {
                ax_[i] = G * sax[i - i0]; ay_[i] = G * say[i - i0]; az_[i] = G * saz[i - i0];
                if constexpr (POT) pot_[i] = -G * sp[i - i0];
            }
        }
    }

    // ---- sort ------------------------------------------------------------------------------
    void sort_bodies() {
        const std::size_t n = n_;
        const long ln = long(n);
        float mnx = x_[0], mny = y_[0], mnz = z_[0], mxx = mnx, mxy = mny, mxz = mnz;
#pragma omp parallel for num_threads(nt_) reduction(min : mnx, mny, mnz) reduction(max : mxx, mxy, mxz)
        for (long i = 0; i < ln; ++i) {
            mnx = std::min(mnx, x_[i]); mny = std::min(mny, y_[i]); mnz = std::min(mnz, z_[i]);
            mxx = std::max(mxx, x_[i]); mxy = std::max(mxy, y_[i]); mxz = std::max(mxz, z_[i]);
        }
        double side = std::max({double(mxx) - mnx, double(mxy) - mny, double(mxz) - mnz});
        side = side > 0 ? side * (1 + 1e-6) : 1.0;  // margin keeps the max coordinate below 2^21
        ox_ = mnx; oy_ = mny; oz_ = mnz; side_ = side;

        grow(key_, n); grow(key2_, n); grow(idx_, n); grow(idx2_, n);
        grow(tmpf_, n); grow(tmpu_, n);
        make_keys(0, n, ox_, oy_, oz_, side, true);
        radix_sort(0, n, nt_);
        // Bodies stay sorted between calls, so the next sort starts from nearly sorted input and
        // the permute is close to a streaming copy.
        permute(0, n, true);
    }

    // Morton keys of bodies [lo, hi) in the cube of the given side at (ox, oy, oz); idx_ = identity.
    void make_keys(std::size_t lo, std::size_t hi, double ox, double oy, double oz, double side, bool par) {
        const double s = double(1u << kMaxLevel) / side;
        const double lim = double((1u << kMaxLevel) - 1);
        const long l0 = long(lo), l1 = long(hi);
#pragma omp parallel for num_threads(nt_) schedule(static) if (par)
        for (long i = l0; i < l1; ++i) {
            auto q = [&](float v, double o) { return uint64_t(std::clamp((double(v) - o) * s, 0.0, lim)); };
            key_[i] = spread21(q(x_[i], ox)) << 2 | spread21(q(y_[i], oy)) << 1 | spread21(q(z_[i], oz));
            idx_[i] = uint32_t(i);
        }
    }

    // Gathers every body array in [lo, hi) into the order idx_ gives.
    void permute(std::size_t lo, std::size_t hi, bool par) {
        const long l0 = long(lo), l1 = long(hi);
        const std::size_t n = hi - lo;
        for (auto* a : {&x_, &y_, &z_, &vx_, &vy_, &vz_, &m_}) {
            float* src = a->data();
            float* dst = tmpf_.data();
#pragma omp parallel for num_threads(nt_) schedule(static) if (par)
            for (long i = l0; i < l1; ++i) dst[i] = src[idx_[i]];
            std::memcpy(src + lo, dst + lo, n * sizeof(float));
        }
#pragma omp parallel for num_threads(nt_) schedule(static) if (par)
        for (long i = l0; i < l1; ++i) tmpu_[i] = id_[idx_[i]];
        std::memcpy(id_.data() + lo, tmpu_.data() + lo, n * sizeof(uint32_t));
        if (cost_.size() == n_) {
#pragma omp parallel for num_threads(nt_) schedule(static) if (par)
            for (long i = l0; i < l1; ++i) tmpf_[i] = cost_[idx_[i]];
            std::memcpy(cost_.data() + lo, tmpf_.data() + lo, n * sizeof(float));
        }
    }

    // Stable LSD radix sort of (key_, idx_) over [lo, hi) with T threads; passes whose digit is
    // constant are skipped.
    void radix_sort(std::size_t lo, std::size_t hi, int T) {
        const std::size_t n = hi - lo;
        hist_.assign(std::size_t(T) * kRadixBuckets, 0);
        uint64_t *ks = key_.data() + lo, *kd = key2_.data() + lo;
        uint32_t *is = idx_.data() + lo, *id = idx2_.data() + lo;
        for (int shift = 0; shift < 63; shift += kRadixBits) {
            bool skip = false;
#pragma omp parallel num_threads(T)
            {
                const int t = omp_get_thread_num(), nth = omp_get_num_threads();
                const std::size_t b0 = n * t / nth, b1 = n * (t + 1) / nth;
                std::size_t* h = &hist_[std::size_t(t) * kRadixBuckets];
                std::fill(h, h + kRadixBuckets, 0);
                for (std::size_t i = b0; i < b1; ++i) ++h[(ks[i] >> shift) & (kRadixBuckets - 1)];
#pragma omp barrier
#pragma omp single
                {
                    std::size_t off = 0;
                    for (int d = 0; d < kRadixBuckets; ++d) {
                        std::size_t tot = 0;
                        for (int u = 0; u < nth; ++u) {
                            std::size_t c = hist_[std::size_t(u) * kRadixBuckets + d];
                            hist_[std::size_t(u) * kRadixBuckets + d] = off;
                            off += c;
                            tot += c;
                        }
                        if (tot == n) skip = true;
                    }
                }
                if (!skip) {
                    for (std::size_t i = b0; i < b1; ++i) {
                        std::size_t dst = h[(ks[i] >> shift) & (kRadixBuckets - 1)]++;
                        kd[dst] = ks[i];
                        id[dst] = is[i];
                    }
                }
            }
            if (!skip) { std::swap(ks, kd); std::swap(is, id); }
        }
        if (ks == key_.data() + lo) return;
        if (lo == 0 && hi == n_) {
            key_.swap(key2_); idx_.swap(idx2_);
        } else {
            std::memcpy(key_.data() + lo, ks, n * sizeof(uint64_t));
            std::memcpy(idx_.data() + lo, is, n * sizeof(uint32_t));
        }
    }

    // Re-keys bodies [lo, hi) in their own bounding cube, re-sorts them and returns that cube.
    // False (nothing changed) if they all coincide.
    bool rekey(uint32_t lo, uint32_t hi, double& gx, double& gy, double& gz, double& half) {
        const long l0 = long(lo), l1 = long(hi);
        float mnx = x_[lo], mny = y_[lo], mnz = z_[lo], mxx = mnx, mxy = mny, mxz = mnz;
#pragma omp parallel for num_threads(nt_) if (hi - lo >= kParMin) \
    reduction(min : mnx, mny, mnz) reduction(max : mxx, mxy, mxz)
        for (long i = l0; i < l1; ++i) {
            mnx = std::min(mnx, x_[i]); mny = std::min(mny, y_[i]); mnz = std::min(mnz, z_[i]);
            mxx = std::max(mxx, x_[i]); mxy = std::max(mxy, y_[i]); mxz = std::max(mxz, z_[i]);
        }
        double side = std::max({double(mxx) - mnx, double(mxy) - mny, double(mxz) - mnz});
        if (side == 0) return false;
        side *= 1 + 1e-6;
        const bool par = hi - lo >= kParMin;
        make_keys(lo, hi, mnx, mny, mnz, side, par);
        if (!par) {
            // Ties broken by index: the same order a stable radix sort gives.
            std::vector<std::pair<uint64_t, uint32_t>> kv(hi - lo);
            for (uint32_t i = lo; i < hi; ++i) kv[i - lo] = {key_[i], i};
            std::sort(kv.begin(), kv.end());
            for (uint32_t i = lo; i < hi; ++i) { key_[i] = kv[i - lo].first; idx_[i] = kv[i - lo].second; }
        } else {
            radix_sort(lo, hi, nt_);
        }
        permute(lo, hi, par);
        half = 0.5 * side;
        gx = mnx + half; gy = mny + half; gz = mnz + half;
        return true;
    }

    // ---- build -----------------------------------------------------------------------------
    // level counts Morton levels within the current key frame, depth counts levels from the root.
    void build_node(uint32_t ni, int level, int depth, double gx, double gy, double gz, double half, bool in_group) {
        const uint32_t lo = nodes_[ni].lo, hi = nodes_[ni].hi;
        max_depth_ = std::max(max_depth_, depth);
        bool leaf = hi - lo <= uint32_t(kLeafMax);
        // Keys exhausted with too many bodies: unless they coincide (kept as one exact leaf),
        // restart the keys in this node's own bounding cube. Otherwise one far body packs a whole
        // cluster into one leaf and one walk group, and the walk goes O(N^2).
        if (!leaf && level == kMaxLevel) {
            if (rekey(lo, hi, gx, gy, gz, half)) level = 0;
            else leaf = true;
        }
        geo_[ni] = {float(gx), float(gy), float(gz), float(half)};
        if (!in_group && (hi - lo <= uint32_t(kGroupMax) || leaf)) {
            groups_.push_back({0, 0, 0, 0, 0, 0, lo, hi});
            in_group = true;
        }
        if (leaf) {
            nodes_[ni].first = lo;
            nodes_[ni].nchild = 0;
            leaves_.push_back(ni);
            return;
        }
        const int shift = 3 * (kMaxLevel - 1 - level);
        const uint64_t* k = key_.data();
        uint32_t bound[9];
        bound[0] = lo;
        for (int o = 1; o < 8; ++o)
            bound[o] = uint32_t(std::partition_point(k + bound[o - 1], k + hi,
                                                     [&](uint64_t v) { return int((v >> shift) & 7) < o; }) - k);
        bound[8] = hi;
        const uint32_t first = uint32_t(nodes_.size());
        uint32_t nc = 0;
        for (int o = 0; o < 8; ++o)
            if (bound[o + 1] > bound[o]) {
                Node c{};
                c.lo = bound[o]; c.hi = bound[o + 1];
                nodes_.push_back(c);
                ++nc;
            }
        geo_.resize(nodes_.size());
        nodes_[ni].first = first;
        nodes_[ni].nchild = nc;
        const double q = 0.5 * half;
        uint32_t c = first;
        for (int o = 0; o < 8; ++o) {
            if (bound[o + 1] == bound[o]) continue;
            build_node(c++, level + 1, depth + 1, gx + ((o & 4) ? q : -q), gy + ((o & 2) ? q : -q),
                       gz + ((o & 1) ? q : -q), q, in_group);
        }
    }

    void build_octree(double theta) {
        nodes_.clear(); geo_.clear(); groups_.clear(); leaves_.clear();
        max_depth_ = 0;
        Node root{};
        root.lo = 0; root.hi = uint32_t(n_);
        nodes_.push_back(root);
        geo_.resize(1);
        const double h = 0.5 * side_;
        build_node(0, 0, 0, ox_ + h, oy_ + h, oz_ + h, h, false);

        const float inv_theta = theta > 0 ? float(1.0 / theta) : std::numeric_limits<float>::max();
        auto finish = [&](Node& nd, const Geo& g) {
            float dx = nd.cx - g.gx, dy = nd.cy - g.gy, dz = nd.cz - g.gz;
            double r = double(2 * g.half) * inv_theta + std::sqrt(double(dx * dx + dy * dy + dz * dz));
            nd.rcrit2 = float(std::min(r * r, double(std::numeric_limits<float>::max())));
        };

        const long ng = long(groups_.size());
#pragma omp parallel for num_threads(nt_) schedule(dynamic, 16)
        for (long g = 0; g < ng; ++g) {
            Group& gr = groups_[g];
            float mnx = x_[gr.lo], mny = y_[gr.lo], mnz = z_[gr.lo], mxx = mnx, mxy = mny, mxz = mnz;
            for (uint32_t i = gr.lo; i < gr.hi; ++i) {
                mnx = std::min(mnx, x_[i]); mny = std::min(mny, y_[i]); mnz = std::min(mnz, z_[i]);
                mxx = std::max(mxx, x_[i]); mxy = std::max(mxy, y_[i]); mxz = std::max(mxz, z_[i]);
            }
            gr.bx = 0.5f * (mnx + mxx); gr.by = 0.5f * (mny + mxy); gr.bz = 0.5f * (mnz + mxz);
            gr.hx = 0.5f * (mxx - mnx); gr.hy = 0.5f * (mxy - mny); gr.hz = 0.5f * (mxz - mnz);
        }

        // Leaf moments (the O(N) part) in parallel, about each leaf's centre of mass in double.
        const long nl = long(leaves_.size());
#pragma omp parallel for num_threads(nt_) schedule(dynamic, 64)
        for (long l = 0; l < nl; ++l) {
            Node& nd = nodes_[leaves_[l]];
            double M = 0, sx = 0, sy = 0, sz = 0;
            for (uint32_t i = nd.lo; i < nd.hi; ++i) {
                double m = m_[i];
                M += m; sx += m * x_[i]; sy += m * y_[i]; sz += m * z_[i];
            }
            const Geo& ge = geo_[leaves_[l]];
            double cx = M > 0 ? sx / M : ge.gx, cy = M > 0 ? sy / M : ge.gy, cz = M > 0 ? sz / M : ge.gz;
            double qxx = 0, qxy = 0, qxz = 0, qyy = 0, qyz = 0, qzz = 0;
            for (uint32_t i = nd.lo; i < nd.hi; ++i) {
                double m = m_[i], dx = x_[i] - cx, dy = y_[i] - cy, dz = z_[i] - cz, r2 = dx * dx + dy * dy + dz * dz;
                qxx += m * (3 * dx * dx - r2); qyy += m * (3 * dy * dy - r2); qzz += m * (3 * dz * dz - r2);
                qxy += m * 3 * dx * dy; qxz += m * 3 * dx * dz; qyz += m * 3 * dy * dz;
            }
            nd.cx = float(cx); nd.cy = float(cy); nd.cz = float(cz); nd.m = float(M);
            nd.qxx = float(qxx); nd.qxy = float(qxy); nd.qxz = float(qxz);
            nd.qyy = float(qyy); nd.qyz = float(qyz); nd.qzz = float(qzz);
            finish(nd, ge);
        }

        // Internal nodes: few (~N/kLeafMax), so a serial reverse sweep is cheap. Combined in double
        // via the parallel-axis theorem for the traceless quadrupole.
        for (long ni = long(nodes_.size()) - 1; ni >= 0; --ni) {
            Node& nd = nodes_[ni];
            if (nd.nchild == 0) continue;
            double M = 0, sx = 0, sy = 0, sz = 0;
            for (uint32_t c = nd.first; c < nd.first + nd.nchild; ++c) {
                const Node& ch = nodes_[c];
                M += ch.m; sx += double(ch.m) * ch.cx; sy += double(ch.m) * ch.cy; sz += double(ch.m) * ch.cz;
            }
            const Geo& ge = geo_[ni];
            double cx = M > 0 ? sx / M : ge.gx, cy = M > 0 ? sy / M : ge.gy, cz = M > 0 ? sz / M : ge.gz;
            double qxx = 0, qxy = 0, qxz = 0, qyy = 0, qyz = 0, qzz = 0;
            for (uint32_t c = nd.first; c < nd.first + nd.nchild; ++c) {
                const Node& ch = nodes_[c];
                double m = ch.m, dx = ch.cx - cx, dy = ch.cy - cy, dz = ch.cz - cz, r2 = dx * dx + dy * dy + dz * dz;
                qxx += ch.qxx + m * (3 * dx * dx - r2); qyy += ch.qyy + m * (3 * dy * dy - r2);
                qzz += ch.qzz + m * (3 * dz * dz - r2);
                qxy += ch.qxy + m * 3 * dx * dy; qxz += ch.qxz + m * 3 * dx * dz; qyz += ch.qyz + m * 3 * dy * dz;
            }
            nd.cx = float(cx); nd.cy = float(cy); nd.cz = float(cz); nd.m = float(M);
            nd.qxx = float(qxx); nd.qxy = float(qxy); nd.qxz = float(qxz);
            nd.qyy = float(qyy); nd.qyz = float(qyz); nd.qzz = float(qzz);
            finish(nd, ge);
        }
    }

    // ---- walk ------------------------------------------------------------------------------
    template <bool POT>
    uint64_t walk(float G, float e2, const std::vector<const LetTree*>& remote) {
        const long ng = long(groups_.size());
        uint64_t inter = 0;
#pragma omp parallel num_threads(nt_) reduction(+ : inter)
        {
            WalkScratch& s = scratch_[omp_get_thread_num()];
            const std::size_t stack_need = 8 * std::size_t(max_depth_ + 2);
            if (s.stack.size() < stack_need) s.stack.resize(stack_need);
#pragma omp for schedule(dynamic, 4)
            for (long g = 0; g < ng; ++g) {
                const Group& gr = groups_[g];
                build_list(gr, s);
                for (const LetTree* t : remote) build_list_let(*t, gr, s);
                pad_lists(s);
                eval_group<POT, false>(gr, s, G, e2);
                const std::size_t len = s.nb + s.nn;
                std::fill(cost_.begin() + gr.lo, cost_.begin() + gr.hi, float(len));
                inter += uint64_t(len) * (gr.hi - gr.lo);
            }
        }
        return inter;
    }

    // Direct method: local forces are already in a_/pot_; remote LETs are added per block of
    // kGroupMax consecutive bodies (unsorted, so the boxes are loose, which only matters for
    // LETs that are not always-open).
    template <bool POT>
    uint64_t direct_remote(float G, float e2, const std::vector<const LetTree*>& remote) {
        const long nblk = long((n_ + kGroupMax - 1) / kGroupMax);
        std::fill(cost_.begin(), cost_.end(), float(n_));
        if (remote.empty()) return 0;
        uint64_t inter = 0;
#pragma omp parallel num_threads(nt_) reduction(+ : inter)
        {
            WalkScratch& s = scratch_[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 1)
            for (long b = 0; b < nblk; ++b) {
                Group gr = bounds_group(uint32_t(b * kGroupMax), uint32_t(std::min<std::size_t>(n_, (b + 1) * kGroupMax)));
                s.nb = s.nn = 0;
                for (const LetTree* t : remote) build_list_let(*t, gr, s);
                pad_lists(s);
                eval_group<POT, true>(gr, s, G, e2);
                const std::size_t len = s.nb + s.nn;
                for (uint32_t i = gr.lo; i < gr.hi; ++i) cost_[i] += float(len);
                inter += uint64_t(len) * (gr.hi - gr.lo);
            }
        }
        return inter;
    }

    Group bounds_group(uint32_t lo, uint32_t hi) const {
        Group gr{0, 0, 0, 0, 0, 0, lo, hi};
        float mnx = x_[lo], mny = y_[lo], mnz = z_[lo], mxx = mnx, mxy = mny, mxz = mnz;
        for (uint32_t i = lo; i < hi; ++i) {
            mnx = std::min(mnx, x_[i]); mny = std::min(mny, y_[i]); mnz = std::min(mnz, z_[i]);
            mxx = std::max(mxx, x_[i]); mxy = std::max(mxy, y_[i]); mxz = std::max(mxz, z_[i]);
        }
        gr.bx = 0.5f * (mnx + mxx); gr.by = 0.5f * (mny + mxy); gr.bz = 0.5f * (mnz + mxz);
        gr.hx = 0.5f * (mxx - mnx); gr.hy = 0.5f * (mxy - mny); gr.hz = 0.5f * (mxz - mnz);
        return gr;
    }

    LetTree export_direct() const {
        LetTree out;
        out.bx = x_; out.by = y_; out.bz = z_; out.bm = m_;
        double M = 0, sx = 0, sy = 0, sz = 0;
        for (std::size_t i = 0; i < n_; ++i) {
            double m = m_[i];
            M += m; sx += m * x_[i]; sy += m * y_[i]; sz += m * z_[i];
        }
        LetNode r{};
        if (M > 0) { r.cx = float(sx / M); r.cy = float(sy / M); r.cz = float(sz / M); }
        r.mass = float(M);
        // Always opened, so the quadrupole is never read and stays zero.
        r.rcrit2 = 3.0e38f;
        r.first_child = -1; r.child_count = 0;
        r.body_first = 0; r.body_count = int32_t(n_);
        out.nodes.push_back(r);
        return out;
    }

    // Zero-mass padding to a multiple of 8 (index -1 never matches a body, Q = 0) lets the kernels
    // run without tails; padding is not counted as interactions.
    static void pad_lists(WalkScratch& s) {
        const std::size_t nb8 = (s.nb + 7) & ~std::size_t(7), nn8 = (s.nn + 7) & ~std::size_t(7);
        s.need_bodies(nb8);
        s.need_nodes(nn8);
        for (std::size_t k = s.nb; k < nb8; ++k) { s.bx[k] = s.by[k] = s.bz[k] = s.bm[k] = 0; s.bi[k] = -1; }
        for (std::size_t k = s.nn; k < nn8; ++k)
            for (auto* v : {&s.nx, &s.ny, &s.nz, &s.nm, &s.qxx, &s.qxy, &s.qxz, &s.qyy, &s.qyz, &s.qzz}) (*v)[k] = 0;
    }

    void push_bodies(WalkScratch& s, uint32_t lo, uint32_t hi) {
        s.need_bodies(s.nb + (hi - lo));
        for (uint32_t j = lo; j < hi; ++j, ++s.nb) {
            s.bx[s.nb] = x_[j]; s.by[s.nb] = y_[j]; s.bz[s.nb] = z_[j]; s.bm[s.nb] = m_[j];
            s.bi[s.nb] = int32_t(j);
        }
    }

    // Remote bodies get index -1 so the self-exclusion in p2p8 never drops one.
    static void push_let_bodies(WalkScratch& s, const LetTree& t, const LetNode& nd) {
        s.need_bodies(s.nb + std::size_t(nd.body_count));
        for (int32_t j = nd.body_first; j < nd.body_first + nd.body_count; ++j, ++s.nb) {
            s.bx[s.nb] = t.bx[j]; s.by[s.nb] = t.by[j]; s.bz[s.nb] = t.bz[j]; s.bm[s.nb] = t.bm[j];
            s.bi[s.nb] = -1;
        }
    }

    template <class N>
    static void push_node(WalkScratch& s, const N& nd, float m) {
        s.need_nodes(s.nn + 1);
        const std::size_t k = s.nn++;
        s.nx[k] = nd.cx; s.ny[k] = nd.cy; s.nz[k] = nd.cz; s.nm[k] = m;
        s.qxx[k] = nd.qxx; s.qxy[k] = nd.qxy; s.qxz[k] = nd.qxz;
        s.qyy[k] = nd.qyy; s.qyz[k] = nd.qyz; s.qzz[k] = nd.qzz;
    }

    // Receiver walk of backend.hpp, against the group box, appending to the group's lists.
    static void build_list_let(const LetTree& t, const Group& gr, WalkScratch& s) {
        if (t.nodes.empty()) return;
        // A LET may come from another backend with a different depth, so this stack grows.
        auto& st = s.let_stack;
        if (st.size() < 64) st.resize(64);
        std::size_t sp = 0;
        st[sp++] = 0;
        while (sp) {
            const LetNode& nd = t.nodes[std::size_t(st[--sp])];
            const bool terminal = nd.first_child < 0 && nd.body_count == 0;
            bool accept = terminal;
            if (!terminal) {
                float dx = std::max(std::fabs(nd.cx - gr.bx) - gr.hx, 0.f);
                float dy = std::max(std::fabs(nd.cy - gr.by) - gr.hy, 0.f);
                float dz = std::max(std::fabs(nd.cz - gr.bz) - gr.hz, 0.f);
                accept = dx * dx + dy * dy + dz * dz >= nd.rcrit2;
            }
            if (accept) {
                if (nd.body_count > 0 && nd.body_count <= kSmallLeaf) push_let_bodies(s, t, nd);
                else push_node(s, nd, nd.mass);
            } else if (nd.body_count > 0) {
                push_let_bodies(s, t, nd);
            } else {
                if (sp + std::size_t(nd.child_count) > st.size()) st.resize(2 * (sp + std::size_t(nd.child_count)));
                for (int32_t c = nd.first_child + nd.child_count; c-- > nd.first_child;) st[sp++] = c;
            }
        }
    }

    void build_list(const Group& gr, WalkScratch& s) {
        s.nb = s.nn = 0;
        uint32_t sp = 0;
        s.stack[sp++] = 0;
        while (sp) {
            const Node& nd = nodes_[s.stack[--sp]];
            float dx = std::max(std::fabs(nd.cx - gr.bx) - gr.hx, 0.f);
            float dy = std::max(std::fabs(nd.cy - gr.by) - gr.hy, 0.f);
            float dz = std::max(std::fabs(nd.cz - gr.bz) - gr.hz, 0.f);
            // A node containing the group is always opened: with a huge theta its multipole would
            // otherwise include the group's own bodies.
            bool contains = nd.lo <= gr.lo && gr.hi <= nd.hi;
            if (!contains && dx * dx + dy * dy + dz * dz >= nd.rcrit2) {
                if (nd.nchild == 0 && nd.hi - nd.lo <= uint32_t(kSmallLeaf)) push_bodies(s, nd.lo, nd.hi);
                else push_node(s, nd, nd.m);
            } else if (nd.nchild == 0) {
                push_bodies(s, nd.lo, nd.hi);
            } else {
                // Depth-first with at most 7 pending siblings per level: bounded by 8*(max_depth_+1).
                for (uint32_t c = nd.first + nd.nchild; c-- > nd.first;) s.stack[sp++] = c;
            }
        }
    }

    // ADD accumulates into the existing accelerations/potential instead of overwriting them.
    template <bool POT, bool ADD>
    void eval_group(const Group& gr, const WalkScratch& s, float G, float e2) {
        const float *bx = s.bx.data(), *by = s.by.data(), *bz = s.bz.data(), *bm = s.bm.data();
        const int32_t* bi = s.bi.data();
        const float *nx = s.nx.data(), *ny = s.ny.data(), *nz = s.nz.data(), *nm = s.nm.data();
        const float *Qxx = s.qxx.data(), *Qxy = s.qxy.data(), *Qxz = s.qxz.data();
        const float *Qyy = s.qyy.data(), *Qyz = s.qyz.data(), *Qzz = s.qzz.data();
        const long nb = long((s.nb + 7) & ~std::size_t(7)), nn = long((s.nn + 7) & ~std::size_t(7));
        for (uint32_t i = gr.lo; i < gr.hi; ++i) {
            const float xi = x_[i], yi = y_[i], zi = z_[i];
            const int32_t ii = int32_t(i);
            float ax = 0, ay = 0, az = 0, p = 0;
            p2p8<POT>(bx, by, bz, bm, bi, 0, nb, xi, yi, zi, ii, e2, ax, ay, az, p);
            v8f vax = {}, vay = {}, vaz = {}, vp = {};
            for (long k = 0; k < nn; k += 8) {
                v8f dx = load8(nx + k) - xi, dy = load8(ny + k) - yi, dz = load8(nz + k) - zi;
                v8f r2 = dx * dx + dy * dy + dz * dz + e2;
                v8f inv = rsqrt_masked(r2, (v8i)(r2 > 0.f));
                v8f inv2 = inv * inv, inv3 = inv * inv2, inv5 = inv3 * inv2;
                v8f xx = load8(Qxx + k), xy = load8(Qxy + k), xz = load8(Qxz + k);
                v8f yy = load8(Qyy + k), yz = load8(Qyz + k), zz = load8(Qzz + k);
                // d points from the body to the node: a = G [M d/r^3 - Q d/r^5 + 5/2 (dQd) d/r^7]
                v8f qx = xx * dx + xy * dy + xz * dz;
                v8f qy = xy * dx + yy * dy + yz * dz;
                v8f qz = xz * dx + yz * dy + zz * dz;
                v8f dqd = dx * qx + dy * qy + dz * qz;
                v8f mk = load8(nm + k);
                v8f f = mk * inv3 + 2.5f * dqd * inv5 * inv2;
                vax += f * dx - qx * inv5; vay += f * dy - qy * inv5; vaz += f * dz - qz * inv5;
                if constexpr (POT) vp += mk * inv + 0.5f * dqd * inv5;
            }
            ax += hsum(vax); ay += hsum(vay); az += hsum(vaz);
            if constexpr (POT) p += hsum(vp);
            if constexpr (ADD) {
                ax_[i] += G * ax; ay_[i] += G * ay; az_[i] += G * az;
                if constexpr (POT) pot_[i] -= G * p;
            } else {
                ax_[i] = G * ax; ay_[i] = G * ay; az_[i] = G * az;
                if constexpr (POT) pot_[i] = -G * p;
            }
        }
    }

    int nt_;
    std::size_t n_ = 0;
    std::vector<float> x_, y_, z_, vx_, vy_, vz_, m_, ax_, ay_, az_, pot_;
    std::vector<uint32_t> id_;
    bool has_pot_ = false;
    bool tree_ok_ = false;   // tree and body order match the current positions
    double pend_sort_ms_ = 0, pend_build_ms_ = 0;  // from build_tree, reported by the next force call
    std::vector<float> cost_;  // interactions per body of the last force call, current body order
    PhaseTimes times_;
    double kick_ms_ = 0, drift_ms_ = 0;

    // Tree buffers, reused across calls (grow-only).
    double ox_ = 0, oy_ = 0, oz_ = 0, side_ = 1;
    std::vector<uint64_t> key_, key2_;
    std::vector<uint32_t> idx_, idx2_, tmpu_;
    std::vector<float> tmpf_;
    std::vector<std::size_t> hist_;
    std::vector<Node> nodes_;
    std::vector<Geo> geo_;
    std::vector<Group> groups_;
    std::vector<uint32_t> leaves_;
    int max_depth_ = 0;  // deepest node; key restarts let it exceed kMaxLevel
    std::vector<WalkScratch> scratch_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(int threads) { return std::make_unique<CpuBackend>(threads); }

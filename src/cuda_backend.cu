// CUDA backend: tiled direct summation and a Barnes-Hut tree rebuilt every force call from
// Morton-sorted bodies (leaf buckets + Karras radix tree + atomic bottom-up moments).
//
// Must compile with both clang (CUDA mode, C++23, libc++) and nvcc (C++20, g++ host): plain
// cuda_runtime API, no nvcc-only extensions, no reliance on fast-math flags for correctness.
// clang's -ffast-math implies finite-math-only, so no infinities or NaN tests appear anywhere.
//
// Device memory, per body (capacity n):
//   state x,y,z,vx,vy,vz,m,id       32 B
//   ax,ay,az,pot                    16 B
//   interaction counts               4 B   per-body cost for body_costs()
//   posm (float4 pos+mass)          16 B   gathered copy for coalesced 16-byte loads in kernels
//                                          (scratch and subkeys while refining long runs)
//   Morton keys, double-buffered    16 B   (the alternate also holds leaf flags/scan/starts)
//   sort indices, double-buffered    8 B   (the alternate is the permutation scratch / group slots)
//   = 92 B, plus tree nodes: 96 B per node (incl. an 8 B body range), 2K-1 nodes for K leaves
//   (K ~ n/8..n/16 for Plummer), plus <= 4 B per leaf of walk-group list
//   ~ 105-117 B/body total, so 80 GB holds ~650M bodies. Nodes are sized from the actual K.
//   Remote LETs (compute_forces_with) add 48 B per LET node and 16 B per LET body, grow-only.
//
// Limits: int body and node indices. upload() refuses n >= 2^31-1 bodies; build_tree_impl()
// refuses a tree of more than 2^31-1 nodes (2K-1 for K leaves).
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"

namespace {

void check_cuda(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + what + ": " +
                                 cudaGetErrorString(e));
}
#define CK(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CK_LAUNCH() check_cuda(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

class DevBuf {
public:
    DevBuf() = default;
    ~DevBuf() {
        if (p_) cudaFree(p_);  // errors ignored: may run after context teardown at exit
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    // Grow-only; contents are not preserved across growth.
    void reserve(std::size_t bytes) {
        if (bytes <= cap_) return;
        if (p_) cudaFree(p_);
        p_ = nullptr;
        cap_ = 0;
        cudaError_t e = cudaMalloc(&p_, bytes);
        if (e != cudaSuccess) {
            p_ = nullptr;
            throw std::runtime_error("cudaMalloc(" + std::to_string(bytes) + " bytes) failed: " +
                                     cudaGetErrorString(e));
        }
        cap_ = bytes;
    }
    template <class T> T* as() const { return static_cast<T*>(p_); }
    std::size_t capacity() const { return cap_; }
    friend void swap(DevBuf& a, DevBuf& b) noexcept {
        std::swap(a.p_, b.p_);
        std::swap(a.cap_, b.cap_);
    }

private:
    void* p_ = nullptr;
    std::size_t cap_ = 0;
};

struct Stream {
    cudaStream_t s = nullptr;
    Stream() { CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t e = nullptr;
    Event() { CK(cudaEventCreate(&e)); }
    ~Event() { if (e) cudaEventDestroy(e); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

constexpr int BS = 256;          // elementwise / build kernels
constexpr int DIRECT_BS = 256;   // direct tile = block; 8 warps hide the rsqrt latency on sm_90
constexpr int WALK_BS = 128;     // smaller blocks balance the uneven per-body walk cost better
constexpr int LEAF_MAX = 32;
// Bounds on the internal nodes of one root-to-leaf radix-tree path. Node prefix lengths (delta)
// strictly increase downwards and take at most 63 values on the 63-bit key, 63 on the 63-bit
// subkey (refined trees only) and 31 on the leaf-index tie-break (K < 2^31).
constexpr int DEPTH_KEY = 96;    // >= 63 + 31
constexpr int DEPTH_SUB = 160;   // >= 63 + 63 + 31
constexpr int STACK = DEPTH_SUB; // per-body walk: at most one pending sibling per internal node above
constexpr int REDUCE_BLOCKS = 1024;
constexpr int DIAG_N = 13;

inline unsigned grid_for(long long n, int bs) { return static_cast<unsigned>((n + bs - 1) / bs); }

// ---------------------------------------------------------------- elementwise kernels

__global__ void pack_posm(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
                          const float* __restrict__ m, float4* __restrict__ pm, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pm[i] = make_float4(x[i], y[i], z[i], m[i]);
}

__global__ void gather_posm(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
                            const float* __restrict__ m, const uint32_t* __restrict__ perm,
                            float4* __restrict__ pm, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t j = perm[i];
        pm[i] = make_float4(x[j], y[j], z[j], m[j]);
    }
}

__global__ void unpack_posm(const float4* __restrict__ pm, float* __restrict__ x, float* __restrict__ y,
                            float* __restrict__ z, float* __restrict__ m, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 p = pm[i];
        x[i] = p.x; y[i] = p.y; z[i] = p.z; m[i] = p.w;
    }
}

__global__ void gather_u32(const uint32_t* __restrict__ src, const uint32_t* __restrict__ perm,
                           uint32_t* __restrict__ dst, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

__global__ void kick_kernel(float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
                            const float* __restrict__ ax, const float* __restrict__ ay,
                            const float* __restrict__ az, float dt, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vx[i] = fmaf(ax[i], dt, vx[i]);
        vy[i] = fmaf(ay[i], dt, vy[i]);
        vz[i] = fmaf(az[i], dt, vz[i]);
    }
}

__global__ void drift_kernel(float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
                             const float* __restrict__ vx, const float* __restrict__ vy,
                             const float* __restrict__ vz, float dt, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fmaf(vx[i], dt, x[i]);
        y[i] = fmaf(vy[i], dt, y[i]);
        z[i] = fmaf(vz[i], dt, z[i]);
    }
}

// ---------------------------------------------------------------- reductions

template <class T> __device__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Sums DIAG_N doubles over the block; result valid in thread 0. Requires blockDim.x == BS.
__device__ void block_sum_diag(double (&v)[DIAG_N], double (*sh)[DIAG_N]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < DIAG_N; ++c) v[c] = warp_sum(v[c]);
    if (lane == 0)
        for (int c = 0; c < DIAG_N; ++c) sh[warp][c] = v[c];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < DIAG_N; ++c) {
            double s = lane < BS / 32 ? sh[lane][c] : 0.0;
            v[c] = warp_sum(s);
        }
    }
}

__global__ void __launch_bounds__(BS) bbox_partial(const float* __restrict__ x, const float* __restrict__ y,
                                                   const float* __restrict__ z, int n, float* __restrict__ out) {
    __shared__ float sh[6][BS];
    float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (long long i = (long long)blockIdx.x * BS + threadIdx.x; i < n; i += (long long)gridDim.x * BS) {
        float p[3] = {x[i], y[i], z[i]};
        for (int c = 0; c < 3; ++c) { lo[c] = fminf(lo[c], p[c]); hi[c] = fmaxf(hi[c], p[c]); }
    }
    for (int c = 0; c < 3; ++c) { sh[c][threadIdx.x] = lo[c]; sh[3 + c][threadIdx.x] = hi[c]; }
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int c = 0; c < 3; ++c) {
                sh[c][threadIdx.x] = fminf(sh[c][threadIdx.x], sh[c][threadIdx.x + s]);
                sh[3 + c][threadIdx.x] = fmaxf(sh[3 + c][threadIdx.x], sh[3 + c][threadIdx.x + s]);
            }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int c = 0; c < 6; ++c) out[blockIdx.x * 6 + c] = sh[c][0];
}

// Single block: reduces `parts` partial boxes into the Morton cube (lo.xyz, cells-per-unit).
__global__ void __launch_bounds__(BS) bbox_final(const float* __restrict__ part, int parts, float4* __restrict__ cube) {
    __shared__ float sh[6][BS];
    float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int b = threadIdx.x; b < parts; b += BS)
        for (int c = 0; c < 3; ++c) {
            lo[c] = fminf(lo[c], part[b * 6 + c]);
            hi[c] = fmaxf(hi[c], part[b * 6 + 3 + c]);
        }
    for (int c = 0; c < 3; ++c) { sh[c][threadIdx.x] = lo[c]; sh[3 + c][threadIdx.x] = hi[c]; }
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int c = 0; c < 3; ++c) {
                sh[c][threadIdx.x] = fminf(sh[c][threadIdx.x], sh[c][threadIdx.x + s]);
                sh[3 + c][threadIdx.x] = fmaxf(sh[3 + c][threadIdx.x], sh[3 + c][threadIdx.x + s]);
            }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float side = fmaxf(sh[3][0] - sh[0][0], fmaxf(sh[4][0] - sh[1][0], sh[5][0] - sh[2][0]));
        // All coincident, or non-finite input (tested on the bits: finite-math may fold float tests).
        if ((__float_as_uint(side) & 0x7f800000u) == 0x7f800000u || side <= 0.f) side = 1.f;
        *cube = make_float4(sh[0][0], sh[1][0], sh[2][0], 2097152.f / side);
    }
}

__global__ void __launch_bounds__(BS) diag_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                                  const float* __restrict__ z, const float* __restrict__ vx,
                                                  const float* __restrict__ vy, const float* __restrict__ vz,
                                                  const float* __restrict__ m, const float* __restrict__ pot,
                                                  int use_pot, int n, double* __restrict__ out) {
    __shared__ double sh[BS / 32][DIAG_N];
    double v[DIAG_N] = {};
    for (long long i = (long long)blockIdx.x * BS + threadIdx.x; i < n; i += (long long)gridDim.x * BS) {
        double mi = m[i], px = x[i], py = y[i], pz = z[i], ux = vx[i], uy = vy[i], uz = vz[i];
        v[0] += mi;
        v[1] += 0.5 * mi * (ux * ux + uy * uy + uz * uz);
        if (use_pot) v[2] += mi * (double)pot[i];
        v[3] += mi * ux; v[4] += mi * uy; v[5] += mi * uz;
        v[6] += mi * (py * uz - pz * uy);
        v[7] += mi * (pz * ux - px * uz);
        v[8] += mi * (px * uy - py * ux);
        v[9] += mi * px; v[10] += mi * py; v[11] += mi * pz;
    }
    block_sum_diag(v, sh);
    if (threadIdx.x == 0)
        for (int c = 0; c < DIAG_N; ++c) out[(long long)blockIdx.x * DIAG_N + c] = v[c];
}

// Exact potential energy sum_i m_i sum_{j!=i} m_j / s_ij, pair terms in double.
__global__ void __launch_bounds__(DIRECT_BS) exact_pe_kernel(const float4* __restrict__ pm, int n, double eps2,
                                                             double* __restrict__ acc) {
    __shared__ float4 tile[DIRECT_BS];
    __shared__ double sh[BS / 32][DIAG_N];
    const long long gi = (long long)blockIdx.x * DIRECT_BS + threadIdx.x;
    const bool valid = gi < n;
    const float4 p = valid ? pm[gi] : make_float4(0.f, 0.f, 0.f, 0.f);
    double phi = 0;
    const int ntiles = (n - 1) / DIRECT_BS + 1;
    for (int t = 0; t < ntiles; ++t) {
        const long long j0 = (long long)t * DIRECT_BS;
        const long long jj = j0 + threadIdx.x;
        tile[threadIdx.x] = jj < n ? pm[jj] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();
        const int cnt = (int)min((long long)DIRECT_BS, (long long)n - j0);
        for (int k = 0; k < cnt; ++k) {
            if (j0 + k == gi) continue;
            const float4 q = tile[k];
            double dx = (double)q.x - p.x, dy = (double)q.y - p.y, dz = (double)q.z - p.z;
            phi += (double)q.w / sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
        __syncthreads();
    }
    double v[DIAG_N] = {};
    v[0] = valid ? (double)p.w * phi : 0.0;
    block_sum_diag(v, sh);
    if (threadIdx.x == 0) atomicAdd(acc, v[0]);
}

// ---------------------------------------------------------------- direct

template <bool POT>
__device__ __forceinline__ void pp(const float4 p, const float4 q, float eps2, float& sx, float& sy, float& sz,
                                   float& sp) {
    const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    const float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
    const float inv = rsqrtf(r2);
    const float minv = q.w * inv;
    const float minv3 = minv * inv * inv;
    sx = fmaf(dx, minv3, sx);
    sy = fmaf(dy, minv3, sy);
    sz = fmaf(dz, minv3, sz);
    if (POT) sp += minv;
}

template <bool POT>
__global__ void __launch_bounds__(DIRECT_BS) direct_kernel(const float4* __restrict__ pm, int n, float eps2, float G,
                                                           float* __restrict__ ax, float* __restrict__ ay,
                                                           float* __restrict__ az, float* __restrict__ pot) {
    __shared__ float4 tile[DIRECT_BS];
    const long long gi = (long long)blockIdx.x * DIRECT_BS + threadIdx.x;
    const bool valid = gi < n;
    const float4 p = valid ? pm[gi] : make_float4(0.f, 0.f, 0.f, 0.f);
    float sx = 0, sy = 0, sz = 0, sp = 0;
    const int ntiles = (n - 1) / DIRECT_BS + 1;
    for (int t = 0; t < ntiles; ++t) {
        const long long j0 = (long long)t * DIRECT_BS;
        const long long jj = j0 + threadIdx.x;
        tile[threadIdx.x] = jj < n ? pm[jj] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();
        const int cnt = (int)min((long long)DIRECT_BS, (long long)n - j0);
        // Only the block's own tile can contain the body itself; excluding it by index (not by
        // subtracting m/eps afterwards) keeps eps = 0 finite and avoids cancellation in phi.
        if (t == (int)blockIdx.x) {
            for (int k = 0; k < cnt; ++k)
                if (k != (int)threadIdx.x) pp<POT>(p, tile[k], eps2, sx, sy, sz, sp);
        } else if (cnt == DIRECT_BS) {
#pragma unroll 16
            for (int k = 0; k < DIRECT_BS; ++k) pp<POT>(p, tile[k], eps2, sx, sy, sz, sp);
        } else {
            for (int k = 0; k < cnt; ++k) pp<POT>(p, tile[k], eps2, sx, sy, sz, sp);
        }
        __syncthreads();
    }
    if (valid) {
        ax[gi] = G * sx;
        ay[gi] = G * sy;
        az[gi] = G * sz;
        if (POT) pot[gi] = -G * sp;
    }
}

// ---------------------------------------------------------------- tree: keys and leaves

__device__ __forceinline__ uint64_t spread3(uint64_t v) {
    v &= 0x1fffffull;
    v = (v | v << 32) & 0x1f00000000ffffull;
    v = (v | v << 16) & 0x1f0000ff0000ffull;
    v = (v | v << 8) & 0x100f00f00f00f00full;
    v = (v | v << 4) & 0x10c30c30c30c30c3ull;
    v = (v | v << 2) & 0x1249249249249249ull;
    return v;
}

__device__ __forceinline__ uint32_t quantize(float v, float lo, float scale) {
    // Clamped: x == box max would otherwise land in cell 2^21, outside the 21-bit grid.
    float q = fminf(fmaxf((v - lo) * scale, 0.f), 2097151.f);
    return static_cast<uint32_t>(q);
}

__device__ __forceinline__ uint64_t morton3(float x, float y, float z, float lx, float ly, float lz, float scale) {
    return spread3(quantize(x, lx, scale)) << 2 | spread3(quantize(y, ly, scale)) << 1 | spread3(quantize(z, lz, scale));
}

__global__ void morton_kernel(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
                              const float4* __restrict__ cube, uint64_t* __restrict__ keys,
                              uint32_t* __restrict__ idx, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 c = *cube;
    keys[i] = morton3(x[i], y[i], z[i], c.x, c.y, c.z, c.w);
    idx[i] = static_cast<uint32_t>(i);
}

// ---------------------------------------------------------------- tree: refining duplicate-key runs
//
// One global cube can put a whole cluster on a few keys (one far escaper is enough). A run of
// > 32 identical keys cut into leaves by sorted index gives spatially random leaves whose boxes
// all span the cluster, and the walk degrades to direct summation. Such runs are refined once:
// each body of a long run gets a subkey, the Morton key of its position in the run's own
// bounding cube, and the run is re-sorted by it. The tree then works on (key, subkey) as one
// longer key; bodies outside long runs have subkey 0. A run whose bodies coincide (zero extent)
// gets subkeys 0 and is still cut into index chunks, as are runs still identical after the subkey.
//
// The re-sort is one radix sort of all n bodies by (segment start << so3 | subkey >> (63 - so3)),
// where the segment start is the run's first index for run bodies and i otherwise: segments keep
// their order and positions, and only long runs change internally. With pbits = bits(n - 1) the
// subkey keeps so3 = 3 min(21, (64 - pbits) / 3) bits (>= 11 levels at n < 2^31), its low bits zero.

// Order-preserving float <-> int map for atomicMin/Max (an involution).
__device__ __forceinline__ int ord_float(float f) {
    const int b = __float_as_int(f);
    return b >= 0 ? b : b ^ 0x7fffffff;
}
__device__ __forceinline__ float unord_float(int o) { return __int_as_float(o >= 0 ? o : o ^ 0x7fffffff); }

// flags[i] = 1 iff body i starts a run of > LEAF_MAX identical keys.
__global__ void long_run_starts(const uint64_t* __restrict__ keys, uint32_t* __restrict__ flags, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t k = keys[i];
    flags[i] = (i == 0 || keys[i - 1] != k) && i + LEAF_MAX < n && keys[i + LEAF_MAX] == k ? 1u : 0u;
}

// incl = inclusive scan of the flags; run r = incl - 1 gets info[7r] = first body, then the
// box as ord_float (lo xyz, hi xyz), initialised empty here.
__global__ void long_run_init(const uint32_t* __restrict__ flags, const uint32_t* __restrict__ incl,
                              int* __restrict__ info, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flags[i]) return;
    int* R = info + 7 * (static_cast<long long>(incl[i]) - 1);
    R[0] = static_cast<int>(i);
    R[1] = R[2] = R[3] = INT_MAX;
    R[4] = R[5] = R[6] = INT_MIN;
}

// Run of body i, or -1. The last long-run start at or before i is incl[i] - 1; i belongs to it
// iff it has the same key (keys are sorted).
__device__ __forceinline__ int long_run_of(const uint64_t* __restrict__ keys, const uint32_t* __restrict__ incl,
                                           const int* __restrict__ info, long long i) {
    const uint32_t c = incl[i];
    if (c == 0) return -1;
    const int r = static_cast<int>(c - 1);
    return keys[i] == keys[info[7 * static_cast<long long>(r)]] ? r : -1;
}

// Run boxes by atomics; a warp whose bodies all lie in one run (the common case: runs are long
// and contiguous) reduces first and issues one set of atomics. Requires blockDim.x % 32 == 0.
__global__ void long_run_box(const uint64_t* __restrict__ keys, const uint32_t* __restrict__ incl,
                             const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
                             int* __restrict__ info, int n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int r = -1;
    int lo[3] = {0, 0, 0};
    if (i < n) {
        r = long_run_of(keys, incl, info, i);
        if (r >= 0) { lo[0] = ord_float(x[i]); lo[1] = ord_float(y[i]); lo[2] = ord_float(z[i]); }
    }
    int hi[3] = {lo[0], lo[1], lo[2]};
    const int r0 = __shfl_sync(0xffffffffu, r, 0);
    const bool uniform = __all_sync(0xffffffffu, r == r0);
    if (uniform) {
        if (r0 < 0) return;
        for (int o = 16; o > 0; o >>= 1)
            for (int c = 0; c < 3; ++c) {
                lo[c] = min(lo[c], __shfl_xor_sync(0xffffffffu, lo[c], o));
                hi[c] = max(hi[c], __shfl_xor_sync(0xffffffffu, hi[c], o));
            }
        if ((threadIdx.x & 31) != 0) return;
    } else if (r < 0) {
        return;
    }
    const int ru = uniform ? r0 : r;
    int* R = info + 7 * static_cast<long long>(ru);
    for (int c = 0; c < 3; ++c) {
        atomicMin(&R[1 + c], lo[c]);
        atomicMax(&R[4 + c], hi[c]);
    }
}

// Sort keys for the re-sort (see above) and the identity permutation.
__global__ void long_run_keys(const uint64_t* __restrict__ keys, const uint32_t* __restrict__ incl,
                              const int* __restrict__ info, const float* __restrict__ x,
                              const float* __restrict__ y, const float* __restrict__ z, int so3,
                              uint64_t* __restrict__ key2, uint32_t* __restrict__ idx, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t seg = static_cast<uint64_t>(i), sub = 0;
    const int r = long_run_of(keys, incl, info, i);
    if (r >= 0) {
        const int* R = info + 7 * static_cast<long long>(r);
        seg = static_cast<uint64_t>(R[0]);
        const float lx = unord_float(R[1]), ly = unord_float(R[2]), lz = unord_float(R[3]);
        const float side = fmaxf(unord_float(R[4]) - lx, fmaxf(unord_float(R[5]) - ly, unord_float(R[6]) - lz));
        float scale = side > 0.f ? 2097152.f / side : 0.f;
        // A denormal side overflows the scale; such a run is coincident for all purposes. Tested on
        // the bits, as in bbox_final.
        if ((__float_as_uint(scale) & 0x7f800000u) == 0x7f800000u) scale = 0.f;
        sub = morton3(x[i], y[i], z[i], lx, ly, lz, scale) >> (63 - so3);
    }
    key2[i] = seg << so3 | sub;
    idx[i] = static_cast<uint32_t>(i);
}

// Sorted sort keys -> subkeys (63-bit, low 63 - so3 bits zero; 0 outside long runs). In place allowed.
__global__ void long_run_subkeys(const uint64_t* key2, uint64_t* sub, int so3, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sub[i] = (key2[i] & ((1ull << so3) - 1)) << (63 - so3);
}

// ---------------------------------------------------------------- tree: leaves
//
// Keys are (key, subkey) pairs; sub == nullptr means all subkeys are 0. Octree levels 0..21 come
// from the key, levels 22..42 from the subkey (only long runs have one: they lie in a single
// level-21 cell, and the subkey's cube is inside it, so cells stay nested and contiguous).

// Common leading octal digits of two keys (42 = identical pairs).
__device__ __forceinline__ int common_octs(uint64_t a, uint64_t b, uint64_t sa, uint64_t sb) {
    if (a != b) return (__clzll(static_cast<long long>(a ^ b)) - 1) / 3;
    return sa == sb ? 42 : 21 + (__clzll(static_cast<long long>(sa ^ sb)) - 1) / 3;
}

__device__ __forceinline__ uint64_t sub_at(const uint64_t* __restrict__ sub, long long i) { return sub ? sub[i] : 0ull; }

// Leaf scheme: a leaf is the smallest octree cell (level L, key prefix of 3L bits) that holds
// <= 32 bodies. Cells are contiguous in sorted order, so the cell of body i at level L has >= 33
// bodies iff some window [j, j+32] containing i has common_octs(key[j], key[j+32]) >= L. Hence
// L_i = 1 + max over those windows (0 if n <= 32), and all bodies of one leaf get the same L.
// L_i = 43 means > 32 identical pairs; such runs are split into fixed chunks of 32.
// Without subkeys, any window of identical keys (a long run) sets *dup: the tree needs refining.
__global__ void window_octs(const uint64_t* __restrict__ keys, const uint64_t* __restrict__ sub,
                            uint8_t* __restrict__ D, int* __restrict__ dup, int n) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    bool eq = false;
    if (j + LEAF_MAX < n) {
        const int d = common_octs(keys[j], keys[j + LEAF_MAX], sub_at(sub, j), sub_at(sub, j + LEAF_MAX));
        D[j] = static_cast<uint8_t>(d);
        eq = !sub && d == 42;
    }
    // One atomic per warp at most: a long run would otherwise hit one address once per body.
    if (__ballot_sync(0xffffffffu, eq) && (threadIdx.x & 31) == 0) atomicOr(dup, 1);
}

__global__ void leaf_level(const uint8_t* __restrict__ D, uint8_t* __restrict__ L, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long lo = max(0ll, i - LEAF_MAX), hi = min(i, (long long)n - LEAF_MAX - 1);
    int best = -1;
    for (long long j = lo; j <= hi; ++j) best = max(best, (int)D[j]);
    L[i] = static_cast<uint8_t>(best + 1);
}

struct LeafId {
    int level;
    uint64_t prefix, sprefix;
    long long run;
};

__device__ LeafId leaf_id(const uint64_t* __restrict__ keys, const uint64_t* __restrict__ sub,
                          const uint8_t* __restrict__ L, long long i) {
    const int l = L[i];
    const uint64_t k = keys[i], s = sub_at(sub, i);
    LeafId r{l, 0, 0, 0};
    if (l <= 21) {
        r.prefix = k >> (63 - 3 * l);
    } else if (l <= 42) {
        r.prefix = k;
        r.sprefix = s >> (63 - 3 * (l - 21));
    } else {
        r.prefix = k;
        r.sprefix = s;
        long long a = 0, b = i;  // lower_bound of (k, s) in [0, i]
        while (a < b) {
            long long mid = a + (b - a) / 2;
            const uint64_t km = keys[mid];
            if (km < k || (km == k && sub_at(sub, mid) < s)) a = mid + 1; else b = mid;
        }
        r.run = (i - a) / LEAF_MAX;
    }
    return r;
}

__global__ void leaf_flags(const uint64_t* __restrict__ keys, const uint64_t* __restrict__ sub,
                           const uint8_t* __restrict__ L, uint32_t* __restrict__ flags, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t f = 1;
    if (i > 0) {
        LeafId a = leaf_id(keys, sub, L, i), b = leaf_id(keys, sub, L, i - 1);
        f = (a.level != b.level || a.prefix != b.prefix || a.sprefix != b.sprefix || a.run != b.run) ? 1u : 0u;
    }
    flags[i] = f;
}

// incl = inclusive scan of the flags. Writes start[leaf] and start[K] = n; K to *nleaves.
__global__ void leaf_starts(const uint32_t* __restrict__ incl, int* __restrict__ start, int* __restrict__ nleaves,
                            int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t c = incl[i];
    const uint32_t prev = i > 0 ? incl[i - 1] : 0u;
    if (c != prev) start[c - 1] = static_cast<int>(i);
    if (i == n - 1) {
        start[c] = n;
        *nleaves = static_cast<int>(c);
    }
}

// ---------------------------------------------------------------- tree: topology (Karras 2012)

// Node numbering: internal nodes 0..K-2 (root 0), leaf l is node K-1+l; root is node 0 either way.
// nodeC = (Qyz, rcrit^2, a, b) with a,b = children for internal nodes, or a = first body,
// b = ~count (< 0) for leaves.

// Common prefix length of the leaves' first (key, subkey, leaf index) as one 64+64+32-bit string
// (key and subkey have a zero top bit). Equal pairs only happen for chunks of one run.
__device__ __forceinline__ int delta(const uint64_t* __restrict__ keys, const uint64_t* __restrict__ sub,
                                     const int* __restrict__ start, int K, int i, long long j) {
    if (j < 0 || j >= K) return -1;
    const int si = start[i], sj = start[j];
    const uint64_t a = keys[si], b = keys[sj];
    if (a != b) return __clzll(static_cast<long long>(a ^ b));
    if (sub) {
        const uint64_t sa = sub[si], sb = sub[sj];
        if (sa != sb) return 64 + __clzll(static_cast<long long>(sa ^ sb));
    }
    return 128 + __clz(static_cast<unsigned>(i ^ static_cast<int>(j)));
}

__global__ void build_radix(const uint64_t* __restrict__ keys, const uint64_t* __restrict__ sub,
                            const int* __restrict__ start, int K, float4* __restrict__ nodeC,
                            int* __restrict__ parent, int2* __restrict__ range) {
    long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= K - 1) return;
    const int i = static_cast<int>(gi);
    const int d = delta(keys, sub, start, K, i, i + 1) - delta(keys, sub, start, K, i, (long long)i - 1) >= 0 ? 1 : -1;
    const int dmin = delta(keys, sub, start, K, i, (long long)i - d);
    long long lmax = 2;
    while (delta(keys, sub, start, K, i, i + lmax * d) > dmin) lmax <<= 1;
    long long l = 0;
    for (long long t = lmax >> 1; t >= 1; t >>= 1)
        if (delta(keys, sub, start, K, i, i + (l + t) * d) > dmin) l += t;
    const long long j = i + l * d;
    const int dnode = delta(keys, sub, start, K, i, j);
    long long s = 0, t = l;
    do {
        t = (t + 1) >> 1;
        if (delta(keys, sub, start, K, i, i + (s + t) * d) > dnode) s += t;
    } while (t > 1);
    const long long gamma = i + s * d + min(d, 0);
    const int left = (min((long long)i, j) == gamma) ? static_cast<int>(K - 1 + gamma) : static_cast<int>(gamma);
    const int right = (max((long long)i, j) == gamma + 1) ? static_cast<int>(K + gamma) : static_cast<int>(gamma + 1);
    nodeC[i] = make_float4(0.f, 0.f, __int_as_float(left), __int_as_float(right));
    parent[left] = i;
    parent[right] = i;
    const int b0 = start[min((long long)i, j)], b1 = start[max((long long)i, j) + 1];
    range[i] = make_int2(b0, b1 - b0);
}

// ---------------------------------------------------------------- tree: moments (bottom-up)

struct Moments {
    float4 com;    // xyz, mass
    float4 qa;     // Qxx, Qyy, Qxy, Qxz   (traceless: Qzz = -Qxx - Qyy), about com
    float qyz;
    float3 lo, hi;
};

constexpr float RCRIT_MAX = 1.8e19f;  // RCRIT_MAX^2 < FLT_MAX

__device__ __forceinline__ float rcrit2(const Moments& M, float inv_theta) {
    // theta = 0 arrives as inv_theta = RCRIT_MAX and means "always open". It must not go through
    // l * inv_theta: a cell of coincident bodies has l = 0, which would give rcrit ~ 0 and let the
    // walk accept it as a far field. RCRIT_MAX^2 is finite (no inf under finite-math) and larger
    // than any d^2 of a sane system.
    if (inv_theta >= RCRIT_MAX) return RCRIT_MAX * RCRIT_MAX;
    const float sx = M.hi.x - M.lo.x, sy = M.hi.y - M.lo.y, sz = M.hi.z - M.lo.z;
    const float l = fmaxf(sx, fmaxf(sy, sz));
    const float cx = 0.5f * (M.lo.x + M.hi.x) - M.com.x, cy = 0.5f * (M.lo.y + M.hi.y) - M.com.y,
                cz = 0.5f * (M.lo.z + M.hi.z) - M.com.z;
    // Clamped for tiny theta: keeps r^2 finite while still "always open".
    const float a = l < RCRIT_MAX / inv_theta ? l * inv_theta : RCRIT_MAX;
    const float r = fminf(a + sqrtf(cx * cx + cy * cy + cz * cz), RCRIT_MAX);
    return r * r;
}

// Adds m (3 s s^T - |s|^2 I) for a point mass offset s from the expansion centre.
__device__ __forceinline__ void add_quad(float m, float sx, float sy, float sz, float4& qa, float& qyz) {
    const float s2 = sx * sx + sy * sy + sz * sz;
    qa.x += m * (3.f * sx * sx - s2);
    qa.y += m * (3.f * sy * sy - s2);
    qa.z += m * 3.f * sx * sy;
    qa.w += m * 3.f * sx * sz;
    qyz += m * 3.f * sy * sz;
}

__device__ void store_node(int node, const Moments& M, float inv_theta, int a, int b, float4* __restrict__ nodeA,
                           float4* __restrict__ nodeB, float4* __restrict__ nodeC, float4* __restrict__ bmin,
                           float4* __restrict__ bmax) {
    nodeA[node] = M.com;
    nodeB[node] = M.qa;
    nodeC[node] = make_float4(M.qyz, rcrit2(M, inv_theta), __int_as_float(a), __int_as_float(b));
    bmin[node] = make_float4(M.lo.x, M.lo.y, M.lo.z, 0.f);
    bmax[node] = make_float4(M.hi.x, M.hi.y, M.hi.z, 0.f);
}

// clang 18 has no generic __ldcg; inline PTX works with both compilers. The memory clobber keeps
// the load after the preceding __threadfence/atomic.
__device__ __forceinline__ float4 load_cg(const float4* p) {
    float4 r;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
                 : "l"(p)
                 : "memory");
    return r;
}

// One thread per leaf computes the leaf, then climbs: the first child to finish a parent stops,
// the second (atomic counter == 1) combines both children. Child data is published with a
// __threadfence before the atomic and read through L2 (ld.global.cg) after it, since L1 is not coherent.
__global__ void __launch_bounds__(BS) moments_kernel(const float4* __restrict__ pm, const int* __restrict__ start,
                                                     int K, float inv_theta, const int* __restrict__ parent,
                                                     int* __restrict__ counter, float4* nodeA, float4* nodeB,
                                                     float4* nodeC, float4* bmin, float4* bmax,
                                                     int2* __restrict__ range) {
    long long gl = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gl >= K) return;
    const int leaf = static_cast<int>(gl);
    const int b0 = start[leaf], b1 = start[leaf + 1];
    Moments M{};
    M.lo = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    M.hi = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float mx = 0, my = 0, mz = 0, mt = 0;
    for (int j = b0; j < b1; ++j) {
        const float4 q = pm[j];
        mx = fmaf(q.w, q.x, mx); my = fmaf(q.w, q.y, my); mz = fmaf(q.w, q.z, mz); mt += q.w;
        M.lo = make_float3(fminf(M.lo.x, q.x), fminf(M.lo.y, q.y), fminf(M.lo.z, q.z));
        M.hi = make_float3(fmaxf(M.hi.x, q.x), fmaxf(M.hi.y, q.y), fmaxf(M.hi.z, q.z));
    }
    // Massless (test-particle) cells: put the expansion centre at the box centre.
    if (mt > 0.f) M.com = make_float4(mx / mt, my / mt, mz / mt, mt);
    else M.com = make_float4(0.5f * (M.lo.x + M.hi.x), 0.5f * (M.lo.y + M.hi.y), 0.5f * (M.lo.z + M.hi.z), 0.f);
    for (int j = b0; j < b1; ++j) {
        const float4 q = pm[j];
        add_quad(q.w, q.x - M.com.x, q.y - M.com.y, q.z - M.com.z, M.qa, M.qyz);
    }
    int node = K - 1 + leaf;
    range[node] = make_int2(b0, b1 - b0);
    store_node(node, M, inv_theta, b0, ~(b1 - b0), nodeA, nodeB, nodeC, bmin, bmax);

    while (true) {
        const int p = parent[node];
        if (p < 0) break;
        __threadfence();
        if (atomicAdd(&counter[p], 1) == 0) break;
        __threadfence();
        const float4 cp = load_cg(&nodeC[p]);
        const int c[2] = {__float_as_int(cp.z), __float_as_int(cp.w)};
        Moments ch[2];
        for (int k = 0; k < 2; ++k) {
            ch[k].com = load_cg(&nodeA[c[k]]);
            ch[k].qa = load_cg(&nodeB[c[k]]);
            ch[k].qyz = load_cg(&nodeC[c[k]]).x;
            const float4 lo = load_cg(&bmin[c[k]]), hi = load_cg(&bmax[c[k]]);
            ch[k].lo = make_float3(lo.x, lo.y, lo.z);
            ch[k].hi = make_float3(hi.x, hi.y, hi.z);
        }
        Moments P{};
        P.lo = make_float3(fminf(ch[0].lo.x, ch[1].lo.x), fminf(ch[0].lo.y, ch[1].lo.y), fminf(ch[0].lo.z, ch[1].lo.z));
        P.hi = make_float3(fmaxf(ch[0].hi.x, ch[1].hi.x), fmaxf(ch[0].hi.y, ch[1].hi.y), fmaxf(ch[0].hi.z, ch[1].hi.z));
        const float m = ch[0].com.w + ch[1].com.w;
        if (m > 0.f)
            P.com = make_float4((ch[0].com.w * ch[0].com.x + ch[1].com.w * ch[1].com.x) / m,
                                (ch[0].com.w * ch[0].com.y + ch[1].com.w * ch[1].com.y) / m,
                                (ch[0].com.w * ch[0].com.z + ch[1].com.w * ch[1].com.z) / m, m);
        else
            P.com = make_float4(0.5f * (P.lo.x + P.hi.x), 0.5f * (P.lo.y + P.hi.y), 0.5f * (P.lo.z + P.hi.z), 0.f);
        for (int k = 0; k < 2; ++k) {
            P.qa.x += ch[k].qa.x; P.qa.y += ch[k].qa.y; P.qa.z += ch[k].qa.z; P.qa.w += ch[k].qa.w;
            P.qyz += ch[k].qyz;
            add_quad(ch[k].com.w, ch[k].com.x - P.com.x, ch[k].com.y - P.com.y, ch[k].com.z - P.com.z, P.qa, P.qyz);
        }
        store_node(p, P, inv_theta, c[0], c[1], nodeA, nodeB, nodeC, bmin, bmax);
        node = p;
    }
}

// ---------------------------------------------------------------- tree: walk

// Far field about the com, d = com - x, s^2 = d^2 + eps^2:
//   a = G [ M d/s^3 - Q d/s^5 + 5/2 (d.Q.d) d/s^7 ],  phi = -G [ M/s + 1/2 d.Q.d/s^5 ]
// A = (com, M), B = (Qxx, Qyy, Qxy, Qxz), qyz; Qzz = -(Qxx + Qyy).
template <bool POT>
__device__ __forceinline__ void cell_term(const float4 p, const float4 A, const float4 B, const float qyz, float eps2,
                                          float& sx, float& sy, float& sz, float& sp) {
    const float dx = A.x - p.x, dy = A.y - p.y, dz = A.z - p.z;
    const float d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    const float qzz = -(B.x + B.y);
    const float qdx = B.x * dx + B.z * dy + B.w * dz;
    const float qdy = B.z * dx + B.y * dy + qyz * dz;
    const float qdz = B.w * dx + qyz * dy + qzz * dz;
    const float dqd = dx * qdx + dy * qdy + dz * qdz;
    const float inv = rsqrtf(d2 + eps2);
    const float inv2 = inv * inv;
    const float inv3 = inv * inv2, inv5 = inv3 * inv2, inv7 = inv5 * inv2;
    const float rad = A.w * inv3 + 2.5f * dqd * inv7;
    sx += rad * dx - qdx * inv5;
    sy += rad * dy - qdy * inv5;
    sz += rad * dz - qdz * inv5;
    if (POT) sp += A.w * inv + 0.5f * dqd * inv5;
}

// Per-body walk (LETTUCE_CUDA_WALK=body), kept for comparison with the group walk.
// One thread per body in Morton order, so a warp's bodies are close and follow similar paths.
// Accept a node iff |com - x|^2 > rcrit^2 with rcrit = l/theta + |com - box centre|, and the
// node does not hold body i. For theta < 2/sqrt(3) the distance test alone rejects a body's own
// cell, except for a cell of coincident bodies: l = 0 makes it the equality |com - x| = |com -
// centre| (x = centre), decided by rounding, and accepting it adds the self pair (phi += m/eps).
// The range is loaded only for nodes that pass the distance test.
template <bool POT>
__global__ void __launch_bounds__(WALK_BS) walk_kernel(const float4* __restrict__ pm, int n,
                                                       const float4* __restrict__ nodeA,
                                                       const float4* __restrict__ nodeB,
                                                       const float4* __restrict__ nodeC,
                                                       const int2* __restrict__ R, float eps2, float G,
                                                       float* __restrict__ ax, float* __restrict__ ay,
                                                       float* __restrict__ az, float* __restrict__ pot,
                                                       uint32_t* __restrict__ counts,
                                                       unsigned long long* __restrict__ interactions,
                                                       int* __restrict__ error) {
    const long long gi = (long long)blockIdx.x * WALK_BS + threadIdx.x;
    unsigned long long count = 0;
    if (gi < n) {
        const int i = static_cast<int>(gi);
        const float4 p = pm[i];
        float sx = 0, sy = 0, sz = 0, sp = 0;
        int stack[STACK];
        int sp_top = 0;
        int node = 0;
        while (true) {
            const float4 A = __ldg(&nodeA[node]);
            const float4 C = __ldg(&nodeC[node]);
            const float dx = A.x - p.x, dy = A.y - p.y, dz = A.z - p.z;
            const float d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
            const int cb = __float_as_int(C.w);
            bool descend = false;
            bool acc = d2 > C.y;
            if (acc) {
                const int2 r = __ldg(&R[node]);
                acc = i < r.x || i >= r.x + r.y;
            }
            if (acc) {
                cell_term<POT>(p, A, __ldg(&nodeB[node]), C.x, eps2, sx, sy, sz, sp);
                ++count;
            } else if (cb < 0) {
                const int b0 = __float_as_int(C.z), cnt = ~cb;
                for (int j = b0; j < b0 + cnt; ++j)
                    if (j != i) pp<POT>(p, __ldg(&pm[j]), eps2, sx, sy, sz, sp);
                count += static_cast<unsigned long long>(cnt) - ((i >= b0 && i < b0 + cnt) ? 1u : 0u);
            } else {
                descend = true;
            }
            if (descend) {
                if (sp_top >= STACK) {
                    atomicExch(error, 1);
                    break;
                }
                stack[sp_top++] = cb;
                node = __float_as_int(C.z);
            } else {
                if (sp_top == 0) break;
                node = stack[--sp_top];
            }
        }
        ax[i] = G * sx;
        ay[i] = G * sy;
        az[i] = G * sz;
        if (POT) pot[i] = -G * sp;
        counts[i] = static_cast<uint32_t>(count);
    }
    // All 32 lanes reach this: WALK_BS is a multiple of 32 and no thread returns early.
    count = warp_sum(count);
    if ((threadIdx.x & 31) == 0 && count) atomicAdd(interactions, count);
}

// ---------------------------------------------------------------- tree: group walk
//
// One warp per group: up to 32*NB consecutive sorted bodies, NB per lane (slots lane, lane+32).
// Local groups are the topmost tree nodes holding <= 32*NB bodies (leaves hold <= 32, so every
// leaf is inside exactly one group). The warp shares one traversal against the group's bounding
// box: a node is accepted only if its com is farther than rcrit from the whole box, so each body
// gets the interactions its own walk would have had, or finer ones (never coarser).
//
// Traversal: a per-warp stack in shared memory. Each step pops up to 32 nodes, one per lane; the
// lanes test them in parallel and ballots compact the outcomes: accepted nodes go to the cell
// list (moments copied into shared memory), opened leaves to the body list (positions copied),
// opened internal nodes push their children. A list is evaluated (every lane, for each of its
// bodies, over the whole list from shared memory) when the next append would overflow it, and at
// the end. All shared-memory hand-offs between lanes are separated by __syncwarp.
//
// Stack bound: a step popping P nodes grows the stack by at most P*(cmax-1) (cmax = most
// children of a node), so P is limited to keep the stack <= thr; above thr the warp pops one node
// per step, a plain depth-first walk, which grows the stack by at most (cmax-1)*height in total.
// With thr = GSTACK - (cmax-1)*height - 1 the stack therefore never overflows (the overflow check
// below is a guard against a wrong height bound, not an expected path).
//
// The same kernel walks remote LETs (LET = true): nodes use the LET encoding below, terminal
// nodes are always accepted, the acceptance test is >= as in backend.hpp, and results are added
// to the local ones.
constexpr int GW_WARPS = 4;
constexpr int GW_BS = 32 * GW_WARPS;
constexpr int GSTACK = 512;
constexpr int CELL_CAP = 64;
constexpr int BODY_CAP = 64;
constexpr unsigned FULL_MASK = 0xffffffffu;

struct GroupSmem {
    float4 cA[CELL_CAP];
    float4 cB[CELL_CAP];
    float cQ[CELL_CAP];
    float4 bP[BODY_CAP];
    int bI[BODY_CAP];
    int lf[32], lc[32];
    int stack[GSTACK];
};

// Local tree: nodeC = (Qyz, rcrit2, a, b) as built above, R = body range per node.
// LET: nodeC = (Qyz, rcrit2, first, code), code > 0: internal with `code` contiguous children at
// `first`; code < 0: leaf with ~code bodies at `first` in `body`; code == 0: terminal.
struct WalkSrc {
    const float4* A;
    const float4* B;
    const float4* C;
    const int2* R;         // local only
    const float4* body;    // what leaf body indices refer to
    const int* roots;      // LET only: up to 32 root nodes
    int nroots;
    int thr;
    int cmax_m1;
};

struct WalkGroups {
    const int* node;   // group -> tree node (its body range is R[node]); null: fixed chunks of gmax
    const int2* R;
    const int* ndev;   // group count on the device; null: nhost
    int nhost;
    int gmax;
    int n;
    int* next;         // work counter, zeroed before each launch
};

struct WalkOut {
    float *ax, *ay, *az, *pot;
    uint32_t* counts;
    unsigned long long* inter;
    int* error;
};

template <int NB, bool POT>
__device__ __forceinline__ void eval_cells(const GroupSmem& S, int ncell, int nslot, const float4 (&p)[NB],
                                           float eps2, float (&sx)[NB], float (&sy)[NB], float (&sz)[NB],
                                           float (&sp)[NB], uint32_t (&cnt)[NB]) {
    __syncwarp();
    if (NB == 2 && nslot == 2) {
        for (int k = 0; k < ncell; ++k) {
            const float4 A = S.cA[k], B = S.cB[k];
            const float q = S.cQ[k];
            cell_term<POT>(p[0], A, B, q, eps2, sx[0], sy[0], sz[0], sp[0]);
            cell_term<POT>(p[NB - 1], A, B, q, eps2, sx[NB - 1], sy[NB - 1], sz[NB - 1], sp[NB - 1]);
        }
        cnt[NB - 1] += ncell;
    } else {
        for (int k = 0; k < ncell; ++k) cell_term<POT>(p[0], S.cA[k], S.cB[k], S.cQ[k], eps2, sx[0], sy[0], sz[0], sp[0]);
    }
    cnt[0] += ncell;
    __syncwarp();
}

template <int NB, bool POT>
__device__ __forceinline__ void eval_bodies(const GroupSmem& S, int nbody, int nslot, const float4 (&p)[NB],
                                            const int (&idx)[NB], float eps2, float (&sx)[NB], float (&sy)[NB],
                                            float (&sz)[NB], float (&sp)[NB], uint32_t (&cnt)[NB]) {
    __syncwarp();
    // Self pair skipped by index: subtracting it afterwards would cancel a term that can exceed
    // the whole potential, and is infinite for eps = 0.
    if (NB == 2 && nslot == 2) {
        for (int k = 0; k < nbody; ++k) {
            const float4 q = S.bP[k];
            const int j = S.bI[k];
            if (j != idx[0]) { pp<POT>(p[0], q, eps2, sx[0], sy[0], sz[0], sp[0]); ++cnt[0]; }
            if (j != idx[NB - 1]) {
                pp<POT>(p[NB - 1], q, eps2, sx[NB - 1], sy[NB - 1], sz[NB - 1], sp[NB - 1]);
                ++cnt[NB - 1];
            }
        }
    } else {
        for (int k = 0; k < nbody; ++k) {
            const int j = S.bI[k];
            if (j != idx[0]) { pp<POT>(p[0], S.bP[k], eps2, sx[0], sy[0], sz[0], sp[0]); ++cnt[0]; }
        }
    }
    __syncwarp();
}

template <int NB, bool POT, bool LET>
__global__ void __launch_bounds__(GW_BS) group_walk_kernel(const float4* __restrict__ pm, const WalkSrc t,
                                                           const WalkGroups gr, float eps2, float G, const WalkOut o) {
    __shared__ GroupSmem smem[GW_WARPS];
    GroupSmem& S = smem[threadIdx.x >> 5];
    const int lane = threadIdx.x & 31;
    const unsigned lt = (1u << lane) - 1u;
    const int ngroups = gr.ndev ? *gr.ndev : gr.nhost;
    unsigned long long total = 0;
    while (true) {
        int g = 0;
        if (lane == 0) g = atomicAdd(gr.next, 1);
        g = __shfl_sync(FULL_MASK, g, 0);
        if (g >= ngroups) break;
        int g0, gn;
        if (gr.node) {
            const int2 r = gr.R[gr.node[g]];
            g0 = r.x;
            gn = r.y;
        } else {
            g0 = g * gr.gmax;
            gn = min(gr.gmax, gr.n - g0);
        }
        const int nslot = min(NB, (gn + 31) >> 5);  // warp-uniform
        float4 p[NB];
        int idx[NB];
        bool act[NB];
        float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int s = 0; s < NB; ++s) {
            const int k = s * 32 + lane;
            act[s] = k < gn;
            // Idle slots shadow the group's first body: inside the box, results never stored.
            idx[s] = act[s] ? g0 + k : g0;
            p[s] = pm[idx[s]];
            lo[0] = fminf(lo[0], p[s].x); lo[1] = fminf(lo[1], p[s].y); lo[2] = fminf(lo[2], p[s].z);
            hi[0] = fmaxf(hi[0], p[s].x); hi[1] = fmaxf(hi[1], p[s].y); hi[2] = fmaxf(hi[2], p[s].z);
        }
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1)
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                lo[c] = fminf(lo[c], __shfl_xor_sync(FULL_MASK, lo[c], o2));
                hi[c] = fmaxf(hi[c], __shfl_xor_sync(FULL_MASK, hi[c], o2));
            }
        float sx[NB], sy[NB], sz[NB], sp[NB];
        uint32_t cnt[NB];
#pragma unroll
        for (int s = 0; s < NB; ++s) { sx[s] = sy[s] = sz[s] = sp[s] = 0.f; cnt[s] = 0; }

        __syncwarp();  // the previous group's last reads of S are done
        int top;
        if (LET) {
            if (lane < t.nroots) S.stack[lane] = t.roots[lane];
            top = t.nroots;
        } else {
            if (lane == 0) S.stack[0] = 0;
            top = 1;
        }
        int ncell = 0, nbody = 0;
        bool ovf = false;
        while (top > 0) {
            const int room = (t.thr - top) / t.cmax_m1;
            const int P = min(32, min(top, max(1, room)));
            __syncwarp();
            const bool active = lane < P;
            const int nd = active ? S.stack[top - P + lane] : 0;
            __syncwarp();  // the window is read before children overwrite it
            top -= P;
            float4 A = make_float4(0.f, 0.f, 0.f, 0.f), C = A;
            bool acc = false, leaf = false;
            int code = 0;
            if (active) {
                A = __ldg(&t.A[nd]);
                C = __ldg(&t.C[nd]);
                const float gx = fmaxf(fmaxf(lo[0] - A.x, A.x - hi[0]), 0.f);
                const float gy = fmaxf(fmaxf(lo[1] - A.y, A.y - hi[1]), 0.f);
                const float gz = fmaxf(fmaxf(lo[2] - A.z, A.z - hi[2]), 0.f);
                const float d2 = fmaf(gx, gx, fmaf(gy, gy, gz * gz));
                code = __float_as_int(C.w);
                if (LET) {
                    acc = code == 0 || d2 >= C.y;
                    leaf = code < 0;
                } else {
                    // A node sharing bodies with the group is always opened. Groups are tree
                    // nodes, so that is a node containing the group (for theta >= 2/sqrt(3) its
                    // multipole could otherwise include the group's own bodies) or one inside it.
                    // The latter's com is in the group box only up to rounding: for coincident
                    // bodies (l = 0, rcrit ~ 0) a com rounded just outside passes the box test
                    // and the group's own bodies, self pairs included, would enter as a cell.
                    const int2 r = __ldg(&t.R[nd]);
                    const bool shares = r.x < g0 + gn && g0 < r.x + r.y;
                    acc = !shares && d2 > C.y;
                    leaf = code < 0;
                }
            }
            const unsigned m_acc = __ballot_sync(FULL_MASK, active && acc);
            const bool open_leaf = active && !acc && leaf;
            const unsigned m_leaf = __ballot_sync(FULL_MASK, open_leaf);
            const bool open_int = active && !acc && !leaf;

            const int nch = open_int ? (LET ? code : 2) : 0;
            int incl = nch;
#pragma unroll
            for (int o2 = 1; o2 < 32; o2 <<= 1) {
                const int v = __shfl_up_sync(FULL_MASK, incl, o2);
                if (lane >= o2) incl += v;
            }
            const int pushed = __shfl_sync(FULL_MASK, incl, 31);
            if (top + pushed > GSTACK) {
                ovf = true;
                break;
            }
            if (open_int) {
                const int base = top + incl - nch, first = __float_as_int(C.z);
                if (LET) {
                    for (int c = 0; c < nch; ++c) S.stack[base + c] = first + c;
                } else {
                    S.stack[base] = first;
                    S.stack[base + 1] = code;
                }
            }
            top += pushed;

            const int na = __popc(m_acc);
            if (na) {
                if (ncell + na > CELL_CAP) {
                    eval_cells<NB, POT>(S, ncell, nslot, p, eps2, sx, sy, sz, sp, cnt);
                    ncell = 0;
                }
                if (active && acc) {
                    const int k = ncell + __popc(m_acc & lt);
                    S.cA[k] = A;
                    S.cB[k] = __ldg(&t.B[nd]);
                    S.cQ[k] = C.x;
                }
                ncell += na;
            }
            const int nl = __popc(m_leaf);
            if (nl) {
                if (open_leaf) {
                    const int k = __popc(m_leaf & lt);
                    S.lf[k] = __float_as_int(C.z);
                    S.lc[k] = ~code;
                }
                __syncwarp();
                for (int r = 0; r < nl; ++r) {
                    const int f = S.lf[r], c = S.lc[r];
                    for (int off = 0; off < c; off += 32) {
                        const int k = min(32, c - off);
                        if (nbody + k > BODY_CAP) {
                            eval_bodies<NB, POT>(S, nbody, nslot, p, idx, eps2, sx, sy, sz, sp, cnt);
                            nbody = 0;
                        }
                        if (lane < k) {
                            S.bP[nbody + lane] = __ldg(&t.body[f + off + lane]);
                            S.bI[nbody + lane] = LET ? -1 : f + off + lane;
                        }
                        nbody += k;
                    }
                }
            }
        }
        if (ovf && lane == 0) atomicExch(o.error, 1);
        if (ncell) eval_cells<NB, POT>(S, ncell, nslot, p, eps2, sx, sy, sz, sp, cnt);
        if (nbody) eval_bodies<NB, POT>(S, nbody, nslot, p, idx, eps2, sx, sy, sz, sp, cnt);
#pragma unroll
        for (int s = 0; s < NB; ++s) {
            if (!act[s]) continue;
            const int i = idx[s];
            if (LET) {
                o.ax[i] += G * sx[s];
                o.ay[i] += G * sy[s];
                o.az[i] += G * sz[s];
                if (POT) o.pot[i] -= G * sp[s];
                o.counts[i] += cnt[s];
            } else {
                o.ax[i] = G * sx[s];
                o.ay[i] = G * sy[s];
                o.az[i] = G * sz[s];
                if (POT) o.pot[i] = -G * sp[s];
                o.counts[i] = cnt[s];
            }
            total += cnt[s];
        }
    }
    total = warp_sum(total);
    if (lane == 0 && total) atomicAdd(o.inter, total);
}

// Groups for the local tree: node v is a group iff it holds <= gmax bodies and its parent more.
// slot[first body] = v; slots of non-first bodies keep -1, and a select compacts them in body order.
__global__ void mark_groups(const int2* __restrict__ R, const int* __restrict__ parent, int nodes, int gmax,
                            int* __restrict__ slot) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nodes) return;
    const int2 r = R[v];
    if (r.y > gmax) return;
    if (v != 0 && R[parent[v]].y <= gmax) return;
    slot[r.x] = static_cast<int>(v);
}

struct NonNegative {
    __host__ __device__ bool operator()(int v) const { return v >= 0; }
};

__global__ void fill_u32(uint32_t* __restrict__ p, uint32_t v, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void gather_float4(const float4* __restrict__ src, const int* __restrict__ idx, float4* __restrict__ dst,
                              int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// ---------------------------------------------------------------- backend

class CudaBackend final : public Backend {
public:
    explicit CudaBackend(int device) : dev_(device) {
        CK(cudaSetDevice(dev_));
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, dev_));
        // Fail here, with a clear message, if this build has no code for the device's architecture.
        cudaFuncAttributes fa{};
        cudaError_t e = cudaFuncGetAttributes(&fa, kick_kernel);
        if (e != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error(std::string("device ") + prop.name + " (sm_" + std::to_string(prop.major) +
                                     std::to_string(prop.minor) + ") has no kernel image in this build (" +
                                     cudaGetErrorString(e) + "); set CMAKE_CUDA_ARCHITECTURES");
        }
        char buf[512];
        std::snprintf(buf, sizeof buf, "cuda: %s (sm_%d%d, %d SMs, %.0f GB)", prop.name, prop.major, prop.minor,
                      prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
        name_ = buf;
        stream_ = std::make_unique<Stream>();
        ev_ = std::make_unique<Event[]>(NEV);
        cube_.reserve(sizeof(float4));
        bbox_part_.reserve(sizeof(float) * 6 * REDUCE_BLOCKS);
        diag_part_.reserve(sizeof(double) * DIAG_N * REDUCE_BLOCKS);
        small_.reserve(128);
        sms_ = prop.multiProcessorCount;
        // LETTUCE_CUDA_WALK=body|group (default group) picks the local tree walk: one traversal per
        // body, or one per group of consecutive sorted bodies. LETTUCE_CUDA_GROUP=64|32 (default 64)
        // is the group size: 64 puts two bodies on each lane. Remote LETs always use the group walk.
        if (const char* w = std::getenv("LETTUCE_CUDA_WALK")) {
            const std::string v = w;
            if (v != "body" && v != "group")
                throw std::runtime_error("LETTUCE_CUDA_WALK must be body or group, not '" + v + "'");
            walk_body_ = v == "body";
        }
        if (const char* g = std::getenv("LETTUCE_CUDA_GROUP")) {
            const std::string v = g;
            if (v != "32" && v != "64") throw std::runtime_error("LETTUCE_CUDA_GROUP must be 32 or 64, not '" + v + "'");
            nb_ = v == "32" ? 1 : 2;
        }
        gmax_ = 32 * nb_;
        name_ += walk_body_ ? ", per-body walk" : (nb_ == 2 ? ", group walk 64" : ", group walk 32");
    }

    std::string name() const override { return name_; }

    // Mirrors reserve_bodies (92 B/body) and build_tree_impl + make_groups (92 B per node, 2K-1
    // nodes, plus 8 B per leaf), with K = n/8 leaves: lettuce's leaf rule replayed on the host gives
    // n/K = 9.2-9.6 for plummer, uniform and disk at 1e6-1e8. Host: export_let's node copy (48 B/node).
    MemoryNeed memory_need(uint64_t n, ForceMethod m) const override {
        MemoryNeed r;
        r.device = 92 * n + (uint64_t(256) << 20);  // + CUB temporary storage and the small buffers
        if (m == ForceMethod::tree) {
            const uint64_t K = n / 8 + 1, nodes = 2 * K - 1;
            r.device += 92 * nodes + 8 * K;
            r.host = 48 * nodes;
        }
        return r;
    }

    bool device_memory(uint64_t& free_bytes, uint64_t& total_bytes) override {
        CK(cudaSetDevice(dev_));
        std::size_t f = 0, t = 0;
        CK(cudaMemGetInfo(&f, &t));
        // What this backend already holds counts as free: a grow-only buffer that must grow is freed
        // before it is reallocated, and the estimate covers every buffer again.
        std::size_t held = 0;
        for (const DevBuf* b : {&x_, &y_, &z_, &vx_, &vy_, &vz_, &m_, &id_, &ax_, &ay_, &az_, &pot_, &posm_, &keys_[0],
                                &keys_[1], &idx_[0], &idx_[1], &nodeA_, &nodeB_, &nodeC_, &bmin_, &bmax_, &parent_,
                                &counter_, &nodeR_, &groups_, &counts_, &temp_})
            held += b->capacity();
        free_bytes = f + held;
        total_bytes = t;
        return true;
    }

    void upload(const Bodies& b) override {
        CK(cudaSetDevice(dev_));
        const std::size_t n = b.size();
        if (n >= static_cast<std::size_t>(INT32_MAX))
            throw std::runtime_error("cuda backend: n = " + std::to_string(n) + " exceeds the 2^31-1 body limit");
        reserve_bodies(n);
        n_ = static_cast<int>(n);
        has_pe_ = false;
        acc_ok_ = true;  // zeroed below
        kick_rec_ = drift_rec_ = false;
        tree_ok_ = host_nodes_ok_ = costs_ok_ = false;
        t_ = PhaseTimes{};
        if (n == 0) return;
        const std::size_t fb = n * sizeof(float);
        cudaStream_t s = stream_->s;
        CK(cudaMemcpyAsync(x_.as<float>(), b.x.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(y_.as<float>(), b.y.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(z_.as<float>(), b.z.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(vx_.as<float>(), b.vx.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(vy_.as<float>(), b.vy.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(vz_.as<float>(), b.vz.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(m_.as<float>(), b.m.data(), fb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(id_.as<uint32_t>(), b.id.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
        // A kick before the first compute_forces then adds zero rather than garbage.
        CK(cudaMemsetAsync(ax_.as<float>(), 0, fb, s));
        CK(cudaMemsetAsync(ay_.as<float>(), 0, fb, s));
        CK(cudaMemsetAsync(az_.as<float>(), 0, fb, s));
        CK(cudaMemsetAsync(pot_.as<float>(), 0, fb, s));
        CK(cudaStreamSynchronize(s));
    }

    void download(Bodies& b) override {
        CK(cudaSetDevice(dev_));
        const std::size_t n = static_cast<std::size_t>(n_);
        b.resize(n);
        if (n == 0) return;
        const std::size_t fb = n * sizeof(float);
        cudaStream_t s = stream_->s;
        CK(cudaMemcpyAsync(b.x.data(), x_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.y.data(), y_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.z.data(), z_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.vx.data(), vx_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.vy.data(), vy_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.vz.data(), vz_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.m.data(), m_.as<float>(), fb, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b.id.data(), id_.as<uint32_t>(), n * sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
    }

    // compute_forces = build_tree + compute_forces_with(no remote): one code path.
    void compute_forces(const ForceParams& p, bool potential) override {
        build_tree(p);
        compute_forces_with(p, potential, {});
    }

    Box3 local_bounds() override {
        CK(cudaSetDevice(dev_));
        Box3 b;
        if (n_ == 0) return b;
        cudaStream_t s = stream_->s;
        const int pb = static_cast<int>(std::min<long long>(grid_for(n_, BS), REDUCE_BLOCKS));
        bbox_partial<<<pb, BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), n_, bbox_part_.as<float>());
        CK_LAUNCH();
        std::vector<float> part(static_cast<std::size_t>(pb) * 6);
        CK(cudaMemcpyAsync(part.data(), bbox_part_.as<float>(), part.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        for (int c = 0; c < 3; ++c) { b.lo[c] = FLT_MAX; b.hi[c] = -FLT_MAX; }
        for (int k = 0; k < pb; ++k)
            for (int c = 0; c < 3; ++c) {
                b.lo[c] = std::min(b.lo[c], part[static_cast<std::size_t>(k) * 6 + c]);
                b.hi[c] = std::max(b.hi[c], part[static_cast<std::size_t>(k) * 6 + 3 + c]);
            }
        return b;
    }

    // Tree: sort, leaves, topology, moments and walk groups. Host-synchronises once (leaf count).
    void build_tree(const ForceParams& p) override {
        CK(cudaSetDevice(dev_));
        tree_ok_ = host_nodes_ok_ = false;
        t_.sort_ms = t_.build_ms = 0;
        if (p.method != ForceMethod::tree || n_ == 0) return;
        cudaStream_t s = stream_->s;
        // The sort reorders the bodies but not ax/ay/az/pot or the counts: all stale until the next
        // compute_forces_with (kick refuses to use them, diagnostics drops the pe).
        costs_ok_ = false;
        has_pe_ = false;
        acc_ok_ = false;
        CK(cudaEventRecord(ev_[0].e, s));
        sort_bodies();
        CK(cudaEventRecord(ev_[1].e, s));
        K_ = build_tree_impl(p.theta);
        make_groups();
        CK(cudaEventRecord(ev_[2].e, s));
        tree_ok_ = true;
    }

    // Synchronises at the end, so the PhaseTimes are final when this returns and walk errors are
    // reported here rather than at some later call.
    void compute_forces_with(const ForceParams& p, bool potential, const std::vector<const LetTree*>& remote) override {
        CK(cudaSetDevice(dev_));
        t_.walk_ms = t_.force_ms = 0;
        t_.interactions = 0;
        costs_ok_ = false;
        if (n_ == 0) {
            t_.sort_ms = t_.build_ms = 0;
            has_pe_ = potential;
            acc_ok_ = true;
            return;
        }
        const bool tree = p.method == ForceMethod::tree;
        if (tree && !tree_ok_)
            throw std::logic_error("cuda compute_forces_with: call build_tree after the last upload/drift first");
        cudaStream_t s = stream_->s;
        const int n = n_;
        const float G = static_cast<float>(p.G), eps2 = static_cast<float>(p.eps * p.eps);
        int* err = small_.as<int>();
        unsigned long long* inter = reinterpret_cast<unsigned long long*>(small_.as<char>() + 16);
        CK(cudaMemsetAsync(small_.as<char>(), 0, 24, s));
        const WalkOut out{ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>(), counts_.as<uint32_t>(),
                          inter, err};
        CK(cudaEventRecord(ev_[8].e, s));
        if (!tree) {
            pack_posm<<<grid_for(n, BS), BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), m_.as<float>(),
                                                     posm_.as<float4>(), n);
            CK_LAUNCH();
            if (potential)
                direct_kernel<true><<<grid_for(n, DIRECT_BS), DIRECT_BS, 0, s>>>(
                    posm_.as<float4>(), n, eps2, G, ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>());
            else
                direct_kernel<false><<<grid_for(n, DIRECT_BS), DIRECT_BS, 0, s>>>(
                    posm_.as<float4>(), n, eps2, G, ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>());
            CK_LAUNCH();
            fill_u32<<<grid_for(n, BS), BS, 0, s>>>(counts_.as<uint32_t>(), static_cast<uint32_t>(n - 1), n);
            CK_LAUNCH();
        } else if (walk_body_) {
            if (potential)
                walk_kernel<true><<<grid_for(n, WALK_BS), WALK_BS, 0, s>>>(
                    posm_.as<float4>(), n, nodeA_.as<float4>(), nodeB_.as<float4>(), nodeC_.as<float4>(),
                    nodeR_.as<int2>(), eps2, G,
                    ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>(), counts_.as<uint32_t>(), inter, err);
            else
                walk_kernel<false><<<grid_for(n, WALK_BS), WALK_BS, 0, s>>>(
                    posm_.as<float4>(), n, nodeA_.as<float4>(), nodeB_.as<float4>(), nodeC_.as<float4>(),
                    nodeR_.as<int2>(), eps2, G,
                    ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>(), counts_.as<uint32_t>(), inter, err);
            CK_LAUNCH();
        } else {
            const WalkSrc src{nodeA_.as<float4>(), nodeB_.as<float4>(), nodeC_.as<float4>(), nodeR_.as<int2>(),
                              posm_.as<float4>(), nullptr, 0, GSTACK - (refined_ ? DEPTH_SUB : DEPTH_KEY) - 1, 1};
            launch_group(potential, false, src, groups(tree), group_bound(tree), eps2, G, out);
        }
        walk_remote(remote, tree, potential, eps2, G, out);
        CK(cudaEventRecord(ev_[3].e, s));
        struct { int err; int pad[3]; unsigned long long inter; } h{};
        CK(cudaMemcpyAsync(&h, small_.as<char>(), 24, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        if (h.err)
            throw std::runtime_error("cuda tree walk: traversal stack overflow; accelerations are invalid");
        t_.sort_ms = tree ? elapsed(0, 1) : 0.0;
        t_.build_ms = tree ? elapsed(1, 2) : 0.0;
        t_.walk_ms = elapsed(8, 3);
        t_.force_ms = t_.sort_ms + t_.build_ms + t_.walk_ms;
        t_.interactions = h.inter + (tree ? 0ull : static_cast<uint64_t>(n) * static_cast<uint64_t>(n - 1));
        has_pe_ = potential;
        acc_ok_ = true;
        costs_ok_ = true;
    }

    // Tree: a host walk over a copy of the node arrays (downloaded once per build, ~48 B/node) plus a
    // device gather of the exported leaf bodies. A device-side export would instead run a
    // level-synchronous BFS on the GPU (frontier queue with atomic appends; children of a parent
    // appended as one block to keep them contiguous), a scan to number the exported bodies, and
    // one download of the finished LET: no full node copy over PCIe and no host loop.
    LetTree export_let(const Box3& target, const ForceParams& p) override {
        CK(cudaSetDevice(dev_));
        LetTree out;
        if (n_ == 0) return out;
        cudaStream_t s = stream_->s;
        const std::size_t n = static_cast<std::size_t>(n_);
        if (p.method == ForceMethod::direct) {
            out.bx.resize(n); out.by.resize(n); out.bz.resize(n); out.bm.resize(n);
            CK(cudaMemcpyAsync(out.bx.data(), x_.as<float>(), n * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(out.by.data(), y_.as<float>(), n * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(out.bz.data(), z_.as<float>(), n * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(out.bm.data(), m_.as<float>(), n * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            double m = 0, cx = 0, cy = 0, cz = 0;
            for (std::size_t i = 0; i < n; ++i) {
                m += out.bm[i]; cx += double(out.bm[i]) * out.bx[i]; cy += double(out.bm[i]) * out.by[i];
                cz += double(out.bm[i]) * out.bz[i];
            }
            LetNode r{};
            if (m > 0) { r.cx = float(cx / m); r.cy = float(cy / m); r.cz = float(cz / m); }
            r.mass = float(m);
            r.rcrit2 = 3.0e38f;
            r.first_child = -1;
            r.child_count = 0;
            r.body_first = 0;
            r.body_count = static_cast<int32_t>(n);
            out.nodes.push_back(r);
            return out;
        }
        if (!tree_ok_) throw std::logic_error("cuda export_let: call build_tree after the last upload/drift first");
        const std::size_t nodes = 2 * static_cast<std::size_t>(K_) - 1;
        if (!host_nodes_ok_) {
            hA_.resize(nodes); hB_.resize(nodes); hC_.resize(nodes);
            CK(cudaMemcpyAsync(hA_.data(), nodeA_.as<float4>(), nodes * sizeof(float4), cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(hB_.data(), nodeB_.as<float4>(), nodes * sizeof(float4), cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(hC_.data(), nodeC_.as<float4>(), nodes * sizeof(float4), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            host_nodes_ok_ = true;
        }
        if (!(hA_[0].w > 0.f)) return out;
        float lo[3], hi[3];
        for (int c = 0; c < 3; ++c) { lo[c] = target.lo[c]; hi[c] = target.hi[c]; }
        // out.nodes doubles as the BFS queue (src[k] = local node behind out.nodes[k]); a parent's
        // children are appended together, so siblings are contiguous and children follow parents.
        std::vector<int> src, gather;
        auto emit = [&](int v) {
            const float4 A = hA_[v], B = hB_[v], C = hC_[v];
            LetNode l{};
            l.cx = A.x; l.cy = A.y; l.cz = A.z; l.mass = A.w;
            l.qxx = B.x; l.qyy = B.y; l.qxy = B.z; l.qxz = B.w; l.qyz = C.x; l.qzz = -(B.x + B.y);
            l.rcrit2 = C.y;
            l.first_child = -1; l.child_count = 0; l.body_first = -1; l.body_count = 0;
            out.nodes.push_back(l);
            src.push_back(v);
        };
        emit(0);
        for (std::size_t k = 0; k < out.nodes.size(); ++k) {
            const int v = src[k];
            const float4 A = hA_[v], C = hC_[v];
            const float gx = std::max(std::max(lo[0] - A.x, A.x - hi[0]), 0.f);
            const float gy = std::max(std::max(lo[1] - A.y, A.y - hi[1]), 0.f);
            const float gz = std::max(std::max(lo[2] - A.z, A.z - hi[2]), 0.f);
            if (gx * gx + gy * gy + gz * gz >= C.y) continue;  // terminal
            const int a = bits(C.z), b = bits(C.w);
            if (b < 0) {
                out.nodes[k].body_first = static_cast<int32_t>(gather.size());
                out.nodes[k].body_count = ~b;
                for (int j = a; j < a + ~b; ++j) gather.push_back(j);
                continue;
            }
            const int32_t first = static_cast<int32_t>(out.nodes.size());
            for (int c : {a, b})
                if (hA_[c].w > 0.f) emit(c);
            out.nodes[k].first_child = first;
            out.nodes[k].child_count = static_cast<int32_t>(out.nodes.size()) - first;
        }
        const std::size_t nb = gather.size();
        if (nb) {
            letIdx_.reserve(nb * sizeof(int));
            letGather_.reserve(nb * sizeof(float4));
            std::vector<float4> hb(nb);
            CK(cudaMemcpyAsync(letIdx_.as<int>(), gather.data(), nb * sizeof(int), cudaMemcpyHostToDevice, s));
            gather_float4<<<grid_for(static_cast<long long>(nb), BS), BS, 0, s>>>(posm_.as<float4>(), letIdx_.as<int>(),
                                                                               letGather_.as<float4>(), static_cast<int>(nb));
            CK_LAUNCH();
            CK(cudaMemcpyAsync(hb.data(), letGather_.as<float4>(), nb * sizeof(float4), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            out.bx.resize(nb); out.by.resize(nb); out.bz.resize(nb); out.bm.resize(nb);
            for (std::size_t j = 0; j < nb; ++j) {
                out.bx[j] = hb[j].x; out.by[j] = hb[j].y; out.bz[j] = hb[j].z; out.bm[j] = hb[j].w;
            }
        }
        return out;
    }

    bool body_costs(std::vector<float>& cost) override {
        CK(cudaSetDevice(dev_));
        if (!costs_ok_ || n_ == 0) return false;
        std::vector<uint32_t> c(static_cast<std::size_t>(n_));
        CK(cudaMemcpyAsync(c.data(), counts_.as<uint32_t>(), c.size() * 4, cudaMemcpyDeviceToHost, stream_->s));
        CK(cudaStreamSynchronize(stream_->s));
        cost.assign(c.begin(), c.end());
        return true;
    }

    void kick(double dt) override {
        CK(cudaSetDevice(dev_));
        if (n_ == 0) return;
        if (!acc_ok_)
            throw std::logic_error("cuda kick: accelerations are stale (build_tree since the last force call)");
        cudaStream_t s = stream_->s;
        CK(cudaEventRecord(ev_[4].e, s));
        kick_kernel<<<grid_for(n_, BS), BS, 0, s>>>(vx_.as<float>(), vy_.as<float>(), vz_.as<float>(), ax_.as<float>(),
                                                     ay_.as<float>(), az_.as<float>(), static_cast<float>(dt), n_);
        CK_LAUNCH();
        CK(cudaEventRecord(ev_[5].e, s));
        kick_rec_ = true;
    }

    void drift(double dt) override {
        CK(cudaSetDevice(dev_));
        if (n_ == 0) return;
        cudaStream_t s = stream_->s;
        CK(cudaEventRecord(ev_[6].e, s));
        drift_kernel<<<grid_for(n_, BS), BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), vx_.as<float>(),
                                                      vy_.as<float>(), vz_.as<float>(), static_cast<float>(dt), n_);
        CK_LAUNCH();
        CK(cudaEventRecord(ev_[7].e, s));
        drift_rec_ = true;
        tree_ok_ = host_nodes_ok_ = false;
    }

    Diag diagnostics() override {
        CK(cudaSetDevice(dev_));
        Diag d;
        d.has_pe = has_pe_;
        if (n_ == 0) return d;
        cudaStream_t s = stream_->s;
        const int blocks = static_cast<int>(std::min<long long>(grid_for(n_, BS), REDUCE_BLOCKS));
        diag_kernel<<<blocks, BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), vx_.as<float>(),
                                          vy_.as<float>(), vz_.as<float>(), m_.as<float>(), pot_.as<float>(),
                                          has_pe_ ? 1 : 0, n_, diag_part_.as<double>());
        CK_LAUNCH();
        std::vector<double> part(static_cast<std::size_t>(blocks) * DIAG_N);
        CK(cudaMemcpyAsync(part.data(), diag_part_.as<double>(), part.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        double v[DIAG_N] = {};
        for (int b = 0; b < blocks; ++b)
            for (int c = 0; c < DIAG_N; ++c) v[c] += part[static_cast<std::size_t>(b) * DIAG_N + c];
        d.mass = v[0];
        d.ke = v[1];
        d.pe = has_pe_ ? 0.5 * v[2] : 0.0;
        d.px = v[3]; d.py = v[4]; d.pz = v[5];
        d.lx = v[6]; d.ly = v[7]; d.lz = v[8];
        if (v[0] > 0) {
            d.cx = v[9] / v[0]; d.cy = v[10] / v[0]; d.cz = v[11] / v[0];
        }
        return d;
    }

    double exact_potential_energy(double G, double eps) override {
        CK(cudaSetDevice(dev_));
        if (n_ < 2) return 0.0;
        cudaStream_t s = stream_->s;
        pack_posm<<<grid_for(n_, BS), BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), m_.as<float>(),
                                                  posm_.as<float4>(), n_);
        CK_LAUNCH();
        double* acc = reinterpret_cast<double*>(small_.as<char>() + 32);
        CK(cudaMemsetAsync(acc, 0, sizeof(double), s));
        exact_pe_kernel<<<grid_for(n_, DIRECT_BS), DIRECT_BS, 0, s>>>(posm_.as<float4>(), n_, eps * eps, acc);
        CK_LAUNCH();
        double h = 0;
        CK(cudaMemcpyAsync(&h, acc, sizeof h, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        return -0.5 * G * h;
    }

    PhaseTimes times() const override {
        PhaseTimes t = t_;
        // Kick/drift do not synchronise; their events are resolved lazily here.
        double ms = 0;
        if (kick_rec_) ms += elapsed_sync(4, 5);
        if (drift_rec_) ms += elapsed_sync(6, 7);
        t.step_ms = ms;
        return t;
    }

private:
    static constexpr int NEV = 9;  // 0 build start, 1 sorted, 2 built, 3 walked; 4-5 kick; 6-7 drift; 8 walk start

    static int bits(float f) {
        int i;
        std::memcpy(&i, &f, sizeof i);
        return i;
    }
    static float as_float(int i) {
        float f;
        std::memcpy(&f, &i, sizeof f);
        return f;
    }

    // Tree: the groups made by make_groups (count on the device). Direct (remote walks only):
    // fixed chunks of gmax_ bodies in the current, unsorted order.
    WalkGroups groups(bool tree) {
        int* next = reinterpret_cast<int*>(small_.as<char>() + 72);
        if (tree)
            return WalkGroups{groups_.as<int>(), nodeR_.as<int2>(), reinterpret_cast<int*>(small_.as<char>() + 64), 0,
                              gmax_, n_, next};
        return WalkGroups{nullptr, nullptr, nullptr, static_cast<int>((n_ + gmax_ - 1) / gmax_), gmax_, n_, next};
    }
    int group_bound(bool tree) const { return tree ? K_ : (n_ + gmax_ - 1) / gmax_; }

    template <int NB, bool POT, bool LET>
    void launch_group_t(const WalkSrc& t, const WalkGroups& g, int ub, float eps2, float G, const WalkOut& o) {
        int& occ = occ_[(NB == 2 ? 4 : 0) + (POT ? 2 : 0) + (LET ? 1 : 0)];
        if (occ == 0) {
            int b = 0;
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, group_walk_kernel<NB, POT, LET>, GW_BS, 0));
            occ = std::max(b, 1);
        }
        // Persistent warps pull groups from a counter: groups differ a lot in cost, and pulling them
        // in body order keeps the warps running at any moment on nearby groups (shared L1/L2 lines).
        const long long want = (static_cast<long long>(ub) + GW_WARPS - 1) / GW_WARPS;
        const int blocks = static_cast<int>(std::max<long long>(1, std::min<long long>(want, static_cast<long long>(occ) * sms_)));
        cudaStream_t s = stream_->s;
        CK(cudaMemsetAsync(g.next, 0, sizeof(int), s));
        group_walk_kernel<NB, POT, LET><<<blocks, GW_BS, 0, s>>>(posm_.as<float4>(), t, g, eps2, G, o);
        CK_LAUNCH();
    }
    void launch_group(bool pot, bool let, const WalkSrc& t, const WalkGroups& g, int ub, float eps2, float G,
                      const WalkOut& o) {
        if (nb_ == 2) {
            if (pot) let ? launch_group_t<2, true, true>(t, g, ub, eps2, G, o) : launch_group_t<2, true, false>(t, g, ub, eps2, G, o);
            else let ? launch_group_t<2, false, true>(t, g, ub, eps2, G, o) : launch_group_t<2, false, false>(t, g, ub, eps2, G, o);
        } else {
            if (pot) let ? launch_group_t<1, true, true>(t, g, ub, eps2, G, o) : launch_group_t<1, true, false>(t, g, ub, eps2, G, o);
            else let ? launch_group_t<1, false, true>(t, g, ub, eps2, G, o) : launch_group_t<1, false, false>(t, g, ub, eps2, G, o);
        }
    }

    // Uploads every remote LET into one set of grow-only device arrays (converted to the walk's
    // node encoding, indices offset) and walks them with the local groups, adding to the forces.
    // LETs arrive over MPI, so every index is validated before it can reach the device.
    void walk_remote(const std::vector<const LetTree*>& remote, bool tree, bool potential, float eps2, float G,
                     const WalkOut& out) {
        std::size_t nn = 0, nbt = 0;
        for (const LetTree* t : remote)
            if (t && !t->nodes.empty()) { nn += t->nodes.size(); nbt += t->bx.size(); }
        if (nn == 0) return;
        if (nn >= static_cast<std::size_t>(INT32_MAX) || nbt >= static_cast<std::size_t>(INT32_MAX))
            throw std::runtime_error("cuda compute_forces_with: remote LETs exceed 2^31 nodes or bodies");
        hLA_.resize(nn); hLB_.resize(nn); hLC_.resize(nn); hLBody_.resize(nbt);
        std::vector<int> roots, depth;
        int cmax = 2, height = 0;
        std::size_t off = 0, boff = 0;
        for (const LetTree* t : remote) {
            if (!t || t->nodes.empty()) continue;
            const std::size_t m = t->nodes.size(), nb = t->bx.size();
            if (t->by.size() != nb || t->bz.size() != nb || t->bm.size() != nb)
                throw std::runtime_error("cuda compute_forces_with: LET body arrays differ in length");
            depth.assign(m, 0);
            roots.push_back(static_cast<int>(off));
            for (std::size_t i = 0; i < m; ++i) {
                const LetNode& L = t->nodes[i];
                int first = 0, code = 0;
                if (L.body_count > 0) {
                    if (L.body_first < 0 || static_cast<std::size_t>(L.body_first) + L.body_count > nb)
                        throw std::runtime_error("cuda compute_forces_with: LET node body range out of bounds");
                    first = static_cast<int>(boff) + L.body_first;
                    code = ~L.body_count;
                } else if (L.first_child >= 0 && L.child_count > 0) {
                    // Children after their parent (both exporters emit breadth-first): rules out
                    // cycles and makes one forward pass compute every depth.
                    if (static_cast<std::size_t>(L.first_child) <= i ||
                        static_cast<std::size_t>(L.first_child) + L.child_count > m)
                        throw std::runtime_error("cuda compute_forces_with: LET node children out of order or bounds");
                    first = static_cast<int>(off) + L.first_child;
                    code = L.child_count;
                    cmax = std::max(cmax, static_cast<int>(L.child_count));
                    for (int32_t c = L.first_child; c < L.first_child + L.child_count; ++c)
                        depth[static_cast<std::size_t>(c)] = std::max(depth[static_cast<std::size_t>(c)], depth[i] + 1);
                }
                height = std::max(height, depth[i]);
                hLA_[off + i] = make_float4(L.cx, L.cy, L.cz, L.mass);
                hLB_[off + i] = make_float4(L.qxx, L.qyy, L.qxy, L.qxz);
                hLC_[off + i] = make_float4(L.qyz, L.rcrit2, as_float(first), as_float(code));
            }
            for (std::size_t j = 0; j < nb; ++j) hLBody_[boff + j] = make_float4(t->bx[j], t->by[j], t->bz[j], t->bm[j]);
            off += m;
            boff += nb;
        }
        const long long thr = GSTACK - static_cast<long long>(height) * (cmax - 1) - 1;
        if (thr < 64)
            throw std::runtime_error("cuda compute_forces_with: LET too deep for the group walk stack (height " +
                                     std::to_string(height) + ", up to " + std::to_string(cmax) + " children)");
        cudaStream_t s = stream_->s;
        letA_.reserve(nn * sizeof(float4));
        letB_.reserve(nn * sizeof(float4));
        letC_.reserve(nn * sizeof(float4));
        letBody_.reserve(std::max<std::size_t>(nbt, 1) * sizeof(float4));
        letRoots_.reserve(roots.size() * sizeof(int));
        CK(cudaMemcpyAsync(letA_.as<float4>(), hLA_.data(), nn * sizeof(float4), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(letB_.as<float4>(), hLB_.data(), nn * sizeof(float4), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(letC_.as<float4>(), hLC_.data(), nn * sizeof(float4), cudaMemcpyHostToDevice, s));
        if (nbt)
            CK(cudaMemcpyAsync(letBody_.as<float4>(), hLBody_.data(), nbt * sizeof(float4), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(letRoots_.as<int>(), roots.data(), roots.size() * sizeof(int), cudaMemcpyHostToDevice, s));
        const WalkGroups g = groups(tree);
        // At most 32 roots per launch: the walk seeds its stack with one root per lane.
        for (std::size_t r0 = 0; r0 < roots.size(); r0 += 32) {
            const int cnt = static_cast<int>(std::min<std::size_t>(32, roots.size() - r0));
            const WalkSrc src{letA_.as<float4>(), letB_.as<float4>(), letC_.as<float4>(), nullptr,
                              letBody_.as<float4>(), letRoots_.as<int>() + r0, cnt, static_cast<int>(thr), cmax - 1};
            launch_group(potential, true, src, g, group_bound(tree), eps2, G, out);
        }
        // The host staging vectors are reused by the next call; the copies above are pageable, so
        // they have completed (staged) by the time cudaMemcpyAsync returns.
    }

    void make_groups() {
        cudaStream_t s = stream_->s;
        const int n = n_;
        const int nodes = 2 * K_ - 1;
        int* slot = idx_[scratch_sel_].as<int>();  // free after the build
        CK(cudaMemsetAsync(slot, 0xff, static_cast<std::size_t>(n) * sizeof(int), s));
        mark_groups<<<grid_for(nodes, BS), BS, 0, s>>>(nodeR_.as<int2>(), parent_.as<int>(), nodes, gmax_, slot);
        CK_LAUNCH();
        groups_.reserve(static_cast<std::size_t>(K_) * sizeof(int));
        int* ng = reinterpret_cast<int*>(small_.as<char>() + 64);
        std::size_t need = 0;
        CK(cub::DeviceSelect::If(nullptr, need, slot, groups_.as<int>(), ng, n, NonNegative(), s));
        reserve_temp(need);
        need = temp_bytes_;
        CK(cub::DeviceSelect::If(temp_.as<void>(), need, slot, groups_.as<int>(), ng, n, NonNegative(), s));
    }

    double elapsed(int a, int b) const {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev_[a].e, ev_[b].e));
        return ms;
    }
    double elapsed_sync(int a, int b) const {
        CK(cudaSetDevice(dev_));
        CK(cudaEventSynchronize(ev_[b].e));
        return elapsed(a, b);
    }

    void reserve_bodies(std::size_t n) {
        const std::size_t f = n * 4;
        for (DevBuf* b : {&x_, &y_, &z_, &vx_, &vy_, &vz_, &m_, &id_, &ax_, &ay_, &az_, &pot_, &counts_, &idx_[0], &idx_[1]})
            b->reserve(f);
        posm_.reserve(n * sizeof(float4));
        // +1 entry: the alternate key buffer also holds leaf starts [0, K] with K <= n.
        keys_[0].reserve((n + 1) * sizeof(uint64_t));
        keys_[1].reserve((n + 1) * sizeof(uint64_t));
    }

    void reserve_temp(std::size_t bytes) { temp_.reserve(std::max<std::size_t>(bytes, 16)); temp_bytes_ = std::max(temp_bytes_, bytes); }

    // Morton keys in the bounding cube, radix sort, and permutation of all body arrays into key
    // order. Bodies stay in that order afterwards, which keeps the next sort nearly presorted.
    void sort_bodies() {
        cudaStream_t s = stream_->s;
        const int n = n_;
        const int pb = static_cast<int>(std::min<long long>(grid_for(n, BS), REDUCE_BLOCKS));
        bbox_partial<<<pb, BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), n, bbox_part_.as<float>());
        CK_LAUNCH();
        bbox_final<<<1, BS, 0, s>>>(bbox_part_.as<float>(), pb, cube_.as<float4>());
        CK_LAUNCH();
        morton_kernel<<<grid_for(n, BS), BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), cube_.as<float4>(),
                                                     keys_[0].as<uint64_t>(), idx_[0].as<uint32_t>(), n);
        CK_LAUNCH();
        cub::DoubleBuffer<uint64_t> kb(keys_[0].as<uint64_t>(), keys_[1].as<uint64_t>());
        cub::DoubleBuffer<uint32_t> vb(idx_[0].as<uint32_t>(), idx_[1].as<uint32_t>());
        std::size_t need = 0;
        CK(cub::DeviceRadixSort::SortPairs(nullptr, need, kb, vb, n, 0, 63, s));
        reserve_temp(need);
        need = temp_bytes_;
        CK(cub::DeviceRadixSort::SortPairs(temp_.as<void>(), need, kb, vb, n, 0, 63, s));
        key_sel_ = kb.selector;
        const int vs = vb.selector;
        const uint32_t* perm = idx_[vs].as<uint32_t>();
        const unsigned g = grid_for(n, BS);
        gather_posm<<<g, BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), m_.as<float>(), perm,
                                     posm_.as<float4>(), n);
        CK_LAUNCH();
        unpack_posm<<<g, BS, 0, s>>>(posm_.as<float4>(), x_.as<float>(), y_.as<float>(), z_.as<float>(), m_.as<float>(), n);
        CK_LAUNCH();
        // All 4-byte body arrays share one capacity, so the free index buffer can trade places with
        // each array as the gather target.
        for (DevBuf* b : {&vx_, &vy_, &vz_, &id_}) {
            gather_u32<<<g, BS, 0, s>>>(b->as<uint32_t>(), perm, idx_[1 - vs].as<uint32_t>(), n);
            CK_LAUNCH();
            swap(*b, idx_[1 - vs]);
        }
        scratch_sel_ = 1 - vs;
    }

    // Leaf levels, flags and starts for the current keys (sub: subkeys or null). Host-synchronises
    // to read K, and, without subkeys, whether some run of identical keys needs refining.
    int build_leaves(const uint64_t* sub, bool& dup) {
        cudaStream_t s = stream_->s;
        const int n = n_;
        const unsigned g = grid_for(n, BS);
        const uint64_t* keys = keys_[key_sel_].as<uint64_t>();
        char* region = keys_[1 - key_sel_].as<char>();
        uint32_t* flags = reinterpret_cast<uint32_t*>(region);                    // n+1 entries
        int* start = reinterpret_cast<int*>(region);                              // reuses flags after the scan
        uint32_t* incl = reinterpret_cast<uint32_t*>(region + (static_cast<std::size_t>(n) + 1) * 4);
        uint8_t* D = idx_[scratch_sel_].as<uint8_t>();
        uint8_t* L = D + n;
        int* dK = reinterpret_cast<int*>(small_.as<char>() + 48);  // K, dup
        CK(cudaMemsetAsync(dK, 0, 2 * sizeof(int), s));
        if (n > LEAF_MAX) {
            window_octs<<<grid_for(n - LEAF_MAX, BS), BS, 0, s>>>(keys, sub, D, dK + 1, n);
            CK_LAUNCH();
        }
        leaf_level<<<g, BS, 0, s>>>(D, L, n);
        CK_LAUNCH();
        leaf_flags<<<g, BS, 0, s>>>(keys, sub, L, flags, n);
        CK_LAUNCH();
        std::size_t need = 0;
        CK(cub::DeviceScan::InclusiveSum(nullptr, need, flags, incl, n, s));
        reserve_temp(need);
        need = temp_bytes_;
        CK(cub::DeviceScan::InclusiveSum(temp_.as<void>(), need, flags, incl, n, s));
        leaf_starts<<<g, BS, 0, s>>>(incl, start, dK, n);
        CK_LAUNCH();
        int h[2] = {0, 0};
        CK(cudaMemcpyAsync(h, dK, sizeof h, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        const int K = h[0];
        dup = h[1] != 0;
        if (K < 1 || K > n) throw std::runtime_error("cuda tree build: bad leaf count " + std::to_string(K));
        if (2 * static_cast<std::size_t>(K) - 1 > static_cast<std::size_t>(INT32_MAX))
            throw std::runtime_error("cuda tree build: " + std::to_string(K) + " leaves exceed the 2^31-1 node index limit");
        return K;
    }

    // Refines the runs of > LEAF_MAX identical keys (see long_run_starts), re-sorts the bodies and
    // returns the subkeys. Uses only buffers that are free between sort_bodies and the leaf build:
    // idx_[0] flags then the identity permutation, idx_[1] their scan, keys_[1 - key_sel_] the
    // alternate sort keys, posm_ bytes [0, 8n) the sort keys then the subkeys and [8n, 8n + 28R)
    // the run info (R <= n/33 runs, so 28R < 8n). posm_ is repacked once the subkeys are dead.
    const uint64_t* refine_runs() {
        cudaStream_t s = stream_->s;
        const int n = n_;
        const unsigned g = grid_for(n, BS);
        const uint64_t* keys = keys_[key_sel_].as<uint64_t>();
        uint32_t* flags = idx_[0].as<uint32_t>();
        uint32_t* incl = idx_[1].as<uint32_t>();
        uint64_t* key2 = posm_.as<uint64_t>();
        int* info = reinterpret_cast<int*>(posm_.as<char>() + static_cast<std::size_t>(n) * 8);
        long_run_starts<<<g, BS, 0, s>>>(keys, flags, n);
        CK_LAUNCH();
        std::size_t need = 0;
        CK(cub::DeviceScan::InclusiveSum(nullptr, need, flags, incl, n, s));
        reserve_temp(need);
        need = temp_bytes_;
        CK(cub::DeviceScan::InclusiveSum(temp_.as<void>(), need, flags, incl, n, s));
        long_run_init<<<g, BS, 0, s>>>(flags, incl, info, n);
        CK_LAUNCH();
        long_run_box<<<g, BS, 0, s>>>(keys, incl, x_.as<float>(), y_.as<float>(), z_.as<float>(), info, n);
        CK_LAUNCH();
        int pbits = 0;  // bits of n - 1, the largest segment start
        while ((1ll << pbits) < n) ++pbits;
        const int so3 = 3 * std::min(21, (64 - pbits) / 3);
        long_run_keys<<<g, BS, 0, s>>>(keys, incl, info, x_.as<float>(), y_.as<float>(), z_.as<float>(), so3, key2,
                                       idx_[0].as<uint32_t>(), n);
        CK_LAUNCH();
        cub::DoubleBuffer<uint64_t> kb(key2, keys_[1 - key_sel_].as<uint64_t>());
        cub::DoubleBuffer<uint32_t> vb(idx_[0].as<uint32_t>(), idx_[1].as<uint32_t>());
        need = 0;
        CK(cub::DeviceRadixSort::SortPairs(nullptr, need, kb, vb, n, 0, pbits + so3, s));
        reserve_temp(need);
        need = temp_bytes_;
        CK(cub::DeviceRadixSort::SortPairs(temp_.as<void>(), need, kb, vb, n, 0, pbits + so3, s));
        long_run_subkeys<<<g, BS, 0, s>>>(kb.Current(), key2, so3, n);
        CK_LAUNCH();
        // Keys are unchanged: only bodies with identical keys moved.
        const int vs = vb.selector;
        const uint32_t* perm = idx_[vs].as<uint32_t>();
        for (DevBuf* b : {&x_, &y_, &z_, &m_, &vx_, &vy_, &vz_, &id_}) {
            gather_u32<<<g, BS, 0, s>>>(b->as<uint32_t>(), perm, idx_[1 - vs].as<uint32_t>(), n);
            CK_LAUNCH();
            swap(*b, idx_[1 - vs]);
        }
        scratch_sel_ = 1 - vs;
        return key2;
    }

    // Returns the number of leaves K. Host-synchronises once to read K for sizing (twice if long
    // runs need refining: the first leaf build then only detects them).
    int build_tree_impl(double theta) {
        cudaStream_t s = stream_->s;
        const int n = n_;
        bool dup = false;
        const uint64_t* sub = nullptr;
        int K = build_leaves(nullptr, dup);
        refined_ = dup;
        if (dup) {
            sub = refine_runs();
            K = build_leaves(sub, dup);
        }
        const uint64_t* keys = keys_[key_sel_].as<uint64_t>();
        const int* start = reinterpret_cast<const int*>(keys_[1 - key_sel_].as<char>());

        const std::size_t nodes = 2 * static_cast<std::size_t>(K) - 1;
        for (DevBuf* b : {&nodeA_, &nodeB_, &nodeC_, &bmin_, &bmax_}) b->reserve(nodes * sizeof(float4));
        parent_.reserve(nodes * sizeof(int));
        nodeR_.reserve(nodes * sizeof(int2));
        counter_.reserve(std::max<std::size_t>(K - 1, 1) * sizeof(int));
        CK(cudaMemsetAsync(parent_.as<int>(), 0xff, sizeof(int), s));  // root's parent = -1
        if (K > 1) {
            CK(cudaMemsetAsync(counter_.as<int>(), 0, static_cast<std::size_t>(K - 1) * sizeof(int), s));
            build_radix<<<grid_for(K - 1, BS), BS, 0, s>>>(keys, sub, start, K, nodeC_.as<float4>(), parent_.as<int>(),
                                                           nodeR_.as<int2>());
            CK_LAUNCH();
        }
        if (sub) {  // the subkeys lived in posm_
            pack_posm<<<grid_for(n, BS), BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), m_.as<float>(),
                                                     posm_.as<float4>(), n);
            CK_LAUNCH();
        }
        const float inv_theta = theta > 1.0 / RCRIT_MAX ? static_cast<float>(1.0 / theta) : RCRIT_MAX;
        moments_kernel<<<grid_for(K, BS), BS, 0, s>>>(posm_.as<float4>(), start, K, inv_theta, parent_.as<int>(),
                                                      counter_.as<int>(), nodeA_.as<float4>(), nodeB_.as<float4>(),
                                                      nodeC_.as<float4>(), bmin_.as<float4>(), bmax_.as<float4>(),
                                                      nodeR_.as<int2>());
        CK_LAUNCH();
        return K;
    }

    int dev_;
    std::string name_;
    std::unique_ptr<Stream> stream_;
    std::unique_ptr<Event[]> ev_;
    int n_ = 0;
    bool has_pe_ = false;
    bool kick_rec_ = false, drift_rec_ = false;
    PhaseTimes t_{};
    int sms_ = 1;
    bool walk_body_ = false;
    int nb_ = 2, gmax_ = 64;      // bodies per lane in the group walk, group size
    int occ_[8] = {};            // resident blocks/SM per group-walk instantiation (lazy)
    int K_ = 0;                  // leaves of the current tree
    bool tree_ok_ = false;       // tree (and groups) match the current positions
    bool host_nodes_ok_ = false; // hA_/hB_/hC_ hold the current tree
    bool costs_ok_ = false;      // counts_ are the last force call's, in the current body order
    bool acc_ok_ = false;        // ax/ay/az are for the current body order
    bool refined_ = false;       // the current tree has subkeys (deeper: DEPTH_SUB)
    std::vector<float4> hA_, hB_, hC_, hLA_, hLB_, hLC_, hLBody_;
    int key_sel_ = 0, scratch_sel_ = 1;
    std::size_t temp_bytes_ = 0;

    DevBuf x_, y_, z_, vx_, vy_, vz_, m_, id_, ax_, ay_, az_, pot_, posm_;
    DevBuf keys_[2], idx_[2];
    DevBuf nodeA_, nodeB_, nodeC_, bmin_, bmax_, parent_, counter_, nodeR_, groups_, counts_;
    DevBuf letA_, letB_, letC_, letBody_, letRoots_, letIdx_, letGather_;
    DevBuf cube_, bbox_part_, diag_part_, small_, temp_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(int device, std::string& why) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        cudaGetLastError();
        why = std::string("no usable CUDA device: ") + cudaGetErrorString(e);
        return nullptr;
    }
    if (count == 0) {
        why = "no CUDA device found";
        return nullptr;
    }
    if (device < 0 || device >= count) {
        why = "CUDA device " + std::to_string(device) + " out of range (" + std::to_string(count) + " devices)";
        return nullptr;
    }
    try {
        return std::make_unique<CudaBackend>(device);
    } catch (const std::exception& ex) {
        why = ex.what();
        return nullptr;
    }
}

bool cuda_compiled() { return true; }

int cuda_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count;
}

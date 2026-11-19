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
//   posm (float4 pos+mass)          16 B   gathered copy for coalesced 16-byte loads in kernels
//   Morton keys, double-buffered    16 B   (the alternate also holds leaf flags/scan/starts)
//   sort indices, double-buffered    8 B   (the alternate is the permutation scratch)
//   = 88 B, plus tree nodes: 88 B per node, 2K-1 nodes for K leaves (K ~ n/8..n/16 for Plummer)
//   ~ 100-110 B/body total, so 80 GB holds ~700M bodies. Nodes are sized from the actual K.
//
// Limits: n < 2^31 (int body/node indices; checked in upload()).
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr int STACK = 96;        // radix-tree depth <= 63 key bits + log2(#duplicate-key runs)
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

__global__ void morton_kernel(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
                              const float4* __restrict__ cube, uint64_t* __restrict__ keys,
                              uint32_t* __restrict__ idx, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 c = *cube;
    keys[i] = spread3(quantize(x[i], c.x, c.w)) << 2 | spread3(quantize(y[i], c.y, c.w)) << 1 |
              spread3(quantize(z[i], c.z, c.w));
    idx[i] = static_cast<uint32_t>(i);
}

// Common leading octal digits of two 63-bit keys (21 = identical).
__device__ __forceinline__ int common_octs(uint64_t a, uint64_t b) {
    return a == b ? 21 : (__clzll(static_cast<long long>(a ^ b)) - 1) / 3;
}

// Leaf scheme: a leaf is the smallest octree cell (level L, key prefix of 3L bits) that holds
// <= 32 bodies. Cells are contiguous in sorted order, so the cell of body i at level L has >= 33
// bodies iff some window [j, j+32] containing i has common_octs(key[j], key[j+32]) >= L. Hence
// L_i = 1 + max over those windows (0 if n <= 32), and all bodies of one leaf get the same L.
// L_i = 22 means > 32 identical keys; such runs are split into fixed chunks of 32.
__global__ void window_octs(const uint64_t* __restrict__ keys, uint8_t* __restrict__ D, int n) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j + LEAF_MAX < n) D[j] = static_cast<uint8_t>(common_octs(keys[j], keys[j + LEAF_MAX]));
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
    uint64_t prefix;
    long long run;
};

__device__ LeafId leaf_id(const uint64_t* __restrict__ keys, const uint8_t* __restrict__ L, long long i) {
    const int l = L[i];
    const uint64_t k = keys[i];
    LeafId r{l, 0, 0};
    if (l <= 21) {
        r.prefix = k >> (63 - 3 * l);
    } else {
        r.prefix = k;
        long long a = 0, b = i;  // lower_bound of k in keys[0..i]
        while (a < b) {
            long long mid = a + (b - a) / 2;
            if (keys[mid] < k) a = mid + 1; else b = mid;
        }
        r.run = (i - a) / LEAF_MAX;
    }
    return r;
}

__global__ void leaf_flags(const uint64_t* __restrict__ keys, const uint8_t* __restrict__ L,
                           uint32_t* __restrict__ flags, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t f = 1;
    if (i > 0) {
        LeafId a = leaf_id(keys, L, i), b = leaf_id(keys, L, i - 1);
        f = (a.level != b.level || a.prefix != b.prefix || a.run != b.run) ? 1u : 0u;
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

__device__ __forceinline__ int delta(const uint64_t* __restrict__ keys, const int* __restrict__ start, int K,
                                     int i, long long j) {
    if (j < 0 || j >= K) return -1;
    const uint64_t a = keys[start[i]], b = keys[start[j]];
    // Equal first keys only happen for chunks of one duplicate run; the index breaks the tie.
    if (a == b) return 64 + __clz(static_cast<unsigned>(i ^ static_cast<int>(j)));
    return __clzll(static_cast<long long>(a ^ b));
}

__global__ void build_radix(const uint64_t* __restrict__ keys, const int* __restrict__ start, int K,
                            float4* __restrict__ nodeC, int* __restrict__ parent) {
    long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= K - 1) return;
    const int i = static_cast<int>(gi);
    const int d = delta(keys, start, K, i, i + 1) - delta(keys, start, K, i, (long long)i - 1) >= 0 ? 1 : -1;
    const int dmin = delta(keys, start, K, i, (long long)i - d);
    long long lmax = 2;
    while (delta(keys, start, K, i, i + lmax * d) > dmin) lmax <<= 1;
    long long l = 0;
    for (long long t = lmax >> 1; t >= 1; t >>= 1)
        if (delta(keys, start, K, i, i + (l + t) * d) > dmin) l += t;
    const long long j = i + l * d;
    const int dnode = delta(keys, start, K, i, j);
    long long s = 0, t = l;
    do {
        t = (t + 1) >> 1;
        if (delta(keys, start, K, i, i + (s + t) * d) > dnode) s += t;
    } while (t > 1);
    const long long gamma = i + s * d + min(d, 0);
    const int left = (min((long long)i, j) == gamma) ? static_cast<int>(K - 1 + gamma) : static_cast<int>(gamma);
    const int right = (max((long long)i, j) == gamma + 1) ? static_cast<int>(K + gamma) : static_cast<int>(gamma + 1);
    nodeC[i] = make_float4(0.f, 0.f, __int_as_float(left), __int_as_float(right));
    parent[left] = i;
    parent[right] = i;
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
    const float sx = M.hi.x - M.lo.x, sy = M.hi.y - M.lo.y, sz = M.hi.z - M.lo.z;
    const float l = fmaxf(sx, fmaxf(sy, sz));
    const float cx = 0.5f * (M.lo.x + M.hi.x) - M.com.x, cy = 0.5f * (M.lo.y + M.hi.y) - M.com.y,
                cz = 0.5f * (M.lo.z + M.hi.z) - M.com.z;
    // theta = 0 arrives as inv_theta = RCRIT_MAX; clamping keeps r^2 finite (no inf under
    // finite-math) while still larger than any d^2 of a sane system, i.e. "always open".
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
                                                     float4* nodeC, float4* bmin, float4* bmax) {
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

// One thread per body in Morton order, so a warp's bodies are close and follow similar paths.
// Accept a node iff |com - x|^2 > rcrit^2 with rcrit = l/theta + |com - box centre|. The strict
// comparison makes a single-body leaf (rcrit = 0) at the body's own position open instead of
// being taken as a far field. For theta < 2/sqrt(3) a body's own cell is always opened.
template <bool POT>
__global__ void __launch_bounds__(WALK_BS) walk_kernel(const float4* __restrict__ pm, int n,
                                                       const float4* __restrict__ nodeA,
                                                       const float4* __restrict__ nodeB,
                                                       const float4* __restrict__ nodeC, float eps2, float G,
                                                       float* __restrict__ ax, float* __restrict__ ay,
                                                       float* __restrict__ az, float* __restrict__ pot,
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
            if (d2 > C.y) {
                // Far field about the com, d = com - x, s^2 = d^2 + eps^2:
                //   a = G [ M d/s^3 - Q d/s^5 + 5/2 (d.Q.d) d/s^7 ],  phi = -G [ M/s + 1/2 d.Q.d/s^5 ]
                const float4 B = __ldg(&nodeB[node]);
                const float qzz = -(B.x + B.y);
                const float qdx = B.x * dx + B.z * dy + B.w * dz;
                const float qdy = B.z * dx + B.y * dy + C.x * dz;
                const float qdz = B.w * dx + C.x * dy + qzz * dz;
                const float dqd = dx * qdx + dy * qdy + dz * qdz;
                const float inv = rsqrtf(d2 + eps2);
                const float inv2 = inv * inv;
                const float inv3 = inv * inv2, inv5 = inv3 * inv2, inv7 = inv5 * inv2;
                const float rad = A.w * inv3 + 2.5f * dqd * inv7;
                sx += rad * dx - qdx * inv5;
                sy += rad * dy - qdy * inv5;
                sz += rad * dz - qdz * inv5;
                if (POT) sp += A.w * inv + 0.5f * dqd * inv5;
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
    }
    // All 32 lanes reach this: WALK_BS is a multiple of 32 and no thread returns early.
    count = warp_sum(count);
    if ((threadIdx.x & 31) == 0 && count) atomicAdd(interactions, count);
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
        small_.reserve(64);
    }

    std::string name() const override { return name_; }

    void upload(const Bodies& b) override {
        CK(cudaSetDevice(dev_));
        const std::size_t n = b.size();
        if (n >= static_cast<std::size_t>(INT32_MAX))
            throw std::runtime_error("cuda backend: n = " + std::to_string(n) + " exceeds the 2^31-1 body limit");
        reserve_bodies(n);
        n_ = static_cast<int>(n);
        has_pe_ = false;
        kick_rec_ = drift_rec_ = false;
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

    // Synchronises at the end (and once mid-build to read the leaf count), so the PhaseTimes are
    // final when this returns and walk errors are reported here rather than at some later call.
    void compute_forces(const ForceParams& p, bool potential) override {
        CK(cudaSetDevice(dev_));
        t_.sort_ms = t_.build_ms = t_.walk_ms = t_.force_ms = 0;
        t_.interactions = 0;
        if (n_ == 0) {
            has_pe_ = potential;
            return;
        }
        cudaStream_t s = stream_->s;
        const int n = n_;
        const float G = static_cast<float>(p.G), eps2 = static_cast<float>(p.eps * p.eps);
        CK(cudaEventRecord(ev_[0].e, s));
        if (p.method == ForceMethod::direct) {
            pack_posm<<<grid_for(n, BS), BS, 0, s>>>(x_.as<float>(), y_.as<float>(), z_.as<float>(), m_.as<float>(),
                                                     posm_.as<float4>(), n);
            CK_LAUNCH();
            CK(cudaEventRecord(ev_[1].e, s));
            if (potential)
                direct_kernel<true><<<grid_for(n, DIRECT_BS), DIRECT_BS, 0, s>>>(
                    posm_.as<float4>(), n, eps2, G, ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>());
            else
                direct_kernel<false><<<grid_for(n, DIRECT_BS), DIRECT_BS, 0, s>>>(
                    posm_.as<float4>(), n, eps2, G, ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>());
            CK_LAUNCH();
            CK(cudaEventRecord(ev_[3].e, s));
            CK(cudaEventSynchronize(ev_[3].e));
            t_.walk_ms = elapsed(1, 3);
            t_.force_ms = elapsed(0, 3);
            t_.interactions = static_cast<uint64_t>(n) * static_cast<uint64_t>(n - 1);
        } else {
            sort_bodies();
            CK(cudaEventRecord(ev_[1].e, s));
            const int K = build_tree(p.theta);
            CK(cudaEventRecord(ev_[2].e, s));
            int* err = small_.as<int>();
            unsigned long long* inter = reinterpret_cast<unsigned long long*>(small_.as<char>() + 16);
            CK(cudaMemsetAsync(small_.as<char>(), 0, 24, s));
            (void)K;
            if (potential)
                walk_kernel<true><<<grid_for(n, WALK_BS), WALK_BS, 0, s>>>(
                    posm_.as<float4>(), n, nodeA_.as<float4>(), nodeB_.as<float4>(), nodeC_.as<float4>(), eps2, G,
                    ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>(), inter, err);
            else
                walk_kernel<false><<<grid_for(n, WALK_BS), WALK_BS, 0, s>>>(
                    posm_.as<float4>(), n, nodeA_.as<float4>(), nodeB_.as<float4>(), nodeC_.as<float4>(), eps2, G,
                    ax_.as<float>(), ay_.as<float>(), az_.as<float>(), pot_.as<float>(), inter, err);
            CK_LAUNCH();
            CK(cudaEventRecord(ev_[3].e, s));
            struct { int err; int pad[3]; unsigned long long inter; } h{};
            CK(cudaMemcpyAsync(&h, small_.as<char>(), 24, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            if (h.err)
                throw std::runtime_error("cuda tree walk: traversal stack overflow (depth > " +
                                         std::to_string(STACK) + "); accelerations are invalid");
            t_.sort_ms = elapsed(0, 1);
            t_.build_ms = elapsed(1, 2);
            t_.walk_ms = elapsed(2, 3);
            t_.force_ms = elapsed(0, 3);
            t_.interactions = h.inter;
        }
        has_pe_ = potential;
    }

    void kick(double dt) override {
        CK(cudaSetDevice(dev_));
        if (n_ == 0) return;
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
    static constexpr int NEV = 8;  // force: 0 start, 1 sorted, 2 built, 3 walked; 4-5 kick; 6-7 drift

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
        for (DevBuf* b : {&x_, &y_, &z_, &vx_, &vy_, &vz_, &m_, &id_, &ax_, &ay_, &az_, &pot_, &idx_[0], &idx_[1]})
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

    // Returns the number of leaves K. Host-synchronises once to read K for sizing.
    int build_tree(double theta) {
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
        if (n > LEAF_MAX) {
            window_octs<<<grid_for(n - LEAF_MAX, BS), BS, 0, s>>>(keys, D, n);
            CK_LAUNCH();
        }
        leaf_level<<<g, BS, 0, s>>>(D, L, n);
        CK_LAUNCH();
        leaf_flags<<<g, BS, 0, s>>>(keys, L, flags, n);
        CK_LAUNCH();
        std::size_t need = 0;
        CK(cub::DeviceScan::InclusiveSum(nullptr, need, flags, incl, n, s));
        reserve_temp(need);
        need = temp_bytes_;
        CK(cub::DeviceScan::InclusiveSum(temp_.as<void>(), need, flags, incl, n, s));
        int* dK = reinterpret_cast<int*>(small_.as<char>() + 48);
        leaf_starts<<<g, BS, 0, s>>>(incl, start, dK, n);
        CK_LAUNCH();
        int K = 0;
        CK(cudaMemcpyAsync(&K, dK, sizeof K, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        if (K < 1 || K > n) throw std::runtime_error("cuda tree build: bad leaf count " + std::to_string(K));

        const std::size_t nodes = 2 * static_cast<std::size_t>(K) - 1;
        for (DevBuf* b : {&nodeA_, &nodeB_, &nodeC_, &bmin_, &bmax_}) b->reserve(nodes * sizeof(float4));
        parent_.reserve(nodes * sizeof(int));
        counter_.reserve(std::max<std::size_t>(K - 1, 1) * sizeof(int));
        CK(cudaMemsetAsync(parent_.as<int>(), 0xff, sizeof(int), s));  // root's parent = -1
        if (K > 1) {
            CK(cudaMemsetAsync(counter_.as<int>(), 0, static_cast<std::size_t>(K - 1) * sizeof(int), s));
            build_radix<<<grid_for(K - 1, BS), BS, 0, s>>>(keys, start, K, nodeC_.as<float4>(), parent_.as<int>());
            CK_LAUNCH();
        }
        const float inv_theta = theta > 1.0 / RCRIT_MAX ? static_cast<float>(1.0 / theta) : RCRIT_MAX;
        moments_kernel<<<grid_for(K, BS), BS, 0, s>>>(posm_.as<float4>(), start, K, inv_theta, parent_.as<int>(),
                                                      counter_.as<int>(), nodeA_.as<float4>(), nodeB_.as<float4>(),
                                                      nodeC_.as<float4>(), bmin_.as<float4>(), bmax_.as<float4>());
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
    int key_sel_ = 0, scratch_sel_ = 1;
    std::size_t temp_bytes_ = 0;

    DevBuf x_, y_, z_, vx_, vy_, vz_, m_, id_, ax_, ay_, az_, pot_, posm_;
    DevBuf keys_[2], idx_[2];
    DevBuf nodeA_, nodeB_, nodeC_, bmin_, bmax_, parent_, counter_;
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

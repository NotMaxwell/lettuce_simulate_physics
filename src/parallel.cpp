#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef LETTUCE_HAVE_MPI
#include <mpi.h>
#endif

namespace par {

namespace {

int g_rank = 0, g_size = 1, g_local_rank = 0, g_local_size = 1;

// Subsample with stride k (k >= 1), masses scaled by k.
void subsample(const Bodies& in, uint64_t k, Bodies& out) {
    out = Bodies{};
    out.reserve(in.size() / k + 1);
    for (size_t i = 0; i < in.size(); i += k) {
        out.push(in.x[i], in.y[i], in.z[i], in.vx[i], in.vy[i], in.vz[i], double(in.m[i]) * double(k));
        out.id.back() = in.id[i];
    }
}

}  // namespace

void even_share(uint64_t n, uint64_t& first, uint64_t& last) {
    const uint64_t p = uint64_t(size()), r = uint64_t(rank());
    first = n / p * r + std::min(r, n % p);
    last = first + n / p + (r < n % p ? 1 : 0);
}

#ifndef LETTUCE_HAVE_MPI

void init(int*, char***) {}
void finalize() {}
void abort(int code) { std::exit(code); }
int rank() { return 0; }
int size() { return 1; }
int local_rank() { return 0; }
int local_size() { return 1; }
bool mpi_compiled() { return false; }
void barrier() {}
double sum(double v) { return v; }
uint64_t sum(uint64_t v) { return v; }
double max(double v) { return v; }
int broadcast(int v) { return v; }

std::unique_ptr<Backend> make_distributed(std::unique_ptr<Backend> local, const DistOptions&) { return local; }

void gather_sample(Backend& be, Bodies& out, uint64_t max_total) {
    Bodies all;
    be.download(all);
    uint64_t k = std::max<uint64_t>(1, (all.size() + max_total - 1) / std::max<uint64_t>(max_total, 1));
    if (k == 1) out = std::move(all);
    else subsample(all, k, out);
}

#else  // LETTUCE_HAVE_MPI

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void check(int rc, const char* what) {
    if (rc != MPI_SUCCESS) throw std::runtime_error(std::string("MPI error in ") + what);
}

// One body as it travels between ranks.
struct Rec {
    float x, y, z, vx, vy, vz, m;
    uint32_t id;
};
static_assert(sizeof(Rec) == 32);

void pack(const Bodies& b, size_t i, Rec& r) {
    r = {b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i], b.m[i], b.id[i]};
}

void append(Bodies& b, const Rec& r) {
    b.x.push_back(r.x); b.y.push_back(r.y); b.z.push_back(r.z);
    b.vx.push_back(r.vx); b.vy.push_back(r.vy); b.vz.push_back(r.vz);
    b.m.push_back(r.m); b.id.push_back(r.id);
}

uint64_t spread21(uint64_t v) {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8) & 0x100f00f00f00f00fULL;
    v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}

// Every rank sends send[r] to rank r and receives recv[r] from it. Large messages go in 1 GiB
// pieces: MPI counts are int.
void exchange_bytes(const std::vector<std::vector<char>>& send, std::vector<std::vector<char>>& recv) {
    const int P = g_size, me = g_rank;
    std::vector<uint64_t> ssz(P), rsz(P);
    for (int r = 0; r < P; ++r) ssz[r] = send[r].size();
    check(MPI_Alltoall(ssz.data(), 1, MPI_UINT64_T, rsz.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD), "Alltoall");
    recv.assign(P, {});
    constexpr uint64_t piece = uint64_t(1) << 30;
    std::vector<MPI_Request> reqs;
    for (int r = 0; r < P; ++r) {
        recv[r].resize(rsz[r]);
        if (r == me) continue;
        int tag = 0;
        for (uint64_t off = 0; off < rsz[r]; off += piece, ++tag) {
            reqs.emplace_back();
            check(MPI_Irecv(recv[r].data() + off, int(std::min(piece, rsz[r] - off)), MPI_BYTE, r, tag, MPI_COMM_WORLD,
                            &reqs.back()),
                  "Irecv");
        }
    }
    for (int r = 0; r < P; ++r) {
        if (r == me) continue;
        int tag = 0;
        for (uint64_t off = 0; off < ssz[r]; off += piece, ++tag) {
            reqs.emplace_back();
            check(MPI_Isend(send[r].data() + off, int(std::min(piece, ssz[r] - off)), MPI_BYTE, r, tag, MPI_COMM_WORLD,
                            &reqs.back()),
                  "Isend");
        }
    }
    recv[me] = send[me];
    if (!reqs.empty()) check(MPI_Waitall(int(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE), "Waitall");
}

template <class T>
void put(std::vector<char>& buf, const T* p, size_t n) {
    const char* c = reinterpret_cast<const char*>(p);
    buf.insert(buf.end(), c, c + n * sizeof(T));
}

std::vector<char> serialize(const LetTree& t) {
    std::vector<char> buf;
    uint64_t hdr[2] = {t.nodes.size(), t.bm.size()};
    buf.reserve(16 + t.nodes.size() * sizeof(LetNode) + t.bm.size() * 16);
    put(buf, hdr, 2);
    put(buf, t.nodes.data(), t.nodes.size());
    put(buf, t.bx.data(), t.bx.size());
    put(buf, t.by.data(), t.by.size());
    put(buf, t.bz.data(), t.bz.size());
    put(buf, t.bm.data(), t.bm.size());
    return buf;
}

void deserialize(const std::vector<char>& buf, LetTree& t) {
    t = LetTree{};
    if (buf.empty()) return;
    uint64_t hdr[2];
    if (buf.size() < sizeof hdr) throw std::runtime_error("LET message truncated");
    std::memcpy(hdr, buf.data(), sizeof hdr);
    const uint64_t nn = hdr[0], nb = hdr[1];
    if (buf.size() != sizeof hdr + nn * sizeof(LetNode) + nb * 16) throw std::runtime_error("LET message has the wrong size");
    const char* p = buf.data() + sizeof hdr;
    t.nodes.resize(nn);
    std::memcpy(t.nodes.data(), p, nn * sizeof(LetNode));
    p += nn * sizeof(LetNode);
    for (auto* v : {&t.bx, &t.by, &t.bz, &t.bm}) {
        v->resize(nb);
        std::memcpy(v->data(), p, nb * 4);
        p += nb * 4;
    }
}

class DistributedBackend final : public Backend {
public:
    DistributedBackend(std::unique_ptr<Backend> local, const DistOptions& o) : local_(std::move(local)), opt_(o) {}

    std::string name() const override {
        return "mpi: " + std::to_string(g_size) + " ranks, rank 0 " + local_->name();
    }

    void upload(const Bodies& b) override {
        local_->upload(b);
        calls_since_balance_ = opt_.rebalance_every;  // the first tree force call rebalances
    }
    void download(Bodies& b) override { local_->download(b); }

    void compute_forces(const ForceParams& p, bool potential) override {
        auto t0 = Clock::now();
        balance_ms_ = 0;
        if (p.method == ForceMethod::tree && calls_since_balance_ >= opt_.rebalance_every) {
            rebalance();
            calls_since_balance_ = 0;
        }
        ++calls_since_balance_;
        force_pass(p, potential);
        last_params_ = p;
        last_potential_ = potential;
        have_last_ = true;
        times_.force_ms = ms_since(t0);
    }

    void kick(double dt) override { local_->kick(dt); }
    void drift(double dt) override { local_->drift(dt); }

    Diag diagnostics() override {
        Diag d = local_->diagnostics();
        double v[12] = {d.mass, d.ke, d.pe, d.px, d.py, d.pz, d.lx, d.ly, d.lz,
                        d.cx * d.mass, d.cy * d.mass, d.cz * d.mass};
        double g[12];
        check(MPI_Allreduce(v, g, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "Allreduce");
        Diag o;
        o.mass = g[0]; o.ke = g[1]; o.pe = g[2]; o.has_pe = d.has_pe;
        o.px = g[3]; o.py = g[4]; o.pz = g[5]; o.lx = g[6]; o.ly = g[7]; o.lz = g[8];
        if (o.mass > 0) { o.cx = g[9] / o.mass; o.cy = g[10] / o.mass; o.cz = g[11] / o.mass; }
        return o;
    }

    // Direct sum over all ranks' bodies for the potential, then the last force evaluation again so
    // the accelerations the integrator holds are the method's own, not the direct ones.
    double exact_potential_energy(double G, double eps) override {
        ForceParams d;
        d.G = G;
        d.eps = eps;
        d.method = ForceMethod::direct;
        PhaseTimes keep = times_;
        force_pass(d, true);
        double pe = local_->diagnostics().pe;
        if (have_last_) force_pass(last_params_, last_potential_);
        times_ = keep;
        double g = 0;
        check(MPI_Allreduce(&pe, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "Allreduce");
        return g;
    }

    PhaseTimes times() const override { return times_; }

    // A distributed backend is never itself a rank of another one; these forward for completeness.
    Box3 local_bounds() override { return local_->local_bounds(); }
    void build_tree(const ForceParams& p) override { local_->build_tree(p); }
    LetTree export_let(const Box3& t, const ForceParams& p) override { return local_->export_let(t, p); }
    void compute_forces_with(const ForceParams& p, bool pot, const std::vector<const LetTree*>& r) override {
        local_->compute_forces_with(p, pot, r);
    }

private:
    void force_pass(const ForceParams& p, bool potential) {
        const int P = g_size, me = g_rank;
        local_->build_tree(p);
        auto tc = Clock::now();
        std::vector<std::vector<char>> send(P), recv;
        if (p.method == ForceMethod::direct) {
            // Everyone needs every body: one export, sent to all.
            std::vector<char> mine = serialize(local_->export_let(Box3{}, p));
            for (int r = 0; r < P; ++r)
                if (r != me) send[r] = mine;
        } else {
            Box3 box = local_->local_bounds();
            std::vector<Box3> boxes(P);
            check(MPI_Allgather(&box, sizeof(Box3), MPI_BYTE, boxes.data(), sizeof(Box3), MPI_BYTE, MPI_COMM_WORLD),
                  "Allgather");
            for (int r = 0; r < P; ++r)
                if (r != me) send[r] = serialize(local_->export_let(boxes[r], p));
        }
        exchange_bytes(send, recv);
        std::vector<LetTree> lets(P);
        std::vector<const LetTree*> remote;
        for (int r = 0; r < P; ++r) {
            if (r == me) continue;
            deserialize(recv[r], lets[r]);
            if (!lets[r].nodes.empty()) remote.push_back(&lets[r]);
        }
        double comm = ms_since(tc);
        local_->compute_forces_with(p, potential, remote);
        PhaseTimes lt = local_->times();
        uint64_t inter = lt.interactions, ginter = 0;
        check(MPI_Allreduce(&inter, &ginter, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD), "Allreduce");
        times_ = lt;
        times_.interactions = ginter;
        times_.comm_ms = comm;
        times_.balance_ms = balance_ms_;
    }

    // Re-split the bodies into contiguous Morton-key ranges of equal cost (sample sort).
    void rebalance() {
        auto t0 = Clock::now();
        const int P = g_size;
        Bodies b;
        local_->download(b);
        const size_t n = b.size();
        std::vector<float> cost;
        if (!local_->body_costs(cost) || cost.size() != n) cost.assign(n, 1.f);

        // Global cube from the local bounding boxes.
        float lo[3] = {3e38f, 3e38f, 3e38f}, hi[3] = {-3e38f, -3e38f, -3e38f};
        for (size_t i = 0; i < n; ++i) {
            lo[0] = std::min(lo[0], b.x[i]); hi[0] = std::max(hi[0], b.x[i]);
            lo[1] = std::min(lo[1], b.y[i]); hi[1] = std::max(hi[1], b.y[i]);
            lo[2] = std::min(lo[2], b.z[i]); hi[2] = std::max(hi[2], b.z[i]);
        }
        float glo[3], ghi[3];
        check(MPI_Allreduce(lo, glo, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD), "Allreduce");
        check(MPI_Allreduce(hi, ghi, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD), "Allreduce");
        double side = std::max({double(ghi[0]) - glo[0], double(ghi[1]) - glo[1], double(ghi[2]) - glo[2], 1e-30});
        const double scale = double((1u << 21) - 1) / side;
        std::vector<uint64_t> key(n);
        for (size_t i = 0; i < n; ++i) {
            auto q = [&](float v, float o) { return uint64_t(std::clamp((double(v) - o) * scale, 0.0, 2097151.0)); };
            key[i] = spread21(q(b.x[i], glo[0])) << 2 | spread21(q(b.y[i], glo[1])) << 1 | spread21(q(b.z[i], glo[2]));
        }
        std::vector<uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t c) { return key[a] < key[c]; });

        // Weighted samples at equal-cost quantiles of the local list; each carries local_cost/S.
        constexpr int S = 256;
        double local_cost = 0;
        for (float c : cost) local_cost += c;
        struct Sample { uint64_t key; double w; };
        std::vector<Sample> mine;
        if (n > 0) {
            double acc = 0;
            size_t j = 0;
            for (int s = 0; s < S; ++s) {
                double target = (s + 0.5) / S * local_cost;
                while (j + 1 < n && acc + cost[order[j]] < target) acc += cost[order[j++]];
                mine.push_back({key[order[j]], local_cost / S});
            }
        }
        int cnt = int(mine.size() * sizeof(Sample));
        std::vector<int> counts(P), displs(P);
        check(MPI_Allgather(&cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD), "Allgather");
        int total_bytes = 0;
        for (int r = 0; r < P; ++r) { displs[r] = total_bytes; total_bytes += counts[r]; }
        std::vector<Sample> all(size_t(total_bytes) / sizeof(Sample));
        check(MPI_Allgatherv(mine.data(), cnt, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE,
                             MPI_COMM_WORLD),
              "Allgatherv");
        std::sort(all.begin(), all.end(), [](const Sample& a, const Sample& c) { return a.key < c.key; });
        double wsum = 0;
        for (auto& s : all) wsum += s.w;
        // Splitter r: the first key that belongs to rank r + 1.
        std::vector<uint64_t> split;
        double acc = 0;
        size_t k = 0;
        for (int r = 1; r < P; ++r) {
            double target = wsum * r / P;
            while (k < all.size() && acc + all[k].w <= target) acc += all[k++].w;
            split.push_back(k < all.size() ? all[k].key : UINT64_MAX);
        }

        std::vector<std::vector<char>> send(P), recv;
        std::vector<size_t> per(P, 0);
        std::vector<int> dest(n);
        for (size_t i = 0; i < n; ++i) {
            dest[i] = int(std::upper_bound(split.begin(), split.end(), key[i]) - split.begin());
            ++per[dest[i]];
        }
        for (int r = 0; r < P; ++r) send[r].reserve(per[r] * sizeof(Rec));
        for (size_t i : order) {
            Rec rec;
            pack(b, i, rec);
            put(send[dest[i]], &rec, 1);
        }
        exchange_bytes(send, recv);
        Bodies nb;
        size_t total = 0;
        for (auto& v : recv) total += v.size() / sizeof(Rec);
        nb.reserve(total);
        for (auto& v : recv) {
            for (size_t off = 0; off + sizeof(Rec) <= v.size(); off += sizeof(Rec)) {
                Rec rec;
                std::memcpy(&rec, v.data() + off, sizeof rec);
                append(nb, rec);
            }
        }
        local_->upload(nb);
        balance_ms_ = ms_since(t0);
    }

    std::unique_ptr<Backend> local_;
    DistOptions opt_;
    uint64_t calls_since_balance_ = 0;
    PhaseTimes times_;
    double balance_ms_ = 0;
    ForceParams last_params_;
    bool last_potential_ = false, have_last_ = false;
};

}  // namespace

void init(int* argc, char*** argv) {
    int inited = 0;
    MPI_Initialized(&inited);
    // FUNNELED: only the main thread calls MPI; OpenMP and the live-view threads never do.
    int provided = 0;
    if (!inited) MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &g_local_rank);
    MPI_Comm_size(node, &g_local_size);
    MPI_Comm_free(&node);
}

void finalize() {
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin) MPI_Finalize();
}

void abort(int code) {
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited && g_size > 1) MPI_Abort(MPI_COMM_WORLD, code);
    std::exit(code);
}

int rank() { return g_rank; }
int size() { return g_size; }
int local_rank() { return g_local_rank; }
int local_size() { return g_local_size; }
bool mpi_compiled() { return true; }

void barrier() {
    if (g_size > 1) MPI_Barrier(MPI_COMM_WORLD);
}

double sum(double v) {
    double g = v;
    if (g_size > 1) MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return g;
}

uint64_t sum(uint64_t v) {
    uint64_t g = v;
    if (g_size > 1) MPI_Allreduce(&v, &g, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    return g;
}

double max(double v) {
    double g = v;
    if (g_size > 1) MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return g;
}

int broadcast(int v) {
    if (g_size > 1) MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return v;
}

std::unique_ptr<Backend> make_distributed(std::unique_ptr<Backend> local, const DistOptions& o) {
    if (g_size == 1) return local;
    return std::make_unique<DistributedBackend>(std::move(local), o);
}

void gather_sample(Backend& be, Bodies& out, uint64_t max_total) {
    Bodies mine;
    be.download(mine);
    uint64_t total = sum(uint64_t(mine.size()));
    uint64_t k = std::max<uint64_t>(1, (total + max_total - 1) / std::max<uint64_t>(max_total, 1));
    Bodies part;
    if (k == 1) part = std::move(mine);
    else subsample(mine, k, part);
    if (g_size == 1) {
        out = std::move(part);
        return;
    }
    std::vector<Rec> recs(part.size());
    for (size_t i = 0; i < part.size(); ++i) pack(part, i, recs[i]);
    int bytes = int(recs.size() * sizeof(Rec));
    std::vector<int> counts(g_size), displs(g_size);
    MPI_Gather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int tot = 0;
    for (int r = 0; r < g_size; ++r) { displs[r] = tot; tot += counts[r]; }
    std::vector<Rec> all(g_rank == 0 ? size_t(tot) / sizeof(Rec) : 0);
    MPI_Gatherv(recs.data(), bytes, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    out = Bodies{};
    if (g_rank != 0) return;
    out.reserve(all.size());
    for (auto& r : all) append(out, r);
}

#endif  // LETTUCE_HAVE_MPI

}  // namespace par

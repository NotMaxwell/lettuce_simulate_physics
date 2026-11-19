// CUDA backend, distributed support: local_bounds, build_tree, export_let, compute_forces_with,
// body_costs. Two backend instances stand in for two ranks. Skips without a usable device.
// Accelerations are read through the interface: zero velocities, force call, kick(1): v == a.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "backend.hpp"
#include "test.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

Bodies plummer(std::size_t n, uint64_t seed, double scale = 1.0, double shift_x = 0.0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto dir = [&](double r, double& x, double& y, double& z) {
        double c = 2 * U(rng) - 1, ph = 2 * kPi * U(rng), s = std::sqrt(1 - c * c);
        x = r * s * std::cos(ph); y = r * s * std::sin(ph); z = r * c;
    };
    Bodies b;
    b.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double r;
        do {
            r = 1 / std::sqrt(std::pow(U(rng), -2.0 / 3.0) - 1);
        } while (!(r < 20));
        double x, y, z;
        dir(r * scale, x, y, z);
        b.push(x + shift_x, y, z, 0, 0, 0, 1.0 / double(n));
    }
    return b;
}

void append(Bodies& dst, const Bodies& src, std::size_t i) {
    dst.x.push_back(src.x[i]); dst.y.push_back(src.y[i]); dst.z.push_back(src.z[i]);
    dst.vx.push_back(0.f); dst.vy.push_back(0.f); dst.vz.push_back(0.f);
    dst.m.push_back(src.m[i]); dst.id.push_back(src.id[i]);
}

// Halves keep their global ids. by_x: x < 0 / x >= 0 (disjoint boxes); otherwise random (the
// two boxes overlap almost completely).
void split(const Bodies& all, bool by_x, Bodies& h0, Bodies& h1) {
    std::mt19937_64 rng(99);
    for (std::size_t i = 0; i < all.size(); ++i) {
        const bool first = by_x ? all.x[i] < 0.f : (rng() & 1) == 0;
        append(first ? h0 : h1, all, i);
    }
}

ForceParams params(ForceMethod m, double eps = 0.01, double theta = 0.6) {
    ForceParams p;
    p.G = 1;
    p.eps = eps;
    p.theta = theta;
    p.method = m;
    return p;
}

// Accelerations after the force call already made on `be`, scattered by id into a 3*nid array.
void read_accel(Backend& be, std::vector<double>& a) {
    be.kick(1.0);
    Bodies out;
    be.download(out);
    for (std::size_t i = 0; i < out.size(); ++i) {
        a[3 * out.id[i]] = out.vx[i];
        a[3 * out.id[i] + 1] = out.vy[i];
        a[3 * out.id[i] + 2] = out.vz[i];
    }
}

double rel_rms(const std::vector<double>& a, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

bool all_finite(const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

struct Full {
    std::vector<double> a;
    double pe = 0;
};

Full single_direct(Backend& be, const Bodies& all) {
    Full f;
    f.a.assign(3 * all.size(), 0.0);
    be.upload(all);
    be.compute_forces(params(ForceMethod::direct), true);
    f.pe = be.diagnostics().pe;
    read_accel(be, f.a);
    return f;
}

struct TwoRank {
    std::vector<double> a;
    double pe = 0;
    uint64_t interactions[2] = {0, 0};
    double cost_sum[2] = {0, 0};
    bool cost_ok[2] = {false, false};
    std::size_t cost_size[2] = {0, 0};
};

// Each half gets its local forces plus the other half's LET, as the MPI layer does.
TwoRank two_rank(Backend& b0, Backend& b1, const Bodies& h0, const Bodies& h1, std::size_t nid,
                 const ForceParams& p, bool pot) {
    TwoRank r;
    r.a.assign(3 * nid, 0.0);
    Backend* be[2] = {&b0, &b1};
    be[0]->upload(h0);
    be[1]->upload(h1);
    be[0]->build_tree(p);
    be[1]->build_tree(p);
    const Box3 box0 = be[0]->local_bounds(), box1 = be[1]->local_bounds();
    const LetTree from1 = be[1]->export_let(box0, p);
    const LetTree from0 = be[0]->export_let(box1, p);
    be[0]->compute_forces_with(p, pot, {&from1});
    be[1]->compute_forces_with(p, pot, {&from0});
    for (int k = 0; k < 2; ++k) {
        r.interactions[k] = be[k]->times().interactions;
        if (pot) r.pe += be[k]->diagnostics().pe;
        std::vector<float> c;
        r.cost_ok[k] = be[k]->body_costs(c);
        r.cost_size[k] = c.size();
        for (float x : c) r.cost_sum[k] += x;
        read_accel(*be[k], r.a);
    }
    std::fprintf(stderr, "    LET sizes: %zu nodes/%zu bodies and %zu nodes/%zu bodies\n", from1.nodes.size(),
                 from1.bx.size(), from0.nodes.size(), from0.bx.size());
    return r;
}

// Every CUDA test starts here: no device (or no CUDA build) means skipped, not passed.
#define CUDA_BACKEND(be)                         \
    std::string why_##be;                        \
    auto be = make_cuda_backend(0, why_##be);    \
    if (!be) SKIP(why_##be)

}  // namespace

TEST("cuda let: local_bounds is the bounding box") {
    CUDA_BACKEND(be);
    Bodies b = plummer(10000, 11);
    be->upload(b);
    const Box3 box = be->local_bounds();
    for (int c = 0; c < 3; ++c) {
        const std::vector<float>& v = c == 0 ? b.x : c == 1 ? b.y : b.z;
        CHECK(box.lo[c] == *std::min_element(v.begin(), v.end()));
        CHECK(box.hi[c] == *std::max_element(v.begin(), v.end()));
    }
}

TEST("cuda let: compute_forces equals build_tree + compute_forces_with({})") {
    CUDA_BACKEND(be);
    Bodies b = plummer(20000, 12);
    for (ForceMethod m : {ForceMethod::tree, ForceMethod::direct}) {
        const ForceParams p = params(m);
        std::vector<double> a0(3 * b.size()), a1(3 * b.size());
        be->upload(b);
        be->compute_forces(p, true);
        const double pe0 = be->diagnostics().pe;
        read_accel(*be, a0);
        be->upload(b);
        be->build_tree(p);
        be->compute_forces_with(p, true, {});
        const double pe1 = be->diagnostics().pe;
        read_accel(*be, a1);
        CHECK(a0 == a1);
        CHECK(pe0 == pe1);
    }
}

TEST("cuda let: two halves with LETs vs single-backend direct, 20k Plummer") {
    CUDA_BACKEND(ref);
    CUDA_BACKEND(b0_);
    CUDA_BACKEND(b1);
    const Bodies all = plummer(20000, 13);
    const Full d = single_direct(*ref, all);
    for (bool by_x : {true, false}) {
        Bodies h0, h1;
        split(all, by_x, h0, h1);
        for (ForceMethod m : {ForceMethod::tree, ForceMethod::direct}) {
            for (bool pot : {false, true}) {
                const bool tree = m == ForceMethod::tree;
                std::fprintf(stderr, "  %s split, %s, potential %d (%zu + %zu bodies)\n", by_x ? "x" : "random",
                             tree ? "tree theta 0.6" : "direct", int(pot), h0.size(), h1.size());
                const TwoRank r = two_rank(*b0_, *b1, h0, h1, all.size(), params(m), pot);
                const double e = rel_rms(r.a, d.a);
                std::fprintf(stderr, "    rel rms %.3g, interactions %llu + %llu\n", e,
                             (unsigned long long)r.interactions[0], (unsigned long long)r.interactions[1]);
                CHECK(all_finite(r.a));
                CHECK(e <= (tree ? 1.5e-3 : 1e-5));
                if (pot) {
                    std::fprintf(stderr, "    pe %.9g vs %.9g\n", r.pe, d.pe);
                    CHECK_NEAR(r.pe / d.pe, 1.0, tree ? 1.5e-3 : 1e-5);
                }
                const std::size_t n[2] = {h0.size(), h1.size()};
                for (int k = 0; k < 2; ++k) {
                    CHECK(r.cost_ok[k]);
                    CHECK(r.cost_size[k] == n[k]);
                    CHECK(r.cost_sum[k] > 0);
                    // Counts are exact integers; their float sum is exact while each is < 2^24.
                    CHECK(r.cost_sum[k] == double(r.interactions[k]));
                }
                if (!tree) {
                    const uint64_t all_pairs = uint64_t(all.size()) * uint64_t(all.size() - 1);
                    CHECK(r.interactions[0] + r.interactions[1] == all_pairs);
                }
            }
        }
    }
}

TEST("cuda let: empty LET and a LET for a far-away box") {
    CUDA_BACKEND(be);
    CUDA_BACKEND(src_);
    const ForceParams p = params(ForceMethod::tree);
    const Bodies b = plummer(5000, 14);

    // An empty LET (what a rank without bodies exports) changes nothing.
    std::vector<double> a0(3 * b.size()), a1(3 * b.size());
    be->upload(b);
    be->compute_forces(p, true);
    const double pe0 = be->diagnostics().pe;
    read_accel(*be, a0);
    const LetTree empty;
    be->upload(b);
    be->build_tree(p);
    be->compute_forces_with(p, true, {&empty, &empty});
    const double pe1 = be->diagnostics().pe;
    read_accel(*be, a1);
    CHECK(a0 == a1);
    CHECK(pe0 == pe1);
    Bodies none;
    src_->upload(none);
    src_->build_tree(p);
    Box3 any;
    CHECK(src_->export_let(any, p).nodes.empty());

    // Sources: 20k Plummer. Targets: a massless, compact cluster 60 length units away, so their
    // forces come from the LET alone and most exported nodes are terminal.
    const Bodies srcb = plummer(20000, 15);
    Bodies tgt = plummer(2000, 16, 0.2, 60.0);
    std::fill(tgt.m.begin(), tgt.m.end(), 0.f);
    src_->upload(srcb);
    src_->build_tree(p);
    be->upload(tgt);
    be->build_tree(p);
    const LetTree let = src_->export_let(be->local_bounds(), p);
    std::size_t terminal = 0;
    for (const LetNode& nd : let.nodes) terminal += nd.first_child < 0 && nd.body_count == 0;
    std::fprintf(stderr, "  far LET: %zu nodes (%zu terminal), %zu bodies\n", let.nodes.size(), terminal, let.bx.size());
    CHECK(!let.nodes.empty());
    CHECK(2 * terminal > let.nodes.size());
    be->compute_forces_with(p, true, {&let});
    std::vector<double> a(3 * tgt.size()), ref(3 * tgt.size(), 0.0);
    read_accel(*be, a);
    const double eps2 = p.eps * p.eps;
    for (std::size_t i = 0; i < tgt.size(); ++i)
        for (std::size_t j = 0; j < srcb.size(); ++j) {
            double dx = double(srcb.x[j]) - tgt.x[i], dy = double(srcb.y[j]) - tgt.y[i], dz = double(srcb.z[j]) - tgt.z[i];
            double inv = 1 / std::sqrt(dx * dx + dy * dy + dz * dz + eps2), f = srcb.m[j] * inv * inv * inv;
            ref[3 * tgt.id[i]] += f * dx; ref[3 * tgt.id[i] + 1] += f * dy; ref[3 * tgt.id[i] + 2] += f * dz;
        }
    const double e = rel_rms(a, ref);
    std::fprintf(stderr, "  far LET rel rms %.3g\n", e);
    CHECK(all_finite(a));
    CHECK(e <= 1.5e-3);
}

TEST("cuda let: LETs cross between the cpu and cuda backends") {
    CUDA_BACKEND(cu);
    CUDA_BACKEND(cu_rx);
    auto cpu = make_cpu_backend(0), cpu_rx = make_cpu_backend(0);
    const Bodies all = plummer(20000, 17);
    Bodies h0, h1;
    split(all, true, h0, h1);
    Backend* exporters[2] = {cpu.get(), cu.get()};
    const char* names[2] = {"cpu", "cuda"};

    // Direct-method LET (one always-open root holding every body), massless receivers: exact sums.
    {
        const ForceParams p = params(ForceMethod::direct);
        Bodies rx = h0;
        std::fill(rx.m.begin(), rx.m.end(), 0.f);
        for (int e = 0; e < 2; ++e) {
            exporters[e]->upload(h1);
            exporters[e]->build_tree(p);
            cu_rx->upload(rx);
            const LetTree let = exporters[e]->export_let(cu_rx->local_bounds(), p);
            std::vector<double> ag(3 * all.size(), 0.0), ac(3 * all.size(), 0.0);
            cu_rx->build_tree(p);
            cu_rx->compute_forces_with(p, false, {&let});
            read_accel(*cu_rx, ag);
            cpu_rx->upload(rx);
            cpu_rx->build_tree(p);
            cpu_rx->compute_forces_with(p, false, {&let});
            read_accel(*cpu_rx, ac);
            const double d = rel_rms(ag, ac);
            std::fprintf(stderr, "  direct LET from %s: cuda vs cpu receiver rel rms %.3g\n", names[e], d);
            CHECK(all_finite(ag));
            CHECK(d <= 1e-5);
        }
    }

    // Tree LET for single-body receivers: the target box is a point, so the receiver's walk makes
    // exactly the exporter's decisions (terminal nodes as multipoles, opened leaves as bodies)
    // and the two receiver backends must agree to rounding, quadrupoles included.
    {
        const ForceParams p = params(ForceMethod::tree);
        for (int e = 0; e < 2; ++e) {
            exporters[e]->upload(h1);
            exporters[e]->build_tree(p);
            std::vector<double> ag, ac;
            std::size_t terminal = 0, nodes = 0;
            for (std::size_t s = 0; s < h0.size(); s += h0.size() / 16) {
                Bodies one;
                append(one, h0, s);
                one.id[0] = 0;
                cu_rx->upload(one);
                cu_rx->build_tree(p);
                const LetTree let = exporters[e]->export_let(cu_rx->local_bounds(), p);
                nodes += let.nodes.size();
                for (const LetNode& nd : let.nodes) terminal += nd.first_child < 0 && nd.body_count == 0;
                std::vector<double> g(3), c(3);
                cu_rx->compute_forces_with(p, false, {&let});
                read_accel(*cu_rx, g);
                cpu_rx->upload(one);
                cpu_rx->build_tree(p);
                cpu_rx->compute_forces_with(p, false, {&let});
                read_accel(*cpu_rx, c);
                ag.insert(ag.end(), g.begin(), g.end());
                ac.insert(ac.end(), c.begin(), c.end());
            }
            const double d = rel_rms(ag, ac);
            std::fprintf(stderr, "  tree LET from %s (%zu nodes, %zu terminal): cuda vs cpu receiver rel rms %.3g\n",
                         names[e], nodes, terminal, d);
            CHECK(terminal > 0);
            CHECK(all_finite(ag));
            CHECK(d <= 1e-5);
        }
    }
}

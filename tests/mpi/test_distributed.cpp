// Distributed force evaluation on the CPU backend: every rank owns part of the bodies; results are
// gathered on rank 0 and checked against a single-process backend over all bodies.
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "backend.hpp"
#include "config.hpp"
#include "parallel.hpp"
#include "scenarios.hpp"
#include "test.hpp"

namespace {

Bodies plummer(uint64_t n, uint64_t seed) {
    Config c;
    c.scenario = "plummer";
    c.n = n;
    c.seed = seed;
    return build_scenario(c);
}

Bodies share(const Bodies& all) {
    uint64_t first, last;
    par::even_share(all.size(), first, last);
    Bodies b;
    for (uint64_t i = first; i < last; ++i) {
        b.push(all.x[i], all.y[i], all.z[i], all.vx[i], all.vy[i], all.vz[i], all.m[i]);
        b.id.back() = all.id[i];
    }
    return b;
}

Bodies at_rest(Bodies b) {
    std::fill(b.vx.begin(), b.vx.end(), 0.f);
    std::fill(b.vy.begin(), b.vy.end(), 0.f);
    std::fill(b.vz.begin(), b.vz.end(), 0.f);
    return b;
}

struct Acc { double x, y, z; };

// Accelerations by id: zero velocities, kick by 1, read velocities back. `gathered`: collect every
// rank's bodies on rank 0 (collective); otherwise just this backend's (a rank-0-only reference).
std::map<uint32_t, Acc> accelerations(Backend& be, const ForceParams& p, const Bodies& mine, bool gathered = true) {
    be.upload(at_rest(mine));
    be.compute_forces(p, false);
    be.kick(1.0);
    Bodies got;
    if (gathered) par::gather_sample(be, got, UINT64_MAX / 2);
    else be.download(got);
    std::map<uint32_t, Acc> out;
    for (size_t i = 0; i < got.size(); ++i) out[got.id[i]] = {got.vx[i], got.vy[i], got.vz[i]};
    return out;
}

double rel_rms(const std::map<uint32_t, Acc>& a, const std::map<uint32_t, Acc>& ref) {
    double num = 0, den = 0;
    for (auto& [id, r] : ref) {
        auto it = a.find(id);
        if (it == a.end()) return 1e30;
        const Acc& v = it->second;
        num += (v.x - r.x) * (v.x - r.x) + (v.y - r.y) * (v.y - r.y) + (v.z - r.z) * (v.z - r.z);
        den += r.x * r.x + r.y * r.y + r.z * r.z;
    }
    return std::sqrt(num / den);
}

std::unique_ptr<Backend> distributed(uint64_t rebalance_every = 10) {
    par::DistOptions o;
    o.rebalance_every = rebalance_every;
    return par::make_distributed(make_cpu_backend(1), o);
}

ForceParams params(ForceMethod m, double theta = 0.6) {
    ForceParams p;
    p.G = 1;
    p.eps = 0.01;
    p.theta = theta;
    p.method = m;
    return p;
}

}  // namespace

TEST("mpi: distributed direct forces equal single-process direct") {
    Bodies all = plummer(3000, 5);
    auto be = distributed();
    auto got = accelerations(*be, params(ForceMethod::direct), share(all));
    if (par::root()) {
        auto ref_be = make_cpu_backend(1);
        auto ref = accelerations(*ref_be, params(ForceMethod::direct), all, false);
        CHECK(got.size() == all.size());
        CHECK(rel_rms(got, ref) <= 1e-5);
    }
}

TEST("mpi: distributed tree forces match direct, every body exactly once") {
    Bodies all = plummer(20000, 7);
    auto be = distributed();
    for (double theta : {0.6, 0.4}) {
        auto got = accelerations(*be, params(ForceMethod::tree, theta), share(all));
        if (par::root()) {
            auto ref_be = make_cpu_backend(0);
            auto ref = accelerations(*ref_be, params(ForceMethod::direct), all, false);
            CHECK(got.size() == all.size());
            double e = rel_rms(got, ref);
            std::fprintf(stderr, "  theta %.1f: rel rms %.3g over %d ranks\n", theta, e, par::size());
            CHECK(e <= (theta > 0.5 ? 1.5e-3 : 4e-4));
        }
    }
}

TEST("mpi: global diagnostics and exact potential equal single-process values") {
    Bodies all = plummer(4000, 11);
    auto be = distributed();
    be->upload(share(all));
    be->compute_forces(params(ForceMethod::tree), true);
    Diag d = be->diagnostics();
    double pe = be->exact_potential_energy(1, 0.01);
    if (par::root()) {
        auto ref = make_cpu_backend(0);
        ref->upload(all);
        ref->compute_forces(params(ForceMethod::direct), true);
        Diag r = ref->diagnostics();
        double rpe = ref->exact_potential_energy(1, 0.01);
        CHECK_NEAR(d.mass, r.mass, 1e-9 * r.mass);
        CHECK_NEAR(d.ke, r.ke, 1e-9 * r.ke);
        CHECK_NEAR(d.lz, r.lz, 1e-6 * std::fabs(r.ke) + 1e-12);
        CHECK_NEAR(pe, rpe, 1e-6 * std::fabs(rpe));
        // Tree potential from the force pass, within the method's error.
        CHECK(d.has_pe);
        CHECK_NEAR(d.pe, rpe, 2e-3 * std::fabs(rpe));
    }
}

TEST("mpi: leapfrog run with frequent rebalancing conserves energy, momentum and bodies") {
    Bodies all = plummer(4000, 13);
    auto be = distributed(7);
    be->upload(share(all));
    ForceParams p = params(ForceMethod::tree, 0.5);
    const double dt = 1.0 / 256;
    be->compute_forces(p, false);
    Diag d0 = be->diagnostics();
    double e0 = d0.ke + be->exact_potential_energy(1, 0.01);
    for (int s = 0; s < 100; ++s) {
        be->kick(0.5 * dt);
        be->drift(dt);
        be->compute_forces(p, false);
        be->kick(0.5 * dt);
    }
    Diag d1 = be->diagnostics();
    double e1 = d1.ke + be->exact_potential_energy(1, 0.01);
    Bodies got;
    par::gather_sample(*be, got, UINT64_MAX / 2);
    if (par::root()) {
        double de = std::fabs((e1 - e0) / e0);
        std::fprintf(stderr, "  |dE/E0| = %.3g after 100 steps, %d ranks\n", de, par::size());
        CHECK(de < 3e-3);
        CHECK(std::sqrt(d1.px * d1.px + d1.py * d1.py + d1.pz * d1.pz) < 1e-4);
        std::set<uint32_t> ids(got.id.begin(), got.id.end());
        CHECK(got.size() == all.size());
        CHECK(ids.size() == all.size());
    }
    // Every rank still holds bodies after rebalancing (cost balancing must not starve a rank).
    Bodies mine;
    be->download(mine);
    CHECK(mine.size() > 0);
}

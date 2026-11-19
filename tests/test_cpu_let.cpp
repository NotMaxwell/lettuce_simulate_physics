// CPU backend, distributed support: forces from a local tree plus locally essential trees (LETs)
// exported by other "ranks" (here: other backend instances in one process) must match a single
// direct summation over all bodies.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "backend.hpp"
#include "test.hpp"

namespace {

// Plummer sphere in Henon units (G = M = 1), Aarseth-Henon-Wielen positions; velocities are not needed.
Bodies plummer_let(std::size_t n, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    const double a = 3 * M_PI / 16;
    Bodies out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double r;
        do r = 1 / std::sqrt(std::pow(u(rng), -2.0 / 3) - 1); while (r > 20);
        double ct = 2 * u(rng) - 1, st = std::sqrt(1 - ct * ct), ph = 2 * M_PI * u(rng);
        out.push(a * r * st * std::cos(ph), a * r * st * std::sin(ph), a * r * ct, 0, 0, 0, 1.0 / double(n));
    }
    return out;
}

const Bodies& sample() {
    static const Bodies b = plummer_let(20000, 7);
    return b;
}

ForceParams params(ForceMethod m, double theta) {
    ForceParams fp;
    fp.method = m; fp.theta = theta; fp.eps = 0.01;
    return fp;
}

double rel_rms(const std::vector<double>& a, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t k = 0; k < a.size(); ++k) { num += (a[k] - ref[k]) * (a[k] - ref[k]); den += ref[k] * ref[k]; }
    return std::sqrt(num / den);
}

// With v = 0 before the force call, kick(1) leaves v == a. Writes a[3 id + k].
void read_accel(Backend& be, std::vector<double>& a) {
    be.kick(1.0);
    Bodies o;
    be.download(o);
    for (std::size_t i = 0; i < o.size(); ++i) {
        a[3 * o.id[i]] = o.vx[i]; a[3 * o.id[i] + 1] = o.vy[i]; a[3 * o.id[i] + 2] = o.vz[i];
    }
}

struct Reference { std::vector<double> a; double pe; };

const Reference& reference() {
    static const Reference r = [] {
        Reference ref;
        auto be = make_cpu_backend(0);
        be->upload(sample());
        be->compute_forces(params(ForceMethod::direct, 0), false);
        ref.a.resize(3 * sample().size());
        read_accel(*be, ref.a);
        ref.pe = be->exact_potential_energy(1.0, 0.01);
        return ref;
    }();
    return r;
}

Bodies subset(const Bodies& b, const std::vector<int>& part, int p) {
    Bodies o;
    for (std::size_t i = 0; i < b.size(); ++i)
        if (part[i] == p) o.push(b.x[i], b.y[i], b.z[i], 0, 0, 0, b.m[i]);
    // push() numbers ids from 0; restore the global ones so results can be matched.
    std::size_t k = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
        if (part[i] == p) o.id[k++] = b.id[i];
    return o;
}

bool valid_let(const LetTree& t) {
    const int32_t nn = int32_t(t.nodes.size()), nb = int32_t(t.bx.size());
    if (t.by.size() != t.bx.size() || t.bz.size() != t.bx.size() || t.bm.size() != t.bx.size()) return false;
    for (int32_t k = 0; k < nn; ++k) {
        const LetNode& n = t.nodes[k];
        if (!(n.mass > 0)) return false;
        if (n.first_child >= 0 && (n.first_child <= k || n.child_count <= 0 || n.first_child + n.child_count > nn))
            return false;
        if (n.body_count > 0 && (n.body_first < 0 || n.body_first + n.body_count > nb)) return false;
    }
    return true;
}

struct Result { double err, pe; std::size_t nodes = 0, bodies = 0; };

// Every part gets its own backend; forces on a part = its local tree (or direct) + the LETs of all other parts.
Result distributed(const std::vector<int>& part, int nparts, const ForceParams& fp) {
    const Bodies& b = sample();
    std::vector<std::unique_ptr<Backend>> be;
    std::vector<Box3> box;
    for (int p = 0; p < nparts; ++p) {
        be.push_back(make_cpu_backend(0));
        be[p]->upload(subset(b, part, p));
        be[p]->build_tree(fp);
        box.push_back(be[p]->local_bounds());
    }
    Result r;
    std::vector<std::vector<LetTree>> let(nparts, std::vector<LetTree>(nparts));
    for (int i = 0; i < nparts; ++i)
        for (int j = 0; j < nparts; ++j) {
            if (i == j) continue;
            let[i][j] = be[j]->export_let(box[i], fp);
            CHECK(valid_let(let[i][j]));
            r.nodes += let[i][j].nodes.size();
            r.bodies += let[i][j].bx.size();
        }
    std::vector<double> a(3 * b.size());
    r.pe = 0;
    for (int i = 0; i < nparts; ++i) {
        std::vector<const LetTree*> remote;
        for (int j = 0; j < nparts; ++j)
            if (j != i) remote.push_back(&let[i][j]);
        be[i]->compute_forces_with(fp, true, remote);
        Diag d = be[i]->diagnostics();
        CHECK(d.has_pe);
        r.pe += d.pe;
        read_accel(*be[i], a);
    }
    r.err = rel_rms(a, reference().a);
    return r;
}

std::vector<int> split_x() {
    const Bodies& b = sample();
    std::vector<int> part(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) part[i] = b.x[i] < 0 ? 0 : 1;
    return part;
}

std::vector<int> split_random() {
    std::mt19937 rng(11);
    std::vector<int> part(sample().size());
    for (auto& p : part) p = int(rng() & 1);
    return part;
}

std::vector<int> split_quadrants() {
    const Bodies& b = sample();
    std::vector<int> part(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) part[i] = (b.x[i] < 0 ? 0 : 1) + (b.y[i] < 0 ? 0 : 2);
    return part;
}

}  // namespace

TEST("cpu let: tree LETs match direct over all bodies (x, random, 4-way splits)") {
    const Reference& ref = reference();
    struct Split { const char* name; std::vector<int> part; int n; };
    const Split splits[] = {{"x", split_x(), 2}, {"random", split_random(), 2}, {"4-way", split_quadrants(), 4}};
    for (const Split& s : splits)
        for (double theta : {0.6, 0.4}) {
            Result r = distributed(s.part, s.n, params(ForceMethod::tree, theta));
            double pe_rel = std::fabs(r.pe - ref.pe) / std::fabs(ref.pe);
            std::fprintf(stderr, "  %-6s theta %.1f: rel rms %.3g, pe rel %.3g, LETs %zu nodes + %zu bodies\n", s.name,
                         theta, r.err, pe_rel, r.nodes, r.bodies);
            CHECK(r.err <= (theta > 0.5 ? 1.5e-3 : 4e-4));
            CHECK(pe_rel <= 2e-3);
        }
}

TEST("cpu let: direct-method LETs reproduce direct forces") {
    const Reference& ref = reference();
    for (auto [part, n] : {std::pair{split_x(), 2}, std::pair{split_quadrants(), 4}}) {
        Result r = distributed(part, n, params(ForceMethod::direct, 0));
        double pe_rel = std::fabs(r.pe - ref.pe) / std::fabs(ref.pe);
        std::fprintf(stderr, "  direct %d parts: rel rms %.3g, pe rel %.3g\n", n, r.err, pe_rel);
        CHECK(r.err <= 1e-6);
        CHECK(pe_rel <= 1e-5);
        CHECK(r.bodies == std::size_t(n - 1) * sample().size());
    }
}

TEST("cpu let: empty remote list equals compute_forces bit-for-bit") {
    for (ForceMethod m : {ForceMethod::tree, ForceMethod::direct}) {
        const ForceParams fp = params(m, 0.6);
        auto a = make_cpu_backend(0), b = make_cpu_backend(0);
        a->upload(sample());
        b->upload(sample());
        a->compute_forces(fp, true);
        b->build_tree(fp);
        b->compute_forces_with(fp, true, {});
        // The per-body potentials are identical, but the OpenMP reduction that sums them may combine
        // thread partials in a different order each run, so the total agrees only to rounding.
        const double pa = a->diagnostics().pe, pb = b->diagnostics().pe;
        CHECK_NEAR(pa, pb, 1e-12 * std::fabs(pa));
        a->kick(1.0);
        b->kick(1.0);
        Bodies oa, ob;
        a->download(oa);
        b->download(ob);
        CHECK(oa.id == ob.id && oa.vx == ob.vx && oa.vy == ob.vy && oa.vz == ob.vz);
        CHECK(a->times().interactions == b->times().interactions);
    }
}

TEST("cpu let: LET for a far box is a few terminal nodes") {
    auto be = make_cpu_backend(0);
    be->upload(sample());
    const ForceParams fp = params(ForceMethod::tree, 0.6);
    be->build_tree(fp);
    Box3 far;
    far.lo[0] = far.lo[1] = far.lo[2] = 200;
    far.hi[0] = far.hi[1] = far.hi[2] = 201;
    LetTree t = be->export_let(far, fp);
    std::fprintf(stderr, "  far-box LET: %zu nodes, %zu bodies\n", t.nodes.size(), t.bx.size());
    CHECK(!t.nodes.empty() && t.nodes.size() <= 16);
    CHECK(t.bx.empty());
    CHECK(valid_let(t));
    double m = 0;
    for (const LetNode& n : t.nodes)
        if (n.first_child < 0) { CHECK(n.body_count == 0); m += n.mass; }
    CHECK_NEAR(m, 1.0, 1e-5);

    // A box containing everything needs the whole tree down to the leaves.
    LetTree all = be->export_let(be->local_bounds(), fp);
    CHECK(valid_let(all));
    CHECK(all.bx.size() == sample().size());
}

TEST("cpu let: body_costs follow the density") {
    const Bodies& b = sample();
    auto be = make_cpu_backend(0);
    std::vector<float> cost;
    CHECK(!be->body_costs(cost));
    be->upload(b);
    be->compute_forces(params(ForceMethod::tree, 0.6), false);
    CHECK(be->body_costs(cost));
    CHECK(cost.size() == b.size());
    Bodies o;
    be->download(o);
    double sum = 0, core = 0, outer = 0;
    int ncore = 0, nouter = 0;
    for (std::size_t i = 0; i < o.size(); ++i) {
        sum += cost[i];
        double r = std::sqrt(double(o.x[i]) * o.x[i] + double(o.y[i]) * o.y[i] + double(o.z[i]) * o.z[i]);
        if (r < 0.2) { core += cost[i]; ++ncore; }
        if (r > 2.0) { outer += cost[i]; ++nouter; }
    }
    std::fprintf(stderr, "  body cost: mean %.0f, core (r<0.2) %.0f, outskirts (r>2) %.0f\n", sum / double(o.size()),
                 core / ncore, outer / nouter);
    CHECK(sum > 0);
    CHECK(ncore > 0 && nouter > 0 && core / ncore > outer / nouter);
    CHECK(std::uint64_t(sum) == be->times().interactions);
}

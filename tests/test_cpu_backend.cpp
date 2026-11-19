// CPU backend: force accuracy (direct vs double oracle, tree vs direct), integration and
// conservation, ordering/identity after tree sorts, degenerate inputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "backend.hpp"
#include "test.hpp"

namespace {

// Plummer sphere in Henon units (G = M = 1, E = -1/4), Aarseth-Henon-Wielen sampling.
Bodies plummer(std::size_t n, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    const double a = 3 * M_PI / 16;
    std::vector<double> p(6 * n);
    double cm[6] = {};
    for (std::size_t i = 0; i < n; ++i) {
        double r;
        do r = 1 / std::sqrt(std::pow(u(rng), -2.0 / 3) - 1); while (r > 20);
        double ct = 2 * u(rng) - 1, st = std::sqrt(1 - ct * ct), ph = 2 * M_PI * u(rng);
        double q, g;
        do { q = u(rng); g = 0.1 * u(rng); } while (g > q * q * std::pow(1 - q * q, 3.5));
        double v = q * std::sqrt(2.0) * std::pow(1 + r * r, -0.25);
        double cv = 2 * u(rng) - 1, sv = std::sqrt(1 - cv * cv), pv = 2 * M_PI * u(rng);
        // Scale from the a = 1 model to Henon units: r *= a, v /= sqrt(a).
        double* b = &p[6 * i];
        b[0] = a * r * st * std::cos(ph); b[1] = a * r * st * std::sin(ph); b[2] = a * r * ct;
        double vs = v / std::sqrt(a);
        b[3] = vs * sv * std::cos(pv); b[4] = vs * sv * std::sin(pv); b[5] = vs * cv;
        for (int k = 0; k < 6; ++k) cm[k] += b[k] / double(n);
    }
    Bodies out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double* b = &p[6 * i];
        out.push(b[0] - cm[0], b[1] - cm[1], b[2] - cm[2], b[3] - cm[3], b[4] - cm[4], b[5] - cm[5], 1.0 / double(n));
    }
    return out;
}

// Accelerations by id: upload with v = 0, compute, kick(1), read v.
std::vector<double> accel(Backend& be, Bodies b, const ForceParams& fp) {
    std::fill(b.vx.begin(), b.vx.end(), 0.f);
    std::fill(b.vy.begin(), b.vy.end(), 0.f);
    std::fill(b.vz.begin(), b.vz.end(), 0.f);
    be.upload(b);
    be.compute_forces(fp, false);
    be.kick(1.0);
    Bodies o;
    be.download(o);
    std::vector<double> a(3 * o.size());
    for (std::size_t i = 0; i < o.size(); ++i) {
        a[3 * o.id[i]] = o.vx[i]; a[3 * o.id[i] + 1] = o.vy[i]; a[3 * o.id[i] + 2] = o.vz[i];
    }
    return a;
}

// sqrt(sum |a - ref|^2 / sum |ref|^2)
double rel_rms(const std::vector<double>& a, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t k = 0; k < a.size(); ++k) { num += (a[k] - ref[k]) * (a[k] - ref[k]); den += ref[k] * ref[k]; }
    return std::sqrt(num / den);
}

double energy(Backend& be, const ForceParams& fp) {
    be.compute_forces(fp, true);
    Diag d = be.diagnostics();
    CHECK(d.has_pe);
    return d.ke + d.pe;
}

void kdk(Backend& be, const ForceParams& fp, double dt, int steps) {
    for (int s = 0; s < steps; ++s) {
        be.kick(0.5 * dt);
        be.drift(dt);
        be.compute_forces(fp, false);
        be.kick(0.5 * dt);
    }
}

ForceParams tree(double theta, double eps = 0.01) {
    ForceParams fp;
    fp.method = ForceMethod::tree; fp.theta = theta; fp.eps = eps;
    return fp;
}

bool all_finite(const Bodies& b) {
    for (std::size_t i = 0; i < b.size(); ++i)
        if (!std::isfinite(b.x[i]) || !std::isfinite(b.vx[i]) || !std::isfinite(b.vy[i]) || !std::isfinite(b.vz[i]))
            return false;
    return true;
}

}  // namespace

TEST("cpu: name") {
    auto be = make_cpu_backend(3);
    CHECK(be->name() == "cpu: 3 threads (OpenMP)");
}

TEST("cpu: direct matches double brute force (10k Plummer)") {
    Bodies b = plummer(10000, 1);
    const std::size_t n = b.size();
    const double e2 = 0.01 * 0.01;
    std::vector<double> ref(3 * n);
#pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < long(n); ++i) {
        double ax = 0, ay = 0, az = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == std::size_t(i)) continue;
            double dx = double(b.x[j]) - b.x[i], dy = double(b.y[j]) - b.y[i], dz = double(b.z[j]) - b.z[i];
            double r2 = dx * dx + dy * dy + dz * dz + e2, f = double(b.m[j]) / (r2 * std::sqrt(r2));
            ax += f * dx; ay += f * dy; az += f * dz;
        }
        ref[3 * i] = ax; ref[3 * i + 1] = ay; ref[3 * i + 2] = az;
    }
    auto be = make_cpu_backend(0);
    ForceParams fp;
    fp.eps = 0.01;
    double err = rel_rms(accel(*be, b, fp), ref);
    std::fprintf(stderr, "  direct vs double: rel rms %.3g\n", err);
    CHECK(err <= 1e-5);
}

TEST("cpu: tree vs direct force error and tree potential (10k Plummer)") {
    Bodies b = plummer(10000, 2);
    auto be = make_cpu_backend(0);
    ForceParams fd;
    fd.eps = 0.01;
    auto ad = accel(*be, b, fd);
    double e6 = rel_rms(accel(*be, b, tree(0.6)), ad);
    double e4 = rel_rms(accel(*be, b, tree(0.4)), ad);
    std::fprintf(stderr, "  tree rel rms: theta=0.6 %.3g, theta=0.4 %.3g\n", e6, e4);
    CHECK(e6 <= 1e-3);
    CHECK(e4 <= 3e-4);

    be->upload(b);
    be->compute_forces(tree(0.5), true);
    Diag d = be->diagnostics();
    double exact = be->exact_potential_energy(1.0, 0.01);
    std::fprintf(stderr, "  tree pe %.8g exact %.8g rel %.3g\n", d.pe, exact, std::fabs(d.pe / exact - 1));
    CHECK(d.has_pe);
    CHECK(std::fabs(d.pe / exact - 1) <= 1e-3);
    CHECK(be->times().interactions > 0);

    // Direct potential agrees with the double oracle too.
    be->upload(b);
    be->compute_forces(fd, true);
    CHECK(std::fabs(be->diagnostics().pe / exact - 1) <= 1e-5);
}

TEST("cpu: two-body circular orbit, KDK, direct and tree") {
    for (ForceMethod meth : {ForceMethod::direct, ForceMethod::tree}) {
        const double m1 = 1, m2 = 1e-3, M = m1 + m2, a = 1;
        const double v = std::sqrt(M / a), T = 2 * M_PI * std::sqrt(a * a * a / M);
        Bodies b;
        b.push(-m2 / M * a, 0, 0, 0, -m2 / M * v, 0, m1);
        b.push(m1 / M * a, 0, 0, 0, m1 / M * v, 0, m2);
        Bodies start = b;
        auto be = make_cpu_backend(2);
        be->upload(b);
        ForceParams fp;
        fp.eps = 0; fp.method = meth; fp.theta = 0.5;
        double e0 = energy(*be, fp);
        const int steps = 1000;
        const double dt = T / steps;
        kdk(*be, fp, dt, steps);
        double e1 = energy(*be, fp);
        Bodies o;
        be->download(o);
        double maxd = 0;
        for (std::size_t i = 0; i < 2; ++i) {
            uint32_t k = o.id[i];
            maxd = std::max({maxd, std::fabs(double(o.x[i]) - start.x[k]), std::fabs(double(o.y[i]) - start.y[k]),
                             std::fabs(double(o.z[i]) - start.z[k])});
        }
        std::fprintf(stderr, "  %s: |dx| %.3g, dE/E %.3g\n", meth == ForceMethod::direct ? "direct" : "tree", maxd,
                     (e1 - e0) / std::fabs(e0));
        CHECK(maxd < 1e-3);
        // Not 1e-6: with float positions near |x| = 1 each drift rounds by ~6e-8, and the same KDK
        // in plain float scalar code drifts by 3.6e-6 (double: 4e-15), so that is the floor.
        CHECK(std::fabs((e1 - e0) / e0) < 2e-5);
    }
}

TEST("cpu: diagnostics match double reductions") {
    Bodies b = plummer(3000, 3);
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    for (auto& m : b.m) m *= u(rng);
    for (auto& x : b.x) x += 0.3f;  // nonzero CoM and angular momentum about the origin
    for (auto& v : b.vy) v += 0.2f;
    double mass = 0, ke = 0, p[3] = {}, l[3] = {}, c[3] = {};
    for (std::size_t i = 0; i < b.size(); ++i) {
        double m = b.m[i], x = b.x[i], y = b.y[i], z = b.z[i], vx = b.vx[i], vy = b.vy[i], vz = b.vz[i];
        mass += m; ke += 0.5 * m * (vx * vx + vy * vy + vz * vz);
        p[0] += m * vx; p[1] += m * vy; p[2] += m * vz;
        l[0] += m * (y * vz - z * vy); l[1] += m * (z * vx - x * vz); l[2] += m * (x * vy - y * vx);
        c[0] += m * x; c[1] += m * y; c[2] += m * z;
    }
    auto be = make_cpu_backend(0);
    be->upload(b);
    be->compute_forces(tree(0.6), false);  // reorders bodies; reductions must not care
    Diag d = be->diagnostics();
    CHECK(!d.has_pe);
    const double tol = 1e-12;
    CHECK_NEAR(d.mass, mass, tol);
    CHECK_NEAR(d.ke, ke, tol);
    CHECK_NEAR(d.px, p[0], tol); CHECK_NEAR(d.py, p[1], tol); CHECK_NEAR(d.pz, p[2], tol);
    CHECK_NEAR(d.lx, l[0], tol); CHECK_NEAR(d.ly, l[1], tol); CHECK_NEAR(d.lz, l[2], tol);
    CHECK_NEAR(d.cx, c[0] / mass, tol); CHECK_NEAR(d.cy, c[1] / mass, tol); CHECK_NEAR(d.cz, c[2] / mass, tol);
}

TEST("cpu: 2000-body Plummer tree KDK conserves energy and momentum; ids survive") {
    Bodies b = plummer(2000, 5);
    auto be = make_cpu_backend(0);
    be->upload(b);
    ForceParams fp = tree(0.5);
    double e0 = energy(*be, fp);
    kdk(*be, fp, 1.0 / 256, 200);
    double e1 = energy(*be, fp);
    Diag d = be->diagnostics();
    double prel = std::sqrt(d.px * d.px + d.py * d.py + d.pz * d.pz);
    std::fprintf(stderr, "  dE/E %.3g, |p| %.3g\n", (e1 - e0) / std::fabs(e0), prel);
    CHECK(std::fabs((e1 - e0) / e0) < 2e-3);
    CHECK(prel < 1e-4);  // |p| scale: M * sigma ~ 0.7
    Bodies o;
    be->download(o);
    CHECK(o.size() == 2000);
    std::vector<int> seen(2000, 0);
    for (uint32_t id : o.id)
        if (id < 2000) ++seen[id];
    CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }));
    CHECK(all_finite(o));
}

TEST("cpu: degenerate inputs stay finite") {
    for (ForceMethod meth : {ForceMethod::direct, ForceMethod::tree}) {
        for (double eps : {0.0, 0.01}) {
            ForceParams fp;
            fp.method = meth; fp.eps = eps; fp.theta = 0.6;
            auto be = make_cpu_backend(0);
            Bodies one;
            one.push(0.5, -1, 2, 0.1, 0, 0, 1);
            be->upload(one);
            be->compute_forces(fp, true);
            CHECK(be->diagnostics().pe == 0);
            kdk(*be, fp, 0.01, 3);
            Bodies o;
            be->download(o);
            CHECK(all_finite(o));
            CHECK_NEAR(o.vx[0], 0.1, 1e-7);

            Bodies two;
            two.push(0, 0, 0, 0, 0, 0, 1);
            two.push(1, 0, 0, 0, 0, 0, 1);
            be->upload(two);
            be->compute_forces(fp, true);
            Diag d = be->diagnostics();
            CHECK_NEAR(d.pe, -1 / std::sqrt(1 + eps * eps), 1e-6);
            be->kick(1);
            be->download(o);
            for (std::size_t i = 0; i < 2; ++i)
                CHECK_NEAR(o.vx[i], (o.x[i] == 0 ? 1 : -1) / std::pow(1 + eps * eps, 1.5), 1e-6);

            // Coincident bodies (some with eps = 0 sit at d = 0): forces must be finite, and with
            // eps > 0 distinct coincident bodies still see each other's -m/eps potential.
            Bodies co;
            for (int i = 0; i < 40; ++i) co.push(0.25, 0.25, 0.25, 0, 0, 0, 1);
            co.push(1, 1, 1, 0, 0, 0, 1);
            be->upload(co);
            be->compute_forces(fp, true);
            be->kick(1);
            be->download(o);
            CHECK(all_finite(o));
            d = be->diagnostics();
            if (eps > 0) CHECK(std::isfinite(d.pe) && d.pe < -40 * 39 / 2 / eps * 0.99);
            else CHECK(std::isfinite(d.pe));
        }
    }
}

TEST("cpu: huge theta is finite and momentum-sane") {
    Bodies b = plummer(5000, 6);
    auto be = make_cpu_backend(0);
    for (double theta : {5.0, 1e6, 1e30}) {
        auto a = accel(*be, b, tree(theta));
        bool ok = true;
        double px = 0;
        for (std::size_t i = 0; i < b.size(); ++i) {
            ok = ok && std::isfinite(a[3 * i]) && std::isfinite(a[3 * i + 1]) && std::isfinite(a[3 * i + 2]);
            px += b.m[i] * a[3 * i];
        }
        CHECK(ok);
        CHECK(std::fabs(px) < 1.0);
    }
}

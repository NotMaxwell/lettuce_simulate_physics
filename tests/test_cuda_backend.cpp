// CUDA backend tests. They skip without a usable device. Accelerations are read through the
// interface by uploading zero velocities, calling compute_forces and kick(1): then v == a.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend.hpp"
#include "test.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

Bodies plummer(std::size_t n, uint64_t seed) {
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
        double q, g;
        do {
            q = U(rng);
            g = 0.1 * U(rng);
        } while (g > q * q * std::pow(1 - q * q, 3.5));
        double v = q * std::sqrt(2.0) * std::pow(1 + r * r, -0.25);
        double x, y, z, vx, vy, vz;
        dir(r, x, y, z);
        dir(v, vx, vy, vz);
        b.push(x, y, z, vx, vy, vz, 1.0 / double(n));
    }
    return b;
}

Bodies cube(std::size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0), M(0.5, 1.5);
    Bodies b;
    for (std::size_t i = 0; i < n; ++i) b.push(U(rng), U(rng), U(rng), U(rng), U(rng), U(rng), M(rng) / double(n));
    return b;
}

struct Ref {
    std::vector<double> a;  // 3n, indexed by id
    double pe = 0;
};

Ref brute(const Bodies& b, double G, double eps) {
    const std::size_t n = b.size();
    Ref r;
    r.a.assign(3 * n, 0.0);
    double pe = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0, ay = 0, az = 0, phi = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            double dx = double(b.x[j]) - b.x[i], dy = double(b.y[j]) - b.y[i], dz = double(b.z[j]) - b.z[i];
            double s2 = dx * dx + dy * dy + dz * dz + eps * eps, inv = 1 / std::sqrt(s2);
            double f = b.m[j] * inv * inv * inv;
            ax += f * dx; ay += f * dy; az += f * dz;
            phi += b.m[j] * inv;
        }
        r.a[3 * b.id[i]] = G * ax;
        r.a[3 * b.id[i] + 1] = G * ay;
        r.a[3 * b.id[i] + 2] = G * az;
        pe += b.m[i] * phi;
    }
    r.pe = -0.5 * G * pe;
    return r;
}

struct Got {
    std::vector<double> a;  // 3n, indexed by id
    Diag diag;
    PhaseTimes times;
};

Got accel(Backend& be, Bodies b, const ForceParams& p, bool potential) {
    std::fill(b.vx.begin(), b.vx.end(), 0.f);
    std::fill(b.vy.begin(), b.vy.end(), 0.f);
    std::fill(b.vz.begin(), b.vz.end(), 0.f);
    be.upload(b);
    be.compute_forces(p, potential);
    Got g;
    g.times = be.times();
    g.diag = be.diagnostics();  // velocities are zero: only pe matters here
    be.kick(1.0);
    Bodies out;
    be.download(out);
    g.a.assign(3 * out.size(), 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        g.a[3 * out.id[i]] = out.vx[i];
        g.a[3 * out.id[i] + 1] = out.vy[i];
        g.a[3 * out.id[i] + 2] = out.vz[i];
    }
    return g;
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
    for (double x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

ForceParams params(ForceMethod m, double eps, double theta = 0.6) {
    ForceParams p;
    p.G = 1;
    p.eps = eps;
    p.theta = theta;
    p.method = m;
    return p;
}

#define CUDA_BACKEND(be)                       \
    std::string why;                           \
    auto be = make_cuda_backend(0, why);       \
    if (!be) SKIP(why)

}  // namespace

TEST("cuda: name and compiled flag") {
    CUDA_BACKEND(be);
    CHECK(cuda_compiled());
    CHECK(be->name().rfind("cuda: ", 0) == 0);
    std::fprintf(stderr, "  %s\n", be->name().c_str());
}

TEST("cuda: direct vs double brute force, 4096 Plummer") {
    CUDA_BACKEND(be);
    Bodies b = plummer(4096, 1);
    const double eps = 0.01;
    Ref ref = brute(b, 1, eps);
    Got g0 = accel(*be, b, params(ForceMethod::direct, eps), false);
    double e0 = rel_rms(g0.a, ref.a);
    std::fprintf(stderr, "  no potential: rel rms %.3g\n", e0);
    CHECK(e0 <= 1e-5);
    CHECK(!g0.diag.has_pe);
    Got g1 = accel(*be, b, params(ForceMethod::direct, eps), true);
    double e1 = rel_rms(g1.a, ref.a);
    std::fprintf(stderr, "  potential: rel rms %.3g, pe %.9g vs %.9g\n", e1, g1.diag.pe, ref.pe);
    CHECK(e1 <= 1e-5);
    CHECK(g1.diag.has_pe);
    CHECK_NEAR(g1.diag.pe / ref.pe, 1.0, 1e-5);
    CHECK_NEAR(be->exact_potential_energy(1, eps) / ref.pe, 1.0, 1e-9);
    CHECK(g1.times.interactions == 4096ull * 4095ull);
}

TEST("cuda: small and odd N, direct and tree(theta=0) vs brute force") {
    CUDA_BACKEND(be);
    for (std::size_t n : {1, 2, 3, 31, 32, 33, 64, 255, 257, 1000, 1023}) {
        Bodies b = cube(n, 100 + n);
        const double eps = 0.05;
        Ref ref = brute(b, 1, eps);
        for (ForceMethod m : {ForceMethod::direct, ForceMethod::tree}) {
            Got g = accel(*be, b, params(m, eps, 0.0), true);
            CHECK(all_finite(g.a));
            if (n == 1) {
                CHECK(g.a[0] == 0 && g.a[1] == 0 && g.a[2] == 0);
                CHECK(g.diag.pe == 0);
                continue;
            }
            double e = rel_rms(g.a, ref.a);
            if (!(e <= 1e-5))
                std::fprintf(stderr, "  n=%zu %s rel rms %.3g\n", n, m == ForceMethod::tree ? "tree" : "direct", e);
            CHECK(e <= 1e-5);
            CHECK_NEAR(g.diag.pe / ref.pe, 1.0, 1e-5);
        }
    }
}

TEST("cuda: tree with coincident bodies (duplicate Morton keys)") {
    CUDA_BACKEND(be);
    // 150 bodies at one point force a > 32 identical-key run; plus a clump of 40 within 1e-7.
    Bodies b = cube(300, 7);
    for (std::size_t i = 0; i < 150; ++i) { b.x[i] = 0.25f; b.y[i] = -0.5f; b.z[i] = 0.125f; }
    for (std::size_t i = 150; i < 190; ++i) { b.x[i] = -0.3f + 1e-7f * float(i - 150); b.y[i] = 0.1f; b.z[i] = 0.2f; }
    const double eps = 0.05;
    Ref ref = brute(b, 1, eps);
    Got g0 = accel(*be, b, params(ForceMethod::tree, eps, 0.0), true);
    CHECK(all_finite(g0.a));
    double e0 = rel_rms(g0.a, ref.a);
    std::fprintf(stderr, "  theta=0 rel rms %.3g\n", e0);
    CHECK(e0 <= 1e-5);
    Got g1 = accel(*be, b, params(ForceMethod::tree, eps, 0.6), true);
    CHECK(all_finite(g1.a));
    double e1 = rel_rms(g1.a, ref.a);
    std::fprintf(stderr, "  theta=0.6 rel rms %.3g\n", e1);
    CHECK(e1 <= 1e-2);
}

TEST("cuda: tree vs cuda direct, 20k Plummer, theta 0.6") {
    CUDA_BACKEND(be);
    Bodies b = plummer(20000, 2);
    const double eps = 0.01;
    Got d = accel(*be, b, params(ForceMethod::direct, eps), true);
    Got t = accel(*be, b, params(ForceMethod::tree, eps, 0.6), true);
    double e = rel_rms(t.a, d.a);
    std::fprintf(stderr, "  rel rms %.3g, pe rel %.3g, interactions/body %.1f, force %.3f ms (sort %.3f build %.3f walk %.3f)\n",
                 e, t.diag.pe / d.diag.pe - 1, double(t.times.interactions) / 20000, t.times.force_ms,
                 t.times.sort_ms, t.times.build_ms, t.times.walk_ms);
    CHECK(all_finite(t.a));
    CHECK(e <= 1e-3);
    CHECK_NEAR(t.diag.pe / d.diag.pe, 1.0, 1e-3);
    CHECK(t.times.interactions > 0 && t.times.interactions < 20000ull * 19999ull);
}

TEST("cuda: tree agrees with cpu tree within combined error") {
    CUDA_BACKEND(be);
    auto cpu = make_cpu_backend(0);
    CHECK(cpu != nullptr);
    if (!cpu) return;
    Bodies b = plummer(20000, 3);
    const double eps = 0.01;
    Got d = accel(*be, b, params(ForceMethod::direct, eps), false);
    Got tg = accel(*be, b, params(ForceMethod::tree, eps, 0.6), false);
    Got tc = accel(*cpu, b, params(ForceMethod::tree, eps, 0.6), false);
    double eg = rel_rms(tg.a, d.a), ec = rel_rms(tc.a, d.a), egc = rel_rms(tg.a, tc.a);
    std::fprintf(stderr, "  gpu err %.3g, cpu err %.3g, gpu-cpu %.3g\n", eg, ec, egc);
    CHECK(ec <= 1e-3);
    CHECK(egc <= eg + ec + 1e-5);
}

TEST("cuda: kick/drift/diagnostics match cpu backend") {
    CUDA_BACKEND(be);
    auto cpu = make_cpu_backend(0);
    CHECK(cpu != nullptr);
    if (!cpu) return;
    Bodies b = plummer(2000, 4);
    const ForceParams p = params(ForceMethod::direct, 0.01);
    Diag dg[2];
    Bodies out[2];
    Backend* bes[2] = {be.get(), cpu.get()};
    for (int k = 0; k < 2; ++k) {
        bes[k]->upload(b);
        bes[k]->compute_forces(p, true);
        bes[k]->kick(0.01);
        bes[k]->drift(0.01);
        bes[k]->compute_forces(p, true);
        bes[k]->kick(0.01);
        dg[k] = bes[k]->diagnostics();
        bes[k]->download(out[k]);
    }
    double pscale = 0, lscale = 0, cscale = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double r = std::sqrt(double(b.x[i]) * b.x[i] + double(b.y[i]) * b.y[i] + double(b.z[i]) * b.z[i]);
        double v = std::sqrt(double(b.vx[i]) * b.vx[i] + double(b.vy[i]) * b.vy[i] + double(b.vz[i]) * b.vz[i]);
        pscale += b.m[i] * v; lscale += b.m[i] * r * v; cscale += b.m[i] * r;
    }
    const double tol = 1e-5;
    CHECK_NEAR(dg[0].mass, dg[1].mass, tol * dg[1].mass);
    CHECK_NEAR(dg[0].ke, dg[1].ke, tol * dg[1].ke);
    CHECK(dg[0].has_pe && dg[1].has_pe);
    CHECK_NEAR(dg[0].pe, dg[1].pe, tol * std::fabs(dg[1].pe));
    CHECK_NEAR(dg[0].px, dg[1].px, tol * pscale);
    CHECK_NEAR(dg[0].py, dg[1].py, tol * pscale);
    CHECK_NEAR(dg[0].pz, dg[1].pz, tol * pscale);
    CHECK_NEAR(dg[0].lx, dg[1].lx, tol * lscale);
    CHECK_NEAR(dg[0].ly, dg[1].ly, tol * lscale);
    CHECK_NEAR(dg[0].lz, dg[1].lz, tol * lscale);
    CHECK_NEAR(dg[0].cx, dg[1].cx, tol * cscale);
    CHECK_NEAR(dg[0].cy, dg[1].cy, tol * cscale);
    CHECK_NEAR(dg[0].cz, dg[1].cz, tol * cscale);
    CHECK(out[0].size() == b.size() && out[1].size() == b.size());
    if (out[0].size() != b.size() || out[1].size() != b.size()) return;
    std::vector<std::size_t> at(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) at[out[1].id[i]] = i;
    int bad = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        std::size_t j = at[out[0].id[i]];
        auto close = [&](float u, float v) { return std::fabs(double(u) - v) <= tol * (std::fabs(double(v)) + 1e-2); };
        if (!(close(out[0].x[i], out[1].x[j]) && close(out[0].y[i], out[1].y[j]) && close(out[0].z[i], out[1].z[j]) &&
              close(out[0].vx[i], out[1].vx[j]) && close(out[0].vy[i], out[1].vy[j]) && close(out[0].vz[i], out[1].vz[j])))
            ++bad;
    }
    CHECK(bad == 0);
}

TEST("cuda: two-body circular orbit returns after one period") {
    CUDA_BACKEND(be);
    for (ForceMethod m : {ForceMethod::direct, ForceMethod::tree}) {
        Bodies b;
        b.push(0.5, 0, 0, 0, 0.5, 0, 0.5);
        b.push(-0.5, 0, 0, 0, -0.5, 0, 0.5);
        const ForceParams p = params(m, 0.0);
        const int steps = 1000;
        const double dt = 2 * kPi / steps;
        be->upload(b);
        be->compute_forces(p, true);
        const Diag d0 = be->diagnostics();
        for (int s = 0; s < steps; ++s) {
            be->kick(dt / 2);
            be->drift(dt);
            be->compute_forces(p, true);
            be->kick(dt / 2);
        }
        const Diag d1 = be->diagnostics();
        Bodies out;
        be->download(out);
        CHECK(out.size() == 2);
        if (out.size() != 2) continue;
        for (std::size_t i = 0; i < 2; ++i) {
            const uint32_t id = out.id[i];
            CHECK(id < 2);
            if (id >= 2) continue;
            double dist = std::hypot(out.x[i] - b.x[id], out.y[i] - b.y[id], out.z[i] - b.z[id]);
            if (!(dist <= 1e-3)) std::fprintf(stderr, "  %s body %u off by %.3g\n", m == ForceMethod::tree ? "tree" : "direct", id, dist);
            CHECK(dist <= 1e-3);
        }
        CHECK_NEAR((d1.ke + d1.pe) / (d0.ke + d0.pe), 1.0, 1e-4);
    }
}

TEST("cuda: download after tree steps returns every id once") {
    CUDA_BACKEND(be);
    Bodies b = plummer(5000, 5);
    const ForceParams p = params(ForceMethod::tree, 0.01, 0.6);
    be->upload(b);
    be->compute_forces(p, false);
    for (int s = 0; s < 5; ++s) {
        be->kick(0.005);
        be->drift(0.01);
        be->compute_forces(p, false);
        be->kick(0.005);
    }
    Bodies out;
    be->download(out);
    CHECK(out.size() == b.size());
    std::vector<int> seen(b.size(), 0);
    bool ok = true;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out.id[i] >= b.size()) { ok = false; continue; }
        ++seen[out.id[i]];
        ok = ok && std::isfinite(out.x[i]) && std::isfinite(out.vx[i]) && out.m[i] == b.m[out.id[i]];
    }
    CHECK(ok);
    CHECK(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }));
}

TEST("cuda: 1M-body tree force call") {
    CUDA_BACKEND(be);
    const std::size_t n = 1u << 20;
    Bodies b = plummer(n, 6);
    const ForceParams p = params(ForceMethod::tree, 0.01, 0.6);
    Got g = accel(*be, b, p, false);  // first call includes allocations
    CHECK(all_finite(g.a));
    double amax = 0;
    for (double x : g.a) amax = std::max(amax, std::fabs(x));
    CHECK(amax > 0);
    auto t0 = std::chrono::steady_clock::now();
    be->compute_forces(p, false);
    double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    PhaseTimes t = be->times();
    std::fprintf(stderr, "  1M tree: wall %.2f ms, force %.2f ms (sort %.2f, build %.2f, walk %.2f), %.1f interactions/body\n",
                 wall, t.force_ms, t.sort_ms, t.build_ms, t.walk_ms, double(t.interactions) / double(n));
}

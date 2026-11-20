#include <cmath>
#include <stdexcept>

#include "scenarios.hpp"
#include "test.hpp"

namespace {

Config make(const std::string& scen, uint64_t n, uint64_t seed = 1) {
    Config c;
    c.scenario = scen;
    c.n = n;
    c.seed = seed;
    return c;
}

// CoM position and momentum must vanish relative to the natural scales of the system.
void check_sane(const Bodies& b) {
    size_t n = b.size();
    double M = 0, c[3] = {}, p[3] = {}, rs = 0, ps = 0;
    for (size_t i = 0; i < n; ++i) {
        CHECK(std::isfinite(b.x[i]) && std::isfinite(b.y[i]) && std::isfinite(b.z[i]));
        CHECK(std::isfinite(b.vx[i]) && std::isfinite(b.vy[i]) && std::isfinite(b.vz[i]));
        CHECK(b.m[i] >= 0 && std::isfinite(b.m[i]));
        CHECK(b.id[i] == i);
        double m = b.m[i];
        M += m;
        c[0] += m * b.x[i]; c[1] += m * b.y[i]; c[2] += m * b.z[i];
        p[0] += m * b.vx[i]; p[1] += m * b.vy[i]; p[2] += m * b.vz[i];
        rs += m * std::sqrt(double(b.x[i]) * b.x[i] + double(b.y[i]) * b.y[i] + double(b.z[i]) * b.z[i]);
        ps += m * std::sqrt(double(b.vx[i]) * b.vx[i] + double(b.vy[i]) * b.vy[i] + double(b.vz[i]) * b.vz[i]);
    }
    for (int k = 0; k < 3; ++k) {
        CHECK(std::fabs(c[k]) <= 1e-6 * rs);
        CHECK(std::fabs(p[k]) <= 1e-6 * ps);
    }
    CHECK(M > 0);
}

bool same(const Bodies& a, const Bodies& b) {
    return a.size() == b.size() && a.x == b.x && a.y == b.y && a.z == b.z && a.vx == b.vx &&
           a.vy == b.vy && a.vz == b.vz && a.m == b.m;
}

void check_scenario(const std::string& name, uint64_t n, uint64_t expect_n, double G = 1, bool seeded = true) {
    Config c = make(name, n);
    c.G = G;
    Bodies a = build_scenario(c);
    CHECK(a.size() == expect_n);
    check_sane(a);
    CHECK(same(a, build_scenario(c)));
    Config d = c;
    d.seed = 2;
    if (seeded) CHECK(!same(a, build_scenario(d)));
}

}  // namespace

TEST("scenarios: list and unknown name") {
    CHECK(scenarios().size() == 7);
    std::string e;
    try { build_scenario(make("nope", 10)); } catch (const std::runtime_error& ex) { e = ex.what(); }
    CHECK(e.find("nope") != std::string::npos);
    CHECK(e.find("plummer") != std::string::npos && e.find("systems") != std::string::npos);
}

TEST("scenarios: plummer") { check_scenario("plummer", 500, 500); }
TEST("scenarios: uniform sphere") {
    Config c = make("uniform", 400);
    c.params["uniform.sigma"] = "0.1";
    c.params["uniform.spin"] = "0.3";
    check_sane(build_scenario(c));
    c.params["uniform.shape"] = "cube";
    check_sane(build_scenario(c));
    check_scenario("uniform", 400, 400);
}
TEST("scenarios: disk") { check_scenario("disk", 600, 600); }
TEST("scenarios: collision") { check_scenario("collision", 601, 601); }
TEST("scenarios: kepler ignores n") { check_scenario("kepler", 77, 2, 1, false); }
TEST("scenarios: solar") { check_scenario("solar", 300, 300, 39.47841760435743); }
TEST("scenarios: systems") { check_scenario("systems", 500, 500); }

TEST("scenarios: plummer virial ratio") {
    Config c = make("plummer", 2000);
    Bodies b = build_scenario(c);
    size_t n = b.size();
    double K = 0, W = 0;
    for (size_t i = 0; i < n; ++i)
        K += 0.5 * b.m[i] * (double(b.vx[i]) * b.vx[i] + double(b.vy[i]) * b.vy[i] + double(b.vz[i]) * b.vz[i]);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            double dx = double(b.x[i]) - b.x[j], dy = double(b.y[i]) - b.y[j], dz = double(b.z[i]) - b.z[j];
            W -= c.G * b.m[i] * b.m[j] / std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    CHECK_NEAR(2 * K / std::fabs(W), 1.0, 0.1);
}

TEST("scenarios: kepler energy") {
    Config c = make("kepler", 2);
    c.G = 1.7;
    c.params["kepler.m1"] = "2";
    c.params["kepler.m2"] = "0.5";
    c.params["kepler.a"] = "1.3";
    c.params["kepler.e"] = "0.4";
    Bodies b = build_scenario(c);
    CHECK(b.size() == 2);
    double K = 0;
    for (int i = 0; i < 2; ++i)
        K += 0.5 * b.m[i] * (double(b.vx[i]) * b.vx[i] + double(b.vy[i]) * b.vy[i] + double(b.vz[i]) * b.vz[i]);
    double dx = double(b.x[0]) - b.x[1], dy = double(b.y[0]) - b.y[1], dz = double(b.z[0]) - b.z[1];
    double U = -c.G * b.m[0] * b.m[1] / std::sqrt(dx * dx + dy * dy + dz * dz);
    double expect = -c.G * 2.0 * 0.5 / (2 * 1.3);
    CHECK_NEAR((K + U) / expect, 1.0, 1e-5);
    CHECK_NEAR(std::sqrt(dx * dx + dy * dy + dz * dz), 1.3 * 1.4, 1e-5);

    Config bad = c;
    bad.params["kepler.e"] = "1.2";
    std::string e;
    try { build_scenario(bad); } catch (const std::runtime_error& ex) { e = ex.what(); }
    CHECK(e.find("kepler.e") != std::string::npos);
}

TEST("scenarios: solar Earth") {
    Config c = make("solar", 9);
    c.G = 39.47841760435743;
    Bodies b = build_scenario(c);
    CHECK(b.size() == 9);
    auto rel = [&](const std::vector<float>& a, size_t i) { return double(a[i]) - a[0]; };
    double r = std::sqrt(rel(b.x, 3) * rel(b.x, 3) + rel(b.y, 3) * rel(b.y, 3) + rel(b.z, 3) * rel(b.z, 3));
    double v = std::sqrt(rel(b.vx, 3) * rel(b.vx, 3) + rel(b.vy, 3) * rel(b.vy, 3) + rel(b.vz, 3) * rel(b.vz, 3));
    CHECK_NEAR(r, 1.0, 0.05);
    CHECK_NEAR(v, 2 * std::acos(-1.0), 0.05 * 2 * std::acos(-1.0));
    CHECK_NEAR(b.m[0], 1.0, 0);

    bool threw = false;
    try { build_scenario(make("solar", 8)); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

TEST("scenarios: systems count and bad params") {
    Config c = make("systems", 1000);
    CHECK(build_scenario(c).size() == 1000);
    c.params["systems.planets"] = "2";
    CHECK(build_scenario(c).size() == 999);
    c.params["systems.r_min"] = "-1";
    std::string e;
    try { build_scenario(c); } catch (const std::runtime_error& ex) { e = ex.what(); }
    CHECK(e.find("systems.r_min") != std::string::npos);
}

TEST("scenarios: disk is rotating and bound by enclosed mass") {
    Config c = make("disk", 2000);
    c.params["disk.dispersion"] = "0";
    c.params["disk.thickness"] = "0";
    Bodies b = build_scenario(c);
    double Lz = 0;
    for (size_t i = 0; i < b.size(); ++i) Lz += b.m[i] * (double(b.x[i]) * b.vy[i] - double(b.y[i]) * b.vx[i]);
    CHECK(Lz > 0.1);
    c.params["collision.retrograde"] = "0";
    c.scenario = "collision";
    check_sane(build_scenario(c));
    c.params["collision.retrograde"] = "1";
    check_sane(build_scenario(c));
}

TEST("scenarios: scenario_body_count is the generated size") {
    for (auto& info : scenarios()) {
        Config c = make(info.name, 1000);
        CHECK(scenario_body_count(c) == build_scenario(c).size());
    }
    Config k = make("kepler", 5000);
    CHECK(scenario_body_count(k) == 2);
    // systems.stars overrides n: the count follows stars * (1 + planets), not n.
    Config s = make("systems", 100);
    s.params["systems.stars"] = "40";
    s.params["systems.planets"] = "3";
    CHECK(scenario_body_count(s) == 160);
    CHECK(build_scenario(s).size() == 160);
    Config d = make("systems", 1003);
    CHECK(scenario_body_count(d) == 1000);  // 200 stars x (1 + 4)
}

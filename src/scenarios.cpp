#include "scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kTwoPi = 2 * std::numbers::pi;

// The standard-library distributions are implementation-defined, so seeds would not reproduce
// across libstdc++/libc++; these helpers depend only on mt19937_64's specified output.
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    double uniform() { return static_cast<double>(g() >> 11) * 0x1.0p-53; }
    double uniform(double a, double b) { return a + (b - a) * uniform(); }
    double normal() {
        double u1 = 1.0 - uniform();
        return std::sqrt(-2 * std::log(u1)) * std::cos(kTwoPi * uniform());
    }
    void unit_vector(double& x, double& y, double& z) {
        z = uniform(-1, 1);
        double s = std::sqrt(std::max(0.0, 1 - z * z)), phi = kTwoPi * uniform();
        x = s * std::cos(phi);
        y = s * std::sin(phi);
    }
};

struct Pts {
    std::vector<double> x, y, z, vx, vy, vz, m;
    size_t size() const { return x.size(); }
    void reserve(size_t n) {
        for (auto* v : {&x, &y, &z, &vx, &vy, &vz, &m}) v->reserve(n);
    }
    void add(double px, double py, double pz, double pvx, double pvy, double pvz, double pm) {
        x.push_back(px); y.push_back(py); z.push_back(pz);
        vx.push_back(pvx); vy.push_back(pvy); vz.push_back(pvz);
        m.push_back(pm);
    }
};

// Subtracts the centre-of-mass position and velocity in double before rounding to float.
// Converts one column at a time and frees its doubles straight away: holding all 56 B/body of
// doubles next to the 32 B/body result set the host-memory peak (88 GB at 1e9 bodies).
Bodies finish(Pts&& p) {
    const size_t n = p.size();
    double M = 0, c[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        M += p.m[i];
        c[0] += p.m[i] * p.x[i]; c[1] += p.m[i] * p.y[i]; c[2] += p.m[i] * p.z[i];
        v[0] += p.m[i] * p.vx[i]; v[1] += p.m[i] * p.vy[i]; v[2] += p.m[i] * p.vz[i];
    }
    for (int k = 0; k < 3; ++k) { c[k] /= M; v[k] /= M; }
    Bodies b;
    auto convert = [n](std::vector<double>& src, std::vector<float>& dst, double shift) {
        dst.resize(n);
        for (size_t i = 0; i < n; ++i) dst[i] = float(src[i] - shift);
        std::vector<double>().swap(src);
    };
    convert(p.x, b.x, c[0]); convert(p.y, b.y, c[1]); convert(p.z, b.z, c[2]);
    convert(p.vx, b.vx, v[0]); convert(p.vy, b.vy, v[1]); convert(p.vz, b.vz, v[2]);
    convert(p.m, b.m, 0);
    b.id.resize(n);
    for (size_t i = 0; i < n; ++i) b.id[i] = static_cast<uint32_t>(i);
    return b;
}

double positive(const Config& c, const char* key, double fallback) {
    double v = c.param(key, fallback);
    if (!(v > 0)) throw std::runtime_error(std::string("parameter '") + key + "' must be > 0");
    return v;
}

double nonneg(const Config& c, const char* key, double fallback) {
    double v = c.param(key, fallback);
    if (!(v >= 0)) throw std::runtime_error(std::string("parameter '") + key + "' must be >= 0");
    return v;
}

uint64_t count_param(const Config& c, const char* key, double fallback) {
    double v = c.param(key, fallback);
    if (!(v >= 0) || v != std::floor(v) || v > 1e12)
        throw std::runtime_error(std::string("parameter '") + key + "' must be a non-negative integer");
    return static_cast<uint64_t>(v);
}

// Aarseth, Henon & Wielen (1974). Radii beyond cut*a are redrawn, so the realised mass is M.
struct PlummerSample { double x, y, z, vx, vy, vz; };

PlummerSample sample_plummer(Rng& rng, double G, double M, double a, double cut) {
    double r = 0;
    do {
        double X = rng.uniform();
        if (X <= 0) continue;
        r = a / std::sqrt(std::pow(X, -2.0 / 3.0) - 1);
    } while (!(r <= cut * a));
    double q;
    for (;;) {
        q = rng.uniform();
        // 0.1 bounds g(q) = q^2 (1 - q^2)^3.5, whose maximum is about 0.092.
        if (0.1 * rng.uniform() < q * q * std::pow(1 - q * q, 3.5)) break;
    }
    double v = q * std::sqrt(2 * G * M / a) * std::pow(1 + r * r / (a * a), -0.25);
    PlummerSample s;
    double ux, uy, uz;
    rng.unit_vector(ux, uy, uz);
    s.x = r * ux; s.y = r * uy; s.z = r * uz;
    rng.unit_vector(ux, uy, uz);
    s.vx = v * ux; s.vy = v * uy; s.vz = v * uz;
    return s;
}

Bodies build_plummer(const Config& c) {
    double M = positive(c, "plummer.mass", 1);
    double a = positive(c, "plummer.a", 3 * kPi / 16);
    double cut = positive(c, "plummer.cut", 20);
    Rng rng(c.seed);
    Pts p;
    p.reserve(c.n);
    for (uint64_t i = 0; i < c.n; ++i) {
        auto s = sample_plummer(rng, c.G, M, a, cut);
        p.add(s.x, s.y, s.z, s.vx, s.vy, s.vz, M / c.n);
    }
    return finish(std::move(p));
}

Bodies build_kepler(const Config& c) {
    double m1 = positive(c, "kepler.m1", 1), m2 = positive(c, "kepler.m2", 1e-3);
    double a = positive(c, "kepler.a", 1), e = c.param("kepler.e", 0);
    if (!(e >= 0 && e < 1)) throw std::runtime_error("parameter 'kepler.e' must be in [0, 1)");
    double M = m1 + m2;
    double ra = a * (1 + e);
    double va = std::sqrt(c.G * M / a * (1 - e) / (1 + e));
    Pts p;
    p.add(-m2 / M * ra, 0, 0, 0, -m2 / M * va, 0, m1);
    p.add(m1 / M * ra, 0, 0, 0, m1 / M * va, 0, m2);
    return finish(std::move(p));
}

Bodies build_uniform(const Config& c) {
    std::string shape = c.param_str("uniform.shape", "sphere");
    if (shape != "sphere" && shape != "cube")
        throw std::runtime_error("parameter 'uniform.shape' must be 'sphere' or 'cube'");
    double R = positive(c, "uniform.radius", 1), M = positive(c, "uniform.mass", 1);
    double sigma = nonneg(c, "uniform.sigma", 0), spin = c.param("uniform.spin", 0);
    double omega = spin * std::sqrt(c.G * M / (R * R * R));
    Rng rng(c.seed);
    Pts p;
    p.reserve(c.n);
    for (uint64_t i = 0; i < c.n; ++i) {
        double x, y, z;
        if (shape == "cube") {
            x = rng.uniform(-R, R); y = rng.uniform(-R, R); z = rng.uniform(-R, R);
        } else {
            double ux, uy, uz, r = R * std::cbrt(rng.uniform());
            rng.unit_vector(ux, uy, uz);
            x = r * ux; y = r * uy; z = r * uz;
        }
        double vx = sigma * rng.normal() - omega * y, vy = sigma * rng.normal() + omega * x;
        double vz = sigma * rng.normal();
        p.add(x, y, z, vx, vy, vz, M / c.n);
    }
    return finish(std::move(p));
}

struct DiskParams {
    double core, mass, scale, rmax, thickness, dispersion;
};

DiskParams read_disk(const Config& c) {
    DiskParams d;
    d.core = nonneg(c, "disk.core_mass", 0.1);
    d.mass = positive(c, "disk.mass", 1);
    d.scale = positive(c, "disk.scale", 1);
    d.rmax = positive(c, "disk.rmax", 5);
    d.thickness = nonneg(c, "disk.thickness", 0.05);
    d.dispersion = nonneg(c, "disk.dispersion", 0.05);
    return d;
}

struct DiskPlacement {
    double cx = 0, cy = 0, cz = 0;
    double vx = 0, vy = 0, vz = 0;
    double incl = 0;  // radians, rotation about the x axis
    double spin = 1;  // -1 for counter-rotating
};

// Radius in scale lengths from the truncated exponential disk: the cumulative mass fraction is
// 1 - (1 + x) exp(-x), inverted by bisection (no closed form; Lambert W avoided on purpose).
double sample_disk_radius(Rng& rng, double xmax, double cdf_max) {
    double target = rng.uniform() * cdf_max, lo = 0, hi = xmax;
    for (int it = 0; it < 48; ++it) {
        double mid = 0.5 * (lo + hi);
        if (1 - (1 + mid) * std::exp(-mid) < target) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Appends a core body and nbodies-1 stars. Circular speed uses the enclosed mass (core plus the
// sampled stars inside r, counted exactly after sorting) under spherical symmetry.
void add_disk(Pts& p, Rng& rng, const Config& c, const DiskParams& d, uint64_t nbodies,
              const DiskPlacement& pl) {
    if (nbodies < 2) throw std::runtime_error("disk needs at least 2 bodies (core + 1 star)");
    const uint64_t ns = nbodies - 1;
    const double ms = d.mass / ns;
    const double xmax = d.rmax, cdf_max = 1 - (1 + xmax) * std::exp(-xmax);
    const double ci = std::cos(pl.incl), si = std::sin(pl.incl);

    auto place = [&](double x, double y, double z, double vx, double vy, double vz, double m) {
        double y2 = y * ci - z * si, z2 = y * si + z * ci;
        double vy2 = vy * ci - vz * si, vz2 = vy * si + vz * ci;
        p.add(x + pl.cx, y2 + pl.cy, z2 + pl.cz, vx + pl.vx, vy2 + pl.vy, vz2 + pl.vz, m);
    };
    place(0, 0, 0, 0, 0, 0, d.core);

    std::vector<double> radii(ns);
    for (auto& r : radii) r = sample_disk_radius(rng, xmax, cdf_max) * d.scale;
    std::sort(radii.begin(), radii.end());
    for (uint64_t i = 0; i < ns; ++i) {
        double r = std::max(radii[i], 1e-12 * d.scale), phi = kTwoPi * rng.uniform();
        double enclosed = d.core + ms * static_cast<double>(i + 1);
        double vc = std::sqrt(c.G * enclosed / r), sg = d.dispersion * vc;
        double cp = std::cos(phi), sp = std::sin(phi);
        double x = r * cp, y = r * sp, z = d.thickness * d.scale * rng.normal();
        double vx = -pl.spin * vc * sp + sg * rng.normal();
        double vy = pl.spin * vc * cp + sg * rng.normal();
        double vz = sg * rng.normal();
        place(x, y, z, vx, vy, vz, ms);
    }
}

Bodies build_disk(const Config& c) {
    DiskParams d = read_disk(c);
    Rng rng(c.seed);
    Pts p;
    p.reserve(c.n);
    add_disk(p, rng, c, d, c.n, {});
    return finish(std::move(p));
}

Bodies build_collision(const Config& c) {
    DiskParams d = read_disk(c);
    double sep = positive(c, "collision.separation", 8) * d.scale;
    double impact = c.param("collision.impact", 2) * d.scale;
    double speed = nonneg(c, "collision.speed", 0.3);
    double incl = c.param("collision.inclination", 30) * kPi / 180;
    bool retro = c.param("collision.retrograde", 0) != 0;

    uint64_t n1 = c.n / 2, n2 = c.n - n1;
    double M1 = d.core + d.mass, M2 = M1;
    double vrel = speed * std::sqrt(2 * c.G * (M1 + M2) / sep);

    Rng rng(c.seed);
    Pts p;
    p.reserve(c.n);
    DiskPlacement a, b;
    a.cx = -sep / 2; a.cy = -impact / 2; a.vx = vrel * M2 / (M1 + M2);
    b.cx = sep / 2;  b.cy = impact / 2;  b.vx = -vrel * M1 / (M1 + M2);
    b.incl = incl;
    b.spin = retro ? -1 : 1;
    add_disk(p, rng, c, d, n1, a);
    add_disk(p, rng, c, d, n2, b);
    return finish(std::move(p));
}

struct Planet { const char* name; double a, e, inc, node, peri, mass; };
// JPL "Keplerian elements for approximate positions" (J2000): a [AU], e, i / node / longitude of
// perihelion [deg], mass [Msun].
const Planet kPlanets[8] = {
    {"Mercury", 0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 1.6601e-7},
    {"Venus", 0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 2.4478e-6},
    {"Earth", 1.00000261, 0.01671123, 0.00001531, 0.0, 102.93768193, 3.0035e-6},
    {"Mars", 1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959 + 360, 3.2272e-7},
    {"Jupiter", 5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 9.5479e-4},
    {"Saturn", 9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831, 2.8588e-4},
    {"Uranus", 19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630, 4.3662e-5},
    {"Neptune", 30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227, 5.1514e-5},
};

double solve_kepler(double M, double e) {
    M = std::fmod(M, kTwoPi);
    double E = M + e * std::sin(M);
    for (int i = 0; i < 60; ++i) {
        double d = (E - e * std::sin(E) - M) / (1 - e * std::cos(E));
        E -= d;
        if (std::fabs(d) < 1e-14) break;
    }
    return E;
}

// Position and velocity relative to the primary for classical elements (angles in radians).
void orbit_state(double mu, double a, double e, double inc, double node, double omega, double M,
                 double pos[3], double vel[3]) {
    double E = solve_kepler(M, e), cE = std::cos(E), sE = std::sin(E), b = std::sqrt(1 - e * e);
    double r = a * (1 - e * cE), f = std::sqrt(mu * a) / r;
    double xp = a * (cE - e), yp = a * b * sE;
    double vxp = -f * sE, vyp = f * b * cE;
    double co = std::cos(omega), so = std::sin(omega), cn = std::cos(node), sn = std::sin(node);
    double ci = std::cos(inc), si = std::sin(inc);
    auto rot = [&](double u, double w, double out[3]) {
        out[0] = (cn * co - sn * so * ci) * u + (-cn * so - sn * co * ci) * w;
        out[1] = (sn * co + cn * so * ci) * u + (-sn * so + cn * co * ci) * w;
        out[2] = (so * si) * u + (co * si) * w;
    };
    rot(xp, yp, pos);
    rot(vxp, vyp, vel);
}

Bodies build_solar(const Config& c) {
    if (c.n < 9) throw std::runtime_error("solar needs n >= 9 (Sun + 8 planets)");
    double belt_in = positive(c, "solar.belt_inner", 2.2), belt_out = positive(c, "solar.belt_outer", 3.3);
    double belt_incl = nonneg(c, "solar.belt_incl", 10) * kPi / 180;
    if (!(belt_out > belt_in)) throw std::runtime_error("parameter 'solar.belt_outer' must exceed 'solar.belt_inner'");
    constexpr double kDeg = kPi / 180;
    Rng rng(c.seed);
    Pts p;
    p.add(0, 0, 0, 0, 0, 0, 1.0);
    double pos[3], vel[3];
    for (const Planet& pl : kPlanets) {
        orbit_state(c.G * (1 + pl.mass), pl.a, pl.e, pl.inc * kDeg, pl.node * kDeg,
                    (pl.peri - pl.node) * kDeg, rng.uniform(0, kTwoPi), pos, vel);
        p.add(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], pl.mass);
    }
    constexpr double kAsteroidMass = 1e-12;
    for (uint64_t i = 9; i < c.n; ++i) {
        double a = rng.uniform(belt_in, belt_out), e = rng.uniform(0, 0.2), inc = rng.uniform(0, belt_incl);
        double node = rng.uniform(0, kTwoPi), omega = rng.uniform(0, kTwoPi), M = rng.uniform(0, kTwoPi);
        orbit_state(c.G * (1 + kAsteroidMass), a, e, inc, node, omega, M, pos, vel);
        p.add(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], kAsteroidMass);
    }
    return finish(std::move(p));
}

Bodies build_systems(const Config& c) {
    uint64_t per = count_param(c, "systems.planets", 4);
    uint64_t stars = count_param(c, "systems.stars", 0);
    if (stars == 0) stars = c.n / (1 + per);
    if (stars == 0) throw std::runtime_error("n is too small for even one planetary system");
    double Mc = positive(c, "systems.cluster_mass", 1), ac = positive(c, "systems.cluster_a", 1);
    double fm = positive(c, "systems.planet_mass", 1e-3);
    double rmin = positive(c, "systems.r_min", 0.002), rmax = positive(c, "systems.r_max", 0.02);
    double cut = positive(c, "systems.cut", 20);
    if (!(rmax >= rmin)) throw std::runtime_error("parameter 'systems.r_max' must be >= 'systems.r_min'");

    Rng rng(c.seed);
    Pts p;
    const double ms = Mc / static_cast<double>(stars);
    std::vector<double> radii(per);
    for (uint64_t s = 0; s < stars; ++s) {
        auto st = sample_plummer(rng, c.G, Mc, ac, cut);
        p.add(st.x, st.y, st.z, st.vx, st.vy, st.vz, ms);
        for (auto& r : radii) r = rmin * std::pow(rmax / rmin, rng.uniform());
        std::sort(radii.begin(), radii.end());
        double nx, ny, nz;
        rng.unit_vector(nx, ny, nz);
        // Orthonormal in-plane basis from a helper axis that is not parallel to the normal.
        double hx = std::fabs(nx) < 0.9 ? 1 : 0, hy = 1 - hx;
        double e1x = ny * 0 - nz * hy, e1y = nz * hx - nx * 0, e1z = nx * hy - ny * hx;
        double l = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        e1x /= l; e1y /= l; e1z /= l;
        double e2x = ny * e1z - nz * e1y, e2y = nz * e1x - nx * e1z, e2z = nx * e1y - ny * e1x;
        for (double r : radii) {
            double th = rng.uniform(0, kTwoPi), ct = std::cos(th), sn = std::sin(th);
            double v = std::sqrt(c.G * ms * (1 + fm) / r);
            p.add(st.x + r * (ct * e1x + sn * e2x), st.y + r * (ct * e1y + sn * e2y),
                  st.z + r * (ct * e1z + sn * e2z),
                  st.vx + v * (-sn * e1x + ct * e2x), st.vy + v * (-sn * e1y + ct * e2y),
                  st.vz + v * (-sn * e1z + ct * e2z), fm * ms);
        }
    }
    return finish(std::move(p));
}

}  // namespace

const std::vector<ScenarioInfo>& scenarios() {
    static const std::vector<ScenarioInfo> list = {
        {"plummer", "Plummer sphere in virial equilibrium (Henon units by default)"},
        {"kepler", "two-body Keplerian orbit, starting at apocentre"},
        {"uniform", "uniform sphere or cube, optional velocity dispersion and spin (cold collapse)"},
        {"disk", "exponential rotating disk around a central point mass"},
        {"collision", "two inclined disk galaxies on a bound-or-flyby encounter"},
        {"solar", "Sun, eight planets (J2000) and a main-belt asteroid swarm; AU, yr, Msun"},
        {"systems", "Plummer star cluster where every star carries circular-orbit planets"},
    };
    return list;
}

Bodies build_scenario(const Config& cfg) {
    const std::string& s = cfg.scenario;
    if (s == "plummer") return build_plummer(cfg);
    if (s == "kepler") return build_kepler(cfg);
    if (s == "uniform") return build_uniform(cfg);
    if (s == "disk") return build_disk(cfg);
    if (s == "collision") return build_collision(cfg);
    if (s == "solar") return build_solar(cfg);
    if (s == "systems") return build_systems(cfg);
    std::string names;
    for (auto& i : scenarios()) names += std::string(names.empty() ? "" : ", ") + i.name;
    throw std::runtime_error("unknown scenario '" + s + "' (valid: " + names + ")");
}

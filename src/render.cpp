#include "render.hpp"

#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "stb_image_write.h"

namespace {

struct Projection {
    // Screen x = ax.(x,y,z), screen y = ay.(x,y,z).
    double ax[3], ay[3];
};

Projection parse_view(const std::string& view) {
    if (view == "xy") return {{1, 0, 0}, {0, 1, 0}};
    if (view == "xz") return {{1, 0, 0}, {0, 0, 1}};
    if (view == "yz") return {{0, 1, 0}, {0, 0, 1}};
    size_t comma = view.find(',');
    auto bad = [&]() { throw std::runtime_error("bad view '" + view + "': expected xy|xz|yz|<yaw>,<pitch>"); };
    if (comma == std::string::npos) bad();
    std::string a = view.substr(0, comma), b = view.substr(comma + 1);
    char *ea = nullptr, *eb = nullptr;
    double yaw = std::strtod(a.c_str(), &ea), pitch = std::strtod(b.c_str(), &eb);
    auto only_space = [](const char* p) { while (*p == ' ' || *p == '\t') ++p; return *p == 0; };
    if (a.empty() || b.empty() || !only_space(ea) || !only_space(eb) || !std::isfinite(yaw) ||
        !std::isfinite(pitch))
        bad();
    const double k = std::numbers::pi / 180;
    double cy = std::cos(yaw * k), sy = std::sin(yaw * k);
    double cp = std::cos(pitch * k), sp = std::sin(pitch * k);
    // Rotate about z by yaw, then about x by pitch; screen x is the rotated x, screen y the rotated y.
    return {{cy, -sy, 0}, {cp * sy, cp * cy, -sp}};
}

struct Rgb { float r, g, b; };

Rgb ramp(float t) {
    static const struct { float t; Rgb c; } stops[] = {
        {0.00f, {10, 12, 70}}, {0.35f, {120, 30, 150}}, {0.70f, {255, 140, 30}}, {1.00f, {255, 255, 255}}};
    t = std::clamp(t, 0.f, 1.f);
    for (int i = 1; i < 4; ++i) {
        if (t <= stops[i].t) {
            float u = (t - stops[i - 1].t) / (stops[i].t - stops[i - 1].t);
            const Rgb &p = stops[i - 1].c, &q = stops[i].c;
            return {p.r + u * (q.r - p.r), p.g + u * (q.g - p.g), p.b + u * (q.b - p.b)};
        }
    }
    return stops[3].c;
}

Rgb speed_ramp(float t) {
    static const struct { float t; Rgb c; } stops[] = {
        {0.00f, {40, 90, 255}}, {0.40f, {80, 230, 255}}, {0.70f, {255, 230, 90}}, {1.00f, {255, 60, 40}}};
    t = std::clamp(t, 0.f, 1.f);
    for (int i = 1; i < 4; ++i) {
        if (t <= stops[i].t) {
            float u = (t - stops[i - 1].t) / (stops[i].t - stops[i - 1].t);
            const Rgb &p = stops[i - 1].c, &q = stops[i].c;
            return {p.r + u * (q.r - p.r), p.g + u * (q.g - p.g), p.b + u * (q.b - p.b)};
        }
    }
    return stops[3].c;
}

void append_bytes(void* ctx, void* data, int size) {
    auto* out = static_cast<std::vector<unsigned char>*>(ctx);
    auto* p = static_cast<unsigned char*>(data);
    out->insert(out->end(), p, p + size);
}

}  // namespace

Image render_image(const Bodies& b, const RenderOptions& o) {
    if (o.width < 1 || o.height < 1 || o.width > 65536 || o.height > 65536)
        throw std::runtime_error("render: bad image size");
    if (!(o.extent >= 0)) throw std::runtime_error("render: extent must be >= 0");
    const Projection P = parse_view(o.view);
    const size_t n = b.size();
    const int W = o.width, H = o.height;
    const bool by_speed = o.color == ColorMode::speed;

    auto sx = [&](size_t i) { return P.ax[0] * b.x[i] + P.ax[1] * b.y[i] + P.ax[2] * b.z[i]; };
    auto sy = [&](size_t i) { return P.ay[0] * b.x[i] + P.ay[1] * b.y[i] + P.ay[2] * b.z[i]; };
    auto speed = [&](size_t i) {
        return std::sqrt(b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i] + b.vz[i] * b.vz[i]);
    };

    double mtot = 0, cx = 0, cy = 0, mv2 = 0;
    for (size_t i = 0; i < n; ++i) {
        mtot += b.m[i];
        cx += b.m[i] * sx(i);
        cy += b.m[i] * sy(i);
        if (by_speed) mv2 += b.m[i] * (b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i] + b.vz[i] * b.vz[i]);
    }
    if (mtot > 0) { cx /= mtot; cy /= mtot; }
    // Hue saturates at twice the rms speed, so a few fast escapers do not wash everything blue.
    const float vref_speed = mtot > 0 && mv2 > 0 ? static_cast<float>(2 * std::sqrt(mv2 / mtot)) : 1.f;

    double extent = o.extent;
    if (extent == 0) {
        std::vector<float> d(n);
        for (size_t i = 0; i < n; ++i) d[i] = static_cast<float>(std::hypot(sx(i) - cx, sy(i) - cy));
        if (n > 0) {
            size_t k = std::min(n - 1, static_cast<size_t>(0.99 * static_cast<double>(n)));
            std::nth_element(d.begin(), d.begin() + k, d.end());
            extent = 1.2 * d[k];
        }
        if (!(extent > 0)) extent = 1;
    }
    const double scale = W / (2 * extent);

    // Mass, then (speed mode) mass * speed, interleaved per pixel.
    const int ch = by_speed ? 2 : 1;
    const size_t npix = static_cast<size_t>(W) * H;
    std::vector<float> img(npix * ch, 0.f);
    auto splat = [&](float* im, size_t i) {
        double px = (sx(i) - cx) * scale + W * 0.5 - 0.5;
        double py = H * 0.5 - (sy(i) - cy) * scale - 0.5;
        double fx0 = std::floor(px), fy0 = std::floor(py);
        if (!(fx0 > -2 && fx0 < W + 1 && fy0 > -2 && fy0 < H + 1)) return;
        int ix = static_cast<int>(fx0), iy = static_cast<int>(fy0);
        float fx = static_cast<float>(px - fx0), fy = static_cast<float>(py - fy0), m = b.m[i];
        float mv = by_speed ? m * speed(i) : 0.f;
        const float w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
        for (int k = 0; k < 4; ++k) {
            int xx = ix + (k & 1), yy = iy + (k >> 1);
            if (xx >= 0 && xx < W && yy >= 0 && yy < H) {
                size_t p = (static_cast<size_t>(yy) * W + xx) * ch;
                im[p] += m * w[k];
                if (by_speed) im[p + 1] += mv * w[k];
            }
        }
    };

    // Per-thread images avoid atomics; the cap bounds their memory at large resolutions.
    const int nt = n >= 200000 ? std::min(omp_get_max_threads(), 16) : 1;
    if (nt > 1) {
        std::vector<std::vector<float>> local(nt);
#pragma omp parallel num_threads(nt)
        {
            auto& im = local[omp_get_thread_num()];
            im.assign(img.size(), 0.f);
#pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); ++i) splat(im.data(), static_cast<size_t>(i));
        }
#pragma omp parallel for schedule(static)
        for (long long p = 0; p < static_cast<long long>(img.size()); ++p) {
            float s = 0;
            for (auto& im : local)
                if (!im.empty()) s += im[p];
            img[p] = s;
        }
    } else {
        for (size_t i = 0; i < n; ++i) splat(img.data(), i);
    }

    double sum = 0;
    std::vector<float> lit;
    lit.reserve(npix / 4);
    for (size_t p = 0; p < npix; ++p) {
        float v = img[p * ch];
        if (v > 0) { sum += v; lit.push_back(v); }
    }
    const size_t nz = lit.size();
    // White point at the 99.5th percentile of lit pixels, not the maximum: one heavy body (a
    // central black hole, the Sun) would otherwise set the scale and leave everything else dark.
    float vmax = 0;
    if (nz > 0) {
        size_t k = std::min(nz - 1, static_cast<size_t>(0.995 * static_cast<double>(nz)));
        std::nth_element(lit.begin(), lit.begin() + k, lit.end());
        vmax = lit[k];
    }
    Image out;
    out.width = W;
    out.height = H;
    out.extent = extent;
    out.rgb.assign(npix * 3, 0);
    if (nz > 0) {
        const float vref = static_cast<float>(sum / nz);
        const float norm = 1.f / std::max(std::log1p(vmax / vref), 1e-6f);
#pragma omp parallel for schedule(static) if (npix >= 500000)
        for (long long pl = 0; pl < static_cast<long long>(npix); ++pl) {
            size_t p = static_cast<size_t>(pl);
            float m = img[p * ch];
            if (m <= 0) continue;
            float t = std::log1p(m / vref) * norm;
            Rgb c;
            if (by_speed) {
                Rgb h = speed_ramp(img[p * ch + 1] / m / vref_speed);
                // Brightness from density, with a floor so sparse regions still show their colour.
                float k = 0.25f + 0.75f * t;
                c = {h.r * k, h.g * k, h.b * k};
            } else {
                c = ramp(t);
            }
            out.rgb[3 * p] = static_cast<unsigned char>(c.r + 0.5f);
            out.rgb[3 * p + 1] = static_cast<unsigned char>(c.g + 0.5f);
            out.rgb[3 * p + 2] = static_cast<unsigned char>(c.b + 0.5f);
        }
    }
    return out;
}

void render_png(const Bodies& b, const RenderOptions& o, const std::string& path) {
    Image img = render_image(b, o);
    if (!stbi_write_png(path.c_str(), img.width, img.height, 3, img.rgb.data(), img.width * 3))
        throw std::runtime_error("render: cannot write '" + path + "'");
}

std::vector<unsigned char> encode_jpeg(const Image& img, int quality) {
    std::vector<unsigned char> out;
    out.reserve(img.rgb.size() / 8);
    if (!stbi_write_jpg_to_func(append_bytes, &out, img.width, img.height, 3, img.rgb.data(), quality))
        throw std::runtime_error("render: JPEG encoding failed");
    return out;
}

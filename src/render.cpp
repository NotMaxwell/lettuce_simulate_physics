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

}  // namespace

void render_png(const Bodies& b, const RenderOptions& o, const std::string& path) {
    if (o.width < 1 || o.height < 1 || o.width > 65536 || o.height > 65536)
        throw std::runtime_error("render: bad image size");
    if (!(o.extent >= 0)) throw std::runtime_error("render: extent must be >= 0");
    const Projection P = parse_view(o.view);
    const size_t n = b.size();
    const int W = o.width, H = o.height;

    auto sx = [&](size_t i) { return P.ax[0] * b.x[i] + P.ax[1] * b.y[i] + P.ax[2] * b.z[i]; };
    auto sy = [&](size_t i) { return P.ay[0] * b.x[i] + P.ay[1] * b.y[i] + P.ay[2] * b.z[i]; };

    double mtot = 0, cx = 0, cy = 0;
    for (size_t i = 0; i < n; ++i) { mtot += b.m[i]; cx += b.m[i] * sx(i); cy += b.m[i] * sy(i); }
    if (mtot > 0) { cx /= mtot; cy /= mtot; }

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

    std::vector<float> img(static_cast<size_t>(W) * H, 0.f);
    auto splat = [&](float* im, size_t i) {
        double px = (sx(i) - cx) * scale + W * 0.5 - 0.5;
        double py = H * 0.5 - (sy(i) - cy) * scale - 0.5;
        double fx0 = std::floor(px), fy0 = std::floor(py);
        if (!(fx0 > -2 && fx0 < W + 1 && fy0 > -2 && fy0 < H + 1)) return;
        int ix = static_cast<int>(fx0), iy = static_cast<int>(fy0);
        float fx = static_cast<float>(px - fx0), fy = static_cast<float>(py - fy0), m = b.m[i];
        const float w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
        for (int k = 0; k < 4; ++k) {
            int xx = ix + (k & 1), yy = iy + (k >> 1);
            if (xx >= 0 && xx < W && yy >= 0 && yy < H) im[static_cast<size_t>(yy) * W + xx] += m * w[k];
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
    size_t nz = 0;
    float vmax = 0;
    for (float v : img)
        if (v > 0) { sum += v; ++nz; vmax = std::max(vmax, v); }
    std::vector<unsigned char> rgb(img.size() * 3, 0);
    if (nz > 0) {
        const float vref = static_cast<float>(sum / nz);
        const float norm = 1.f / std::log1p(vmax / vref);
        for (size_t p = 0; p < img.size(); ++p) {
            if (img[p] <= 0) continue;
            Rgb c = ramp(std::log1p(img[p] / vref) * norm);
            rgb[3 * p] = static_cast<unsigned char>(c.r + 0.5f);
            rgb[3 * p + 1] = static_cast<unsigned char>(c.g + 0.5f);
            rgb[3 * p + 2] = static_cast<unsigned char>(c.b + 0.5f);
        }
    }
    if (!stbi_write_png(path.c_str(), W, H, 3, rgb.data(), W * 3))
        throw std::runtime_error("render: cannot write '" + path + "'");
}

// lettuce-render: turn a snapshot into a PNG.
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "render.hpp"
#include "snapshot.hpp"

namespace {

// Full-consume parses: std::stoi/stod would take "1e3" as 1.
double parse_full(const std::string& k, const std::string& v) {
    char* end = nullptr;
    double x = std::strtod(v.c_str(), &end);
    if (v.empty() || *end != 0 || !std::isfinite(x)) throw std::runtime_error("option '" + k + "': bad number '" + v + "'");
    return x;
}

// Same bounds as the render_width/render_height config keys.
int parse_size(const std::string& k, const std::string& v) {
    char* end = nullptr;
    errno = 0;
    long x = std::strtol(v.c_str(), &end, 10);
    if (v.empty() || *end != 0 || errno != 0 || x < 16 || x > 16384)
        throw std::runtime_error("option '" + k + "': expected an integer in 16..16384, got '" + v + "'");
    return static_cast<int>(x);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fputs("usage: lettuce-render <snapshot> <out.png> [width=N] [height=N] [view=xy|xz|yz|yaw,pitch] [extent=X]\n",
                   stderr);
        return 2;
    }
    try {
        RenderOptions o;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            auto eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value, got '" + a + "'");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "width") o.width = parse_size(k, v);
            else if (k == "height") o.height = parse_size(k, v);
            else if (k == "view") o.view = v;
            else if (k == "extent") o.extent = parse_full(k, v);
            else throw std::runtime_error("unknown option '" + k + "'");
        }
        Bodies b;
        SnapshotHeader h = read_snapshot(argv[1], b);
        render_png(b, o, argv[2]);
        std::printf("%s: %s n=%llu step=%llu t=%g -> %s\n", argv[1], h.scenario.c_str(), static_cast<unsigned long long>(h.n),
                    static_cast<unsigned long long>(h.step), h.time, argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lettuce-render: %s\n", e.what());
        return 1;
    }
}

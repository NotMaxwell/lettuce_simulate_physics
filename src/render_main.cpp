// lettuce-render: turn a snapshot into a PNG.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "render.hpp"
#include "snapshot.hpp"

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
            if (k == "width") o.width = std::stoi(v);
            else if (k == "height") o.height = std::stoi(v);
            else if (k == "view") o.view = v;
            else if (k == "extent") o.extent = std::stod(v);
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

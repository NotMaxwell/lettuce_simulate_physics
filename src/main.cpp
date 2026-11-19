// lettuce: run one simulation from a config file plus key=value overrides.
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "scenarios.hpp"
#include "simulation.hpp"

namespace {

constexpr const char* usage =
    "usage: lettuce [config.ini] [key=value ...]\n"
    "       lettuce --list            list scenarios\n"
    "       lettuce --print [args]    print the effective configuration and exit\n"
    "\n"
    "examples:\n"
    "  lettuce configs/plummer.ini\n"
    "  lettuce configs/disk.ini n=1000000 method=tree steps=2000 render_every=50\n"
    "  lettuce scenario=kepler kepler.e=0.6 steps=20000 dt=0.001 eps=0\n"
    "  lettuce configs/plummer.ini restart=out/snap_00001000.snap steps=1000\n"
    "  lettuce configs/collision.ini steps=0 live=8080      watch in a browser at http://localhost:8080/\n"
    "\n"
    "Every key a config file accepts can be given on the command line; later ones win.\n"
    "See configs/README.md for the keys and each scenario's parameters.\n";

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg;
        bool print_only = false;
        bool have_file = false;
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
                std::fputs(usage, stdout);
                return 0;
            }
            if (!std::strcmp(a, "--list")) {
                for (auto& s : scenarios()) std::printf("  %-12s %s\n", s.name, s.summary);
                return 0;
            }
            if (!std::strcmp(a, "--print")) {
                print_only = true;
            } else if (std::strchr(a, '=')) {
                apply_override(cfg, a);
            } else if (!have_file && i == 1) {
                cfg = load_config(a);
                have_file = true;
            } else {
                std::fprintf(stderr, "lettuce: unexpected argument '%s' (overrides are key=value)\n%s", a, usage);
                return 2;
            }
        }
        validate(cfg);
        if (print_only) {
            std::fputs(dump_config(cfg).c_str(), stdout);
            return 0;
        }
        run_simulation(cfg, stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lettuce: %s\n", e.what());
        return 1;
    }
}

// Runs every registered test, or only those whose name contains argv[1].
#include <cstring>
#include <exception>

#include "test.hpp"

int main(int argc, char** argv) {
    int ran = 0, failed = 0, skipped = 0;
    for (auto& c : t::registry()) {
        if (argc > 1 && !std::strstr(c.name, argv[1])) continue;
        int before = t::failures();
        std::fprintf(stderr, "[ RUN  ] %s\n", c.name);
        try {
            c.fn();
        } catch (const t::Skip& s) {
            std::fprintf(stderr, "[ SKIP ] %s: %s\n", c.name, s.why.c_str());
            ++skipped;
            continue;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  threw: %s\n", e.what());
            ++t::failures();
        }
        ++ran;
        bool ok = t::failures() == before;
        if (!ok) ++failed;
        std::fprintf(stderr, "[ %s ] %s\n", ok ? " OK " : "FAIL", c.name);
    }
    std::fprintf(stderr, "%d run, %d failed, %d skipped\n", ran, failed, skipped);
    return failed ? 1 : 0;
}

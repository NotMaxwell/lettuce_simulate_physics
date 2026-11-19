// MPI test runner: every rank runs every test in the same order (tests are collective); failures
// are summed over ranks. Run with mpirun -n <1|2|4> lettuce-mpi-tests [filter].
#include <cstring>
#include <exception>

#include "parallel.hpp"
#include "test.hpp"

int main(int argc, char** argv) {
    par::init(&argc, &argv);
    const bool root = par::root();
    int ran = 0, failed = 0;
    for (auto& c : t::registry()) {
        if (argc > 1 && !std::strstr(c.name, argv[1])) continue;
        int before = t::failures();
        if (root) std::fprintf(stderr, "[ RUN  ] %s (%d ranks)\n", c.name, par::size());
        try {
            c.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  rank %d threw: %s\n", par::rank(), e.what());
            ++t::failures();
        }
        ++ran;
        bool ok = par::sum(uint64_t(t::failures() - before)) == 0;
        if (!ok) ++failed;
        if (root) std::fprintf(stderr, "[ %s ] %s\n", ok ? " OK " : "FAIL", c.name);
    }
    if (root) std::fprintf(stderr, "%d run, %d failed (%d ranks)\n", ran, failed, par::size());
    par::finalize();
    return failed ? 1 : 0;
}

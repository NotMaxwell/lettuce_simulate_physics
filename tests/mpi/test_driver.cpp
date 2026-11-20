// The run driver end to end (run_simulation) over however many ranks the runner was started with:
// restart from snapshot pieces, diag.csv on restart, final outputs on a live-view stop, and the
// per-rank body cap of the rebalance.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

#include "backend.hpp"
#include "config.hpp"
#include "parallel.hpp"
#include "scenarios.hpp"
#include "simulation.hpp"
#include "snapshot.hpp"
#include "test.hpp"

namespace fs = std::filesystem;

namespace {

std::string piece(const std::string& dir, uint64_t step, int r) {
    char name[64];
    if (par::size() == 1 && r < 0)
        std::snprintf(name, sizeof name, "snap_%08llu.snap", static_cast<unsigned long long>(step));
    else
        std::snprintf(name, sizeof name, "snap_%08llu_r%d.snap", static_cast<unsigned long long>(step), r);
    return (fs::path(dir) / name).string();
}

// A fresh, empty directory every rank agrees on.
std::string fresh_dir(const char* name) {
    std::string dir = std::string(".scratch/mpi_driver_") + name;
    if (par::root()) {
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    par::barrier();
    return dir;
}

Config base(const std::string& dir) {
    Config c;
    c.scenario = "plummer";
    c.n = 2000;
    c.dt = 1.0 / 128;
    c.method = Method::tree;
    c.backend = BackendKind::cpu;
    c.threads = 1;
    c.out_dir = dir;
    return c;
}

RunSummary run(const Config& c) {
    std::FILE* devnull = std::fopen("/dev/null", "w");
    RunSummary s = run_simulation(c, devnull);
    std::fclose(devnull);
    return s;
}

std::vector<uint64_t> diag_steps(const std::string& dir) {
    std::ifstream in(fs::path(dir) / "diag.csv");
    std::vector<uint64_t> out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) out.push_back(std::stoull(line));
    return out;
}

}  // namespace

TEST("mpi: restart reads the recorded pieces, not stale higher ones, and diag.csv has no repeats") {
    const int P = par::size();
    std::string dir = fresh_dir("restart");
    Config c = base(dir);
    c.steps = 20;
    c.diag_every = 10;
    c.snap_every = 10;
    run(c);
    // A stale piece beyond this run's count, as a run with more ranks into the same dir leaves behind.
    if (par::root() && P > 1) {
        Bodies extra;
        for (int i = 0; i < 500; ++i) extra.push(1.f + i, 0, 0, 0, 0, 0, 1e-3);
        SnapshotHeader h;
        h.step = 10;
        h.pieces = uint32_t(P + 1);
        write_snapshot(piece(dir, 10, P), h, extra);
    }
    par::barrier();
    Config r = c;
    r.steps = 10;
    r.restart = P == 1 ? piece(dir, 10, -1) : piece(dir, 10, 0);
    r.snap_every = 0;
    r.snap_final = true;
    run(r);
    // The restarted run's final snapshot holds exactly the original bodies.
    uint64_t n = 0;
    std::set<uint32_t> ids;
    for (int k = 0; k < (P == 1 ? 1 : P); ++k) {
        Bodies b;
        read_snapshot(P == 1 ? piece(dir, 20, -1) : piece(dir, 20, k), b);
        n += b.size();
        ids.insert(b.id.begin(), b.id.end());
    }
    CHECK(n == 2000);
    CHECK(ids.size() == 2000);
    if (par::root()) CHECK((diag_steps(dir) == std::vector<uint64_t>{0, 10, 20}));
    par::barrier();
}

TEST("mpi: restart refuses a piece from a different snapshot") {
    const int P = par::size();
    if (P == 1) return;  // a single-rank snapshot is one file
    std::string dir = fresh_dir("mixed");
    Config c = base(dir);
    c.steps = 0;
    c.snap_final = true;
    run(c);
    if (par::root()) {
        Bodies b;
        SnapshotHeader h = read_snapshot(piece(dir, 0, 1), b);
        h.pieces = uint32_t(P + 2);
        write_snapshot(piece(dir, 0, 1), h, b);
    }
    par::barrier();
    Config r = c;
    r.restart = piece(dir, 0, 0);
    bool threw = false;
    try { run(r); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

TEST("mpi: stop from the live view still writes the final snapshot") {
    std::string dir = fresh_dir("stop");
    Config c = base(dir);
    c.steps = 0;
    c.snap_final = true;
    c.live_paused = true;
    c.live = par::broadcast(20000 + int(::getpid() % 20000));
    std::thread stopper;
    if (par::root()) {
        stopper = std::thread([port = c.live] {
            // Retries until the server is up; the request that gets through stops the paused run.
            for (int attempt = 0; attempt < 200; ++attempt) {
                int fd = ::socket(AF_INET, SOCK_STREAM, 0);
                sockaddr_in a{};
                a.sin_family = AF_INET;
                a.sin_port = htons(uint16_t(port));
                a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0) {
                    const char req[] = "GET /control?stop=1 HTTP/1.0\r\n\r\n";
                    (void)!::write(fd, req, sizeof req - 1);
                    char buf[256];
                    while (::read(fd, buf, sizeof buf) > 0) {}
                    ::close(fd);
                    return;
                }
                ::close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });
    }
    run(c);
    if (stopper.joinable()) stopper.join();
    CHECK(fs::exists(piece(dir, 0, par::size() == 1 ? -1 : par::rank())));
    par::barrier();
}

TEST("mpi: rebalance never gives a rank more than max_rank_bodies") {
    // Most bodies in one spot smaller than a Morton cell: one key range would take them all.
    Config c;
    c.n = 2000;
    c.seed = 3;
    Bodies all = build_scenario(c);
    for (int i = 0; i < 6000; ++i) all.push(1e-7f * float(i % 7), 1e-7f * float(i % 5), 1e-7f * float(i % 3), 0, 0, 0, 1e-4);
    for (size_t i = 0; i < all.size(); ++i) all.id[i] = uint32_t(i);
    uint64_t first, last;
    par::even_share(all.size(), first, last);
    Bodies mine;
    for (uint64_t i = first; i < last; ++i) {
        mine.push(all.x[i], all.y[i], all.z[i], 0, 0, 0, all.m[i]);
        mine.id.back() = all.id[i];
    }
    par::DistOptions o;
    o.rebalance_every = 1;
    auto be = par::make_distributed(make_cpu_backend(1), o);
    be->upload(mine);
    ForceParams p;
    p.eps = 0.01;
    p.method = ForceMethod::tree;
    for (int s = 0; s < 3; ++s) be->compute_forces(p, false);
    Bodies got;
    be->download(got);
    CHECK(got.size() <= par::max_rank_bodies(all.size()));
    CHECK(par::sum(uint64_t(got.size())) == all.size());
    // The capped split must follow key order: cut by source rank inside the clump's key range,
    // neighbouring ranks would both span that whole range, their bounding boxes would overlap
    // and each would send the other most of its bodies as LET. Keys as the rebalance computes them
    // (positions have not moved: no kicks or drifts here).
    float lo[3] = {3e38f, 3e38f, 3e38f}, hi[3] = {-3e38f, -3e38f, -3e38f};
    for (size_t i = 0; i < all.size(); ++i) {
        const float v[3] = {all.x[i], all.y[i], all.z[i]};
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], v[k]);
            hi[k] = std::max(hi[k], v[k]);
        }
    }
    const double side = std::max({double(hi[0]) - lo[0], double(hi[1]) - lo[1], double(hi[2]) - lo[2], 1e-30});
    const double scale = double((1u << 21) - 1) / side;
    auto spread = [](uint64_t v) {
        uint64_t r = 0;
        for (int b = 0; b < 21; ++b) r |= ((v >> b) & 1) << (3 * b);
        return r;
    };
    auto morton = [&](size_t i) {
        auto q = [&](float v, float o) { return uint64_t(std::clamp((double(v) - o) * scale, 0.0, 2097151.0)); };
        return spread(q(got.x[i], lo[0])) << 2 | spread(q(got.y[i], lo[1])) << 1 | spread(q(got.z[i], lo[2]));
    };
    uint64_t kmin = UINT64_MAX, kmax = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        kmin = std::min(kmin, morton(i));
        kmax = std::max(kmax, morton(i));
    }
    bool ordered = true;
    uint64_t prev_max = 0;
    for (int r = 0; r < par::size(); ++r) {
        const bool me = par::rank() == r;
        const uint64_t nr = par::sum(uint64_t(me ? got.size() : 0));
        const uint64_t rmin = par::sum(me ? kmin : uint64_t(0)), rmax = par::sum(me ? kmax : uint64_t(0));
        if (nr == 0) continue;
        ordered = ordered && rmin >= prev_max;  // neighbours may share only the boundary key
        prev_max = rmax;
    }
    CHECK(ordered);
    Bodies gathered;
    par::gather_sample(*be, gathered, UINT64_MAX / 2);
    if (par::root()) CHECK(std::set<uint32_t>(gathered.id.begin(), gathered.id.end()).size() == all.size());
}

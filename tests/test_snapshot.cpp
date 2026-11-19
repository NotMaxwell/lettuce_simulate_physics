#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "scenarios.hpp"
#include "snapshot.hpp"
#include "test.hpp"

namespace {
std::string tmp_path(const char* name) {
    std::filesystem::create_directories(".scratch");
    return std::string(".scratch/") + name;
}
}  // namespace

TEST("snapshot: bit-exact round trip") {
    Config c;
    c.scenario = "plummer";
    c.n = 1234;
    c.seed = 5;
    Bodies b = build_scenario(c);
    for (size_t i = 0; i < b.size(); ++i) b.id[i] = static_cast<uint32_t>(b.size() - 1 - i);

    SnapshotHeader h;
    h.n = b.size();
    h.step = 987654321012ull;
    h.time = 12.3456789012345;
    h.G = 6.5;
    h.eps = 0.0123;
    h.dt = 1.0 / 128;
    h.seed = 5;
    h.scenario = "plummer";
    std::string path = tmp_path("snap_rt.bin");
    write_snapshot(path, h, b);
    CHECK(!std::filesystem::exists(path + ".tmp"));
    CHECK(std::filesystem::file_size(path) >= 4096u * 8);

    Bodies r;
    SnapshotHeader g = read_snapshot(path, r);
    CHECK(g.n == h.n && g.step == h.step && g.seed == h.seed && g.scenario == h.scenario);
    CHECK(g.time == h.time && g.G == h.G && g.eps == h.eps && g.dt == h.dt);
    CHECK(r.x == b.x && r.y == b.y && r.z == b.z);
    CHECK(r.vx == b.vx && r.vy == b.vy && r.vz == b.vz);
    CHECK(r.m == b.m && r.id == b.id);
}

TEST("snapshot: empty and long scenario name") {
    Bodies b, r;
    SnapshotHeader h;
    h.scenario = std::string(50, 'q');
    std::string path = tmp_path("snap_empty.bin");
    write_snapshot(path, h, b);
    SnapshotHeader g = read_snapshot(path, r);
    CHECK(g.n == 0 && r.size() == 0);
    CHECK(g.scenario == std::string(31, 'q'));
}

TEST("snapshot: rejects bad files") {
    Config c;
    c.n = 50;
    Bodies b = build_scenario(c), r;
    std::string path = tmp_path("snap_bad.bin");
    write_snapshot(path, SnapshotHeader{}, b);

    auto read_all = [&] {
        std::ifstream f(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), {});
    };
    auto expect_throw = [&](const std::string& bytes) {
        std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
        try { read_snapshot(path, r); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    std::string good = read_all();
    std::string badmagic = good;
    badmagic[0] = 'X';
    CHECK(expect_throw(badmagic));
    std::string badver = good;
    badver[8] = 9;
    CHECK(expect_throw(badver));
    CHECK(expect_throw(good.substr(0, good.size() - 8)));
    CHECK(expect_throw(good.substr(0, 100)));
    CHECK(expect_throw(""));
    std::string hugen = good;
    uint64_t huge = ~0ull;
    std::memcpy(&hugen[16], &huge, 8);
    CHECK(expect_throw(hugen));
    bool missing = false;
    try { read_snapshot(tmp_path("does_not_exist.bin"), r); } catch (const std::runtime_error&) { missing = true; }
    CHECK(missing);
}

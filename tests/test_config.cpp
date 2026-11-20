#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "config.hpp"
#include "scenarios.hpp"
#include "test.hpp"

namespace {

std::string write_tmp(const std::string& name, const std::string& text) {
    std::filesystem::create_directories(".scratch");
    std::string path = ".scratch/" + name;
    std::ofstream(path) << text;
    return path;
}

template <class F> std::string error_of(F f) {
    try { f(); } catch (const std::runtime_error& e) { return e.what(); }
    return "";
}

}  // namespace

TEST("config: sections, comments, bools, enums") {
    auto path = write_tmp("cfg_basic.ini",
        "# full-line comment\n"
        "; another\n"
        "\n"
        "scenario = disk   # trailing\n"
        "n = 1_000_000\n"
        "G = 2.5 ; trailing\n"
        "method = tree\n"
        "backend = CPU\n"
        "exact_pe = yes\n"
        "snap_final = off\n"
        "[disk]\n"
        "radius = 2\n"
        "[]\n"
        "dt = 0.25\n"
        "[general]\n"
        "steps = 7\n"
        "[collision]\n"
        "speed = 0.5\n");
    Config c = load_config(path);
    CHECK(c.scenario == "disk");
    CHECK(c.n == 1000000);
    CHECK_NEAR(c.G, 2.5, 0);
    CHECK(c.method == Method::tree);
    CHECK(c.backend == BackendKind::cpu);
    CHECK(c.exact_pe);
    CHECK(!c.snap_final);
    CHECK_NEAR(c.dt, 0.25, 0);
    CHECK(c.steps == 7);
    CHECK(c.params.at("disk.radius") == "2");
    CHECK_NEAR(c.param("collision.speed", 0), 0.5, 0);
}

TEST("config: integer forms") {
    Config c;
    apply_override(c, "n=1e6");
    CHECK(c.n == 1000000);
    apply_override(c, "n=1_000");
    CHECK(c.n == 1000);
    apply_override(c, "n = 18446744073709551615");
    CHECK(c.n == 18446744073709551615ull);
    apply_override(c, "steps=2.5e3");
    CHECK(c.steps == 2500);
    CHECK(!error_of([&] { apply_override(c, "n=1.5"); }).empty());
    CHECK(!error_of([&] { apply_override(c, "n=-3"); }).empty());
    CHECK(!error_of([&] { apply_override(c, "n=abc"); }).empty());
    CHECK(!error_of([&] { apply_override(c, "n=1e30"); }).empty());
    CHECK(!error_of([&] { apply_override(c, "exact_pe=maybe"); }).empty());
    CHECK(!error_of([&] { apply_override(c, "method=fast"); }).empty());
    CHECK(!error_of([&] { apply_override(c, "novalue"); }).empty());
}

TEST("config: overrides and params") {
    Config c;
    apply_override(c, "disk.thickness=0.1");
    apply_override(c, "theta = 0.4");
    CHECK_NEAR(c.theta, 0.4, 0);
    CHECK_NEAR(c.param("disk.thickness", 9), 0.1, 0);
    CHECK_NEAR(c.param("disk.other", 9), 9, 0);
    CHECK(c.param_str("disk.thickness", "x") == "0.1");
    CHECK(!error_of([&] { apply_override(c, "bad.key=abc"); c.param("bad.key", 0); }).empty());
}

TEST("config: unknown key names the key and line") {
    auto path = write_tmp("cfg_bad.ini", "n = 5\n\nfrobnicate = 3\n");
    std::string e = error_of([&] { load_config(path); });
    CHECK(e.find("frobnicate") != std::string::npos);
    CHECK(e.find("cfg_bad.ini:3") != std::string::npos);
    std::string e2 = error_of([] { Config c; apply_override(c, "wibble=1"); });
    CHECK(e2.find("wibble") != std::string::npos);
    CHECK(!error_of([] { load_config(".scratch/does_not_exist.ini"); }).empty());
}

TEST("config: unused_params") {
    Config c;
    apply_override(c, "disk.mass=2");
    apply_override(c, "disk.typo=3");
    CHECK_NEAR(c.param("disk.mass", 1), 2, 0);
    auto u = c.unused_params();
    CHECK(u.size() == 1);
    CHECK(!u.empty() && u[0] == "disk.typo");
}

TEST("config: validate") {
    Config c;
    validate(c);
    auto bad = [](auto mutate) {
        Config c;
        mutate(c);
        return !error_of([&] { validate(c); }).empty();
    };
    CHECK(bad([](Config& c) { c.n = 0; }));
    CHECK(bad([](Config& c) { c.dt = 0; }));
    CHECK(bad([](Config& c) { c.eps = -1; }));
    CHECK(bad([](Config& c) { c.theta = 0; }));
    CHECK(bad([](Config& c) { c.G = 0; }));
    CHECK(bad([](Config& c) { c.render_width = 8; }));
    CHECK(bad([](Config& c) { c.render_height = 20000; }));
    Config ok;
    ok.steps = 0;
    ok.eps = 0;
    validate(ok);
}

TEST("config: dump round trip") {
    Config c;
    c.scenario = "collision";
    c.n = 123456;
    c.seed = 99;
    c.G = 39.47841760435743;
    c.eps = 1.0 / 3.0;
    c.dt = 1e-7;
    c.steps = 42;
    c.method = Method::direct;
    c.theta = 0.55;
    c.auto_direct_max = 777;
    c.backend = BackendKind::cuda;
    c.device = 2;
    c.threads = 3;
    c.diag_every = 5;
    c.exact_pe = true;
    c.out_dir = "some/dir";
    c.snap_every = 11;
    c.snap_final = true;
    c.render_every = 13;
    c.render_width = 640;
    c.render_height = 480;
    c.view = "30,60";
    c.render_extent = 2.5;
    c.restart = "a/b.snap";
    c.params["disk.scale"] = "1.5";
    c.params["collision.retrograde"] = "1";
    auto path = write_tmp("cfg_dump.ini", dump_config(c));
    Config d = load_config(path);
    CHECK(d.scenario == c.scenario && d.n == c.n && d.seed == c.seed);
    CHECK(d.G == c.G && d.eps == c.eps && d.dt == c.dt && d.steps == c.steps);
    CHECK(d.method == c.method && d.theta == c.theta && d.auto_direct_max == c.auto_direct_max);
    CHECK(d.backend == c.backend && d.device == c.device && d.threads == c.threads);
    CHECK(d.diag_every == c.diag_every && d.exact_pe == c.exact_pe);
    CHECK(d.out_dir == c.out_dir && d.snap_every == c.snap_every && d.snap_final == c.snap_final);
    CHECK(d.render_every == c.render_every && d.render_width == c.render_width);
    CHECK(d.render_height == c.render_height && d.view == c.view && d.render_extent == c.render_extent);
    CHECK(d.restart == c.restart && d.params == c.params);
    CHECK(dump_config(d) == dump_config(c));
}

TEST("config: enum to_string") {
    CHECK(std::string(to_string(Method::auto_)) == "auto");
    CHECK(std::string(to_string(Method::tree)) == "tree");
    CHECK(std::string(to_string(Method::direct)) == "direct");
    CHECK(std::string(to_string(BackendKind::cpu)) == "cpu");
    CHECK(std::string(to_string(BackendKind::cuda)) == "cuda");
    CHECK(std::string(to_string(BackendKind::auto_)) == "auto");
}

TEST("config: shipped configs load, validate and use all their params") {
    for (auto& e : std::filesystem::directory_iterator("configs")) {
        if (e.path().extension() != ".ini") continue;
        Config c = load_config(e.path().string());
        validate(c);
        c.n = c.scenario == "solar" ? 50 : 200;
        CHECK(build_scenario(c).size() > 0);
        CHECK(c.unused_params().empty());
    }
}

TEST("config: comment markers need leading whitespace; view and n are validated") {
    auto path = write_tmp("cfg_hash.ini",
        "out_dir = runs/exp#2\n"
        "view = xz   # trailing\n"
        "G = 2 ;c\n");
    Config c = load_config(path);
    CHECK(c.out_dir == "runs/exp#2");
    CHECK(c.view == "xz");
    CHECK_NEAR(c.G, 2, 0);

    Config d;
    apply_override(d, "out_dir=runs/exp#2");
    CHECK(d.out_dir == "runs/exp#2");

    CHECK(!error_of([&] { apply_override(d, "view=bogus"); }).empty());
    CHECK(!error_of([&] { load_config(write_tmp("cfg_badview.ini", "view = 1,x\n")); }).empty());
    Config big;
    big.n = 1ull << 32;
    CHECK(!error_of([&] { validate(big); }).empty());
    big.n = (1ull << 32) - 1;
    validate(big);
}

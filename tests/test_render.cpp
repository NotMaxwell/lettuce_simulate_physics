#include <cmath>
#include <cstdint>
#include <limits>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "render.hpp"
#include "scenarios.hpp"
#include "test.hpp"

namespace {

uint32_t be32(const unsigned char* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

void check_png(const std::string& path, uint32_t w, uint32_t h) {
    std::ifstream f(path, std::ios::binary);
    unsigned char hdr[24] = {};
    f.read(reinterpret_cast<char*>(hdr), 24);
    static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    for (int i = 0; i < 8; ++i) CHECK(hdr[i] == sig[i]);
    CHECK(hdr[12] == 'I' && hdr[13] == 'H' && hdr[14] == 'D' && hdr[15] == 'R');
    CHECK(be32(hdr + 16) == w);
    CHECK(be32(hdr + 20) == h);
    CHECK(std::filesystem::file_size(path) > 8);
}

}  // namespace

TEST("render: writes png of the requested size") {
    std::filesystem::create_directories(".scratch");
    Config c;
    c.scenario = "disk";
    c.n = 3000;
    Bodies b = build_scenario(c);
    RenderOptions o;
    o.width = 200;
    o.height = 120;
    for (const char* v : {"xy", "xz", "yz", "30,60", "0, 90"}) {
        o.view = v;
        o.extent = 0;
        std::string path = ".scratch/render_test.png";
        render_png(b, o, path);
        check_png(path, 200, 120);
    }
    o.extent = 3;
    render_png(b, o, ".scratch/render_test.png");
    check_png(".scratch/render_test.png", 200, 120);
}

TEST("render: degenerate inputs still write") {
    Bodies empty, one;
    one.push(1, 2, 3, 0, 0, 0, 1);
    RenderOptions o;
    o.width = 32;
    o.height = 32;
    render_png(empty, o, ".scratch/render_empty.png");
    check_png(".scratch/render_empty.png", 32, 32);
    render_png(one, o, ".scratch/render_one.png");
    check_png(".scratch/render_one.png", 32, 32);
}

TEST("render: bad view throws") {
    Bodies b;
    b.push(0, 0, 0, 0, 0, 0, 1);
    RenderOptions o;
    for (const char* v : {"ab", "", "xyz", "10", "10,", ",10", "a,b", "10,20,30"}) {
        o.view = v;
        bool threw = false;
        try { render_png(b, o, ".scratch/render_bad.png"); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
}

TEST("render: unwritable path throws") {
    Bodies b;
    b.push(0, 0, 0, 0, 0, 0, 1);
    bool threw = false;
    try { render_png(b, RenderOptions{}, ".scratch/no_such_dir/x.png"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

TEST("render: non-finite bodies are skipped, not poisoning the frame") {
    Bodies b;
    for (int i = 0; i < 200; ++i) {
        double t = i * 0.3;
        b.push(std::cos(t), std::sin(t), 0, 0, 0, 0, 1.0);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    RenderOptions o;
    o.width = 64;
    o.height = 64;
    Image clean = render_image(b, o);
    b.push(nan, 0, 0, 0, 0, 0, 1.0);
    b.push(0, inf, 0, 0, 0, 0, 1.0);
    b.push(0, 0, 0, nan, 0, 0, 1.0);
    Image dirty = render_image(b, o);
    CHECK_NEAR(dirty.extent, clean.extent, 0.01);
    int lit = 0;
    for (unsigned char v : dirty.rgb) lit += v > 40;
    CHECK(lit > 50);
    o.color = ColorMode::speed;
    CHECK(render_image(b, o).extent > 0);
}

TEST("render: validate_view") {
    validate_view("xy");
    validate_view("10,20");
    bool threw = false;
    try { validate_view("nope"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

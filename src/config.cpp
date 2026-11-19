#include "config.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

double parse_double(const std::string& key, const std::string& v) {
    if (v.empty()) fail("key '" + key + "': empty value");
    char* end = nullptr;
    errno = 0;
    double d = std::strtod(v.c_str(), &end);
    if (end != v.c_str() + v.size() || !std::isfinite(d))
        fail("key '" + key + "': bad number '" + v + "'");
    return d;
}

// Accepts 1000000, 1_000_000 and 1e6, but not 1.5 or -3: a body count is never fractional.
uint64_t parse_uint(const std::string& key, const std::string& raw) {
    std::string v;
    for (char c : raw)
        if (c != '_') v.push_back(c);
    auto bad = [&]() { fail("key '" + key + "': bad integer '" + raw + "'"); };
    if (v.empty()) bad();
    if (std::all_of(v.begin(), v.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        errno = 0;
        unsigned long long u = std::strtoull(v.c_str(), nullptr, 10);
        if (errno == ERANGE) bad();
        return u;
    }
    char* end = nullptr;
    long double d = std::strtold(v.c_str(), &end);
    if (end != v.c_str() + v.size() || !std::isfinite(d) || d < 0 || d != std::floor(d) ||
        d >= 18446744073709551616.0L)
        bad();
    return static_cast<uint64_t>(d);
}

int parse_int(const std::string& key, const std::string& v) {
    uint64_t u = parse_uint(key, v);
    if (u > 2147483647u) fail("key '" + key + "': integer out of range '" + v + "'");
    return static_cast<int>(u);
}

bool parse_bool(const std::string& key, const std::string& v) {
    std::string s = lower(v);
    if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
    if (s == "false" || s == "0" || s == "no" || s == "off") return false;
    fail("key '" + key + "': bad boolean '" + v + "'");
}

Method parse_method(const std::string& v) {
    std::string s = lower(v);
    if (s == "direct") return Method::direct;
    if (s == "tree") return Method::tree;
    if (s == "auto") return Method::auto_;
    fail("key 'method': expected direct|tree|auto, got '" + v + "'");
}

BackendKind parse_backend(const std::string& v) {
    std::string s = lower(v);
    if (s == "cpu") return BackendKind::cpu;
    if (s == "cuda") return BackendKind::cuda;
    if (s == "auto") return BackendKind::auto_;
    fail("key 'backend': expected cpu|cuda|auto, got '" + v + "'");
}

std::string strip_comment(const std::string& line) {
    size_t p = line.find_first_of("#;");
    return trim(p == std::string::npos ? line : line.substr(0, p));
}

std::string fmt_double(double d) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", d);
    return buf;
}

}  // namespace

double Config::param(const std::string& key, double fallback) const {
    used_params.insert(key);
    auto it = params.find(key);
    return it == params.end() ? fallback : parse_double(key, it->second);
}

std::string Config::param_str(const std::string& key, const std::string& fallback) const {
    used_params.insert(key);
    auto it = params.find(key);
    return it == params.end() ? fallback : it->second;
}

std::vector<std::string> Config::unused_params() const {
    std::vector<std::string> out;
    for (auto& kv : params)
        if (!used_params.count(kv.first)) out.push_back(kv.first);
    return out;
}

void set_key(Config& c, const std::string& key, const std::string& v) {
    if (key.find('.') != std::string::npos) { c.params[key] = v; return; }
    if (key == "scenario") c.scenario = v;
    else if (key == "n") c.n = parse_uint(key, v);
    else if (key == "seed") c.seed = parse_uint(key, v);
    else if (key == "G") c.G = parse_double(key, v);
    else if (key == "eps") c.eps = parse_double(key, v);
    else if (key == "dt") c.dt = parse_double(key, v);
    else if (key == "steps") c.steps = parse_uint(key, v);
    else if (key == "method") c.method = parse_method(v);
    else if (key == "theta") c.theta = parse_double(key, v);
    else if (key == "auto_direct_max") c.auto_direct_max = parse_uint(key, v);
    else if (key == "backend") c.backend = parse_backend(v);
    else if (key == "device") c.device = parse_int(key, v);
    else if (key == "threads") c.threads = parse_int(key, v);
    else if (key == "diag_every") c.diag_every = parse_uint(key, v);
    else if (key == "exact_pe") c.exact_pe = parse_bool(key, v);
    else if (key == "out_dir") c.out_dir = v;
    else if (key == "snap_every") c.snap_every = parse_uint(key, v);
    else if (key == "snap_final") c.snap_final = parse_bool(key, v);
    else if (key == "render_every") c.render_every = parse_uint(key, v);
    else if (key == "render_width") c.render_width = parse_int(key, v);
    else if (key == "render_height") c.render_height = parse_int(key, v);
    else if (key == "view") c.view = v;
    else if (key == "render_extent") c.render_extent = parse_double(key, v);
    else if (key == "restart") c.restart = v;
    else if (key == "live") c.live = parse_int(key, v);
    else if (key == "live_bind") c.live_bind = v;
    else if (key == "live_width") c.live_width = parse_int(key, v);
    else if (key == "live_height") c.live_height = parse_int(key, v);
    else if (key == "live_fps") c.live_fps = parse_double(key, v);
    else if (key == "live_quality") c.live_quality = parse_int(key, v);
    else if (key == "live_hold") c.live_hold = parse_bool(key, v);
    else if (key == "live_paused") c.live_paused = parse_bool(key, v);
    else fail("unknown key '" + key + "'");
}

void apply_override(Config& cfg, const std::string& assignment) {
    size_t eq = assignment.find('=');
    if (eq == std::string::npos) fail("override '" + assignment + "' is not key=value");
    std::string key = trim(assignment.substr(0, eq));
    if (key.empty()) fail("override '" + assignment + "' has an empty key");
    set_key(cfg, key, trim(assignment.substr(eq + 1)));
}

Config load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) fail("cannot open config file '" + path + "'");
    Config cfg;
    std::string prefix, raw;
    int lineno = 0;
    while (std::getline(in, raw)) {
        ++lineno;
        std::string line = strip_comment(raw);
        if (line.empty()) continue;
        try {
            if (line.front() == '[') {
                if (line.back() != ']') fail("unterminated section header '" + line + "'");
                std::string name = trim(line.substr(1, line.size() - 2));
                prefix = (name.empty() || name == "general") ? "" : name + ".";
                continue;
            }
            size_t eq = line.find('=');
            if (eq == std::string::npos) fail("expected 'key = value', got '" + line + "'");
            std::string key = trim(line.substr(0, eq));
            if (key.empty()) fail("empty key");
            set_key(cfg, prefix + key, trim(line.substr(eq + 1)));
        } catch (const std::runtime_error& e) {
            fail(path + ":" + std::to_string(lineno) + ": " + e.what());
        }
    }
    return cfg;
}

void validate(const Config& c) {
    if (c.n < 1) fail("n must be >= 1");
    if (!(c.G > 0)) fail("G must be > 0");
    if (!(c.dt > 0)) fail("dt must be > 0");
    if (!(c.eps >= 0)) fail("eps must be >= 0");
    if (!(c.theta > 0)) fail("theta must be > 0");
    if (c.render_width < 16 || c.render_width > 16384) fail("render_width must be in 16..16384");
    if (c.render_height < 16 || c.render_height > 16384) fail("render_height must be in 16..16384");
    if (!(c.render_extent >= 0)) fail("render_extent must be >= 0");
    if (c.threads < 0) fail("threads must be >= 0");
    if (c.live < 0 || c.live > 65535) fail("live must be a TCP port 1..65535, or 0 for off");
    if (c.live_width < 64 || c.live_width > 4096) fail("live_width must be in 64..4096");
    if (c.live_height < 64 || c.live_height > 4096) fail("live_height must be in 64..4096");
    if (!(c.live_fps > 0 && c.live_fps <= 240)) fail("live_fps must be in (0, 240]");
    if (c.live_quality < 1 || c.live_quality > 100) fail("live_quality must be in 1..100");
}

std::string dump_config(const Config& c) {
    std::string s;
    auto kv = [&](const char* k, const std::string& v) { s += std::string(k) + " = " + v + "\n"; };
    auto b = [](bool v) { return std::string(v ? "true" : "false"); };
    kv("scenario", c.scenario);
    kv("n", std::to_string(c.n));
    kv("seed", std::to_string(c.seed));
    kv("G", fmt_double(c.G));
    kv("eps", fmt_double(c.eps));
    kv("dt", fmt_double(c.dt));
    kv("steps", std::to_string(c.steps));
    kv("method", to_string(c.method));
    kv("theta", fmt_double(c.theta));
    kv("auto_direct_max", std::to_string(c.auto_direct_max));
    kv("backend", to_string(c.backend));
    kv("device", std::to_string(c.device));
    kv("threads", std::to_string(c.threads));
    kv("diag_every", std::to_string(c.diag_every));
    kv("exact_pe", b(c.exact_pe));
    kv("out_dir", c.out_dir);
    kv("snap_every", std::to_string(c.snap_every));
    kv("snap_final", b(c.snap_final));
    kv("render_every", std::to_string(c.render_every));
    kv("render_width", std::to_string(c.render_width));
    kv("render_height", std::to_string(c.render_height));
    kv("view", c.view);
    kv("render_extent", fmt_double(c.render_extent));
    kv("restart", c.restart);
    kv("live", std::to_string(c.live));
    kv("live_bind", c.live_bind);
    kv("live_width", std::to_string(c.live_width));
    kv("live_height", std::to_string(c.live_height));
    kv("live_fps", fmt_double(c.live_fps));
    kv("live_quality", std::to_string(c.live_quality));
    kv("live_hold", b(c.live_hold));
    kv("live_paused", b(c.live_paused));
    for (auto& p : c.params) kv(p.first.c_str(), p.second);
    return s;
}

const char* to_string(Method m) {
    switch (m) {
        case Method::direct: return "direct";
        case Method::tree: return "tree";
        case Method::auto_: return "auto";
    }
    return "?";
}

const char* to_string(BackendKind b) {
    switch (b) {
        case BackendKind::cpu: return "cpu";
        case BackendKind::cuda: return "cuda";
        case BackendKind::auto_: return "auto";
    }
    return "?";
}

// Run configuration: an .ini-style file of `key = value` lines, then `key=value` overrides from
// the command line. Keys that belong to one scenario are prefixed with its name
// (`disk.radius`); they are stored in `params` and read by the scenario builder.
#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

enum class Method { direct, tree, auto_ };
enum class BackendKind { cpu, cuda, auto_ };

struct Config {
    std::string scenario = "plummer";
    uint64_t n = 10000;
    uint64_t seed = 1;

    // Physics. Units are whatever the scenario documents (most use G = M = 1 N-body units).
    double G = 1.0;
    double eps = 0.01;        // Plummer softening length
    double dt = 0.001;
    uint64_t steps = 1000;

    Method method = Method::auto_;
    double theta = 0.6;       // Barnes-Hut opening angle
    // `method = auto` uses direct summation up to this many bodies.
    uint64_t auto_direct_max = 50000;

    BackendKind backend = BackendKind::auto_;
    int device = 0;           // CUDA device ordinal
    int threads = 0;          // CPU backend threads, 0 = all hardware threads

    uint64_t diag_every = 100;  // 0 = only at start and end
    // Exact O(N^2) potential energy for diagnostics; off by default above auto_direct_max,
    // where the potential comes from the force method itself (tree potential for `tree`).
    bool exact_pe = false;

    std::string out_dir = "out";
    uint64_t snap_every = 0;    // 0 = no snapshots except a final one when snap_final is set
    bool snap_final = false;
    uint64_t render_every = 0;  // 0 = no PNG frames
    int render_width = 1024;
    int render_height = 1024;
    std::string view = "xy";    // xy | xz | yz | <yaw_deg>,<pitch_deg>
    double render_extent = 0;   // half-width of the view in world units, 0 = fit to the bodies
    std::string restart;        // snapshot to resume from; overrides scenario, n and seed

    // Live view: an HTTP server streaming frames to a browser (see live.hpp). 0 = off.
    int live = 0;                        // TCP port
    std::string live_bind = "127.0.0.1"; // localhost only: reach it through an SSH tunnel
    int live_width = 960;
    int live_height = 720;
    double live_fps = 30;                // frame cap; frames are skipped, never queued
    int live_quality = 80;               // JPEG quality 1..100
    bool live_hold = true;               // keep serving the last state after the final step
    bool live_paused = false;            // start paused (press Run in the browser)

    std::map<std::string, std::string> params;  // scenario keys, e.g. "disk.radius" -> "1.5"

    // Scenario parameter lookup. Records the key as used so leftover (misspelt) keys can be reported.
    double param(const std::string& key, double fallback) const;
    std::string param_str(const std::string& key, const std::string& fallback) const;
    // Prefixed keys that no scenario builder asked for.
    std::vector<std::string> unused_params() const;

    mutable std::set<std::string> used_params;
};

// Throws std::runtime_error with the file name and line number on a malformed or unknown key.
Config load_config(const std::string& path);
// Applies one `key=value` override; throws std::runtime_error on an unknown key or bad value.
void apply_override(Config& cfg, const std::string& assignment);
// Applies `key = value` without the file/argv plumbing; shared by the two functions above.
void set_key(Config& cfg, const std::string& key, const std::string& value);
// Throws std::runtime_error if values are out of range (n = 0, dt <= 0, ...).
void validate(const Config& cfg);
// The effective configuration as an .ini file that load_config reads back to the same Config.
std::string dump_config(const Config& cfg);

const char* to_string(Method m);
const char* to_string(BackendKind b);

// The run driver: builds or restores the initial state, picks a backend, integrates with a global
// kick-drift-kick leapfrog, and writes diagnostics, snapshots and frames on their cadences.
#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "backend.hpp"
#include "config.hpp"

// backend=auto takes CUDA when a device is usable, else the CPU. Throws if an explicitly requested
// backend is unavailable. `note` receives a line saying what was chosen and why.
std::unique_ptr<Backend> select_backend(const Config& cfg, std::string& note);
ForceMethod resolve_method(const Config& cfg, uint64_t n);

struct RunSummary {
    uint64_t steps = 0;
    double e0 = 0, e1 = 0;     // total energy at the first and last diagnostic
    double wall_s = 0;         // integration only, without output I/O
    double ms_per_step = 0;
};

// Throws std::runtime_error on bad configuration, unknown scenario parameters or I/O errors.
RunSummary run_simulation(const Config& cfg, std::FILE* log);

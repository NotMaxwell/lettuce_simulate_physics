// Initial conditions. Each scenario reads its own `<name>.<key>` parameters from the Config and
// documents its units; positions and velocities are in the centre-of-mass frame.
#pragma once
#include <string>
#include <vector>

#include "bodies.hpp"
#include "config.hpp"

struct ScenarioInfo {
    const char* name;
    const char* summary;
};

const std::vector<ScenarioInfo>& scenarios();

// Throws std::runtime_error for an unknown scenario or a bad parameter. Uses cfg.n and cfg.seed
// (some scenarios fix or round the count; the returned size is authoritative) and cfg.G.
Bodies build_scenario(const Config& cfg);

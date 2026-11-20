// Pre-flight memory checks: refuse a body count that will not fit, before spending minutes
// generating it and before a host out-of-memory can wedge the machine.
#pragma once
#include <cstdint>
#include <string>

#include "backend.hpp"

// Host bytes this process can still claim: MemAvailable, capped by the cgroup limit of a container
// if one is set. 0 if unknown.
uint64_t host_available_bytes();

// Peak host bytes of build_scenario for n bodies, including the returned Bodies.
uint64_t scenario_peak_bytes(uint64_t n, const std::string& scenario);

// Collective: every rank calls it with the same arguments. `be` is this rank's backend (or the
// distributed wrapper), `ntotal` the body count over all ranks. `scenario` empty means the bodies
// come from a restart snapshot. Returns "" if the run fits on every rank, else a message (the same
// verdict on every rank) saying what is short and how many bodies would fit.
// LETTUCE_SKIP_MEMCHECK=1 skips the check.
std::string memory_preflight(Backend& be, uint64_t ntotal, ForceMethod m, const std::string& scenario);

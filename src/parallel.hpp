// Process-level parallelism: one MPI rank per GPU. Built without MPI (LETTUCE_MPI=OFF or no MPI
// found) every function degrades to a single rank, so callers never need #ifdefs.
//
// Distributed force evaluation (make_distributed) follows the Bonsai scheme: bodies are split into
// contiguous Morton-key ranges balanced by measured cost, each rank builds its own tree, and every
// rank sends every other rank a locally essential tree (LET, see backend.hpp) for that rank's
// bounding box. All communication is host-staged in this version.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "backend.hpp"

namespace par {

void init(int* argc, char*** argv);
void finalize();
// Kills every rank: one rank failing must not leave the others blocked in a collective.
[[noreturn]] void abort(int code);

int rank();
int size();
int local_rank();  // rank among the processes on this node (selects the GPU)
int local_size();
inline bool root() { return rank() == 0; }
bool mpi_compiled();

void barrier();
double sum(double v);
uint64_t sum(uint64_t v);
double max(double v);
// Root's value on every rank.
int broadcast(int v);

struct DistOptions {
    uint64_t rebalance_every = 10;  // force calls between domain rebalances (tree method)
};

// Wraps a per-rank backend so it computes forces over all ranks' bodies. Returns `local` unchanged
// on a single rank. The wrapper's upload/download take and return this rank's bodies only;
// diagnostics(), exact_potential_energy() and times().interactions are global (collective calls).
std::unique_ptr<Backend> make_distributed(std::unique_ptr<Backend> local, const DistOptions& o);

// Collective. Root receives a subsample of every rank's bodies, at most about `max_total` in all
// (every k-th body, masses multiplied by k so projected densities are unchanged); others get none.
void gather_sample(Backend& be, Bodies& out, uint64_t max_total);

// This rank's share [first, last) of `n` items split evenly.
void even_share(uint64_t n, uint64_t& first, uint64_t& last);

}  // namespace par

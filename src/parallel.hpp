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

// The most bodies one rank may hold after a rebalance: 3/2 of an even share (n itself on one rank).
// The balancer equalises measured cost, so ranks over cheap outer regions take more bodies than
// even. Measured worst max/even share (plummer, uniform, disk, collision, systems; 2e5-4e5 bodies):
// 1.03-1.14 at 4 ranks, 1.09-1.24 at 8, 1.45 (disk) at 16. 3/2 leaves that free while bounding
// what the memory pre-flight must plan for and keeping coincident bodies (one Morton key) from
// piling onto one rank.
uint64_t max_rank_bodies(uint64_t n);

// Host bytes per body a rebalance stages, counted at max_rank_bodies: download 32, cost 4, key 8,
// order 4, dest 4, send buffers 32, receive buffers 32 (including the copy to itself), the new
// Bodies 32.
constexpr uint64_t kRebalanceHostBytesPerBody = 148;

}  // namespace par

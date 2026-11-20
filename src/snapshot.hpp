// Snapshot files: a 256-byte little-endian header, then planar arrays (f32 x, y, z, vx, vy, vz,
// m, then u32 id), each starting on a 4096-byte boundary. Velocities are synchronous with positions.
#pragma once
#include <cstdint>
#include <string>

#include "bodies.hpp"

struct SnapshotHeader {
    uint64_t n = 0;
    uint64_t step = 0;
    double time = 0;
    double G = 1;
    double eps = 0;
    double dt = 0;
    uint64_t seed = 0;
    std::string scenario;  // at most 31 bytes are stored
    // Number of pieces (one per rank) the snapshot was written in; 0 in files that predate the field.
    uint32_t pieces = 0;
};

// All throw std::runtime_error with the path on I/O failure or a bad/foreign file. The file is
// written to <path>.tmp, fsync'ed and renamed into place, so `path` is never left half-written.
void write_snapshot(const std::string& path, const SnapshotHeader& h, const Bodies& b);
SnapshotHeader read_snapshot(const std::string& path, Bodies& b);
// The header only (validated against the file size), without reading the arrays.
SnapshotHeader read_snapshot_header(const std::string& path);

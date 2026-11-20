#include "snapshot.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <unistd.h>

static_assert(std::endian::native == std::endian::little, "snapshots are little-endian on disk");

namespace {

constexpr char kMagic[8] = {'L', 'T', 'C', 'S', 'N', 'A', 'P', '3'};
constexpr uint32_t kVersion = 3;
constexpr uint32_t kHeaderBytes = 256;
constexpr uint64_t kAlign = 4096;
constexpr int kArrays = 8;
// kOffPieces was spare (zero) in files written before it existed, so 0 reads as "unrecorded".
constexpr size_t kOffN = 16, kOffScenario = 72, kOffOffsets = 104, kOffPieces = 168;

// Only for reads and error paths: a successful write closes explicitly and checks fclose.
struct FileCloser { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
using File = std::unique_ptr<std::FILE, FileCloser>;

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error("snapshot '" + path + "': " + msg);
}

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

template <class T> void put(unsigned char* h, size_t off, T v) { std::memcpy(h + off, &v, sizeof v); }
template <class T> T get(const unsigned char* h, size_t off) { T v; std::memcpy(&v, h + off, sizeof v); return v; }

}  // namespace

void write_snapshot(const std::string& path, const SnapshotHeader& h, const Bodies& b) {
    // The arrays are the truth: h.n is ignored so a stale header field cannot corrupt the file.
    const uint64_t n = b.size();
    if (b.y.size() != n || b.z.size() != n || b.vx.size() != n || b.vy.size() != n ||
        b.vz.size() != n || b.m.size() != n || b.id.size() != n)
        fail(path, "body arrays have different lengths");

    const void* data[kArrays] = {b.x.data(),  b.y.data(),  b.z.data(),  b.vx.data(),
                                 b.vy.data(), b.vz.data(), b.m.data(),  b.id.data()};
    uint64_t offsets[kArrays];
    uint64_t next = kAlign;
    for (int i = 0; i < kArrays; ++i) {
        offsets[i] = next;
        next = align_up(next + n * 4);
    }

    unsigned char hdr[kHeaderBytes] = {};
    std::memcpy(hdr, kMagic, 8);
    put<uint32_t>(hdr, 8, kVersion);
    put<uint32_t>(hdr, 12, kHeaderBytes);
    put<uint64_t>(hdr, kOffN, n);
    put<uint64_t>(hdr, 24, h.step);
    put<double>(hdr, 32, h.time);
    put<double>(hdr, 40, h.G);
    put<double>(hdr, 48, h.eps);
    put<double>(hdr, 56, h.dt);
    put<uint64_t>(hdr, 64, h.seed);
    std::memcpy(hdr + kOffScenario, h.scenario.data(), std::min<size_t>(h.scenario.size(), 31));
    std::memcpy(hdr + kOffOffsets, offsets, sizeof offsets);
    put<uint32_t>(hdr, kOffPieces, h.pieces);

    const std::string tmp = path + ".tmp";
    bool ok = false;
    {
        File f(std::fopen(tmp.c_str(), "wb"));
        if (!f) fail(path, "cannot create '" + tmp + "'");
        ok = std::fwrite(hdr, 1, kHeaderBytes, f.get()) == kHeaderBytes;
        uint64_t pos = kHeaderBytes;
        static const unsigned char zeros[kAlign] = {};
        for (int i = 0; ok && i < kArrays; ++i) {
            for (uint64_t pad = offsets[i] - pos; ok && pad > 0;) {
                size_t k = static_cast<size_t>(std::min<uint64_t>(pad, kAlign));
                ok = std::fwrite(zeros, 1, k, f.get()) == k;
                pad -= k;
            }
            ok = ok && (n == 0 || std::fwrite(data[i], 4, n, f.get()) == n);
            pos = offsets[i] + n * 4;
        }
        ok = (std::fflush(f.get()) == 0) && ok;
        if (std::ferror(f.get())) ok = false;
        // Without fsync a crash after the rename can leave the final name pointing at an empty or
        // partial file; fclose reports write-back errors (NFS, quota) that fflush may not.
        ok = ok && ::fsync(fileno(f.get())) == 0;
        ok = (std::fclose(f.release()) == 0) && ok;
    }
    if (!ok) {
        std::remove(tmp.c_str());
        fail(path, "write failed (disk full or I/O error)");
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        fail(path, "cannot rename '" + tmp + "' into place");
    }
}

namespace {

// Opens and validates `path`; leaves the array offsets in `offsets`.
SnapshotHeader read_header(const std::string& path, File& f, uint64_t (&offsets)[kArrays]) {
    f.reset(std::fopen(path.c_str(), "rb"));
    if (!f) fail(path, "cannot open");
    if (std::fseek(f.get(), 0, SEEK_END) != 0) fail(path, "cannot seek");
    const long end = std::ftell(f.get());
    if (end < 0) fail(path, "cannot determine size");
    const uint64_t file_size = static_cast<uint64_t>(end);
    std::rewind(f.get());

    unsigned char hdr[kHeaderBytes];
    if (file_size < kHeaderBytes || std::fread(hdr, 1, kHeaderBytes, f.get()) != kHeaderBytes)
        fail(path, "file too short for a header");
    if (std::memcmp(hdr, kMagic, 8) != 0) fail(path, "bad magic (not a LTCSNAP3 snapshot)");
    if (get<uint32_t>(hdr, 8) != kVersion) fail(path, "unsupported version");
    if (get<uint32_t>(hdr, 12) != kHeaderBytes) fail(path, "unexpected header size");

    SnapshotHeader h;
    h.n = get<uint64_t>(hdr, kOffN);
    h.step = get<uint64_t>(hdr, 24);
    h.time = get<double>(hdr, 32);
    h.G = get<double>(hdr, 40);
    h.eps = get<double>(hdr, 48);
    h.dt = get<double>(hdr, 56);
    h.seed = get<uint64_t>(hdr, 64);
    h.scenario.assign(reinterpret_cast<const char*>(hdr + kOffScenario),
                      strnlen(reinterpret_cast<const char*>(hdr + kOffScenario), 32));
    h.pieces = get<uint32_t>(hdr, kOffPieces);

    // Bound n before multiplying so a corrupt count cannot overflow the size arithmetic.
    if (h.n > file_size / 4) fail(path, "body count exceeds file size");
    std::memcpy(offsets, hdr + kOffOffsets, sizeof offsets);
    for (int i = 0; i < kArrays; ++i)
        if (offsets[i] < kHeaderBytes || offsets[i] > file_size || file_size - offsets[i] < h.n * 4)
            fail(path, "array " + std::to_string(i) + " extends past end of file");
    return h;
}

}  // namespace

SnapshotHeader read_snapshot_header(const std::string& path) {
    File f;
    uint64_t offsets[kArrays];
    return read_header(path, f, offsets);
}

SnapshotHeader read_snapshot(const std::string& path, Bodies& b) {
    File f;
    uint64_t offsets[kArrays];
    SnapshotHeader h = read_header(path, f, offsets);
    Bodies tmp;
    tmp.resize(h.n);
    void* data[kArrays] = {tmp.x.data(),  tmp.y.data(),  tmp.z.data(),  tmp.vx.data(),
                           tmp.vy.data(), tmp.vz.data(), tmp.m.data(),  tmp.id.data()};
    for (int i = 0; i < kArrays && h.n > 0; ++i) {
        if (std::fseek(f.get(), static_cast<long>(offsets[i]), SEEK_SET) != 0 ||
            std::fread(data[i], 4, h.n, f.get()) != h.n)
            fail(path, "short read in array " + std::to_string(i));
    }
    b = std::move(tmp);
    return h;
}

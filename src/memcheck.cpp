#include "memcheck.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "parallel.hpp"

namespace {

// Fraction of the free memory a run may plan to use. Host: the rest is left to the OS, sshd and
// the page cache; a VM with no swap that runs out does not always get to the OOM killer, it can
// stop responding instead. Device: CUB temporary storage and allocator rounding.
constexpr double kHostUse = 0.90;
constexpr double kDeviceUse = 0.97;

bool read_u64(const std::string& path, uint64_t& v) {
    std::ifstream in(path);
    std::string s;
    if (!(in >> s) || s == "max") return false;
    try {
        v = std::stoull(s);
    } catch (...) {
        return false;
    }
    return true;
}

// Value of `key` in a "key value [unit]" file such as memory.stat or /proc/meminfo, or 0.
uint64_t stat_field(const std::string& path, const std::string& key) {
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        std::istringstream ls(line);
        std::string k;
        uint64_t v;
        if (ls >> k >> v && k == key) return v;
    }
    return 0;
}

// Headroom under every cgroup memory limit between this process and the root, or 0 if none is set.
// The inactive page cache is counted as available: the kernel reclaims it before charging a limit.
uint64_t cgroup_available() {
    uint64_t best = 0;
    bool any = false;
    auto consider = [&](uint64_t limit, uint64_t usage, uint64_t reclaimable) {
        uint64_t used = usage > reclaimable ? usage - reclaimable : 0;
        uint64_t room = limit > used ? limit - used : 0;
        best = any ? std::min(best, room) : room;
        any = true;
    };
    std::ifstream self("/proc/self/cgroup");
    std::string line, v2path;
    while (std::getline(self, line))
        if (line.rfind("0::", 0) == 0) v2path = line.substr(3);
    if (!v2path.empty()) {
        // Inside a container with its own cgroup namespace the path is "/" and the limit sits at the root.
        for (std::string p = v2path;;) {
            std::string dir = "/sys/fs/cgroup" + (p == "/" ? std::string() : p);
            uint64_t limit, usage;
            if (read_u64(dir + "/memory.max", limit) && read_u64(dir + "/memory.current", usage))
                consider(limit, usage, stat_field(dir + "/memory.stat", "inactive_file"));
            if (p == "/" || p.empty()) break;
            auto slash = p.rfind('/');
            p = slash == 0 ? "/" : p.substr(0, slash);
        }
    }
    uint64_t limit, usage;
    // cgroup v1; "unlimited" reads as a huge number, which never wins the min below.
    if (read_u64("/sys/fs/cgroup/memory/memory.limit_in_bytes", limit) &&
        read_u64("/sys/fs/cgroup/memory/memory.usage_in_bytes", usage) && limit < (uint64_t(1) << 60))
        consider(limit, usage, stat_field("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file"));
    return any ? best : 0;
}

std::string gb(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f GB", double(bytes) / 1e9);
    return buf;
}

}  // namespace

uint64_t host_available_bytes() {
    uint64_t avail = stat_field("/proc/meminfo", "MemAvailable:") * 1024;
    uint64_t cg = cgroup_available();
    if (cg && (!avail || cg < avail)) avail = cg;
    return avail;
}

uint64_t scenario_peak_bytes(uint64_t n, const std::string& scenario) {
    // Measured peak RSS of build_scenario (doubles, 56 B/body, then converted column by column to the
    // 32 B/body Bodies): 60 B/body for plummer and uniform at 1e6-1e8. disk measured 64 B/body (its
    // sorted radii add 8); the others are bounded by the same 56 + 8.
    const uint64_t per = scenario == "plummer" || scenario == "uniform" ? 60 : 66;
    return n * per;
}

std::string memory_preflight(Backend& be, uint64_t ntotal, ForceMethod m, const std::string& scenario) {
    // Root's environment decides: mpirun need not forward it to every node, and ranks taking
    // different paths here would deadlock in the collective below.
    const char* s = std::getenv("LETTUCE_SKIP_MEMCHECK");
    if (par::broadcast(s && *s && std::string(s) != "0" ? 1 : 0)) return "";
    const uint64_t P = uint64_t(par::size()), R = uint64_t(par::local_size());
    uint64_t dfree = 0, dtotal = 0;
    const bool dev = be.device_memory(dfree, dtotal);

    // Per node: every rank on it builds all n bodies (generation is deterministic, each rank keeps its
    // share), then holds its share while the backend allocates. The node peak is the worst phase.
    auto host_rank = [&](uint64_t n) {
        const uint64_t nl = (n + P - 1) / P;
        const uint64_t gen = scenario.empty() ? 32 * n : scenario_peak_bytes(n, scenario);
        const uint64_t split = 32 * n + (P > 1 ? 32 * nl : 0);
        const uint64_t run = 32 * nl + be.memory_need(nl, m).host;
        return std::max({gen, split, run});
    };
    // One rank per GPU, as the launch scripts set up; ranks sharing a device are not detected.
    auto dev_need = [&](uint64_t n) { return be.memory_need((n + P - 1) / P, m).device; };
    const uint64_t havail = host_available_bytes();
    auto host_ok = [&](uint64_t n) { return havail == 0 || double(R * host_rank(n)) <= kHostUse * double(havail); };
    auto dev_ok = [&](uint64_t n) { return !dev || double(dev_need(n)) <= kDeviceUse * double(dfree); };
    auto max_fit = [&]() {
        uint64_t lo = 0, hi = ntotal;
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo + 1) / 2;
            if (host_ok(mid) && dev_ok(mid)) lo = mid; else hi = mid - 1;
        }
        return lo;
    };

    std::ostringstream msg;
    if (!host_ok(ntotal)) {
        msg << "n=" << ntotal << " needs ~" << gb(R * host_rank(ntotal)) << " of host memory";
        if (R > 1) msg << " (" << gb(host_rank(ntotal)) << " per rank x " << R << " ranks on this node)";
        msg << " at its peak, but " << gb(havail) << " is available (MemAvailable or the container's limit) and runs "
            << "plan to use at most " << int(kHostUse * 100) << "% of it";
    } else if (!dev_ok(ntotal)) {
        msg << "n=" << ntotal << " needs ~" << gb(dev_need(ntotal)) << " of GPU memory per device ("
            << dev_need(ntotal) / std::max<uint64_t>((ntotal + P - 1) / P, 1) << " B/body), but " << gb(dfree)
            << " of " << gb(dtotal) << " is free (counting what this run already holds)";
    }
    if (!msg.str().empty())
        msg << "; at most ~" << max_fit() << " bodies fit (LETTUCE_SKIP_MEMCHECK=1 skips this check)";
    std::string out = msg.str();
    if (par::sum(uint64_t(out.empty() ? 0 : 1)) != 0 && out.empty())
        out = "the memory pre-flight check failed on another rank";
    return out;
}

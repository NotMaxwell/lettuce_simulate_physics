// lettuce-bench: force-evaluation and step timings across backends, methods and body counts.
//   lettuce-bench [n=1e4,1e5,1e6] [backend=cpu,cuda] [method=direct,tree] [steps=5] [theta=0.6]
//                 [scenario=plummer] [eps=0.01] [direct_max=200000] [csv=path] [<scenario>.<key>=v ...]
// Under mpirun every rank drives one backend (one GPU) and n is the total over all ranks; times are
// the slowest rank's, and the sort/build/walk split is rank 0's.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "config.hpp"
#include "memcheck.hpp"
#include "parallel.hpp"
#include "scenarios.hpp"

namespace {

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}

struct Row {
    std::string backend, method;
    uint64_t n;
    double force_ms, sort_ms, build_ms, walk_ms, comm_ms, step_ms, ginter;
};

}  // namespace

int main(int argc, char** argv) {
    par::init(&argc, &argv);
    const bool root = par::root();
    try {
        std::vector<std::string> ns = {"1e4", "1e5", "1e6"};
        std::vector<std::string> backends = {"cpu", "cuda"};
        std::vector<std::string> methods = {"direct", "tree"};
        int steps = 5;
        // Direct summation is O(N^2): past this many bodies it is skipped unless asked for.
        uint64_t direct_max = 200000;
        std::string csv_path;
        Config cfg;
        cfg.eps = 0.01;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto eq = a.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected key=value, got '" + a + "'");
            std::string k = a.substr(0, eq), v = a.substr(eq + 1);
            if (k == "n") ns = split(v);
            else if (k == "backend") backends = split(v);
            else if (k == "method") methods = split(v);
            else if (k == "steps") steps = std::stoi(v);
            else if (k == "direct_max") direct_max = std::stoull(v);
            else if (k == "csv") csv_path = v;
            else set_key(cfg, k, v);  // scenario, eps, theta, G, seed, threads, device, scenario params
        }
        // A tiny dry build reads the scenario's parameters, so typos fail before memcheck can skip
        // every n. If it cannot run (a parameter it rejects first), the check after each real build
        // still applies; reporting unused keys from a build that stopped early would be wrong.
        std::vector<std::string> unused;
        try {
            Config dry = cfg;
            dry.n = 64;
            if (dry.params.count("systems.stars")) dry.params["systems.stars"] = "1";
            if (scenario_body_count(dry) <= 4096) {
                build_scenario(dry);
                unused = dry.unused_params();
            }
        } catch (const std::exception&) {
        }
        if (!unused.empty()) {
            std::string msg = "unknown parameter(s) for scenario '" + cfg.scenario + "':";
            for (auto& k : unused) msg += " " + k;
            throw std::runtime_error(msg);
        }

        std::FILE* csv = nullptr;
        if (!csv_path.empty() && root) {
            csv = std::fopen(csv_path.c_str(), "w");
            if (!csv) throw std::runtime_error("cannot write " + csv_path);
            std::fprintf(csv, "ranks,backend,method,n,force_ms,sort_ms,build_ms,walk_ms,comm_ms,step_ms,ginteractions_per_s\n");
        }

        if (root) {
            std::printf("lettuce-bench: scenario %s, eps %g, theta %g, %d timed steps (after 1 warm-up), %d rank%s\n",
                        cfg.scenario.c_str(), cfg.eps, cfg.theta, steps, par::size(), par::size() == 1 ? "" : "s");
            std::printf("%-6s %-6s %10s %10s %8s %8s %9s %8s %8s %10s\n", "backend", "method", "n", "force ms", "sort",
                        "build", "walk", "comm", "step", "Ginter/s");
        }
        for (auto& bname : backends) {
            std::unique_ptr<Backend> be;
            if (bname == "cpu") {
                be = make_cpu_backend(cfg.threads);
            } else if (bname == "cuda") {
                std::string why;
                int count = cuda_device_count();
                int device = par::size() > 1 && count > 1 ? (cfg.device + par::local_rank()) % count : cfg.device;
                be = make_cuda_backend(device, why);
                // Every rank must agree, or the collectives below would deadlock.
                if (par::sum(uint64_t(be ? 0 : 1)) != 0) {
                    if (root) std::printf("# cuda skipped: %s\n", be ? "unavailable on some rank" : why.c_str());
                    continue;
                }
            } else {
                throw std::runtime_error("unknown backend '" + bname + "'");
            }
            if (root) std::printf("# %s\n", be->name().c_str());
            par::DistOptions dopt;
            dopt.rebalance_every = cfg.rebalance_every;
            be = par::make_distributed(std::move(be), dopt);
            for (auto& nstr : ns) {
                Config c = cfg;
                set_key(c, "n", nstr);
                const uint64_t nbodies = scenario_body_count(c);
                bool any_tree = false, any_run = false;
                for (auto& mname : methods) {
                    any_tree = any_tree || mname == "tree";
                    any_run = any_run || mname == "tree" || (mname == "direct" && nbodies <= direct_max);
                }
                if (!any_run) continue;
                // Checked before generating: building 1e9 bodies alone takes minutes.
                std::string short_of =
                    memory_preflight(*be, nbodies, any_tree ? ForceMethod::tree : ForceMethod::direct, c.scenario);
                if (!short_of.empty()) {
                    if (root) std::printf("# %s n=%s skipped: %s\n", bname.c_str(), nstr.c_str(), short_of.c_str());
                    std::fflush(stdout);
                    continue;
                }
                Bodies all = build_scenario(c);
                // Same on every rank (same config), so every rank throws together.
                auto unused = c.unused_params();
                if (!unused.empty()) {
                    std::string msg = "unknown parameter(s) for scenario '" + c.scenario + "':";
                    for (auto& k : unused) msg += " " + k;
                    throw std::runtime_error(msg);
                }
                const uint64_t ntotal = all.size();
                Bodies b;
                if (par::size() == 1) {
                    b = std::move(all);
                } else {
                    uint64_t first, last;
                    par::even_share(ntotal, first, last);
                    b.reserve(last - first);
                    for (uint64_t i = first; i < last; ++i) {
                        b.push(all.x[i], all.y[i], all.z[i], all.vx[i], all.vy[i], all.vz[i], all.m[i]);
                        b.id.back() = all.id[i];
                    }
                    all = Bodies{};
                }
                for (auto& mname : methods) {
                    ForceParams fp;
                    fp.G = c.G;
                    fp.eps = c.eps;
                    fp.theta = c.theta;
                    if (mname == "direct") fp.method = ForceMethod::direct;
                    else if (mname == "tree") fp.method = ForceMethod::tree;
                    else throw std::runtime_error("unknown method '" + mname + "'");
                    if (fp.method == ForceMethod::direct && ntotal > direct_max) continue;

                    be->upload(b);
                    be->compute_forces(fp, false);
                    Row r{bname, mname, ntotal, 0, 0, 0, 0, 0, 0, 0};
                    uint64_t inter = 0;
                    double wall = 0;
                    for (int s = 0; s < steps; ++s) {
                        auto t0 = std::chrono::steady_clock::now();
                        be->kick(0.5 * c.dt);
                        be->drift(c.dt);
                        be->compute_forces(fp, false);
                        be->kick(0.5 * c.dt);
                        (void)be->times();  // synchronises the device before the clock is read
                        wall += par::max(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                        PhaseTimes pt = be->times();
                        r.force_ms += par::max(pt.force_ms);
                        r.comm_ms += pt.comm_ms;
                        r.sort_ms += pt.sort_ms;
                        r.build_ms += pt.build_ms;
                        r.walk_ms += pt.walk_ms;
                        inter += pt.interactions;
                    }
                    double k = 1.0 / std::max(steps, 1);
                    r.force_ms *= k;
                    r.sort_ms *= k;
                    r.build_ms *= k;
                    r.walk_ms *= k;
                    r.comm_ms *= k;
                    r.step_ms = 1e3 * wall * k;
                    if (fp.method == ForceMethod::direct && inter == 0) inter = uint64_t(steps) * ntotal * ntotal;
                    r.ginter = r.force_ms > 0 ? double(inter) * k / (r.force_ms * 1e-3) / 1e9 : 0;
                    if (root) {
                        std::printf("%-7s %-6s %10llu %10.3f %8.3f %8.3f %9.3f %8.3f %8.3f %10.2f\n", r.backend.c_str(),
                                    r.method.c_str(), static_cast<unsigned long long>(r.n), r.force_ms, r.sort_ms,
                                    r.build_ms, r.walk_ms, r.comm_ms, r.step_ms, r.ginter);
                        std::fflush(stdout);
                    }
                    if (csv)
                        std::fprintf(csv, "%d,%s,%s,%llu,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", par::size(), r.backend.c_str(),
                                     r.method.c_str(), static_cast<unsigned long long>(r.n), r.force_ms, r.sort_ms,
                                     r.build_ms, r.walk_ms, r.comm_ms, r.step_ms, r.ginter);
                }
            }
        }
        if (csv) std::fclose(csv);
        par::finalize();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lettuce-bench: %s\n", e.what());
        par::abort(1);
    }
}

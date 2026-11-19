// lettuce-bench: force-evaluation and step timings across backends, methods and body counts.
//   lettuce-bench [n=1e4,1e5,1e6] [backend=cpu,cuda] [method=direct,tree] [steps=5] [theta=0.6]
//                 [scenario=plummer] [eps=0.01] [direct_max=200000] [csv=path] [<scenario>.<key>=v ...]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "config.hpp"
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
    double force_ms, sort_ms, build_ms, walk_ms, step_ms, ginter;
};

}  // namespace

int main(int argc, char** argv) {
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

        std::FILE* csv = nullptr;
        if (!csv_path.empty()) {
            csv = std::fopen(csv_path.c_str(), "w");
            if (!csv) throw std::runtime_error("cannot write " + csv_path);
            std::fprintf(csv, "backend,method,n,force_ms,sort_ms,build_ms,walk_ms,step_ms,ginteractions_per_s\n");
        }

        std::printf("lettuce-bench: scenario %s, eps %g, theta %g, %d timed steps (after 1 warm-up)\n",
                    cfg.scenario.c_str(), cfg.eps, cfg.theta, steps);
        std::printf("%-6s %-6s %10s %10s %8s %8s %9s %8s %10s\n", "backend", "method", "n", "force ms", "sort", "build",
                    "walk", "step", "Ginter/s");
        for (auto& bname : backends) {
            std::unique_ptr<Backend> be;
            if (bname == "cpu") {
                be = make_cpu_backend(cfg.threads);
            } else if (bname == "cuda") {
                std::string why;
                be = make_cuda_backend(cfg.device, why);
                if (!be) {
                    std::printf("# cuda skipped: %s\n", why.c_str());
                    continue;
                }
            } else {
                throw std::runtime_error("unknown backend '" + bname + "'");
            }
            std::printf("# %s\n", be->name().c_str());
            for (auto& nstr : ns) {
                Config c = cfg;
                set_key(c, "n", nstr);
                Bodies b = build_scenario(c);
                for (auto& mname : methods) {
                    ForceParams fp;
                    fp.G = c.G;
                    fp.eps = c.eps;
                    fp.theta = c.theta;
                    if (mname == "direct") fp.method = ForceMethod::direct;
                    else if (mname == "tree") fp.method = ForceMethod::tree;
                    else throw std::runtime_error("unknown method '" + mname + "'");
                    if (fp.method == ForceMethod::direct && b.size() > direct_max) continue;

                    be->upload(b);
                    be->compute_forces(fp, false);
                    Row r{bname, mname, b.size(), 0, 0, 0, 0, 0, 0};
                    uint64_t inter = 0;
                    double wall = 0;
                    for (int s = 0; s < steps; ++s) {
                        auto t0 = std::chrono::steady_clock::now();
                        be->kick(0.5 * c.dt);
                        be->drift(c.dt);
                        be->compute_forces(fp, false);
                        be->kick(0.5 * c.dt);
                        (void)be->times();  // synchronises the device before the clock is read
                        wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        PhaseTimes pt = be->times();
                        r.force_ms += pt.force_ms;
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
                    r.step_ms = 1e3 * wall * k;
                    if (fp.method == ForceMethod::direct && inter == 0) inter = uint64_t(steps) * b.size() * b.size();
                    r.ginter = r.force_ms > 0 ? double(inter) * k / (r.force_ms * 1e-3) / 1e9 : 0;
                    std::printf("%-7s %-6s %10llu %10.3f %8.3f %8.3f %9.3f %8.3f %10.2f\n", r.backend.c_str(),
                                r.method.c_str(), static_cast<unsigned long long>(r.n), r.force_ms, r.sort_ms, r.build_ms,
                                r.walk_ms, r.step_ms, r.ginter);
                    std::fflush(stdout);
                    if (csv)
                        std::fprintf(csv, "%s,%s,%llu,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", r.backend.c_str(), r.method.c_str(),
                                     static_cast<unsigned long long>(r.n), r.force_ms, r.sort_ms, r.build_ms, r.walk_ms,
                                     r.step_ms, r.ginter);
                }
            }
        }
        if (csv) std::fclose(csv);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lettuce-bench: %s\n", e.what());
        return 1;
    }
}

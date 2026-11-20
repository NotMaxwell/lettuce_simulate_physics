#include "simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>

#include "live.hpp"
#include "memcheck.hpp"
#include "parallel.hpp"
#include "render.hpp"
#include "scenarios.hpp"
#include "snapshot.hpp"

namespace fs = std::filesystem;

std::unique_ptr<Backend> select_backend(const Config& cfg, std::string& note) {
    if (cfg.backend != BackendKind::cpu) {
        // One GPU per rank. Under Slurm with --gpus-per-task each rank sees exactly one device
        // (CUDA_VISIBLE_DEVICES), so device 0; otherwise ranks on a node take consecutive devices.
        int device = cfg.device;
        int count = cuda_device_count();
        if (par::size() > 1 && count > 1) {
            device = (cfg.device + par::local_rank()) % count;
            if (par::local_size() > count && par::local_rank() == 0)
                std::fprintf(stderr, "lettuce: warning: %d ranks on this node share %d GPUs\n", par::local_size(), count);
        }
        std::string why;
        auto be = make_cuda_backend(device, why);
        if (be) {
            note = "backend " + be->name();
            return be;
        }
        if (cfg.backend == BackendKind::cuda) throw std::runtime_error("backend=cuda requested but unavailable: " + why);
        note = "CUDA unavailable (" + why + "), falling back to the CPU; ";
    }
    auto be = make_cpu_backend(cfg.threads);
    note += "backend " + be->name();
    return be;
}

ForceMethod resolve_method(const Config& cfg, uint64_t n) {
    switch (cfg.method) {
        case Method::direct: return ForceMethod::direct;
        case Method::tree: return ForceMethod::tree;
        case Method::auto_: break;
    }
    return n <= cfg.auto_direct_max ? ForceMethod::direct : ForceMethod::tree;
}

namespace {

std::string numbered(const std::string& dir, const char* stem, uint64_t step, const char* ext, int piece = -1) {
    char name[80];
    if (piece < 0)
        std::snprintf(name, sizeof name, "%s_%08llu.%s", stem, static_cast<unsigned long long>(step), ext);
    else
        std::snprintf(name, sizeof name, "%s_%08llu_r%d.%s", stem, static_cast<unsigned long long>(step), piece, ext);
    return (fs::path(dir) / name).string();
}

// A multi-rank run writes one snapshot piece per rank (snap_<step>_r<k>.snap). Naming any piece
// restarts from all of them, with any number of ranks. Piece 0's header says how many pieces the
// snapshot has: a later run with fewer ranks into the same out_dir overwrites only the low pieces,
// and the stale high ones must not be read. Files without the count fall back to every piece found.
std::vector<std::string> restart_pieces(const std::string& path, std::FILE* log) {
    static const std::regex piece(R"((.*)_r(\d+)\.snap)");
    std::smatch m;
    if (!std::regex_match(path, m, piece)) return {path};
    const std::string prefix = fs::path(m[1].str()).filename().string();
    fs::path dir = fs::path(path).parent_path();
    std::vector<std::pair<int, std::string>> found;
    for (auto& e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir)) {
        std::string fname = e.path().filename().string();
        std::smatch q;
        if (std::regex_match(fname, q, piece) && q[1].str() == prefix)
            found.emplace_back(std::stoi(q[2].str()), (dir / fname).string());
    }
    std::sort(found.begin(), found.end());
    size_t want = found.size();
    if (!found.empty() && found[0].first == 0) {
        uint32_t recorded = read_snapshot_header(found[0].second).pieces;
        if (recorded > 0) want = recorded;
    }
    std::vector<std::string> out;
    size_t extra = 0;
    for (size_t k = 0; k < found.size(); ++k) {
        if (k >= want) {
            ++extra;
            continue;
        }
        if (found[k].first != int(k)) throw std::runtime_error("restart: snapshot piece _r" + std::to_string(k) + " is missing");
        out.push_back(found[k].second);
    }
    if (out.size() < want) throw std::runtime_error("restart: snapshot piece _r" + std::to_string(out.size()) + " is missing");
    if (extra)
        std::fprintf(log, "# warning: ignoring %zu stale snapshot piece%s beyond _r%zu (piece 0 records %zu pieces)\n", extra,
                     extra == 1 ? "" : "s", want - 1, want);
    return out;
}

// On restart, drops the rows at or after the restart step (a previous run's, about to be redone)
// and returns diag.csv open for appending; a fresh run starts the file over.
std::FILE* open_diag(const fs::path& path, uint64_t step0, const char* header) {
    if (step0 && fs::exists(path)) {
        const fs::path tmp = path.string() + ".tmp";
        {
            std::ifstream in(path);
            std::ofstream out(tmp, std::ios::trunc);
            bool have_header = false;
            for (std::string line; std::getline(in, line);) {
                if (line.rfind("step,", 0) == 0) {
                    if (!have_header) out << line << '\n';
                    have_header = true;
                    continue;
                }
                char* end = nullptr;
                unsigned long long s = std::strtoull(line.c_str(), &end, 10);
                if (end != line.c_str() && *end == ',' && s < step0) out << line << '\n';
            }
            if (!have_header) throw std::runtime_error("unexpected contents in " + path.string() + " (no header line)");
            out.flush();
            if (!out) throw std::runtime_error("cannot write " + tmp.string());
        }
        fs::rename(tmp, path);
        return std::fopen(path.string().c_str(), "a");
    }
    std::FILE* f = std::fopen(path.string().c_str(), "w");
    if (f) std::fprintf(f, "%s\n", header);
    return f;
}

// Appends the bodies with global index in [first, last) from `src`, whose first body has global index `base`.
void keep_range(const Bodies& src, uint64_t base, uint64_t first, uint64_t last, Bodies& dst) {
    uint64_t lo = std::max(first, base), hi = std::min(last, base + src.size());
    for (uint64_t g = lo; g < hi; ++g) {
        size_t i = size_t(g - base);
        dst.x.push_back(src.x[i]); dst.y.push_back(src.y[i]); dst.z.push_back(src.z[i]);
        dst.vx.push_back(src.vx[i]); dst.vy.push_back(src.vy[i]); dst.vz.push_back(src.vz[i]);
        dst.m.push_back(src.m[i]); dst.id.push_back(src.id[i]);
    }
}

double norm3(double a, double b, double c) { return std::sqrt(a * a + b * b + c * c); }

struct DiagRow {
    uint64_t step;
    double time, e, ke, pe, virial, p, l;
};

// Frames and the live view render on rank 0 from a gathered subsample of at most this many bodies.
constexpr uint64_t frame_sample_max = 16'000'000;

}  // namespace

RunSummary run_simulation(const Config& cfg, std::FILE* log_in) {
    validate(cfg);
    const bool root = par::root();
    const int nranks = par::size();
    std::FILE* devnull = root ? nullptr : std::fopen("/dev/null", "w");
    std::FILE* log = root ? log_in : devnull;

    // Chosen first so the memory pre-flight can ask it what n bodies will cost.
    std::string note;
    auto local = select_backend(cfg, note);
    auto preflight = [&](uint64_t total, const std::string& scenario) {
        std::string why = memory_preflight(*local, total, resolve_method(cfg, total), scenario);
        if (!why.empty()) throw std::runtime_error(why);
    };

    // Every rank ends up with an even share of the bodies; the first tree force call rebalances.
    Bodies bodies;
    uint64_t step0 = 0;
    double time = 0;
    if (!cfg.restart.empty()) {
        auto pieces = restart_pieces(cfg.restart, log);
        // Pass 1: counts from the headers; pass 2: keep this rank's range. One piece in memory at a time.
        std::vector<uint64_t> counts;
        const SnapshotHeader h = read_snapshot_header(pieces[0]);
        for (auto& p : pieces) {
            SnapshotHeader hp = read_snapshot_header(p);
            if (hp.step != h.step || hp.pieces != h.pieces)
                throw std::runtime_error("restart: " + p + " is from a different snapshot than " + pieces[0] +
                                         " (step " + std::to_string(hp.step) + ", " + std::to_string(hp.pieces) +
                                         " pieces recorded vs step " + std::to_string(h.step) + ", " +
                                         std::to_string(h.pieces) + ")");
            counts.push_back(hp.n);
        }
        uint64_t total = 0;
        for (auto c : counts) total += c;
        preflight(total, "");
        uint64_t first, last;
        par::even_share(total, first, last);
        uint64_t base = 0;
        for (size_t k = 0; k < pieces.size(); ++k) {
            if (base < last && base + counts[k] > first) {
                Bodies tmp;
                read_snapshot(pieces[k], tmp);
                keep_range(tmp, base, first, last, bodies);
            }
            base += counts[k];
        }
        step0 = h.step;
        time = h.time;
        std::fprintf(log, "# restart from %s (%zu piece%s): %s, n=%llu, step %llu, t=%.6g\n", cfg.restart.c_str(),
                     pieces.size(), pieces.size() == 1 ? "" : "s", h.scenario.c_str(),
                     static_cast<unsigned long long>(total), static_cast<unsigned long long>(h.step), h.time);
        if (h.G != cfg.G || h.eps != cfg.eps || h.dt != cfg.dt)
            std::fprintf(log, "# note: snapshot had G=%g eps=%g dt=%g; using the config's G=%g eps=%g dt=%g\n", h.G, h.eps,
                         h.dt, cfg.G, cfg.eps, cfg.dt);
    } else {
        // Deterministic from the seed, so every rank builds the same set and keeps its share.
        // Costs N bodies of host memory per rank at start-up (distributed generation would fix that).
        preflight(scenario_body_count(cfg), cfg.scenario);
        Bodies all = build_scenario(cfg);
        auto unused = cfg.unused_params();
        if (!unused.empty()) {
            std::string msg = "unknown parameter(s) for scenario '" + cfg.scenario + "':";
            for (auto& k : unused) msg += " " + k;
            throw std::runtime_error(msg);
        }
        if (nranks == 1) {
            bodies = std::move(all);
        } else {
            uint64_t first, last;
            par::even_share(all.size(), first, last);
            keep_range(all, 0, first, last, bodies);
        }
    }
    const uint64_t n = par::sum(uint64_t(bodies.size()));
    if (n < uint64_t(nranks)) throw std::runtime_error("fewer bodies than MPI ranks");

    std::FILE* csv = nullptr;
    if (root) {
        fs::create_directories(cfg.out_dir);
        std::ofstream(fs::path(cfg.out_dir) / "config.ini") << dump_config(cfg);
        const fs::path diag = fs::path(cfg.out_dir) / "diag.csv";
        csv = open_diag(diag, step0,
                        "step,time,energy,kinetic,potential,virial,momentum,angular_momentum,ms_per_step,interactions");
        if (!csv) throw std::runtime_error("cannot write " + diag.string());
    }
    par::barrier();  // out_dir exists before any rank writes into it

    par::DistOptions dopt;
    dopt.rebalance_every = cfg.rebalance_every;
    auto be = par::make_distributed(std::move(local), dopt);
    if (nranks > 1) note = std::to_string(nranks) + " MPI ranks; rank 0 " + note;
    ForceParams fp;
    fp.G = cfg.G;
    fp.eps = cfg.eps;
    fp.theta = cfg.theta;
    fp.method = resolve_method(cfg, n);
    // The tree potential carries the force method's error (~1e-3 at theta 0.6), which would swamp
    // the integrator's energy drift; small runs get the exact sum, which costs one direct pass.
    const bool exact_pe = cfg.exact_pe || n <= cfg.auto_direct_max;

    std::fprintf(log, "# %s\n# scenario %s, n=%llu, method %s%s, G=%g, eps=%g, dt=%g, steps=%llu, potential %s\n",
                 note.c_str(), cfg.restart.empty() ? cfg.scenario.c_str() : "(restart)", static_cast<unsigned long long>(n),
                 fp.method == ForceMethod::tree ? "tree" : "direct",
                 fp.method == ForceMethod::tree ? (" theta=" + std::to_string(cfg.theta).substr(0, 4)).c_str() : "", cfg.G,
                 cfg.eps, cfg.dt, static_cast<unsigned long long>(cfg.steps), exact_pe ? "exact" : "from force method");
    std::fprintf(log, "%10s %12s %14s %11s %12s %12s %8s %11s %11s %10s\n", "step", "time", "energy", "dE/E0", "kinetic",
                 "potential", "2K/|W|", "|P|", "dL/L0", "ms/step");

    be->upload(bodies);

    double e0 = 0, l0 = 0;
    bool have_ref = false;
    RunSummary sum;
    double integrate_s = 0;
    double interval_s = 0;
    uint64_t interval_steps = 0;
    double diag_virial = 0;

    // Collective: every rank calls it at the same steps.
    auto diagnose = [&](uint64_t step, bool have_tree_pe) {
        Diag d = be->diagnostics();
        double pe = exact_pe ? be->exact_potential_energy(cfg.G, cfg.eps) : (have_tree_pe && d.has_pe ? d.pe : NAN);
        uint64_t inter = be->times().interactions;
        DiagRow r{step, time, d.ke + pe, d.ke, pe, 2 * d.ke / std::fabs(pe), norm3(d.px, d.py, d.pz),
                  norm3(d.lx, d.ly, d.lz)};
        if (!have_ref) {
            e0 = r.e;
            l0 = r.l;
            have_ref = true;
        }
        // The slowest rank sets the pace.
        double ms = par::max(interval_steps ? 1e3 * interval_s / double(interval_steps) : 0);
        double de = e0 != 0 ? (r.e - e0) / std::fabs(e0) : 0;
        double dl = l0 > 0 ? (r.l - l0) / l0 : 0;
        std::fprintf(log, "%10llu %12.6g %14.8g %11.3e %12.6g %12.6g %8.4f %11.3e %11.3e %10.3f\n",
                     static_cast<unsigned long long>(step), time, r.e, de, r.ke, r.pe, r.virial, r.p, dl, ms);
        std::fflush(log);
        if (csv) {
            std::fprintf(csv, "%llu,%.17g,%.17g,%.17g,%.17g,%.9g,%.9g,%.9g,%.6g,%llu\n",
                         static_cast<unsigned long long>(step), time, r.e, r.ke, r.pe, r.virial, r.p, r.l, ms,
                         static_cast<unsigned long long>(inter));
            std::fflush(csv);
        }
        sum.e1 = r.e;
        diag_virial = r.virial;
        interval_s = 0;
        interval_steps = 0;
    };

    auto periodic_snap = [&](uint64_t step) { return cfg.snap_every && step % cfg.snap_every == 0; };
    // Collective.
    auto write_snap = [&](uint64_t step) {
        be->download(bodies);
        SnapshotHeader h;
        h.n = bodies.size();
        h.step = step;
        h.time = time;
        h.G = cfg.G;
        h.eps = cfg.eps;
        h.dt = cfg.dt;
        h.seed = cfg.seed;
        h.scenario = cfg.scenario;
        h.pieces = uint32_t(nranks);
        write_snapshot(numbered(cfg.out_dir, "snap", step, "snap", nranks > 1 ? par::rank() : -1), h, bodies);
    };
    // Collective.
    auto write_outputs = [&](uint64_t step, bool final_step) {
        bool frame = cfg.render_every && step % cfg.render_every == 0;
        if (periodic_snap(step) || (final_step && cfg.snap_final)) write_snap(step);
        if (frame) {
            Bodies sample;
            par::gather_sample(*be, sample, frame_sample_max);
            if (root) {
                RenderOptions ro;
                ro.width = cfg.render_width;
                ro.height = cfg.render_height;
                ro.view = cfg.view;
                ro.extent = cfg.render_extent;
                render_png(sample, ro, numbered(cfg.out_dir, "frame", step, "png"));
            }
        }
    };

    // With a live view, steps = 0 runs until Stop is pressed in the browser.
    const bool unbounded = cfg.live && cfg.steps == 0;
    const uint64_t last = unbounded ? UINT64_MAX : step0 + cfg.steps;
    auto diag_due = [&](uint64_t step) {
        return step == last || (cfg.diag_every && step % cfg.diag_every == 0);
    };

    // The live view runs on rank 0; its decisions (run/stop, frame due) are broadcast each step.
    std::unique_ptr<LiveView> live;
    Bodies live_buf;
    LiveStats ls;
    if (cfg.live && root) {
        LiveOptions lo;
        lo.port = cfg.live;
        lo.bind = cfg.live_bind;
        lo.width = cfg.live_width;
        lo.height = cfg.live_height;
        lo.fps = cfg.live_fps;
        lo.quality = cfg.live_quality;
        lo.view = cfg.view;
        lo.start_paused = cfg.live_paused;
        live = std::make_unique<LiveView>(lo, log);
        std::fprintf(log, "# live view at %s  (remote machine: ssh -L %d:localhost:%d <host>, then open http://localhost:%d/)\n",
                     live->url().c_str(), cfg.live, cfg.live, cfg.live);
        std::fflush(log);
        ls.scenario = cfg.restart.empty() ? cfg.scenario : "restart";
        ls.backend = be->name();
        ls.method = fp.method == ForceMethod::tree ? "tree" : "direct";
        ls.n = n;
        ls.last_step = unbounded ? 0 : last;
    }
    double last_de = 0, last_virial = 0, last_ms = 0;
    uint64_t last_diag_step = step0;
    // Collective when cfg.live is set.
    auto publish = [&](uint64_t step, bool force_frame) {
        if (!cfg.live) return;
        int want = 0;
        if (live) {
            ls.step = step;
            ls.time = time;
            ls.diag_step = last_diag_step;
            ls.de = last_de;
            ls.virial = last_virial;
            ls.ms_per_step = last_ms;
            live->set_stats(ls);
            want = force_frame || live->frame_due();
        }
        if (par::broadcast(want)) {
            par::gather_sample(*be, live_buf, frame_sample_max);
            if (live) live->submit(live_buf);
        }
    };

    be->compute_forces(fp, !exact_pe);
    diagnose(step0, true);
    sum.e0 = e0;
    write_outputs(step0, step0 == last);
    last_virial = diag_virial;
    publish(step0, true);

    uint64_t done = 0;
    bool stopped = false;
    for (uint64_t step = step0 + 1; step <= last; ++step) {
        if (cfg.live) {
            int go = live ? int(live->gate()) : 0;
            if (!par::broadcast(go)) {
                stopped = true;
                break;
            }
        }
        bool due = diag_due(step);
        auto t0 = std::chrono::steady_clock::now();
        be->kick(0.5 * cfg.dt);
        be->drift(cfg.dt);
        be->compute_forces(fp, due && !exact_pe);
        be->kick(0.5 * cfg.dt);
        // kick/drift may be asynchronous on the GPU; diagnostics() and times() synchronise, so
        // the interval timing is only exact across a diagnostic boundary, which is where it is read.
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        integrate_s += s;
        interval_s += s;
        ++interval_steps;
        ++done;
        time += cfg.dt;
        last_ms = 0.9 * last_ms + 0.1 * 1e3 * s;
        if (due) {
            diagnose(step, true);
            last_de = e0 != 0 ? (sum.e1 - e0) / std::fabs(e0) : 0;
            last_virial = diag_virial;
            last_diag_step = step;
        }
        write_outputs(step, step == last);
        publish(step, false);
    }

    if (stopped) {
        std::fprintf(log, "# stopped from the live view after %llu steps\n", static_cast<unsigned long long>(done));
        if (interval_steps) {
            // The last force call only had a potential if it was a diagnostic step.
            if (!exact_pe) be->compute_forces(fp, true);
            diagnose(step0 + done, true);
        }
        // The step's periodic outputs went out in the loop; only the final snapshot is still owed.
        // The stop decision was broadcast, so every rank takes this branch.
        if (cfg.snap_final && !periodic_snap(step0 + done)) write_snap(step0 + done);
    } else if (cfg.live && cfg.live_hold) {
        publish(step0 + done, true);
        if (live) {
            std::fprintf(log, "# run finished; still serving %s (press Stop in the browser or Ctrl-C to exit)\n",
                         live->url().c_str());
            std::fflush(log);
            live->hold();
        }
        par::barrier();
    }
    if (csv) std::fclose(csv);
    sum.steps = done;
    sum.wall_s = par::max(integrate_s);
    sum.ms_per_step = done ? 1e3 * sum.wall_s / double(done) : 0;
    std::fprintf(log, "# done: %llu steps in %.3f s (%.3f ms/step), dE/E0 = %.3e\n",
                 static_cast<unsigned long long>(sum.steps), sum.wall_s, sum.ms_per_step,
                 e0 != 0 ? (sum.e1 - sum.e0) / std::fabs(sum.e0) : 0.0);
    if (devnull) std::fclose(devnull);
    return sum;
}

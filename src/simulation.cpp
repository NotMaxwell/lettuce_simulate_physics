#include "simulation.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "render.hpp"
#include "scenarios.hpp"
#include "snapshot.hpp"

namespace fs = std::filesystem;

std::unique_ptr<Backend> select_backend(const Config& cfg, std::string& note) {
    if (cfg.backend != BackendKind::cpu) {
        std::string why;
        auto be = make_cuda_backend(cfg.device, why);
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

std::string numbered(const std::string& dir, const char* stem, uint64_t step, const char* ext) {
    char name[64];
    std::snprintf(name, sizeof name, "%s_%08llu.%s", stem, static_cast<unsigned long long>(step), ext);
    return (fs::path(dir) / name).string();
}

double norm3(double a, double b, double c) { return std::sqrt(a * a + b * b + c * c); }

struct DiagRow {
    uint64_t step;
    double time, e, ke, pe, virial, p, l;
};

}  // namespace

RunSummary run_simulation(const Config& cfg, std::FILE* log) {
    validate(cfg);

    Bodies bodies;
    uint64_t step0 = 0;
    double time = 0;
    if (!cfg.restart.empty()) {
        SnapshotHeader h = read_snapshot(cfg.restart, bodies);
        step0 = h.step;
        time = h.time;
        std::fprintf(log, "# restart from %s: %s, n=%llu, step %llu, t=%.6g\n", cfg.restart.c_str(), h.scenario.c_str(),
                     static_cast<unsigned long long>(h.n), static_cast<unsigned long long>(h.step), h.time);
        if (h.G != cfg.G || h.eps != cfg.eps || h.dt != cfg.dt)
            std::fprintf(log, "# note: snapshot had G=%g eps=%g dt=%g; using the config's G=%g eps=%g dt=%g\n", h.G, h.eps,
                         h.dt, cfg.G, cfg.eps, cfg.dt);
    } else {
        bodies = build_scenario(cfg);
        auto unused = cfg.unused_params();
        if (!unused.empty()) {
            std::string msg = "unknown parameter(s) for scenario '" + cfg.scenario + "':";
            for (auto& k : unused) msg += " " + k;
            throw std::runtime_error(msg);
        }
    }
    const uint64_t n = bodies.size();

    fs::create_directories(cfg.out_dir);
    {
        std::ofstream(fs::path(cfg.out_dir) / "config.ini") << dump_config(cfg);
    }
    std::FILE* csv = std::fopen((fs::path(cfg.out_dir) / "diag.csv").string().c_str(), step0 ? "a" : "w");
    if (!csv) throw std::runtime_error("cannot write " + (fs::path(cfg.out_dir) / "diag.csv").string());
    if (!step0) std::fprintf(csv, "step,time,energy,kinetic,potential,virial,momentum,angular_momentum,ms_per_step,interactions\n");

    std::string note;
    auto be = select_backend(cfg, note);
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

    auto diagnose = [&](uint64_t step, bool have_tree_pe) {
        Diag d = be->diagnostics();
        double pe = exact_pe ? be->exact_potential_energy(cfg.G, cfg.eps) : (have_tree_pe && d.has_pe ? d.pe : NAN);
        DiagRow r{step, time, d.ke + pe, d.ke, pe, 2 * d.ke / std::fabs(pe), norm3(d.px, d.py, d.pz),
                  norm3(d.lx, d.ly, d.lz)};
        if (!have_ref) {
            e0 = r.e;
            l0 = r.l;
            have_ref = true;
        }
        double ms = interval_steps ? 1e3 * interval_s / double(interval_steps) : 0;
        double de = e0 != 0 ? (r.e - e0) / std::fabs(e0) : 0;
        double dl = l0 > 0 ? (r.l - l0) / l0 : 0;
        std::fprintf(log, "%10llu %12.6g %14.8g %11.3e %12.6g %12.6g %8.4f %11.3e %11.3e %10.3f\n",
                     static_cast<unsigned long long>(step), time, r.e, de, r.ke, r.pe, r.virial, r.p, dl, ms);
        std::fflush(log);
        std::fprintf(csv, "%llu,%.17g,%.17g,%.17g,%.17g,%.9g,%.9g,%.9g,%.6g,%llu\n", static_cast<unsigned long long>(step),
                     time, r.e, r.ke, r.pe, r.virial, r.p, r.l, ms,
                     static_cast<unsigned long long>(be->times().interactions));
        std::fflush(csv);
        sum.e1 = r.e;
        interval_s = 0;
        interval_steps = 0;
    };

    auto write_outputs = [&](uint64_t step, bool final_step) {
        bool snap = (cfg.snap_every && step % cfg.snap_every == 0) || (final_step && cfg.snap_final);
        bool frame = cfg.render_every && step % cfg.render_every == 0;
        if (!snap && !frame) return;
        be->download(bodies);
        if (snap) {
            SnapshotHeader h;
            h.n = n;
            h.step = step;
            h.time = time;
            h.G = cfg.G;
            h.eps = cfg.eps;
            h.dt = cfg.dt;
            h.seed = cfg.seed;
            h.scenario = cfg.scenario;
            write_snapshot(numbered(cfg.out_dir, "snap", step, "snap"), h, bodies);
        }
        if (frame) {
            RenderOptions ro;
            ro.width = cfg.render_width;
            ro.height = cfg.render_height;
            ro.view = cfg.view;
            ro.extent = cfg.render_extent;
            render_png(bodies, ro, numbered(cfg.out_dir, "frame", step, "png"));
        }
    };

    const uint64_t last = step0 + cfg.steps;
    auto diag_due = [&](uint64_t step) {
        return step == last || (cfg.diag_every && step % cfg.diag_every == 0);
    };

    be->compute_forces(fp, !exact_pe);
    diagnose(step0, true);
    sum.e0 = e0;
    write_outputs(step0, step0 == last);

    for (uint64_t step = step0 + 1; step <= last; ++step) {
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
        time += cfg.dt;
        if (due) diagnose(step, true);
        write_outputs(step, step == last);
    }

    std::fclose(csv);
    sum.steps = cfg.steps;
    sum.wall_s = integrate_s;
    sum.ms_per_step = cfg.steps ? 1e3 * integrate_s / double(cfg.steps) : 0;
    std::fprintf(log, "# done: %llu steps in %.3f s (%.3f ms/step), dE/E0 = %.3e\n",
                 static_cast<unsigned long long>(sum.steps), sum.wall_s, sum.ms_per_step,
                 e0 != 0 ? (sum.e1 - sum.e0) / std::fabs(sum.e0) : 0.0);
    return sum;
}

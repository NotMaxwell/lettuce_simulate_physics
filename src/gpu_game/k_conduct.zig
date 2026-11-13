//! Heat conduction between touching balls, Jacobi style: every ball reads the old temperatures of
//! its neighbours and writes only its own new temperature to `tmp`, so there are no races.
//! Neighbours come from the solver's broad phase of this tick: `counts` holds the scanned cell
//! starts, `sorted_id[slot]` the ball ids in cell order, `ball_cell[id]` the cell a ball was
//! binned in (gpu_sim/collide.zig Sim).
//! Bindings: ro 0 pos, 1 body, 2 ball_cell, 3 starts, 4 sorted_id, 5 meta, 6 heat, 7 fx;
//!           rw 0 tmp.
//! Deviations from physics.zig resolvePair: impact heat is not applied (the solver does not
//! expose the approach speed), and a pair transfers heat once per tick, from the post-solve
//! positions, instead of at each sequential contact.
const c = @import("common.zig");
const L = c.L;

const pos = c.ro(L.Vec2, 0, "pos");
const body = c.ro(L.BallBody, 1, "body");
const ball_cell = c.ro(u32, 2, "ball_cell");
const starts = c.ro(u32, 3, "starts");
const sorted_id = c.ro(u32, 4, "sorted_id");
const meta = c.ro(L.BallMeta, 5, "meta");
const heat = c.ro(L.BallHeat, 6, "heat");
const fx = c.ro(u32, 7, "fx");
const tmp = c.rw(f32, 0, "tmp");

export fn conduct() callconv(c.kernel_cc) void {
    const P = c.params;
    const i = c.gid();
    if (i >= P.ball_count) return;
    const t = heat.v[i].temp;
    tmp.v[i] = t;
    // Nothing above room temperature anywhere: no ball can receive heat.
    if (fx.v[c.fx_prev_warm] == 0) return;
    if (meta.v[i] & L.flag_alive == 0) return;

    const b = body.v[i];
    const p = pos.v[i];
    const cell = ball_cell.v[i];
    const nx = P.grid_cells_x;
    const cx = cell % nx;
    const cy = cell / nx;
    const x0 = if (cx > 0) cx - 1 else cx;
    const x1 = if (cx + 1 < nx) cx + 1 else cx;
    const y0 = if (cy > 0) cy - 1 else cy;
    const y1 = if (cy + 1 < P.grid_cells_y) cy + 1 else cy;
    const dens_i = c.density(b.mass, b.radius);

    var delta: f32 = 0;
    var y = y0;
    while (y <= y1) : (y += 1) {
        const row = y * nx;
        const end = starts.v[row + x1 + 1];
        var s = starts.v[row + x0];
        while (s < end) : (s += 1) {
            const j = sorted_id.v[s];
            if (j == i) continue;
            const tj = heat.v[j].temp;
            if (!(tj > t)) continue;
            const q = pos.v[j];
            const bj = body.v[j];
            const dx = q.x - p.x;
            const dy = q.y - p.y;
            const rsum = b.radius + bj.radius;
            // The solver leaves touching balls just apart, so contact has a small tolerance.
            const reach = rsum * 1.02 + 0.5;
            if (dx * dx + dy * dy >= reach * reach) continue;
            const cond = (dens_i + c.density(bj.mass, bj.radius)) * 0.5 * rsum * 0.5 * c.cfg.base_conduct_mult;
            delta += (tj - t) * cond / b.mass * 10.0;
        }
    }
    tmp.v[i] = t + delta;
}

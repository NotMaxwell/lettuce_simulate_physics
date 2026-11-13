//! Per-ball fire state each tick (Ball.update_fire / increase_burn), the per-tick counter
//! rollover, and right-click ignition.
//! Bindings: ro 0 tmp (temperatures after conduction);
//!           rw 0 heat, 1 meta, 2 body, 3 out, 4 fx, 5 pick.
//! `out` uses the gpu_sim event layout (element 0 = header).
const c = @import("common.zig");
const L = c.L;
const cfg = c.cfg;

const tmp = c.ro(f32, 0, "tmp");
const heat = c.rw(L.BallHeat, 0, "heat");
const meta = c.rw(L.BallMeta, 1, "meta");
const body = c.rw(L.BallBody, 2, "body");
const out = c.rw(L.Event, 3, "out");
const fx = c.rw(u32, 4, "fx");
const pick_buf = c.rw(u32, 5, "pick");

export fn fxReset() callconv(c.kernel_cc) void {
    if (c.gid() != 0) return;
    fx.v[c.fx_prev_burning] = fx.v[c.fx_burning];
    fx.v[c.fx_prev_warm] = fx.v[c.fx_warm];
    fx.v[c.fx_prev_deaths] = fx.v[c.fx_deaths];
    fx.v[c.fx_burning] = 0;
    fx.v[c.fx_warm] = 0;
    fx.v[c.fx_deaths] = 0;
    fx.v[c.fx_fire_alive] = 0;
    fx.v[c.fx_ash_alive] = 0;
}

/// main.onRightClick: flame temperature, then Ball.update_temp's -150 cooling, which leaves the
/// ball burning at 250. Runs after `pick` found the ball under the mouse.
export fn igniteGrab() callconv(c.kernel_cc) void {
    if (c.gid() != 0) return;
    const k = pick_buf.v[0];
    if (k == 0) return;
    const i = k & c.id_mask;
    heat.v[i].temp = cfg.flame_temp - 150.0;
    meta.v[i] = meta.v[i] | L.flag_burning;
}

/// Every thread reaches the reduction at the end, so nothing returns early.
export fn burn() callconv(c.kernel_cc) void {
    const P = c.params;
    const i = c.gid();
    var burning: u32 = 0;
    var warm: u32 = 0;
    var died: u32 = 0;
    const m = if (i < P.ball_count) meta.v[i] else 0;
    if (m & L.flag_alive != 0) {
        const h = heat.v[i];
        var t = if (P.fx_flags & 1 != 0) tmp.v[i] else h.temp;
        var burn_amt = h.burn;
        var flags = m;
        if (t > cfg.room_temp) {
            warm = 1;
            t = @max(cfg.room_temp, t - 0.1);
        }
        if (t >= cfg.ignition_temp) {
            flags |= L.flag_burning;
            var rng = c.hash(i *% 0x9e3779b1 +% P.stamp *% 0x85ebca6b);
            burn_amt += 0.1 + c.next(&rng) * 0.2;
        } else {
            t = cfg.room_temp;
            flags &= ~L.flag_burning;
        }
        if (burn_amt >= cfg.max_burn) {
            // A full event list defers the death to the next tick, so the CPU mirror never
            // misses one.
            const slot = c.atomicAdd(&out.v[0].kind, 1);
            if (slot < P.out_cap) {
                died = 1;
                flags = (flags & ~(L.flag_alive | L.flag_burning | L.flag_grabbed)) | c.flag_ashing;
                body.v[i].mass = 0;
                out.v[slot + 1] = .{ .kind = c.event_burnout, .a = i, .b = 0, .value = body.v[i].radius };
            } else {
                _ = c.atomicAdd(&out.v[0].a, 1);
            }
        }
        if (died == 0 and flags & L.flag_burning != 0) burning = 1;
        heat.v[i] = .{ .temp = t, .burn = burn_amt };
        if (flags != m) meta.v[i] = flags;
    }
    const total = c.blockSum(burning | warm << 10 | died << 20);
    if (c.lid() == 0 and total != 0) {
        if (total & 1023 != 0) _ = c.atomicAdd(&fx.v[c.fx_burning], total & 1023);
        if (total >> 10 & 1023 != 0) _ = c.atomicAdd(&fx.v[c.fx_warm], total >> 10 & 1023);
        if (total >> 20 != 0) _ = c.atomicAdd(&fx.v[c.fx_deaths], total >> 20);
    }
}

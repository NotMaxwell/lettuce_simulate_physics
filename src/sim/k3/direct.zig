//! Direct-sum kick: every body against every body, Plummer-softened, in shared-memory tiles.
//! Positions are u32 box fractions; pair differences are wrapped integer subtractions turned
//! into f32 and scaled by box_size / 2^32, so near pairs stay exact whatever the box.
const C = @import("common.zig");
const L = C.L;

const pos = C.ro(L.Pos3, 0, "pos");
const mass = C.ro(f32, 1, "mass");
const vel = C.rw(L.Vel3, 0, "vel");
const amag = C.rw(f32, 1, "amag");
const pot = C.rw(f32, 2, "pot");
const params = C.uniform(L.Params3, "params");

const tile = C.group_size;
const sx = C.Shared(u32, tile, "sx");
const sy = C.Shared(u32, tile, "sy");
const sz = C.Shared(u32, tile, "sz");
const sm = C.Shared(f32, tile, "sm");

/// Threads outside [0, n) still walk the tiles: the barriers must be reached by the whole group.
export fn kick() callconv(C.cc) void {
    // Group-uniform, so it precedes the barriers safely (see diag in step.zig).
    if (params.first + C.groupId() * C.group_size >= params.n) return;
    const l = C.lid();
    const i = params.first + C.gid();
    const active = i < params.n;
    var me: L.Pos3 = .{ .x = 0, .y = 0, .z = 0 };
    if (active) me = C.load(pos, i);
    var ax: f32 = 0;
    var ay: f32 = 0;
    var az: f32 = 0;
    var inv_sum: f32 = 0;
    const eps2 = params.eps2;
    const tiles = (params.n + tile - 1) / tile;
    var t: u32 = 0;
    while (t < tiles) : (t += 1) {
        const j = t * tile + l;
        if (j < params.n) {
            const q = C.load(pos, j);
            sx[l] = q.x;
            sy[l] = q.y;
            sz[l] = q.z;
            sm[l] = C.load(mass, j);
        } else {
            sx[l] = 0;
            sy[l] = 0;
            sz[l] = 0;
            sm[l] = 0;
        }
        C.barrier();
        var k: u32 = 0;
        while (k < tile) : (k += 1) {
            const dx = C.delta(sx[k], me.x);
            const dy = C.delta(sy[k], me.y);
            const dz = C.delta(sz[k], me.z);
            // The 2^32 conversion scale is applied once after the sum, not per pair.
            const r2 = (dx * dx + dy * dy + dz * dz) * (params.box_scale * params.box_scale) + eps2;
            const inv = 1.0 / @sqrt(r2);
            const w = sm[k] * inv * inv * inv;
            ax += w * dx;
            ay += w * dy;
            az += w * dz;
            // The body's own softened term is excluded here, not subtracted afterwards (that loses precision).
            inv_sum += if (t * tile + k == i) 0 else sm[k] * inv;
        }
        C.barrier();
    }
    if (!active) return;
    const s = params.g * params.box_scale;
    const gx = ax * s;
    const gy = ay * s;
    const gz = az * s;
    if (params.flags & L.sim3_flag_store_accel != 0) {
        C.store(vel, i, .{ .x = gx, .y = gy, .z = gz });
    } else {
        const v = C.load(vel, i);
        C.store(vel, i, .{ .x = v.x + gx * params.kick, .y = v.y + gy * params.kick, .z = v.z + gz * params.kick });
    }
    C.store(amag, i, @sqrt(gx * gx + gy * gy + gz * gz));
    if (params.flags & L.sim3_flag_potential != 0) {
        C.store(pot, i, -params.g * inv_sum);
    }
}

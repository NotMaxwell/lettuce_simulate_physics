//! Contact count per sorted slot before the first collide iteration (see k_collide: pair
//! responses are divided by the larger contact count of the two balls).
const c = @import("common.zig");
const L = c.L;

const s_pos = c.ro(L.Vec2, 0, "s_pos");
const s_body = c.ro(L.BallBody, 1, "s_body");
const starts = c.ro(u32, 2, "starts");
const sorted_id = c.ro(u32, 3, "sorted_id");
const meta = c.ro(L.BallMeta, 4, "meta");
const cnt = c.rw(u32, 0, "cnt");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const k = c.gid();
    if (k >= P.ball_count) return;
    const b = s_body.v[k];
    if (b.radius <= 0) {
        cnt.v[k] = 0;
        return;
    }
    const p = s_pos.v[k];
    const merging = P.flags & c.flag_merging != 0;
    const my_level = if (merging) c.level(meta.v[sorted_id.v[k]]) else 0;
    const g = c.neighbourhood(@bitCast(b.friction), P);
    var contacts: u32 = 0;
    var y = g.y0;
    while (y <= g.y1) : (y += 1) {
        const row = y * P.grid_cells_x;
        const j_end = starts.v[row + g.x1 + 1];
        var j = starts.v[row + g.x0];
        while (j < j_end) : (j += 1) {
            if (j == k) continue;
            const q = s_pos.v[j];
            const dx = q.x - p.x;
            const dy = q.y - p.y;
            const d2 = dx * dx + dy * dy;
            const rsum = b.radius + s_body.v[j].radius;
            if (!(d2 > 0 and d2 < rsum * rsum)) continue;
            if (merging and my_level < c.max_level and c.level(meta.v[sorted_id.v[j]]) == my_level) continue;
            contacts += 1;
        }
    }
    cnt.v[k] = contacts;
}

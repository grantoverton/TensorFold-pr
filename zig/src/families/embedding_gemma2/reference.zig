//! The encoder on the host in fp32 at the GPU's bf16 steps: the reference its Metal layers are checked against.
const std = @import("std");
const cfg = @import("config.zig");
const bf = @import("encoder.zig").bf;

/// A layer's weights as bf16 values in fp32, at their stored shapes.
pub const Layer = struct {
    in_norm: []const f32,
    post_attn: []const f32,
    pre_ff: []const f32,
    post_ff: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    gate: []const f32,
    up: []const f32,
    down: []const f32,
    ple_gate: []const f32,
    ple_proj: []const f32,
    ple_norm: []const f32,
    scalar: f32,
};

pub const Model = struct { c: cfg.Config, embed: []const f32, ple_project: []const f32, ple_norm: []const f32, layers: []const Layer };

/// mx.fast.rms_norm of one row: bf16(w bf16(v r)), v = bf16(x pre) when pre is not 0.
pub fn rms(x: []const f32, w: []const f32, eps: f32, pre: f32, out: []f32) void {
    var s: f32 = 0;
    for (x) |v0| {
        const v = if (pre != 0) bf(v0 * pre) else v0;
        s += v * v;
    }
    const r: f32 = @floatCast(1.0 / @sqrt(@as(f64, s / @as(f32, @floatFromInt(x.len)) + eps)));
    for (x, w, out) |v0, wi, *o| {
        const v = if (pre != 0) bf(v0 * pre) else v0;
        o.* = bf(wi * bf(v * r));
    }
}

/// y [m, n] = bf16(x [m, k] w^T), w [n, k].
pub fn linear(x: []const f32, w: []const f32, m: usize, n: usize, k: usize, y: []f32) void {
    for (0..m) |r| for (0..n) |j| {
        var s: f32 = 0;
        for (x[r * k ..][0..k], w[j * k ..][0..k]) |a, b| s += a * b;
        y[r * n + j] = bf(s);
    };
}

/// mlx.nn.gelu_approx in bf16 steps, as the kernel's eg2_gelu.
pub fn gelu(x: f32) f32 {
    const c0 = bf(0.7978845608028654);
    const c1 = bf(0.044715);
    const cube = bf(x * x * x);
    const inner = bf(x + bf(c1 * cube));
    const th = bf(std.math.tanh(bf(c0 * inner)));
    return bf(bf(0.5 * x) * bf(1.0 + th));
}

fn addNorm(x: []f32, y: []const f32, w: []const f32, eps: f32, scalar: ?f32, tmp: []f32) void {
    rms(y, w, eps, 0, tmp[0..y.len]);
    for (x, tmp[0..x.len]) |*r, v| {
        const s = bf(r.* + v);
        r.* = if (scalar) |k| bf(s * k) else s;
    }
}

/// The scaled embeddings of `tokens` [rows, hidden].
pub fn embed(m: Model, tokens: []const u32, x: []f32) void {
    const d = m.c.hidden;
    const scale = bf(@sqrt(@as(f32, @floatFromInt(d))));
    for (tokens, 0..) |t, r| for (0..d) |j| {
        x[r * d + j] = bf(m.embed[t * d + j] * scale);
    };
}

/// Layer `i` over one input's rows `x` [rows, hidden] in place; `x0` its scaled embeddings.
pub fn layer(a: std.mem.Allocator, m: Model, i: usize, x0: []const f32, x: []f32) !void {
    const c = m.c;
    const l = m.layers[i];
    const kind = c.kind(i);
    const d: usize = c.hidden;
    const rows = x.len / d;
    const dh: usize = kind.head_dim;
    const nq: usize = c.query_heads;
    const nk: usize = kind.kv_heads;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const t = arena.allocator();
    const xn = try t.alloc(f32, rows * d);
    const tmp = try t.alloc(f32, @max(d, c.intermediate));
    for (0..rows) |r| rms(x[r * d ..][0..d], l.in_norm, c.eps, 0, xn[r * d ..][0..d]);
    const q = try t.alloc(f32, rows * nq * dh);
    const k = try t.alloc(f32, rows * nk * dh);
    const v = try t.alloc(f32, rows * nk * dh);
    linear(xn, l.q, rows, nq * dh, d, q);
    linear(xn, l.k, rows, nk * dh, d, k);
    linear(xn, l.v, rows, nk * dh, d, v);
    var inv: [cfg.max_head_dim / 2]f32 = undefined;
    cfg.inverseFrequencies(kind, inv[0 .. dh / 2]);
    for (0..rows) |r| {
        for (0..nq) |h| prep(q[(r * nq + h) * dh ..][0..dh], l.q_norm, c.eps, r, inv[0 .. dh / 2], tmp);
        for (0..nk) |h| {
            prep(k[(r * nk + h) * dh ..][0..dh], l.k_norm, c.eps, r, inv[0 .. dh / 2], tmp);
            normOnly(v[(r * nk + h) * dh ..][0..dh], c.eps);
        }
    }
    const att = try t.alloc(f32, rows * nq * dh);
    const p = try t.alloc(f32, rows);
    const window: usize = if (c.global[i]) 0 else c.window;
    for (0..nq) |h| for (0..rows) |r| {
        const kh = h / (nq / nk);
        var mx: f32 = -std.math.inf(f32);
        for (0..rows) |j| {
            const live = window == 0 or (if (r > j) r - j else j - r) <= window;
            var s: f32 = -std.math.inf(f32);
            if (live) {
                s = 0;
                for (q[(r * nq + h) * dh ..][0..dh], k[(j * nk + kh) * dh ..][0..dh]) |a0, b0| s += a0 * b0;
            }
            p[j] = s;
            mx = @max(mx, s);
        }
        var z: f32 = 0;
        for (p) |*e| {
            e.* = if (e.* == -std.math.inf(f32)) 0 else @exp(e.* - mx);
            z += e.*;
        }
        for (0..dh) |dd| {
            var o: f32 = 0;
            for (0..rows) |j| o += p[j] / z * v[(j * nk + kh) * dh + dd];
            att[r * nq * dh + h * dh + dd] = bf(o);
        }
    };
    const y = try t.alloc(f32, rows * d);
    linear(att, l.o, rows, d, nq * dh, y);
    for (0..rows) |r| addNorm(x[r * d ..][0..d], y[r * d ..][0..d], l.post_attn, c.eps, null, tmp);
    for (0..rows) |r| rms(x[r * d ..][0..d], l.pre_ff, c.eps, 0, xn[r * d ..][0..d]);
    const f: usize = c.intermediate;
    const g = try t.alloc(f32, rows * f);
    const u = try t.alloc(f32, rows * f);
    linear(xn, l.gate, rows, f, d, g);
    linear(xn, l.up, rows, f, d, u);
    for (g, u) |*a0, b0| a0.* = bf(gelu(a0.*) * b0);
    linear(g, l.down, rows, d, f, y);
    for (0..rows) |r| addNorm(x[r * d ..][0..d], y[r * d ..][0..d], l.post_ff, c.eps, null, tmp);
    const pw: usize = c.layer_input;
    const pli = try t.alloc(f32, rows * pw);
    linear(x0, m.ple_project[i * pw * d ..][0 .. pw * d], rows, pw, d, pli);
    const pre = bf(1.0 / @sqrt(@as(f32, @floatFromInt(d))));
    for (0..rows) |r| rms(pli[r * pw ..][0..pw], m.ple_norm, c.eps, pre, pli[r * pw ..][0..pw]);
    const pg = try t.alloc(f32, rows * pw);
    linear(x, l.ple_gate, rows, pw, d, pg);
    for (pg, pli) |*a0, b0| a0.* = bf(gelu(a0.*) * b0);
    linear(pg, l.ple_proj, rows, d, pw, y);
    for (0..rows) |r| addNorm(x[r * d ..][0..d], y[r * d ..][0..d], l.ple_norm, c.eps, l.scalar, tmp);
}

/// A head's rms_norm by `w`, then the half-split rotation at position `pos`, in bf16 steps.
fn prep(x: []f32, w: []const f32, eps: f32, pos: usize, inv: []const f32, tmp: []f32) void {
    const n = tmp[0..x.len];
    rms(x, w, eps, 0, n);
    const half = x.len / 2;
    for (x, 0..) |*o, d| {
        const other = if (d < half) -n[d + half] else n[d - half];
        const angle = @as(f32, @floatFromInt(pos)) * inv[d % half];
        o.* = bf(bf(n[d] * bf(@cos(angle))) + bf(other * bf(@sin(angle))));
    }
}

/// The value head's unweighted rms_norm.
fn normOnly(x: []f32, eps: f32) void {
    var s: f32 = 0;
    for (x) |v| s += v * v;
    const r: f32 = @floatCast(1.0 / @sqrt(@as(f64, s / @as(f32, @floatFromInt(x.len)) + eps)));
    for (x) |*v| v.* = bf(v.* * r);
}

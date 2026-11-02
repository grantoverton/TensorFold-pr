//! The Metal encoder against the host reference on seeded weights, and an input batched against it alone.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wt = @import("weights.zig");
const ref = @import("reference.zig");
const Encoder = @import("encoder.zig").Encoder;
const bf = @import("encoder.zig").bf;

/// The release's widths at two layers (sliding, then full), a small vocabulary and a window of 3 to exercise masks.
fn tiny() cfg.Config {
    var c: cfg.Config = .{
        .hidden = 512,
        .intermediate = 2048,
        .layers = 2,
        .query_heads = 4,
        .sliding = .{ .head_dim = 256, .kv_heads = 2, .theta = 10000 },
        .full = .{ .head_dim = 512, .kv_heads = 1, .theta = 1000000 },
        .layer_input = 512,
        .out = 768,
        .eps = 1e-6,
        .window = 3,
        .vocab = 64,
        .context = 1024,
    };
    c.global[1] = true;
    return c;
}

/// Seeded bf16 tensors in one shared buffer, each with its host copy.
const Synth = struct {
    arena: std.heap.ArenaAllocator,
    buf: mtl.Buffer,
    off: usize = 0,
    rng: std.Random.DefaultPrng,

    fn tensor(s: *Synth, n: usize, scale: f32, shift: f32) !struct { r: wt.Ref, h: []f32 } {
        const h = try s.arena.allocator().alloc(f32, n);
        const bits = s.buf.slice(u8, s.buf.length())[s.off..][0 .. 2 * n];
        const out: []align(1) u16 = std.mem.bytesAsSlice(u16, bits);
        const random = s.rng.random();
        for (h, out) |*v, *o| {
            v.* = bf(shift + scale * (2 * random.float(f32) - 1));
            o.* = @intCast(@as(u32, @bitCast(v.*)) >> 16);
        }
        const r: wt.Ref = .{ .buf = s.buf, .off = s.off };
        s.off = std.mem.alignForward(usize, s.off + 2 * n, 256);
        return .{ .r = r, .h = h };
    }
};

const Built = struct { w: wt.Weights, m: ref.Model, layers: [2]ref.Layer };

fn build(s: *Synth, c: cfg.Config, out: *Built) !void {
    const d: usize = c.hidden;
    const embed = try s.tensor(@as(usize, c.vocab) * d, 0.05, 0);
    const pp = try s.tensor(@as(usize, c.layers) * c.layer_input * d, 0.05, 0);
    const pn = try s.tensor(c.layer_input, 0.1, 1);
    const norm = try s.tensor(d, 0.1, 1);
    const proj = try s.tensor(@as(usize, c.out) * d, 0.05, 0);
    out.w = .{ .embed = embed.r, .ple_project = pp.r, .ple_norm = pn.r, .norm = norm.r, .projection = proj.r };
    for (0..c.layers) |i| {
        const k = c.kind(i);
        const qw: usize = @as(usize, c.query_heads) * k.head_dim;
        const kw: usize = @as(usize, k.kv_heads) * k.head_dim;
        const f: usize = c.intermediate;
        const p: usize = c.layer_input;
        const t = .{
            try s.tensor(d, 0.1, 1),      try s.tensor(d, 0.1, 1),      try s.tensor(d, 0.1, 1),      try s.tensor(d, 0.1, 1),
            try s.tensor(qw * d, 0.05, 0), try s.tensor(kw * d, 0.05, 0), try s.tensor(kw * d, 0.05, 0), try s.tensor(d * qw, 0.05, 0),
            try s.tensor(k.head_dim, 0.1, 1), try s.tensor(k.head_dim, 0.1, 1), try s.tensor(f * d, 0.05, 0), try s.tensor(f * d, 0.05, 0),
            try s.tensor(d * f, 0.03, 0), try s.tensor(p * d, 0.05, 0), try s.tensor(d * p, 0.05, 0),  try s.tensor(d, 0.1, 1),
            try s.tensor(1, 0.1, 0.6),
        };
        out.w.layers[i] = .{ .in_norm = t[0].r, .post_attn = t[1].r, .pre_ff = t[2].r, .post_ff = t[3].r, .q = t[4].r, .k = t[5].r, .v = t[6].r, .o = t[7].r, .q_norm = t[8].r, .k_norm = t[9].r, .gate = t[10].r, .up = t[11].r, .down = t[12].r, .ple_gate = t[13].r, .ple_proj = t[14].r, .ple_norm = t[15].r, .scalar = t[16].r };
        out.layers[i] = .{ .in_norm = t[0].h, .post_attn = t[1].h, .pre_ff = t[2].h, .post_ff = t[3].h, .q = t[4].h, .k = t[5].h, .v = t[6].h, .o = t[7].h, .q_norm = t[8].h, .k_norm = t[9].h, .gate = t[10].h, .up = t[11].h, .down = t[12].h, .ple_gate = t[13].h, .ple_proj = t[14].h, .ple_norm = t[15].h, .scalar = t[16].h[0] };
    }
    out.m = .{ .c = c, .embed = embed.h, .ple_project = pp.h, .ple_norm = pn.h, .layers = &out.layers };
}

const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    synth: Synth,
    built: Built,
    enc: *Encoder,

    fn init(r: *Rig, rows: u32) !void {
        r.device = try mtl.Device.init();
        errdefer r.device.deinit();
        r.queue = try r.device.queue();
        errdefer r.queue.deinit();
        r.synth = .{ .arena = .init(std.testing.allocator), .buf = try r.device.buffer(64 << 20, mtl.ResourceOptions.shared), .rng = .init(0xE6B2) };
        errdefer r.synth.buf.deinit();
        errdefer r.synth.arena.deinit();
        try build(&r.synth, tiny(), &r.built);
        r.enc = try Encoder.init(std.testing.allocator, r.device, r.queue, tiny(), r.built.w, .{ .rows = rows, .inputs = 8 });
    }

    fn deinit(r: *Rig) void {
        r.enc.deinit();
        r.synth.buf.deinit();
        r.synth.arena.deinit();
        r.queue.deinit();
        r.device.deinit();
    }
};

fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "embedding_gemma2: a sliding layer and a full layer on the GPU follow the host reference at bf16 precision" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest; // a recording Metal stand-in has no device
    _ = mtl.Device.init() catch return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var rig: Rig = undefined;
    try rig.init(64);
    defer rig.deinit();
    const a = std.testing.allocator;
    const tokens = [_]u32{ 2, 17, 5, 63, 9, 9, 40, 1, 33, 12, 7 }; // 11 rows: the window of 3 masks most pairs
    const d: usize = 512;
    const x0 = try a.alloc(f32, tokens.len * d);
    defer a.free(x0);
    const x = try a.alloc(f32, tokens.len * d);
    defer a.free(x);
    ref.embed(rig.built.m, &tokens, x0);
    @memcpy(x, x0);
    for (0..2) |layers| {
        try ref.layer(a, rig.built.m, layers, x0, x);
        try rig.enc.forward(&.{&tokens}, tokens.len, @intCast(layers + 1));
        const got = rig.enc.residual(tokens.len);
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var worst: f32 = 0;
        var scale: f32 = 0;
        for (got, x) |g, w| {
            const gv = value(g);
            dot += @as(f64, gv) * w;
            gg += @as(f64, gv) * gv;
            ww += @as(f64, w) * w;
            worst = @max(worst, @abs(gv - w));
            scale = @max(scale, @abs(w));
        }
        const cosine = dot / @sqrt(gg * ww);
        std.debug.print("embedding_gemma2 layer {d}: cosine {d:.7}, worst |gpu - host| {e:.3} of max {e:.3}\n", .{ layers, cosine, worst, scale });
        try std.testing.expect(cosine > 0.9999);
        try std.testing.expect(worst <= scale * 0x1p-5); // a few bf16 steps of the largest value
    }
}

test "embedding_gemma2: each input's projected rows batched are bit for bit its rows alone, across attention blocks and windows" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    _ = mtl.Device.init() catch return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var rig: Rig = undefined;
    try rig.init(2048);
    defer rig.deinit();
    const a = std.testing.allocator;
    var long: [600]u32 = undefined; // three attention blocks of 256 query rows
    for (&long, 0..) |*t, i| t.* = @intCast((i * 37 + 11) % 64);
    const inputs = [_][]const u32{ &.{ 2, 17, 5 }, long[0..300], &.{9}, long[100..] };
    const len: u32 = 500;
    var total: usize = 0;
    for (inputs) |t| total += t.len;
    try rig.enc.forward(&inputs, len, null);
    const batched = try a.dupe(u16, rig.enc.projected(total));
    defer a.free(batched);
    var at: usize = 0;
    for (inputs) |t| {
        try rig.enc.forward(&.{t}, @intCast(t.len), null);
        const solo = rig.enc.projected(t.len);
        try std.testing.expectEqualSlices(u16, solo, batched[at * 768 ..][0 .. t.len * 768]);
        at += t.len;
    }
    for (batched) |b| try std.testing.expect(std.math.isFinite(value(b)));
}

//! EmbeddingGemma 2's bidirectional text encoder on Metal: packed rows through the linears, padded inputs to attend.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const cfg = @import("config.zig");
const wt = @import("weights.zig");
const Ref = wt.Ref;

const names = [_][:0]const u8{ "eg2_gemm_bf16_nt", "eg2_gemm_f32_nt", "eg2_gemm_f32_nn", "eg2_embed", "eg2_rms", "eg2_rms_add", "eg2_qkv", "eg2_softmax", "eg2_heads", "eg2_geglu" };
const Pipe = enum { gemm_bf16, gemm_nt, gemm_nn, embed, rms, rms_add, qkv, softmax, heads, geglu };

/// Query rows an attention block scores at once: fixed, so an input's blocks and key ranges never depend on its batch.
pub const block = 256;

/// A forward's bounds: packed and padded rows, and inputs.
pub const Limits = struct { rows: u32 = 16384, inputs: u32 = 128 };

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    c: cfg.Config,
    w: wt.Weights,
    limits: Limits,
    lib: mtl.Library,
    pipes: [names.len]mtl.Pipeline,
    inv: [2]mtl.Buffer, // RoPE inverse frequencies: sliding, full
    scratch: mtl.Buffer,
    x: Ref,
    x0: Ref,
    xn: Ref,
    y: Ref,
    qkv: Ref,
    att: Ref,
    gu: Ref,
    act: Ref,
    pli: Ref,
    pg: Ref,
    emb: Ref, // the projected rows [rows, out] bf16, read by the host after a forward
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    s: Ref,
    ids: Ref, // host-written: packed tokens, each row's input and position, the padded map, input lengths
    input: Ref,
    pos: Ref,
    map: Ref,
    lens: Ref,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, c: cfg.Config, w: wt.Weights, limits: Limits) !*Encoder {
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .device = device, .queue = queue, .c = c, .w = w, .limits = limits, .lib = undefined, .pipes = undefined, .inv = undefined, .scratch = undefined, .x = undefined, .x0 = undefined, .xn = undefined, .y = undefined, .qkv = undefined, .att = undefined, .gu = undefined, .act = undefined, .pli = undefined, .pg = undefined, .emb = undefined, .q = undefined, .k = undefined, .v = undefined, .o = undefined, .s = undefined, .ids = undefined, .input = undefined, .pos = undefined, .map = undefined, .lens = undefined };
        const body = try frags.source(device, gpa, ks.embedding_gemma2);
        defer gpa.free(body);
        const src = try std.fmt.allocPrint(gpa, "#define EG2_D {d}\n#define EG2_NQ {d}\n{s}", .{ c.hidden, c.query_heads, body });
        defer gpa.free(src);
        e.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer e.lib.deinit();
        var made: usize = 0;
        errdefer for (e.pipes[0..made]) |p| p.deinit();
        for (names, 0..) |n, i| {
            e.pipes[i] = try mtl.Pipeline.init(device, e.lib, n, false);
            made += 1;
        }
        var held: usize = 0;
        errdefer for (e.inv[0..held]) |b| b.deinit();
        for ([_]cfg.Kind{ c.sliding, c.full }, 0..) |k, i| {
            e.inv[i] = try device.buffer(k.head_dim / 2 * 4, mtl.ResourceOptions.shared);
            held += 1;
            cfg.inverseFrequencies(k, e.inv[i].slice(f32, k.head_dim / 2));
        }
        try e.carve();
        return e;
    }

    /// The scratch: every packed and padded region at the limits, 256-byte aligned, in one shared buffer.
    fn carve(e: *Encoder) !void {
        const c = e.c;
        const r: usize = e.limits.rows;
        const dh = @max(c.sliding.head_dim, c.full.head_dim);
        const nk = @max(c.sliding.kv_heads, c.full.kv_heads);
        const qkvw = @max(c.sliding.qkvWidth(c.query_heads), c.full.qkvWidth(c.query_heads));
        const qw: usize = @as(usize, c.query_heads) * dh;
        const fields = [_]*Ref{ &e.x, &e.x0, &e.xn, &e.y, &e.qkv, &e.att, &e.gu, &e.act, &e.pli, &e.pg, &e.emb, &e.q, &e.k, &e.v, &e.o, &e.s, &e.ids, &e.input, &e.pos, &e.map, &e.lens };
        const sizes = [_]usize{ r * c.hidden * 2, r * c.hidden * 2, r * c.hidden * 2, r * c.hidden * 2, r * qkvw * 2, r * qw * 2, r * 2 * c.intermediate * 2, r * @max(c.intermediate, c.layer_input) * 2, r * c.layer_input * 2, r * c.layer_input * 2, r * c.out * 2, r * qw * 4, r * nk * dh * 4, r * nk * dh * 4, r * qw * 4, @as(usize, c.query_heads) * block * r * 4, r * 4, r * 4, r * 4, r * 4, @as(usize, e.limits.inputs) * 4 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        e.scratch = try e.device.buffer(total, mtl.ResourceOptions.shared);
        var off: usize = 0;
        for (fields, sizes) |f, s| {
            f.* = .{ .buf = e.scratch, .off = off };
            off += std.mem.alignForward(usize, s, 256);
        }
    }

    pub fn deinit(e: *Encoder) void {
        e.scratch.deinit();
        for (e.inv) |b| b.deinit();
        for (e.pipes) |p| p.deinit();
        e.lib.deinit();
        e.gpa.destroy(e);
    }

    /// The scratch's bytes.
    pub fn bytes(e: *const Encoder) usize {
        return e.scratch.length();
    }

    fn host(e: *Encoder, comptime T: type, r: Ref, n: usize) []T {
        const all = e.scratch.slice(u8, e.scratch.length());
        return @alignCast(std.mem.bytesAsSlice(T, all[r.off..][0 .. n * @sizeOf(T)]));
    }

    /// The last forward's projected rows [rows, out] as bf16 bits.
    pub fn projected(e: *Encoder, rows: usize) []const u16 {
        return e.host(u16, e.emb, rows * e.c.out);
    }

    /// The residual rows [rows, hidden] as bf16 bits (after a forward that stopped early).
    pub fn residual(e: *Encoder, rows: usize) []const u16 {
        return e.host(u16, e.x, rows * e.c.hidden);
    }

    /// One forward over `inputs` (each nonempty, at most `len` tokens), their rows packed in order; `stop`: layers run.
    pub fn forward(e: *Encoder, inputs: []const []const u32, len: u32, stop: ?u32) !void {
        const n: u32 = @intCast(inputs.len);
        if (n == 0 or n > e.limits.inputs or len == 0 or @as(u64, n) * len > e.limits.rows) return error.EmbeddingForwardTooLarge;
        var rows: u32 = 0;
        const ids = e.host(u32, e.ids, e.limits.rows);
        const input = e.host(u32, e.input, e.limits.rows);
        const pos = e.host(u32, e.pos, e.limits.rows);
        const map = e.host(i32, e.map, e.limits.rows);
        const lens = e.host(u32, e.lens, n);
        @memset(map[0 .. n * len], -1);
        for (inputs, 0..) |tokens, i| {
            if (tokens.len == 0 or tokens.len > len) return error.EmbeddingForwardTooLarge;
            lens[i] = @intCast(tokens.len);
            for (tokens, 0..) |t, p| {
                if (t >= e.c.vocab) return error.InvalidEmbeddingToken;
                ids[rows] = t;
                input[rows] = @intCast(i);
                pos[rows] = @intCast(p);
                map[i * len + p] = @intCast(rows);
                rows += 1;
            }
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        e.encode(enc, rows, n, len, stop orelse e.c.layers);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("embedding_gemma2: forward failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    fn encode(e: *Encoder, enc: mtl.ComputeEncoder, rows: u32, n: u32, len: u32, layers: u32) void {
        const c = e.c;
        enc.setPipeline(e.pipe(.embed));
        for ([_]Ref{ e.ids, e.w.embed, e.x, e.x0 }, 0..) |r, b| enc.setBuffer(r.buf, r.off, b);
        enc.setValue(bf(@sqrt(@as(f32, @floatFromInt(c.hidden)))), 4);
        enc.dispatchThreads(mtl.Size.of(c.hidden, rows, 1), mtl.Size.of(256, 1, 1));
        for (0..@min(layers, c.layers)) |i| e.layer(enc, i, rows, n, len);
        if (layers < c.layers) return;
        e.rms(enc, e.x, e.w.norm, e.xn, rows, c.hidden, 0);
        e.linear(enc, e.xn, e.w.projection, e.emb, rows, c.out, c.hidden, c.out);
    }

    fn layer(e: *Encoder, enc: mtl.ComputeEncoder, i: usize, rows: u32, n: u32, len: u32) void {
        const c = e.c;
        const l = e.w.layers[i];
        const kind = c.kind(i);
        const dh = kind.head_dim;
        const nk = kind.kv_heads;
        const nq = c.query_heads;
        const qkvw = kind.qkvWidth(nq);
        e.rms(enc, e.x, l.in_norm, e.xn, rows, c.hidden, 0);
        e.linear(enc, e.xn, l.q, e.qkv, rows, nq * dh, c.hidden, qkvw);
        e.linear(enc, e.xn, l.k, e.qkv.at(nq * dh * 2), rows, nk * dh, c.hidden, qkvw);
        e.linear(enc, e.xn, l.v, e.qkv.at((nq + nk) * dh * 2), rows, nk * dh, c.hidden, qkvw);
        const padded = n * len;
        enc.setPipeline(e.pipe(.qkv));
        for ([_]Ref{ e.qkv, l.q_norm, l.k_norm, e.map, .{ .buf = e.inv[@intFromBool(c.global[i])] }, e.q, e.k, e.v }, 0..) |r, b| enc.setBuffer(r.buf, r.off, b);
        enc.setValue(extern struct { dh: u32, nk: u32, len: u32, padded: u32, eps: f32 }{ .dh = dh, .nk = nk, .len = len, .padded = padded, .eps = c.eps }, 8);
        enc.dispatchGroups(mtl.Size.of(@as(usize, padded) * (nq + 2 * nk), 1, 1), mtl.Size.of(32, 1, 1));
        e.attend(enc, kind, n, len, if (c.global[i]) 0 else c.window);
        enc.setPipeline(e.pipe(.heads));
        for ([_]Ref{ e.o, e.att, e.input, e.pos }, 0..) |r, b| enc.setBuffer(r.buf, r.off, b);
        enc.setValue([2]u32{ dh, len }, 4);
        enc.dispatchThreads(mtl.Size.of(@as(usize, nq) * dh, rows, 1), mtl.Size.of(256, 1, 1));
        e.linear(enc, e.att, l.o, e.y, rows, c.hidden, nq * dh, c.hidden);
        e.rmsAdd(enc, l.post_attn, l.scalar, rows, false);
        e.rms(enc, e.x, l.pre_ff, e.xn, rows, c.hidden, 0);
        const f = c.intermediate;
        e.linear(enc, e.xn, l.gate, e.gu, rows, f, c.hidden, 2 * f);
        e.linear(enc, e.xn, l.up, e.gu.at(f * 2), rows, f, c.hidden, 2 * f);
        e.geglu(enc, e.gu, e.gu.at(f * 2), e.act, rows, f, 2 * f, 2 * f);
        e.linear(enc, e.act, l.down, e.y, rows, c.hidden, f, c.hidden);
        e.rmsAdd(enc, l.post_ff, l.scalar, rows, false);
        const p = c.layer_input;
        const slice = e.w.ple_project.at(i * @as(usize, p) * c.hidden * 2);
        e.linear(enc, e.x0, slice, e.pli, rows, p, c.hidden, p);
        e.rms(enc, e.pli, e.w.ple_norm, e.pli, rows, p, bf(1.0 / @sqrt(@as(f32, @floatFromInt(c.hidden)))));
        e.linear(enc, e.x, l.ple_gate, e.pg, rows, p, c.hidden, p);
        e.geglu(enc, e.pg, e.pli, e.act, rows, p, p, p);
        e.linear(enc, e.act, l.ple_proj, e.y, rows, c.hidden, p, c.hidden);
        e.rmsAdd(enc, l.ple_norm, l.scalar, rows, true);
    }

    /// Every input's heads over its own keys: scores, softmax and values a block of query rows at a time.
    fn attend(e: *Encoder, enc: mtl.ComputeEncoder, kind: cfg.Kind, n: u32, len: u32, window: u32) void {
        const dh = kind.head_dim;
        const nq = e.c.query_heads;
        const group = nq / kind.kv_heads;
        const batch = n * nq;
        var q0: u32 = 0;
        while (q0 < len) : (q0 += block) {
            const lc = @min(block, len - q0);
            const k0 = if (window == 0) 0 else q0 -| window;
            const k1 = if (window == 0) len else @min(len, q0 + lc + window);
            const keys = k1 - k0;
            const col: usize = @as(usize, dh) * 4;
            const head: usize = @as(usize, len) * dh;
            e.gemm(enc, .gemm_nt, e.q.at(q0 * col), e.k.at(k0 * col), e.s, .{ lc, keys, dh, dh, dh, keys }, batch, .{ head, head, @as(usize, lc) * keys }, group);
            enc.setPipeline(e.pipe(.softmax));
            for ([_]Ref{ e.s, e.lens }, 0..) |r, b| enc.setBuffer(r.buf, r.off, b);
            enc.setValue([5]u32{ keys, lc, q0, k0, window }, 2);
            enc.dispatchGroups(mtl.Size.of(@as(usize, batch) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            e.gemm(enc, .gemm_nn, e.s, e.v.at(k0 * col), e.o.at(q0 * col), .{ lc, dh, keys, keys, dh, dh }, batch, .{ @as(usize, lc) * keys, head, head }, group);
        }
    }

    fn pipe(e: *const Encoder, p: Pipe) mtl.Pipeline {
        return e.pipes[@backingInt(p)];
    }

    /// y = rms_norm(x) by `w` [rows, width], from bf16(x * pre) when pre is not 0.
    fn rms(e: *Encoder, enc: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: u32, width: u32, pre: f32) void {
        enc.setPipeline(e.pipe(.rms));
        for ([_]Ref{ x, w, y }, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue(extern struct { width: u32, weighted: u32, eps: f32, pre: f32 }{ .width = width, .weighted = 1, .eps = e.c.eps, .pre = pre }, 3);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// x = bf16(x + rms_norm(y) w), then times the layer scalar when `scaled`.
    fn rmsAdd(e: *Encoder, enc: mtl.ComputeEncoder, w: Ref, scalar: Ref, rows: u32, scaled: bool) void {
        enc.setPipeline(e.pipe(.rms_add));
        for ([_]Ref{ e.x, e.y, w, scalar }, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue(extern struct { eps: f32, scaled: u32 }{ .eps = e.c.eps, .scaled = @intFromBool(scaled) }, 4);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn geglu(e: *Encoder, enc: mtl.ComputeEncoder, gate: Ref, up: Ref, act: Ref, rows: u32, width: u32, gs: u32, us: u32) void {
        enc.setPipeline(e.pipe(.geglu));
        for ([_]Ref{ gate, up, act }, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue([3]u32{ width, gs, us }, 3);
        enc.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [M, N] (rows `ldy` apart) = x [M, K] w^T in bf16, w as stored [N, K].
    fn linear(e: *Encoder, enc: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, m: u32, n: u32, k: u32, ldy: u32) void {
        e.gemm(enc, .gemm_bf16, x, w, y, .{ m, n, k, k, k, ldy }, 1, .{ 0, 0, 0 }, 1);
    }

    /// gemm_nax: d = a b (or a b^T) for `batch` z, a and d z apart, b z / `group` apart; dims M, N, K, lda, ldb, ldd.
    fn gemm(e: *Encoder, enc: mtl.ComputeEncoder, p: Pipe, a: Ref, b: Ref, d: Ref, dims: [6]u32, batch: u32, strides: [3]usize, group: u32) void {
        const tn = (dims[1] + 127) / 128;
        const tm = (dims[0] + 63) / 64;
        enc.setPipeline(e.pipe(p));
        enc.setBuffer(a.buf, a.off, 0);
        enc.setBuffer(b.buf, b.off, 1);
        var v: [16]i32 = @splat(0);
        for (dims, 0..) |x, i| v[i] = @intCast(x);
        v[6] = @intCast(tn);
        v[7] = @intCast(tm);
        v[8] = 2;
        v[9] = @intCast(dims[2] / 256);
        for (strides, 10..) |s, i| v[i] = @intCast(s);
        v[15] = @intCast(group);
        enc.setBytes(std.mem.asBytes(&v), 2);
        enc.setBuffer(d.buf, d.off, 3);
        enc.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }
};

/// fp32 to the nearest bf16 value, ties to even.
pub fn bf(x: f32) f32 {
    const u: u32 = @bitCast(x);
    const r: u32 = (u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000;
    return @bitCast(r);
}

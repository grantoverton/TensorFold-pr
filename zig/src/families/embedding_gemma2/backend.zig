//! EmbeddingGemma 2 behind the lane engine: a forward of inputs per prompt step, then mean pooling and normalizing.
const std = @import("std");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const plan = @import("plan.zig");
const Encoder = @import("encoder.zig").Encoder;
const be = lanes.backend;
const Stream = lanes.Stream;

/// A request's vectors as its forwards finish them, and the order they run in.
const Job = struct { plan: plan.Plan, vectors: []f32 };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    enc: *Encoder,
    jobs: std.AutoHashMapUnmanaged(*Stream, Job) = .empty,
    rows: []f32 = &.{}, // one forward's projected rows as fp32, for pooling

    pub fn init(gpa: std.mem.Allocator, enc: *Encoder) !*Metal {
        const m = try gpa.create(Metal);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .enc = enc };
        m.rows = try gpa.alloc(f32, @as(usize, enc.limits.rows) * enc.c.out);
        return m;
    }

    pub fn deinit(m: *Metal) void {
        var it = m.jobs.valueIterator();
        while (it.next()) |j| free(m.gpa, j);
        m.jobs.deinit(m.gpa);
        m.gpa.free(m.rows);
        m.gpa.destroy(m);
    }

    pub fn backend(m: *Metal) be.Backend {
        return .{ .ptr = m, .vtable = &.{ .prefill = prefillFn, .prefill_step = stepFn, .embedding = embeddingFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .reports_fill = true }, .chunk_invariant = true };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn free(gpa: std.mem.Allocator, j: *Job) void {
        j.plan.deinit(gpa);
        gpa.free(j.vectors);
    }

    fn job(m: *Metal, s: *Stream) !*Job {
        const batch = s.embedding orelse return error.EmbeddingOnly;
        const got = try m.jobs.getOrPut(m.gpa, s);
        if (got.found_existing) return got.value_ptr;
        errdefer _ = m.jobs.remove(s);
        var p = try plan.Plan.init(m.gpa, batch.ends);
        errdefer p.deinit(m.gpa);
        got.value_ptr.* = .{ .plan = p, .vectors = try m.gpa.alloc(f32, batch.ends.len * m.enc.c.out) };
        return got.value_ptr;
    }

    /// One forward of the stream's next inputs (shortest first, as many as the encoder's rows hold); true at the end.
    fn step(m: *Metal, s: *Stream) !bool {
        if (s.isCancelled()) return error.Cancelled;
        const j = try m.job(s);
        if (j.plan.done()) return true;
        const g = j.plan.take(m.enc.limits.rows, m.enc.limits.inputs);
        const prompt = s.prompt();
        var inputs: [256][]const u32 = undefined;
        for (0..g.count) |i| {
            const at = j.plan.order[g.first + i];
            inputs[i] = prompt[j.plan.starts[at]..][0..j.plan.lens[at]];
        }
        try m.enc.forward(inputs[0..g.count], g.len, null);
        const out = m.enc.c.out;
        var total: usize = 0;
        for (inputs[0..g.count]) |t| total += t.len;
        for (m.enc.projected(total), m.rows[0 .. total * out]) |h, *r| r.* = @bitCast(@as(u32, h) << 16);
        var row: usize = 0;
        var filled: usize = 0;
        for (0..g.count) |i| {
            const at = j.plan.order[g.first + i];
            const n = inputs[i].len;
            const v = j.vectors[at * out ..][0..out];
            plan.meanPool(m.rows[row * out ..][0 .. n * out], out, v);
            plan.normalize(v);
            row += n;
        }
        for (j.plan.order[0..j.plan.next]) |at| filled += j.plan.lens[at];
        s.fill_at.store(@intCast(filled), .monotonic);
        return j.plan.done();
    }

    fn stepFn(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        return cast(ptr).step(s);
    }

    fn prefillFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const m = cast(ptr);
        while (!try m.step(s)) {}
    }

    fn embeddingFn(ptr: *anyopaque, s: *Stream, out: []f32) anyerror!void {
        const m = cast(ptr);
        const j = m.jobs.getPtr(s) orelse return error.EmbeddingPromptMissing;
        if (!j.plan.done() or out.len != j.vectors.len) return error.IncompleteEmbeddingPrompt;
        @memcpy(out, j.vectors);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const m = cast(ptr);
        if (m.jobs.fetchRemove(s)) |kv| {
            var j = kv.value;
            free(m.gpa, &j);
        }
    }

    fn firstFn(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
        return error.EmbeddingOnly;
    }
    fn queueFn(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.EmbeddingOnly;
    }
    fn readFn(_: *anyopaque, _: u64) anyerror!u32 {
        return error.EmbeddingOnly;
    }
    fn verifyFn(_: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {
        return error.EmbeddingOnly;
    }
    fn keepFn(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {
        return error.EmbeddingOnly;
    }
    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.EmbeddingOnly;
    }
};

/// What the lane host serves: `out`-wide vectors (Matryoshka widths below it), inputs of at most the context.
pub fn info(c: cfg.Config, inputs: u32) lanes.embedding.Info {
    return .{ .dimensions = c.out, .vocabulary = c.vocab, .max_batch_tokens = inputs * c.context, .max_inputs = inputs, .mrl_dimensions = &cfg.mrl };
}

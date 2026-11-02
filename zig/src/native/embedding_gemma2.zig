//! EmbeddingGemma 2 on the lane host: /v1/embeddings only, every stream's inputs encoded a forward per prompt step.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const eg = tf.embedding_gemma2;
const lanes = tf.lanes;

/// Inputs one request may carry (upstream's default), each at most the model's context.
const max_inputs = 128;

const Host = struct {
    gpa: std.mem.Allocator,
    model: *eg.Model,
    metal: *eg.backend.Metal,
    warm: mtl.keepalive.Target = undefined,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (o.drafter != null or o.vision != null) {
        problem.* = "EmbeddingGemma 2 serves text embeddings only: serve it without --drafter or --vision";
        return null;
    }
    const model = eg.Model.load(gpa, io, o.dir, .{}) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native EmbeddingGemma 2 engine cannot load this checkpoint ({s}): it serves the bf16 text model", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const window: u32 = if (o.context) |c| @intCast(@max(1, @min(c, model.config.context))) else model.config.context;
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.metal = try eg.backend.Metal.init(gpa, model.enc);
    errdefer h.metal.deinit();
    h.config = try lanes.Config.init(gpa, .{}, 1, 0);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const cap = eg.backend.info(model.config, max_inputs);
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "embedding_gemma2", .text_generation = false, .embeddings = cap, .lanes = @max(o.lanes, 1), .context_window = window });
    h.warm = .{ .queue = model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    std.log.info("embedding_gemma2: {d} layers, {d}-wide vectors, inputs of at most {d} tokens, {d:.0} MiB scratch", .{ model.config.layers, model.config.out, window, @as(f64, @floatFromInt(model.enc.bytes())) / (1 << 20) });
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

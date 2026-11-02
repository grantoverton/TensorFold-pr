//! EmbeddingGemma 2 text embeddings on Metal: config admission, the bidirectional encoder, pooling, the lane backend.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
pub const config = @import("config.zig");
pub const plan = @import("plan.zig");
pub const weights = @import("weights.zig");
pub const encoder = @import("encoder.zig");
pub const reference = @import("reference.zig");
pub const backend = @import("backend.zig");

/// The checkpoint's text tensors in Metal buffers, its config and the encoder over them.
pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: config.Config,
    checkpoint: ck.Checkpoint,
    enc: *encoder.Encoder,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, limits: encoder.Limits) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = try config.Config.read(gpa, io, dir);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try m.checkpoint.addFileSelected(m.device, path, "", weights.prefix);
        const w = try weights.load(&m.checkpoint, &m.config);
        m.enc = try encoder.Encoder.init(gpa, m.device, m.queue, m.config, w, limits);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.enc.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

test {
    _ = config;
    _ = plan;
    _ = @import("encoder_test.zig");
}

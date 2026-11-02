//! EmbeddingGemma 2's text tensors as the encoder reads them: bf16, at their stored shapes, by buffer and offset.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");

/// A tensor's place: its buffer and byte offset.
pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }
};

/// One encoder layer: its norms, q, k, v and o, the MLP, the per-layer block and the layer scalar.
pub const Layer = struct {
    in_norm: Ref,
    post_attn: Ref,
    pre_ff: Ref,
    post_ff: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    q_norm: Ref,
    k_norm: Ref,
    gate: Ref,
    up: Ref,
    down: Ref,
    ple_gate: Ref, // per_layer_input_gate [layer_input, hidden]
    ple_proj: Ref, // per_layer_projection [hidden, layer_input]
    ple_norm: Ref, // post_per_layer_input_norm [hidden]
    scalar: Ref, // layer_scalar [1]
};

pub const Weights = struct {
    embed: Ref, // embed_tokens [vocab, hidden]
    ple_project: Ref, // per_layer_model_projection [layers * layer_input, hidden]
    ple_norm: Ref, // per_layer_projection_norm [layer_input]
    norm: Ref,
    projection: Ref, // embedding_projection [out, hidden]
    layers: [cfg.max_layers]Layer = undefined,
};

pub const prefix = "language_model.";

/// The tensor `name` as bf16 of exactly `dims`.
fn get(c: *const ck.Checkpoint, name: []const u8, dims: []const usize) !Ref {
    const t = c.get(name) catch {
        std.log.err("embedding_gemma2: the checkpoint has no {s}", .{name});
        return error.MissingEmbeddingGemmaTensor;
    };
    var bytes: usize = 2;
    for (dims) |d| bytes *= d;
    if (t.dtype != .bf16 or t.rank != dims.len or !std.mem.eql(usize, t.shape[0..t.rank], dims) or t.bytes != bytes) {
        std.log.err("embedding_gemma2: {s} is not bf16{any}", .{ name, dims });
        return error.UnexpectedEmbeddingGemmaTensor;
    }
    return .{ .buf = t.buffer, .off = t.offset };
}

fn layerGet(c: *const ck.Checkpoint, i: usize, suffix: []const u8, dims: []const usize) !Ref {
    var b: [160]u8 = undefined;
    return get(c, try std.fmt.bufPrint(&b, prefix ++ "layers.{d}.{s}", .{ i, suffix }), dims);
}

/// Every text tensor from `c` (loaded with the `language_model.` names), checked against `g`.
pub fn load(c: *const ck.Checkpoint, g: *const cfg.Config) !Weights {
    const d: usize = g.hidden;
    var w: Weights = .{
        .embed = try get(c, prefix ++ "embed_tokens.weight", &.{ g.vocab, d }),
        .ple_project = try get(c, prefix ++ "ple.per_layer_model_projection.weight", &.{ @as(usize, g.layers) * g.layer_input, d }),
        .ple_norm = try get(c, prefix ++ "ple.per_layer_projection_norm.weight", &.{g.layer_input}),
        .norm = try get(c, prefix ++ "norm.weight", &.{d}),
        .projection = try get(c, prefix ++ "embedding_projection.weight", &.{ g.out, d }),
    };
    for (0..g.layers) |i| {
        const k = g.kind(i);
        const qw: usize = @as(usize, g.query_heads) * k.head_dim;
        const kw: usize = @as(usize, k.kv_heads) * k.head_dim;
        const f: usize = g.intermediate;
        const p: usize = g.layer_input;
        w.layers[i] = .{
            .in_norm = try layerGet(c, i, "input_layernorm.weight", &.{d}),
            .post_attn = try layerGet(c, i, "post_attention_layernorm.weight", &.{d}),
            .pre_ff = try layerGet(c, i, "pre_feedforward_layernorm.weight", &.{d}),
            .post_ff = try layerGet(c, i, "post_feedforward_layernorm.weight", &.{d}),
            .q = try layerGet(c, i, "self_attn.q_proj.weight", &.{ qw, d }),
            .k = try layerGet(c, i, "self_attn.k_proj.weight", &.{ kw, d }),
            .v = try layerGet(c, i, "self_attn.v_proj.weight", &.{ kw, d }),
            .o = try layerGet(c, i, "self_attn.o_proj.weight", &.{ d, qw }),
            .q_norm = try layerGet(c, i, "self_attn.q_norm.weight", &.{k.head_dim}),
            .k_norm = try layerGet(c, i, "self_attn.k_norm.weight", &.{k.head_dim}),
            .gate = try layerGet(c, i, "mlp.gate_proj.weight", &.{ f, d }),
            .up = try layerGet(c, i, "mlp.up_proj.weight", &.{ f, d }),
            .down = try layerGet(c, i, "mlp.down_proj.weight", &.{ d, f }),
            .ple_gate = try layerGet(c, i, "ple_block.per_layer_input_gate.weight", &.{ p, d }),
            .ple_proj = try layerGet(c, i, "ple_block.per_layer_projection.weight", &.{ d, p }),
            .ple_norm = try layerGet(c, i, "ple_block.post_per_layer_input_norm.weight", &.{d}),
            .scalar = try layerGet(c, i, "layer_scalar", &.{1}),
        };
    }
    return w;
}

//! EmbeddingGemma 2 (embedding_gemma2) text admission: its geometry, layer kinds, RoPE and the embedding width.
const std = @import("std");

pub const max_layers = 64;
/// The longest input oMLX embeds (mlx_vlm's max_input_length for embedding_gemma2): the served context.
pub const max_context = 8192;
/// The widest head the qkv kernel's threadgroup row holds.
pub const max_head_dim = 512;
/// Matryoshka widths the checkpoint's README names (128, 256, 512 and the full 768).
pub const mrl = [_]u32{ 128, 256, 512 };

/// One attention kind's heads: sliding layers' or full layers'.
pub const Kind = struct {
    head_dim: u32,
    kv_heads: u32,
    theta: f64,

    pub fn qkvWidth(k: Kind, query_heads: u32) u32 {
        return (query_heads + 2 * k.kv_heads) * k.head_dim;
    }
};

pub const Config = struct {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    query_heads: u32,
    sliding: Kind,
    full: Kind,
    layer_input: u32, // hidden_size_per_layer_input: each layer's slice of per_layer_model_projection
    out: u32, // embedding_dim: embedding_projection's rows
    eps: f32,
    window: u32,
    vocab: u32,
    context: u32,
    global: [max_layers]bool = @splat(false),

    pub fn kind(c: *const Config, layer: usize) Kind {
        return if (c.global[layer]) c.full else c.sliding;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

pub const Error = error{ BadEmbeddingGemmaConfig, UnsupportedEmbeddingGemmaGeometry } || std.json.ParseError(std.json.Scanner);

fn object(v: ?std.json.Value) Error!std.json.ObjectMap {
    const x = v orelse return error.BadEmbeddingGemmaConfig;
    return if (x == .object) x.object else error.BadEmbeddingGemmaConfig;
}

fn number(v: ?std.json.Value) Error!f64 {
    return switch (v orelse return error.BadEmbeddingGemmaConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadEmbeddingGemmaConfig,
    };
}

/// A positive integer field of at most 2^31.
fn count(o: std.json.ObjectMap, key: []const u8) Error!u32 {
    const v = o.get(key) orelse return error.BadEmbeddingGemmaConfig;
    if (v != .integer or v.integer < 1 or v.integer > 1 << 31) return error.BadEmbeddingGemmaConfig;
    return @intCast(v.integer);
}

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

fn theta(ropes: std.json.ObjectMap, key: []const u8) Error!f64 {
    const r = try object(ropes.get(key));
    if (r.get("rope_type")) |t| if (!isString(t, "default")) return error.UnsupportedEmbeddingGemmaGeometry;
    if (r.get("partial_rotary_factor")) |f| if (try number(f) != 1.0) return error.UnsupportedEmbeddingGemmaGeometry;
    return number(r.get("rope_theta"));
}

/// The full layers' heads: per_layer_config's entry for every full layer, all alike.
fn fullKind(text: std.json.ObjectMap, c: *const Config, sliding: Kind) Error!Kind {
    const per = if (text.get("per_layer_config")) |v| try object(v) else return sliding;
    var found: ?Kind = null;
    for (0..c.layers) |i| {
        var key: [8]u8 = undefined;
        const name = std.fmt.bufPrint(&key, "{d:0>2}", .{i}) catch unreachable;
        const entry = per.get(name) orelse continue;
        if (!c.global[i]) return error.UnsupportedEmbeddingGemmaGeometry; // a sliding layer of its own shape
        const o = try object(entry);
        var it = o.iterator();
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            if (!std.mem.eql(u8, k, "head_dim") and !std.mem.eql(u8, k, "num_key_value_heads")) return error.UnsupportedEmbeddingGemmaGeometry;
        }
        const here: Kind = .{ .head_dim = try count(o, "head_dim"), .kv_heads = try count(o, "num_key_value_heads"), .theta = 0 };
        if (found) |f| if (f.head_dim != here.head_dim or f.kv_heads != here.kv_heads) return error.UnsupportedEmbeddingGemmaGeometry;
        found = here;
    }
    return found orelse sliding;
}

fn heads(k: Kind, query_heads: u32) Error!void {
    if (k.head_dim > max_head_dim or k.head_dim % 64 != 0) return error.UnsupportedEmbeddingGemmaGeometry;
    if (query_heads % k.kv_heads != 0) return error.UnsupportedEmbeddingGemmaGeometry;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    if (!isString(root.get("model_type"), "embedding_gemma2")) return error.BadEmbeddingGemmaConfig;
    const text = try object(root.get("text_config"));
    if (text.get("model_type")) |t| if (!isString(t, "embedding_gemma2_text")) return error.BadEmbeddingGemmaConfig;
    if (!isString(text.get("hidden_activation"), "gelu_pytorch_tanh")) return error.UnsupportedEmbeddingGemmaGeometry;
    if (text.get("attention_bias")) |b| if (b != .bool or b.bool) return error.UnsupportedEmbeddingGemmaGeometry;
    var c: Config = .{
        .hidden = try count(text, "hidden_size"),
        .intermediate = try count(text, "intermediate_size"),
        .layers = try count(text, "num_hidden_layers"),
        .query_heads = try count(text, "num_attention_heads"),
        .sliding = .{ .head_dim = try count(text, "head_dim"), .kv_heads = try count(text, "num_key_value_heads"), .theta = 0 },
        .full = undefined,
        .layer_input = try count(text, "hidden_size_per_layer_input"),
        .out = try count(text, "embedding_dim"),
        .eps = @floatCast(try number(text.get("rms_norm_eps"))),
        .window = try count(text, "sliding_window"),
        .vocab = try count(text, "vocab_size"),
        .context = @min(max_context, try count(text, "max_position_embeddings")),
    };
    if (c.layers > max_layers or c.hidden % 128 != 0 or c.layer_input % 128 != 0) return error.UnsupportedEmbeddingGemmaGeometry;
    const types = text.get("layer_types") orelse return error.BadEmbeddingGemmaConfig;
    if (types != .array or types.array.items.len != c.layers) return error.BadEmbeddingGemmaConfig;
    for (types.array.items, 0..) |t, i| {
        c.global[i] = isString(t, "full_attention");
        if (!c.global[i] and !isString(t, "sliding_attention")) return error.UnsupportedEmbeddingGemmaGeometry;
    }
    c.global[c.layers - 1] = true; // mlx_vlm's TextConfig makes the last layer full whatever the list says
    c.full = try fullKind(text, &c, c.sliding);
    const ropes = try object(text.get("rope_parameters"));
    c.sliding.theta = try theta(ropes, "sliding_attention");
    c.full.theta = try theta(ropes, "full_attention");
    try heads(c.sliding, c.query_heads);
    try heads(c.full, c.query_heads);
    for (mrl) |d| if (d >= c.out) return error.UnsupportedEmbeddingGemmaGeometry;
    return c;
}

/// RoPE inverse frequencies [head_dim / 2] in fp32 as mlx_vlm makes them: 1 / theta^(2i / head_dim).
pub fn inverseFrequencies(k: Kind, out: []f32) void {
    for (out, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(k.head_dim));
        f.* = 1.0 / std.math.pow(f32, @floatCast(k.theta), e);
    }
}

const fixture = @embedFile("fixtures/config.json");

test "embedding_gemma2: the release's config is admitted: 24 layers, every sixth full at head width 512 with one key head" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(u32, 512), c.hidden);
    try std.testing.expectEqual(@as(u32, 24), c.layers);
    try std.testing.expectEqual(@as(u32, 768), c.out);
    try std.testing.expectEqual(@as(u32, 512), c.layer_input);
    try std.testing.expectEqual(@as(u32, 8192), c.context);
    try std.testing.expectEqual(@as(u32, 262144), c.vocab);
    for (0..24) |i| try std.testing.expectEqual(i % 6 == 5, c.global[i]);
    try std.testing.expectEqual(Kind{ .head_dim = 256, .kv_heads = 2, .theta = 10000 }, c.sliding);
    try std.testing.expectEqual(Kind{ .head_dim = 512, .kv_heads = 1, .theta = 1000000 }, c.full);
    try std.testing.expectEqual(@as(u32, 2048), c.sliding.qkvWidth(c.query_heads));
    try std.testing.expectEqual(@as(u32, 3072), c.full.qkvWidth(c.query_heads));
}

test "embedding_gemma2: other models and shapes are refused" {
    const a = std.testing.allocator;
    const swaps = [_][2][]const u8{
        .{ "\"model_type\": \"embedding_gemma2\"", "\"model_type\": \"gemma4\"" },
        .{ "\"gelu_pytorch_tanh\"", "\"silu\"" },
        .{ "\"attention_bias\": false", "\"attention_bias\": true" },
        .{ "\"head_dim\": 512", "\"head_dim\": 1024" },
        .{ "\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"" },
        .{ "\"embedding_dim\": 768", "\"embedding_dim\": 256" },
    };
    for (swaps) |s| {
        const changed = try std.mem.replaceOwned(u8, a, fixture, s[0], s[1]);
        defer a.free(changed);
        try std.testing.expect(!std.mem.eql(u8, changed, fixture));
        if (parse(a, changed)) |_| return error.TestExpectedRefusal else |_| {}
    }
}

test "embedding_gemma2: inverse frequencies fall from 1 by theta over the head" {
    var f: [128]f32 = undefined;
    inverseFrequencies(.{ .head_dim = 256, .kv_heads = 2, .theta = 10000 }, &f);
    try std.testing.expectEqual(@as(f32, 1), f[0]);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / 10000.0), f[64] * f[64], 1e-5);
    try std.testing.expect(f[127] < f[126]);
}

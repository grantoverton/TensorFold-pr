//! A request's inputs in forwards: shortest first, each forward's inputs padded to its longest within the row budget.
const std = @import("std");

/// The inputs a forward takes, as positions in `order`, and the padded length L they share.
pub const Group = struct { first: usize, count: usize, len: u32 };

/// The inputs' order by length (stable), each input's token count, and where the next forward starts.
pub const Plan = struct {
    order: []u32,
    lens: []u32,
    starts: []u32, // each input's first token in the request's prompt
    next: usize = 0,

    /// `ends` are the request's input boundaries (Batch.ends).
    pub fn init(gpa: std.mem.Allocator, ends: []const u32) !Plan {
        const n = ends.len;
        const order = try gpa.alloc(u32, n);
        errdefer gpa.free(order);
        const lens = try gpa.alloc(u32, n);
        errdefer gpa.free(lens);
        const starts = try gpa.alloc(u32, n);
        var at: u32 = 0;
        for (ends, 0..) |end, i| {
            starts[i] = at;
            lens[i] = end - at;
            order[i] = @intCast(i);
            at = end;
        }
        std.sort.insertion(u32, order, lens, struct {
            fn less(l: []u32, x: u32, y: u32) bool {
                return l[x] < l[y];
            }
        }.less);
        return .{ .order = order, .lens = lens, .starts = starts };
    }

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(p.order);
        gpa.free(p.lens);
        gpa.free(p.starts);
    }

    pub fn done(p: *const Plan) bool {
        return p.next >= p.order.len;
    }

    /// The next forward: as many inputs as fit `rows` padded rows and `max_inputs`, at least one; then past them.
    pub fn take(p: *Plan, rows: u32, max_inputs: usize) Group {
        const first = p.next;
        var n: usize = 0;
        var len: u32 = 0;
        while (first + n < p.order.len and n < max_inputs) {
            const l = @max(len, p.lens[p.order[first + n]]);
            if (n > 0 and @as(u64, l) * (n + 1) > rows) break;
            len = l;
            n += 1;
        }
        p.next = first + n;
        return .{ .first = first, .count = n, .len = len };
    }
};

/// sentence-transformers' mean pooling of `rows` [n, width] (bf16 values as fp32) in fp32, into `out` [width].
pub fn meanPool(rows: []const f32, width: usize, out: []f32) void {
    @memset(out, 0);
    const n = rows.len / width;
    for (0..n) |r| for (out, rows[r * width ..][0..width]) |*o, v| {
        o.* += v;
    };
    const inv: f32 = @floatFromInt(@max(n, 1));
    for (out) |*o| o.* /= inv;
}

/// mlx_vlm's normalize_embeddings: x / max(||x||, 1e-9), the norm in fp32.
pub fn normalize(v: []f32) void {
    var s: f32 = 0;
    for (v) |x| s += x * x;
    const n = @max(@sqrt(s), 1e-9);
    for (v) |*x| x.* /= n;
}

test "embedding_gemma2: inputs go shortest first and a forward stops before its padded rows pass the budget" {
    const a = std.testing.allocator;
    var p = try Plan.init(a, &.{ 5, 7, 17, 18, 22 }); // lengths 5, 2, 10, 1, 4
    defer p.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 4, 0, 2 }, p.order);
    try std.testing.expectEqualSlices(u32, &.{ 0, 5, 7, 17, 18 }, p.starts);
    const g1 = p.take(16, 128); // 1, 2, 4, 5: four at L 5 is 20 > 16, so three at L 4
    try std.testing.expectEqual(Group{ .first = 0, .count = 3, .len = 4 }, g1);
    const g2 = p.take(16, 128); // 5, then 10 makes 20 > 16
    try std.testing.expectEqual(Group{ .first = 3, .count = 1, .len = 5 }, g2);
    const g3 = p.take(16, 128);
    try std.testing.expectEqual(Group{ .first = 4, .count = 1, .len = 10 }, g3);
    try std.testing.expect(p.done());
}

test "embedding_gemma2: one input longer than the budget still forms its own forward, and the input cap holds" {
    const a = std.testing.allocator;
    var p = try Plan.init(a, &.{ 1, 2, 3, 40 });
    defer p.deinit(a);
    try std.testing.expectEqual(Group{ .first = 0, .count = 2, .len = 1 }, p.take(100, 2));
    try std.testing.expectEqual(Group{ .first = 2, .count = 1, .len = 1 }, p.take(40, 2)); // 37 twice is 74 > 40
    try std.testing.expectEqual(Group{ .first = 3, .count = 1, .len = 37 }, p.take(8, 2));
}

test "embedding_gemma2: mean pooling averages each column over the rows and normalizing gives a unit vector" {
    const rows = [_]f32{ 1, 2, 3, 4, 5, 6 }; // three rows of width 2
    var out: [2]f32 = undefined;
    meanPool(&rows, 2, &out);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, &out);
    normalize(&out);
    try std.testing.expectEqualSlices(f32, &.{ 0.6, 0.8 }, &out);
    var zero = [_]f32{ 0, 0 };
    normalize(&zero);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, &zero);
}

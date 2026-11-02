// EmbeddingGemma 2's text encoder at MLX's bf16 steps (mlx_vlm embedding_gemma2), after gemm_nax and EG2_D, EG2_NQ.

// gemm_nax's bounded tiles, batch z at A + P[10] z, B + P[11] (z / P[15]), D + P[12] z (P[15]: z's a key head).
#define EG2_GEMM(NAME, T, TB)                                                                                     \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device T* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    const long z = long(tg.z);                                                                                    \
    A += long(P[10]) * z;                                                                                         \
    B += long(P[11]) * (z / long(P[15]));                                                                         \
    D += long(P[12]) * z;                                                                                         \
    tfp::gemm_nax<T, false, TB, false, false, false, 64, 128, 256, 2, 4>(A, B, D, P, uint3(tg.x, tg.y, 0), sg, lane); \
  }
EG2_GEMM(eg2_gemm_bf16_nt, bfloat16_t, true)
EG2_GEMM(eg2_gemm_f32_nt, float, true)
EG2_GEMM(eg2_gemm_f32_nn, float, false)

inline float eg2_bf(float x) { return float(bfloat(x)); }

inline float eg2_group_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float w = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    w = simd_sum(w);
    if (lane == 0) part[32] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

inline float eg2_group_max(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_max(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float w = lane < (n + 31) / 32 ? part[lane] : -INFINITY;
    w = simd_max(w);
    if (lane == 0) part[32] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// mlx.nn.gelu_approx as MLX's bf16 ops: 0.5 x (1 + tanh(c0 (x + c1 x^3))), each step rounded to bf16.
inline float eg2_gelu(float x) {
  const float c0 = eg2_bf(0.7978845608028654f), c1 = eg2_bf(0.044715f);
  const float cube = eg2_bf(x * x * x);
  const float inner = eg2_bf(x + eg2_bf(c1 * cube));
  const float th = eg2_bf(metal::precise::tanh(eg2_bf(c0 * inner)));
  return eg2_bf(eg2_bf(0.5f * x) * eg2_bf(1.0f + th));
}

// Token rows times the bf16 scale sqrt(D), into x and x0 (the per-layer inputs read x0).
[[kernel]] void eg2_embed(const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
                          device bfloat* x [[buffer(2)]], device bfloat* x0 [[buffer(3)]],
                          constant float& scale [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= EG2_D) return;
  const bfloat v = bfloat(float(table[ulong(ids[p.y]) * EG2_D + p.x]) * scale);
  x[ulong(p.y) * EG2_D + p.x] = v;
  x0[ulong(p.y) * EG2_D + p.x] = v;
}

struct Eg2Rows {
  uint width;
  uint weighted;
  float eps;
  float pre; // nonzero: the row is bf16(x * pre) first (a bf16 scalar product before the norm)
};

// mx.fast.rms_norm: fp32 sums, bf16(x r), then times the weight in bf16 when `weighted`.
[[kernel]] void eg2_rms(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
                        constant Eg2Rows& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float v = p.pre != 0.0f ? eg2_bf(float(xr[i]) * p.pre) : float(xr[i]);
    s += v * v;
  }
  const float r = metal::precise::rsqrt(eg2_group_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    const float v = p.pre != 0.0f ? eg2_bf(float(xr[i]) * p.pre) : float(xr[i]);
    const float o = eg2_bf(v * r);
    yr[i] = bfloat(p.weighted ? float(w[i]) * o : o);
  }
}

struct Eg2Add {
  float eps;
  uint scaled; // 1: then x = bf16(x * s[0]) (the layer's layer_scalar)
};

// A branch's post norm into the residual: x = bf16(x + rms_norm(y) w), then times the layer scalar when `scaled`.
[[kernel]] void eg2_rms_add(device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                            const device bfloat* scalar [[buffer(3)]], constant Eg2Add& p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                            uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device bfloat* xr = x + ulong(row) * EG2_D;
  const device bfloat* yr = y + ulong(row) * EG2_D;
  float s = 0.0f;
  for (uint i = t; i < EG2_D; i += n) s += float(yr[i]) * float(yr[i]);
  const float r = metal::precise::rsqrt(eg2_group_sum(s, part, n, lane, sg) / float(EG2_D) + p.eps);
  const float k = p.scaled ? float(scalar[0]) : 1.0f;
  for (uint i = t; i < EG2_D; i += n) {
    const float v = eg2_bf(float(xr[i]) + eg2_bf(float(w[i]) * eg2_bf(float(yr[i]) * r)));
    xr[i] = bfloat(p.scaled ? v * k : v);
  }
}

struct Eg2Qkv {
  uint dh;     // the layer kind's head width
  uint nk;     // its key heads
  uint len;    // the padded input length L
  uint padded; // inputs times L
  float eps;
};

// Packed q|k|v rows to padded fp32 heads [input][head][L][dh]: per-head norms (v unweighted), q and k rotated.
[[kernel]] void eg2_qkv(const device bfloat* qkv [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
                        const device bfloat* kw [[buffer(2)]], const device int* map [[buffer(3)]],
                        const device float* inv [[buffer(4)]], device float* Q [[buffer(5)]], device float* K [[buffer(6)]],
                        device float* V [[buffer(7)]], constant Eg2Qkv& p [[buffer(8)]],
                        uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float nrm[512];
  const uint parts = EG2_NQ + 2 * p.nk, part = g % parts, t = g / parts;
  if (t >= p.padded) return;
  const uint input = t / p.len, pos = t % p.len;
  const bool isq = part < EG2_NQ, isk = !isq && part < EG2_NQ + p.nk;
  const uint head = isq ? part : isk ? part - EG2_NQ : part - EG2_NQ - p.nk;
  const uint heads = isq ? EG2_NQ : p.nk;
  device float* out = (isq ? Q : isk ? K : V) + ((ulong(input) * heads + head) * p.len + pos) * p.dh;
  const int row = map[t];
  if (row < 0) {
    for (uint d = lane; d < p.dh; d += 32) out[d] = 0.0f;
    return;
  }
  const device bfloat* x = qkv + ulong(row) * (parts * p.dh) + part * p.dh;
  float s = 0.0f;
  for (uint d = lane; d < p.dh; d += 32) s += float(x[d]) * float(x[d]);
  const float r = metal::precise::rsqrt(simd_sum(s) / float(p.dh) + p.eps);
  const device bfloat* w = isq ? qw : kw;
  for (uint d = lane; d < p.dh; d += 32) {
    const float v = eg2_bf(float(x[d]) * r);
    nrm[d] = (isq || isk) ? eg2_bf(float(w[d]) * v) : v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint hd = p.dh / 2;
  for (uint d = lane; d < p.dh; d += 32) {
    if (!isq && !isk) {
      out[d] = nrm[d];
      continue;
    }
    const float other = d < hd ? -nrm[d + hd] : nrm[d - hd];
    const float angle = float(pos) * inv[d % hd];
    const float c = eg2_bf(metal::precise::cos(angle)), sn = eg2_bf(metal::precise::sin(angle));
    out[d] = eg2_bf(eg2_bf(nrm[d] * c) + eg2_bf(other * sn));
  }
}

struct Eg2Soft {
  uint n;      // keys a row scores
  uint lc;     // query rows a head's block holds
  uint q0;     // the block's first query position
  uint k0;     // the first key's position
  uint window; // 0: every key of the input; else |q - k| <= window
};

// Each score row to its softmax over its input's real keys (and the window), in place; masked keys get 0.
[[kernel]] void eg2_softmax(device float* S [[buffer(0)]], const device uint* lens [[buffer(1)]],
                            constant Eg2Soft& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const uint block = row / p.lc, q = p.q0 + row % p.lc, real = lens[block / EG2_NQ];
  device float* s = S + ulong(row) * p.n;
  float m = -INFINITY;
  for (uint i = t; i < p.n; i += tn) {
    const uint k = p.k0 + i;
    const bool live = k < real && (p.window == 0 || (q > k ? q - k : k - q) <= p.window);
    if (live) m = max(m, s[i]);
  }
  m = eg2_group_max(m, part, tn, lane, sg);
  float z = 0.0f;
  for (uint i = t; i < p.n; i += tn) {
    const uint k = p.k0 + i;
    const bool live = m != -INFINITY && k < real && (p.window == 0 || (q > k ? q - k : k - q) <= p.window);
    const float e = live ? metal::precise::exp(s[i] - m) : 0.0f;
    s[i] = e;
    z += e;
  }
  z = eg2_group_sum(z, part, tn, lane, sg);
  const float inv = z > 0.0f ? 1.0f / z : 0.0f;
  for (uint i = t; i < p.n; i += tn) s[i] *= inv;
}

struct Eg2Heads {
  uint dh;
  uint len; // the padded input length L
};

// Padded fp32 heads [input][head][L][dh] back to packed rows [rows][NQ dh] bf16.
[[kernel]] void eg2_heads(const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]],
                          const device uint* input [[buffer(2)]], const device uint* pos [[buffer(3)]],
                          constant Eg2Heads& p [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint width = EG2_NQ * p.dh;
  if (at.x >= width) return;
  const uint h = at.x / p.dh, d = at.x % p.dh, r = at.y;
  out[ulong(r) * width + at.x] = bfloat(O[((ulong(input[r]) * EG2_NQ + h) * p.len + pos[r]) * p.dh + d]);
}

struct Eg2Glu {
  uint n;  // values a row
  uint gs; // the gate's row pitch
  uint us; // the multiplier's row pitch
};

// bf16(gelu_approx(gate) * up) [rows][n]: the MLP's and the per-layer block's gate.
[[kernel]] void eg2_geglu(const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
                          device bfloat* act [[buffer(2)]], constant Eg2Glu& p [[buffer(3)]],
                          uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) return;
  const float g = float(gate[ulong(at.y) * p.gs + at.x]), u = float(up[ulong(at.y) * p.us + at.x]);
  act[ulong(at.y) * p.n + at.x] = bfloat(eg2_gelu(g) * u);
}

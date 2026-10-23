#pragma once
// Opt-in Qwen3.6 KV only. Format follows Strata PR #3678's int8 path:
// signed codes [-127,127], one stored FP16 scale per 64 values, separately
// for K and V, [head][context][dimension/group]. No FP32 kernels are replaced.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace q36kv8 {
constexpr int HD = 256, GROUP = 64, NG = HD / GROUP, GQA = 8;
struct Cache { int8_t *k = nullptr, *v = nullptr; __half *ks = nullptr, *vs = nullptr; };
struct Rows { const int32_t *cell = nullptr, *slot = nullptr, *delta = nullptr; };

__device__ __forceinline__ float reduce_sum(float v, float* sh) {
    const int j = threadIdx.x; sh[j] = v; __syncthreads();
    for (int s = 128; s; s >>= 1) { if (j < s) sh[j] += sh[j+s]; __syncthreads(); }
    return sh[0];
}
__device__ __forceinline__ float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
__device__ __forceinline__ int8_t code(float x, float scale) {
    if (!(scale > 0.f)) return 0;
    const float r = rintf(div_rn(x, scale));
    return (int8_t) fminf(127.f, fmaxf(-127.f, r));
}

// Same query/gate, RMS and RoPE arithmetic as the FP32 prep. KV quantization
// happens only AFTER key rotation. All 256 threads participate in reductions.
static __global__ void prepare(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                        float* q, float* gate, Cache cache, const Cache* table, int H, int HKV, int n_rot,
                        const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                        const int32_t* mtab, Rows rows) {
    __shared__ float sh[HD], buf[HD], rk[8], rv[8];
    const int head = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = rows.cell ? rows.cell[t] : *pos0 + t;
    const int dl = rows.delta ? rows.delta[t] : (rdelta ? *rdelta : 0);
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t)t*H+head)*2*HD : kx + ((size_t)t*HKV+head-H)*HD;
    const float v0 = src[i];
    const float sum = reduce_sum(v0*v0, sh);
    const float r = rsqrtf(sum / (float)HD + eps);
    buf[i] = v0*r*(isq ? qw[i] : kw[i]); __syncthreads();
    float v = buf[i];
    const int half = n_rot/2;
    if (i < n_rot) {
        const int p = i < half ? i : i-half;
        const int rp = mtab ? __ldg(mtab+(size_t)pos*3+p%3) : pos+dl;
        const float theta = (float)rp * powf(base, -2.f*(float)p/(float)n_rot);
        float sn, cs; sincosf(theta, &sn, &cs);
        v = i < half ? buf[i]*cs-buf[i+half]*sn : buf[i-half]*sn+buf[i]*cs;
    }
    if (isq) {
        q[((size_t)t*H+head)*HD+i] = v;
        gate[((size_t)t*H+head)*HD+i] = qg[((size_t)t*H+head)*2*HD+HD+i];
        return;
    }
    if (table) cache = table[rows.slot[t]];
    const int kh = head-H, group = i/GROUP;
    const float vv = vx[((size_t)t*HKV+kh)*HD+i];
    float ak = fabsf(v), av = fabsf(vv);
    for (int s = 16; s; s >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, s));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, s));
    }
    if ((i & 31) == 0) { rk[i/32] = ak; rv[i/32] = av; }
    __syncthreads();
    const __half sk = __float2half_rn(div_rn(fmaxf(rk[2*group],rk[2*group+1]),127.f));
    const __half sv = __float2half_rn(div_rn(fmaxf(rv[2*group],rv[2*group+1]),127.f));
    const size_t cell = (size_t)kh*ctx+pos;
    cache.k[cell*HD+i] = code(v, __half2float(sk));
    cache.v[cell*HD+i] = code(vv, __half2float(sv));
    if ((i & 63) == 0) { cache.ks[cell*NG+group] = sk; cache.vs[cell*NG+group] = sv; }
}

// One block per (KV head, 256-cell chunk, query token), sharing K/V reads
// among all eight GQA query heads. FP32 softmax and accumulators. Same partial
// layout as k_attn_merge, including row-selected batch/MTP verification windows.
static __global__ void attention(const float* q, Cache cache, const Cache* table, const int32_t* pos0, int t0,
                          int H, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, Rows rows) {
    __shared__ __align__(16) float qs[GQA][HD];
    __shared__ float pr[GQA][HD], red[GQA][HD];
    const int kvh = blockIdx.x, c = blockIdx.y, t = t0+blockIdx.z, j = threadIdx.x;
    const int npos = (rows.cell ? rows.cell[t] : *pos0+t)+1, p0 = c*HD;
    if (p0 >= npos) return;
    if (table) cache = table[rows.slot[t]];
#pragma unroll
    for (int h = 0; h < GQA; ++h) qs[h][j] = q[((size_t)t*H+kvh*GQA+h)*HD+j];
    __syncthreads();
    float a[GQA] = {};
    const int p = p0+j;
    if (p < npos) {
        const size_t cell = (size_t)kvh*ctx+p;
        float sk[NG];
#pragma unroll
        for (int g = 0; g < NG; ++g) sk[g] = __half2float(cache.ks[cell*NG+g]);
        for (int i0 = 0; i0 < HD; i0 += 16) {
            const int4 pack = __ldg(reinterpret_cast<const int4*>(cache.k+cell*HD+i0));
            const int word[4] = {pack.x,pack.y,pack.z,pack.w};
            for (int f = 0; f < 4; ++f) {
                const float s = sk[(i0+f*4)/GROUP];
                float k[4];
                for (int x = 0; x < 4; ++x) k[x] = (float)(int8_t)((word[f] >> (8*x)) & 255)*s;
#pragma unroll
                for (int h = 0; h < GQA; ++h) {
                    const float4 v = *reinterpret_cast<const float4*>(&qs[h][i0+f*4]);
                    a[h] += v.x*k[0]; a[h] += v.y*k[1]; a[h] += v.z*k[2]; a[h] += v.w*k[3];
                }
            }
        }
    }
    float mx[GQA], sums[GQA];
#pragma unroll
    for (int h = 0; h < GQA; ++h) red[h][j] = p < npos ? a[h]*scale : -INFINITY;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (j < s) {
#pragma unroll
            for (int h = 0; h < GQA; ++h) red[h][j] = fmaxf(red[h][j],red[h][j+s]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < GQA; ++h) mx[h] = red[h][0];
    __syncthreads();
#pragma unroll
    for (int h = 0; h < GQA; ++h) { const float e = p < npos ? expf(a[h]*scale-mx[h]) : 0.f; pr[h][j] = e; red[h][j] = e; }
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (j < s) {
#pragma unroll
            for (int h = 0; h < GQA; ++h) red[h][j] += red[h][j+s];
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < GQA; ++h) sums[h] = red[h][0];
    float acc[GQA] = {};
    const int n = min(HD,npos-p0);
    for (int k = 0; k < n; ++k) {
        const size_t cell = (size_t)kvh*ctx+p0+k;
        const float v = (float)cache.v[cell*HD+j]*__half2float(cache.vs[cell*NG+j/GROUP]);
#pragma unroll
        for (int h = 0; h < GQA; ++h) acc[h] += pr[h][k]*v;
    }
#pragma unroll
    for (int h = 0; h < GQA; ++h) {
        const size_t o = ((size_t)(t-t0)*H+kvh*GQA+h)*n_chunk+c;
        pacc[o*HD+j] = acc[h];
        if (j == 0) { pm[o] = mx[h]; pl[o] = sums[h]; }
    }
}

// Compact FP16 staging for the UNCHANGED cuBLAS prompt attention. Both decode
// and prefill read the same quantized history; no persistent FP32 KV mirror.
static __global__ void stage_f16(Cache cache, __half* k16, __half* v16, int ctx, int P, int HKV) {
    const int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= (int64_t)HKV*P*HD) return;
    const int64_t h = i/((int64_t)P*HD), rest = i%((int64_t)P*HD);
    const size_t src = (size_t)h*ctx*HD+rest, sc = (size_t)h*ctx*NG+rest/GROUP;
    k16[i] = __float2half((float)cache.k[src]*__half2float(cache.ks[sc]));
    v16[i] = __float2half((float)cache.v[src]*__half2float(cache.vs[sc]));
}
inline cudaError_t copy_prefix(Cache dst, Cache src, size_t ctx, size_t cells, size_t heads, cudaStream_t stream) {
    for (size_t h = 0; h < heads; ++h) {
        cudaError_t e = cudaMemcpyAsync(dst.k+h*ctx*HD,src.k+h*ctx*HD,cells*HD,cudaMemcpyDeviceToDevice,stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(dst.v+h*ctx*HD,src.v+h*ctx*HD,cells*HD,cudaMemcpyDeviceToDevice,stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(dst.ks+h*ctx*NG,src.ks+h*ctx*NG,cells*NG*sizeof(__half),cudaMemcpyDeviceToDevice,stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(dst.vs+h*ctx*NG,src.vs+h*ctx*NG,cells*NG*sizeof(__half),cudaMemcpyDeviceToDevice,stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}
} // namespace q36kv8

// Focused, model-free native gate. A missing CUDA device is a skip, not a pass.
#include "strata/core/unified_kv_vmm.hpp"
#include "strata/core/unified_kv_protocol.hpp"
#include "../../src/qwen35/q36_kv_int8.cuh"
#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace strata::core;
using q36kv8::Cache;
constexpr int CTX=16384, H=16, HKV=2, HD=256, NG=4, T=257, NC=2;
static void check(cudaError_t e,const char* s,int line){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+":"+std::to_string(line)+" "+cudaGetErrorName(e)+": "+cudaGetErrorString(e));}
#define CK(e) check((e),#e,__LINE__)
template<class X> X* alloc(size_t n){X* p=nullptr;CK(cudaMalloc(&p,n*sizeof(X)));return p;}
template<class X> X* upload(const std::vector<X>& h){auto* p=alloc<X>(h.size());CK(cudaMemcpy(p,h.data(),h.size()*sizeof(X),cudaMemcpyHostToDevice));return p;}
template<class X> std::vector<X> read(const X* p,size_t n){std::vector<X> h(n);CK(cudaMemcpy(h.data(),p,n*sizeof(X),cudaMemcpyDeviceToHost));return h;}
static Cache reserve_cache(UnifiedKvVmm& a){
 auto plane=[&](uint64_t n){return a.reserve({n,1,CTX,HKV,(uint64_t)CTX*n});};
 return {(int8_t*)plane(HD),(int8_t*)plane(HD),(__half*)plane(NG*2),(__half*)plane(NG*2)};
}
struct Snapshot{std::vector<int8_t> k,v;std::vector<__half> ks,vs;};
static Snapshot snapshot(Cache c){
 Snapshot r; r.k.resize(HKV*T*HD);r.v.resize(r.k.size());r.ks.resize(HKV*T*NG);r.vs.resize(r.ks.size());
 for(int h=0;h<HKV;++h){
  CK(cudaMemcpy(r.k.data()+h*T*HD,c.k+h*CTX*HD,T*HD,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(r.v.data()+h*T*HD,c.v+h*CTX*HD,T*HD,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(r.ks.data()+h*T*NG,c.ks+h*CTX*NG,T*NG*2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(r.vs.data()+h*T*NG,c.vs+h*CTX*NG,T*NG*2,cudaMemcpyDeviceToHost));
 } return r;
}
static void equal(const Snapshot& a,const Snapshot& b){assert(a.k==b.k&&a.v==b.v);assert(std::memcmp(a.ks.data(),b.ks.data(),a.ks.size()*2)==0);assert(std::memcmp(a.vs.data(),b.vs.data(),a.vs.size()*2)==0);}
static float value(const Snapshot& c,bool key,int h,int p,int d){
 return (float)(key?c.k:c.v)[((size_t)h*T+p)*HD+d]*__half2float((key?c.ks:c.vs)[((size_t)h*T+p)*NG+d/64]);
}
static std::vector<float> normalized_rotary(const float* input,int pos){
 double sum=0;for(int d=0;d<HD;++d)sum+=(double)input[d]*input[d];
 const float inv=1.f/std::sqrt((float)(sum/HD)+1e-6f);
 std::vector<float> before(HD),out(HD);for(int d=0;d<HD;++d)out[d]=before[d]=input[d]*inv;
 for(int d=0;d<64;++d){const int p=d%32;const float theta=(float)pos*std::pow(10000.f,-2.f*(float)p/64.f);
  out[d]=d<32?before[d]*std::cos(theta)-before[d+32]*std::sin(theta):before[d-32]*std::sin(theta)+before[d]*std::cos(theta);}
 return out;
}
static void verify_written(const Snapshot& s,const std::vector<float>& kx,const std::vector<float>& vx,int input_row,int cell,int rotary){
 for(int h=0;h<HKV;++h){const auto key=normalized_rotary(kx.data()+((size_t)input_row*HKV+h)*HD,rotary);
  for(int j=0;j<HD;++j){const size_t sc=((size_t)h*T+cell)*NG+j/64;
   assert(std::abs(value(s,true,h,cell,j)-key[j])<=.51f*__half2float(s.ks[sc])+4e-5f);
   assert(std::abs(value(s,false,h,cell,j)-vx[((size_t)input_row*HKV+h)*HD+j])<=.51f*__half2float(s.vs[sc])+1e-6f);
  }
 }
}
static void near(double actual,double expected){if(std::abs(actual-expected)>5e-5*(1+std::abs(expected)))throw std::runtime_error("INT8 attention/reference mismatch: "+std::to_string(actual)+" vs "+std::to_string(expected));}
// Independent FP64 softmax/accumulation over the ACTUALLY stored quantized KV,
// not over original FP32 values. This isolates attention-reader correctness.
static void reference(const std::vector<float>& q,const Snapshot& cache,const std::vector<float>& acc,
                      const std::vector<float>& mx,const std::vector<float>& sums,int rows,
                      const std::vector<int>& cells={},const std::vector<int>& slots={},const Snapshot* other=nullptr){
 std::vector<int> samples=rows==T?std::vector<int>{0,63,255,256}:std::vector<int>{0,1,2,3};
 for(int t:samples)for(int h=0;h<H;++h){
  const auto& data=other&&!slots.empty()&&slots[t]==1?*other:cache;
  const int npos=cells.empty()?t+1:cells[t]+1;
  for(int c=0;c<(npos+255)/256;++c){
   const int start=c*256,n=std::min(256,npos-start),kvh=h/8;
   std::vector<double> scores(n);double m=-INFINITY;
   for(int p=0;p<n;++p){double dot=0;for(int d=0;d<HD;++d)dot+=(double)q[((size_t)t*H+h)*HD+d]*value(data,true,kvh,start+p,d);scores[p]=dot/16.;m=std::max(m,scores[p]);}
   double den=0;for(auto& s:scores){s=std::exp(s-m);den+=s;}
   const size_t o=((size_t)t*H+h)*NC+c;near(mx[o],m);near(sums[o],den);
   for(int d=0;d<HD;++d){double a=0;for(int p=0;p<n;++p)a+=scores[p]*value(data,false,kvh,start+p,d);near(acc[o*HD+d],a);}
  }
 }
}
int main(){
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count){std::cout<<"SKIP: native CUDA device required\n";return 77;}
 try{
 CK(cudaSetDevice(0));if(!vmm_available()){std::cout<<"SKIP: CUDA VMM required\n";return 77;}
 cudaStream_t st=nullptr;CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
 UnifiedKvVmm a(CTX),b(CTX),c(CTX),d(CTX);UnifiedKvVmm* owners[]={&a,&b,&c,&d};Cache main[4],draft[4];
 for(int s=0;s<4;++s){main[s]=reserve_cache(*owners[s]);draft[s]=reserve_cache(*owners[s]);assert(owners[s]->committed_bytes()==0);}
 const auto small=a.bytes_for(T);assert(small>0);
 UnifiedKvPool pool(CTX,4*CTX,2*small);KvIdentity ids[4];for(int s=0;s<4;++s)ids[s]=pool.add(*owners[s]);
 auto grow=[&](int s,int n){kv_require(pool.grow(ids[s],n,[]{return UINT64_MAX;}));};
 kv_require(pool.begin(ids[0],CTX));grow(0,T);kv_require(pool.begin(ids[1],CTX));grow(1,T);
 assert(a.committed_bytes()==small&&b.committed_bytes()==small);
 std::vector<float> qg(T*H*2*HD),kx(T*HKV*HD),vx(kx.size()),weights(HD,1.f);
 for(size_t i=0;i<qg.size();++i)qg[i]=std::sin((float)i*.017f);
 for(size_t i=0;i<kx.size();++i){kx[i]=std::sin((float)i*.013f);vx[i]=std::cos((float)i*.019f)*.7f;}
 std::fill(kx.begin(),kx.begin()+HD,0);std::fill(vx.begin(),vx.begin()+HD,0);
 auto* dqg=upload(qg);auto* dk=upload(kx);auto* dv=upload(vx);auto* dw=upload(weights);
 auto* q=alloc<float>(T*H*HD);auto* gate=alloc<float>(T*H*HD);auto* pos=upload(std::vector<int>{0});
 CK(cudaDeviceSynchronize()); // pageable/default-stream uploads before nonblocking-stream kernels
 for(int s=0;s<2;++s){q36kv8::prepare<<<dim3(H+HKV,T),256,0,st>>>(dqg,dk,dv,dw,dw,q,gate,main[s],nullptr,H,HKV,64,pos,CTX,10000.f,1e-6f,nullptr,nullptr,{});CK(cudaGetLastError());}
 CK(cudaStreamSynchronize(st));auto snap=snapshot(main[0]);auto qhost=read(q,T*H*HD);
 for(int t=0;t<T;++t)verify_written(snap,kx,vx,t,t,t);
 for(int t:{0,128,256})for(int h=0;h<H;++h){const auto expected=normalized_rotary(qg.data()+((size_t)t*H+h)*2*HD,t);
  for(int j=0;j<HD;++j)assert(std::abs(qhost[((size_t)t*H+h)*HD+j]-expected[j])<4e-5f);}
 // Check stored V against original values, including zero scales and groups.
 for(int h=0;h<HKV;++h)for(int t=0;t<T;++t)for(int j=0;j<HD;++j){
  const auto i=((size_t)h*T+t)*HD+j;const float scale=__half2float(snap.vs[((size_t)h*T+t)*NG+j/64]);
  assert(snap.v[i]>=-127&&snap.k[i]>=-127);
  assert(std::abs(value(snap,false,h,t,j)-vx[((size_t)t*HKV+h)*HD+j])<=.51f*scale+1e-6f);
 }
 for(int j=0;j<HD;++j)assert(snap.k[j]==0&&snap.v[j]==0);
 auto* acc=alloc<float>(T*H*NC*HD);auto* mx=alloc<float>(T*H*NC);auto* sums=alloc<float>(T*H*NC);
 q36kv8::attention<<<dim3(HKV,NC,T),256,0,st>>>(q,main[0],nullptr,pos,0,H,CTX,1.f/16,NC,acc,mx,sums,{});CK(cudaGetLastError());CK(cudaStreamSynchronize(st));
 reference(qhost,snap,read(acc,T*H*NC*HD),read(mx,T*H*NC),read(sums,T*H*NC),T);
 // Prompt staging reads the same codes/scales, retaining cuBLAS/MMQ inference.
 auto* k16=alloc<__half>(HKV*T*HD);auto* v16=alloc<__half>(HKV*T*HD);
 q36kv8::stage_f16<<<(HKV*T*HD+255)/256,256,0,st>>>(main[0],k16,v16,CTX,T,HKV);CK(cudaGetLastError());CK(cudaStreamSynchronize(st));
 auto kh=read(k16,HKV*T*HD),vh=read(v16,HKV*T*HD);
 for(int h=0;h<HKV;++h)for(int t=0;t<T;++t)for(int j=0;j<HD;++j){size_t i=((size_t)h*T+t)*HD+j;assert(__half2float(kh[i])==__half2float(__float2half(value(snap,true,h,t,j))));assert(__half2float(vh[i])==__half2float(__float2half(value(snap,false,h,t,j))));}
 // Row-selected batch/MTP verification positions must choose all FOUR planes.
 auto* table=upload(std::vector<Cache>(main,main+4));auto* cell=upload(std::vector<int>{0,64,255,256});auto* slot=upload(std::vector<int>{0,1,0,1});auto* delta=upload(std::vector<int>(4,3));
 q36kv8::Rows rows{cell,slot,delta};
 CK(cudaDeviceSynchronize());
 q36kv8::prepare<<<dim3(H+HKV,4),256,0,st>>>(dqg,dk,dv,dw,dw,q,gate,{},table,H,HKV,64,pos,CTX,10000.f,1e-6f,nullptr,nullptr,rows);CK(cudaGetLastError());
 q36kv8::attention<<<dim3(HKV,NC,4),256,0,st>>>(q,{},table,pos,0,H,CTX,1.f/16,NC,acc,mx,sums,rows);CK(cudaGetLastError());CK(cudaStreamSynchronize(st));
 const auto before_rows=snap;
 snap=snapshot(main[0]);auto snap1=snapshot(main[1]);
 const int positions[]={0,64,255,256},selected[]={0,1,0,1};
 for(int r=0;r<4;++r)verify_written(selected[r]?snap1:snap,kx,vx,r,positions[r],positions[r]+3);
 for(int s=0;s<2;++s)for(int p=0;p<T;++p)if((s==0&&p!=0&&p!=255)||(s==1&&p!=64&&p!=256)){
  const auto& after=s?snap1:snap;
  for(int h=0;h<HKV;++h){const size_t i=((size_t)h*T+p)*HD,sc=((size_t)h*T+p)*NG;
   assert(std::memcmp(after.k.data()+i,before_rows.k.data()+i,HD)==0&&std::memcmp(after.v.data()+i,before_rows.v.data()+i,HD)==0);
   assert(std::memcmp(after.ks.data()+sc,before_rows.ks.data()+sc,NG*2)==0&&std::memcmp(after.vs.data()+sc,before_rows.vs.data()+sc,NG*2)==0);}
 }
 reference(read(q,4*H*HD),snap,read(acc,4*H*NC*HD),read(mx,4*H*NC),read(sums,4*H*NC),4,{0,64,255,256},{0,1,0,1},&snap1);
 // Session cloning copies main and draft codes AND scales, not head gaps.
 CK(q36kv8::copy_prefix(draft[0],main[0],CTX,T,HKV,st));
 CK(q36kv8::copy_prefix(main[1],main[0],CTX,T,HKV,st));
 CK(q36kv8::copy_prefix(draft[1],draft[0],CTX,T,HKV,st));CK(cudaStreamSynchronize(st));
 equal(snap,snapshot(main[1]));equal(snap,snapshot(draft[1]));assert(a.committed_bytes()==small&&b.committed_bytes()==small);
 kv_require(pool.end(ids[1],T));kv_require(pool.end(ids[0],T));
 // The captured INT8 reader keeps its VA across growth, eviction and replay.
 const int final_pos=256;CK(cudaMemcpyAsync(pos,&final_pos,4,cudaMemcpyHostToDevice,st));CK(cudaStreamSynchronize(st));
 cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
 CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
 q36kv8::attention<<<dim3(HKV,NC,1),256,0,st>>>(q,main[0],nullptr,pos,0,H,CTX,1.f/16,NC,acc,mx,sums,{});CK(cudaGetLastError());
 CK(cudaStreamEndCapture(st,&graph));CK(cudaGraphInstantiate(&exec,graph,0));CK(cudaGraphDestroy(graph));
 kv_require(pool.begin(ids[0],CTX));grow(0,8193);equal(snap,snapshot(main[0]));
 assert(a.committed_bytes()==a.bytes_for(8193)&&a.committed_bytes()>small);
 assert(pool.session(1).state==KvResidency::evicted);assert(pool.begin(ids[1],CTX).code==KvCode::cache_miss);
 CK(cudaGraphLaunch(exec,st));CK(cudaStreamSynchronize(st));kv_require(pool.end(ids[0],T));
 kv_require(pool.begin(ids[3],CTX));grow(3,8193);kv_require(pool.end(ids[3],T));assert(pool.session(0).state==KvResidency::evicted);
 kv_require(pool.begin(ids[0],CTX,true));grow(0,T);CK(cudaGraphLaunch(exec,st));CK(cudaStreamSynchronize(st));
 const auto zeros=read(acc,H*NC*HD);for(float x:zeros)assert(x==0.f);
 kv_require(pool.end(ids[0],T));CK(cudaGraphExecDestroy(exec));CK(cudaStreamDestroy(st));
 for(void* p:{(void*)dqg,(void*)dk,(void*)dv,(void*)dw,(void*)q,(void*)gate,(void*)pos,(void*)acc,(void*)mx,(void*)sums,(void*)k16,(void*)v16,(void*)table,(void*)cell,(void*)slot,(void*)delta})CK(cudaFree(p));
 std::cout<<"Q36 INT8: quantization, FP64 stored-KV reference, prefill staging, batch rows, main/draft copy, VMM accounting, LRU and graph replay passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

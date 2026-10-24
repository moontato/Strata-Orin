#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstring>
#include <map>
#include <vector>
#include <sys/mman.h>
namespace fake_cuda {
int fail_create=0, fail_map=0, fail_access=0, fail_unmap=0, fail_release=0, fail_zero=0;
size_t creates=0, releases=0, live_handles=0, maps=0, unmaps=0;
}
namespace {
constexpr size_t gran = 2u << 20;
struct Page { std::vector<unsigned char> bytes; unsigned refs=0; };
std::map<CUmemGenericAllocationHandle, Page> pages;
std::map<CUdeviceptr, CUmemGenericAllocationHandle> mapped;
CUmemGenericAllocationHandle sequence=0;
bool fails(int& n) { return n<0 || (n>0 && --n==0); }
void* ptr(CUdeviceptr p) { return reinterpret_cast<void*>(p); }
}
CUresult cuDeviceGetAttribute(int* n,int,CUdevice) { *n=1; return 0; }
CUresult cuMemGetAllocationGranularity(size_t* n,const CUmemAllocationProp*,int) { *n=gran; return 0; }
CUresult cuMemAddressReserve(CUdeviceptr* p,size_t n,size_t,CUdeviceptr,unsigned long long) {
    auto* a=mmap(nullptr,n,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(a==MAP_FAILED) return 1;
    *p=reinterpret_cast<CUdeviceptr>(a); return 0;
}
CUresult cuMemAddressFree(CUdeviceptr p,size_t n) { return munmap(ptr(p),n)==0 ? 0:1; }
CUresult cuMemCreate(CUmemGenericAllocationHandle* h,size_t,const CUmemAllocationProp*,unsigned long long) {
    if(fails(fake_cuda::fail_create)) return 1;
    *h=++sequence; pages.emplace(*h,Page{}); ++fake_cuda::creates; ++fake_cuda::live_handles; return 0;
}
CUresult cuMemRelease(CUmemGenericAllocationHandle h) {
    assert(pages.count(h) && !pages.at(h).refs); // double release / mapped release is a test failure
    if(fails(fake_cuda::fail_release)) return 1;
    pages.erase(h); ++fake_cuda::releases; --fake_cuda::live_handles; return 0;
}
CUresult cuMemMap(CUdeviceptr p,size_t n,size_t,CUmemGenericAllocationHandle h,unsigned long long) {
    if(fails(fake_cuda::fail_map)) return 1;
    assert(!mapped.count(p) && pages.count(h));
    auto& page=pages.at(h); assert(!page.refs); page.bytes.resize(n,0xAB);
    if(mprotect(ptr(p),n,PROT_READ|PROT_WRITE)!=0) return 1;
    std::memcpy(ptr(p),page.bytes.data(),n); mapped.emplace(p,h); ++page.refs; ++fake_cuda::maps; return 0;
}
CUresult cuMemUnmap(CUdeviceptr p,size_t n) {
    if(fails(fake_cuda::fail_unmap)) return 1;
    assert(mapped.count(p)); auto& page=pages.at(mapped.at(p));
    std::memcpy(page.bytes.data(),ptr(p),n); --page.refs; mapped.erase(p);
    assert(mprotect(ptr(p),n,PROT_NONE)==0); ++fake_cuda::unmaps; return 0;
}
CUresult cuMemSetAccess(CUdeviceptr,size_t,const CUmemAccessDesc*,size_t) { return fails(fake_cuda::fail_access) ? 1:0; }
cudaError_t cudaGetDevice(int* n) { *n=0; return 0; }
cudaError_t cudaGetDeviceCount(int* n) { *n=1; return 0; }
cudaError_t cudaSetDevice(int) { return 0; }
cudaError_t cudaDeviceGetAttribute(int* n,cudaDeviceAttr,int) { *n=1; return 0; }
cudaError_t cudaGetDeviceFlags(unsigned* n) { *n=0; return 0; }
cudaError_t cudaGetLastError() { return 0; }
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
const char* cudaGetErrorName(cudaError_t) { return "fake_cuda"; }
cudaError_t cudaDeviceSynchronize() { return 0; }
cudaError_t cudaMemset(void* p,int value,size_t n) { if(fails(fake_cuda::fail_zero)) return 1; std::memset(p,value,n); return 0; }
cudaError_t cudaMemcpy(void* p,const void* q,size_t n,int) { std::memcpy(p,q,n); return 0; }
cudaError_t cudaHostRegister(void*,size_t,unsigned) { return 1; }
cudaError_t cudaHostUnregister(void*) { return 1; }
cudaError_t cudaHostGetDevicePointer(void**,void*,unsigned) { return 1; }
cudaError_t cudaGetDriverEntryPointByVersion(const char* name,void** out,unsigned,unsigned long long,cudaDriverEntryPointQueryResult* q) {
#define ENTRY(f) if(std::strcmp(name,#f)==0) { *out=reinterpret_cast<void*>(&f); *q=0; return 0; }
    ENTRY(cuDeviceGetAttribute) ENTRY(cuMemGetAllocationGranularity) ENTRY(cuMemAddressReserve)
    ENTRY(cuMemAddressFree) ENTRY(cuMemCreate) ENTRY(cuMemRelease) ENTRY(cuMemMap) ENTRY(cuMemUnmap) ENTRY(cuMemSetAccess)
#undef ENTRY
    return 1;
}

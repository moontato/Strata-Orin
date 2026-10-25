#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstring>
#include <cstdint>
#include <map>
#include <vector>
#include <sys/mman.h>
namespace fake_cuda {
int fail_create=0, fail_map=0, fail_access=0, fail_unmap=0, fail_release=0, fail_zero=0, fail_sync=0;
size_t creates=0, releases=0, live_handles=0, maps=0, unmaps=0, memsets=0;
int host_pointer_attribute=0, fail_host_attribute=0, fail_host_register=0, fail_host_alias=0, fail_host_unregister=0;
bool host_alias_null=false;
std::vector<bool> host_alias_identity;
size_t host_registers=0, host_unregisters=0, host_alias_queries=0, live_host_registrations=0;
}
namespace {
constexpr size_t gran = 2u << 20;
struct Page { std::vector<unsigned char> bytes; unsigned refs=0; };
std::map<CUmemGenericAllocationHandle, Page> pages;
std::map<CUdeviceptr, CUmemGenericAllocationHandle> mapped;
std::map<void*, size_t> host_registered;
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
cudaError_t cudaDeviceGetAttribute(int* n,cudaDeviceAttr attr,int) {
    if(attr==cudaDevAttrCanUseHostPointerForRegisteredMem) {
        if(fails(fake_cuda::fail_host_attribute)) return 1;
        *n=fake_cuda::host_pointer_attribute;
    } else *n=1;
    return 0;
}
cudaError_t cudaGetDeviceFlags(unsigned* n) { *n=0; return 0; }
cudaError_t cudaGetLastError() { return 0; }
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
const char* cudaGetErrorName(cudaError_t) { return "fake_cuda"; }
cudaError_t cudaDeviceSynchronize() { return fails(fake_cuda::fail_sync) ? 1:0; }
cudaError_t cudaMemset(void* p,int value,size_t n) { ++fake_cuda::memsets; if(fails(fake_cuda::fail_zero)) return 1; std::memset(p,value,n); return 0; }
cudaError_t cudaMemcpy(void* p,const void* q,size_t n,int) { std::memcpy(p,q,n); return 0; }
cudaError_t cudaHostRegister(void* p,size_t n,unsigned) {
    ++fake_cuda::host_registers;
    if(fails(fake_cuda::fail_host_register)) return 1;
    assert(!host_registered.count(p)); host_registered.emplace(p,n); ++fake_cuda::live_host_registrations; return 0;
}
cudaError_t cudaHostUnregister(void* p) {
    ++fake_cuda::host_unregisters;
    if(fails(fake_cuda::fail_host_unregister)) return 1;
    assert(host_registered.count(p)); host_registered.erase(p); --fake_cuda::live_host_registrations; return 0;
}
cudaError_t cudaHostGetDevicePointer(void** alias,void* p,unsigned) {
    const auto query=fake_cuda::host_alias_queries++;
    if(fails(fake_cuda::fail_host_alias)) return 1;
    assert(host_registered.count(p));
    const bool identity=query>=fake_cuda::host_alias_identity.size() || fake_cuda::host_alias_identity[query];
    *alias=fake_cuda::host_alias_null ? nullptr : identity ? p :
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p)^uintptr_t{0x100000000ULL});
    return 0;
}
cudaError_t cudaGetDriverEntryPointByVersion(const char* name,void** out,unsigned,unsigned long long,cudaDriverEntryPointQueryResult* q) {
#define ENTRY(f) if(std::strcmp(name,#f)==0) { *out=reinterpret_cast<void*>(&f); *q=0; return 0; }
    ENTRY(cuDeviceGetAttribute) ENTRY(cuMemGetAllocationGranularity) ENTRY(cuMemAddressReserve)
    ENTRY(cuMemAddressFree) ENTRY(cuMemCreate) ENTRY(cuMemRelease) ENTRY(cuMemMap) ENTRY(cuMemUnmap) ENTRY(cuMemSetAccess)
#undef ENTRY
    return 1;
}

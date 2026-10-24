#pragma once
#include <cstddef>
#define CUDART_VERSION 13020
using cudaError_t = int;
using cudaDeviceAttr = int;
constexpr int cudaSuccess = 0, cudaDriverEntryPointSuccess = 0, cudaEnableDefault = 0;
using cudaDriverEntryPointQueryResult = int;
constexpr int cudaDevAttrIntegrated = 1, cudaDevAttrUnifiedAddressing = 2, cudaDevAttrCanMapHostMemory = 3;
constexpr int cudaDevAttrHostRegisterSupported = 4, cudaDevAttrCanUseHostPointerForRegisteredMem = 5;
constexpr int cudaDevAttrPageableMemoryAccess = 6, cudaDevAttrPageableMemoryAccessUsesHostPageTables = 7;
constexpr unsigned cudaHostRegisterMapped = 1, cudaHostRegisterPortable = 2;
constexpr int cudaMemcpyDefault = 0;
cudaError_t cudaGetDevice(int*);
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t cudaGetDeviceFlags(unsigned*);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
const char* cudaGetErrorName(cudaError_t);
cudaError_t cudaGetDriverEntryPointByVersion(const char*, void**, unsigned, unsigned long long, cudaDriverEntryPointQueryResult*);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaMemcpy(void*, const void*, size_t, int);
cudaError_t cudaHostRegister(void*, size_t, unsigned);
cudaError_t cudaHostUnregister(void*);
cudaError_t cudaHostGetDevicePointer(void**, void*, unsigned);
namespace fake_cuda {
extern int fail_create, fail_map, fail_access, fail_unmap, fail_release, fail_zero;
extern size_t creates, releases, live_handles, maps, unmaps;
}

#pragma once
// Test-only driver ABI shim; src/core/vmm.cpp itself is compiled unchanged.
#include <cstddef>
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;
using CUresult = int;
constexpr int CUDA_SUCCESS = 0, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 1;
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED = 1, CU_MEM_LOCATION_TYPE_DEVICE = 1;
constexpr int CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0, CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3;
struct CUmemLocation { int type = 0, id = 0; };
struct CUmemAllocationProp { int type = 0; CUmemLocation location; };
struct CUmemAccessDesc { CUmemLocation location; int flags = 0; };
CUresult cuDeviceGetAttribute(int*, int, CUdevice);
CUresult cuMemGetAllocationGranularity(size_t*, const CUmemAllocationProp*, int);
CUresult cuMemAddressReserve(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
CUresult cuMemAddressFree(CUdeviceptr, size_t);
CUresult cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
CUresult cuMemRelease(CUmemGenericAllocationHandle);
CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
CUresult cuMemUnmap(CUdeviceptr, size_t);
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);

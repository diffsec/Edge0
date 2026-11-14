#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA headers.
#if defined(_WIN32)
// hip_runtime.h pulls in <windows.h> on Windows: keep its min/max macros and the rarely used APIs out of the engine.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
// Do not let HIP's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#if HIP_VERSION_MAJOR < 7
#define cudaFreeHost hipHostFree
#else
#define cudaFreeHost hipFreeHost
#endif
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraph_t hipGraph_t
#define cudaHostAlloc hipHostMalloc
#if HIP_VERSION_MAJOR < 7
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocPortable hipHostMallocPortable
#else
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#endif
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(...) (::strata::hip_compat::malloc_host(__VA_ARGS__))
#if defined(_WIN32)
// Windows: hipMemGetInfo counts this process's own allocations only, so the desktop's and other programs' share of
// the card is invisible to it, and VRAM sized by it overfills the card (WDDM then moves memory out to system RAM and
// the GPU slows down).  Its free figure is lowered by what Windows' video memory budget for this process withholds
// (src/core/device.cu; STRATA_WDDM_BUDGET=0: hipMemGetInfo as it is).
namespace strata::hip_compat {
hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes);
}
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info
#else
#define cudaMemGetInfo hipMemGetInfo
#endif
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

namespace strata::hip_compat {
template <typename T>
inline hipError_t malloc_host(T** pointer, size_t bytes) {
    return hipHostMalloc(reinterpret_cast<void**>(pointer), bytes, hipHostMallocDefault);
}
}  // namespace strata::hip_compat

// ---- copies and fills on the buffer's own device.
// CUDA takes a memcpy or a memset of any device's buffer whatever the current device is, and the engine's layer
// split relies on that.  HIP runs such a call on the CURRENT device's queue.  A blocking hipMemset of another
// device's buffer then returns before that device sees the bytes (measured on two gfx1100 cards, ROCm 7.2.4: a
// kernel launched on the buffer's device right after the call still read the old bytes), so the fill lands late,
// over whatever that device wrote in between.  With more than one device, these wrappers make the buffer's device
// current for the call.  A call on a stream of its own is left alone: the stream names the device.
// STRATA_HIP_XDEV_LOG=1 prints the first calls that needed the switch.
namespace strata::hip_compat {
inline bool multi_device() {
    static const bool multi = [] {
        int n = 0;
        return hipGetDeviceCount(&n) == hipSuccess && n > 1;
    }();
    return multi;
}
// the device that owns `p`, or -1 for host memory and for a pointer HIP does not know
inline int device_of(const void* p) {
    hipPointerAttribute_t a{};
    if (p == nullptr) return -1;
    if (hipPointerGetAttributes(&a, p) != hipSuccess) {
        (void) hipGetLastError();   // a plain host pointer: not an error of the caller's
        return -1;
    }
    return a.type == hipMemoryTypeDevice ? a.device : -1;
}
struct DeviceScope {
    int previous = -1;
    DeviceScope(int device, const char* what, const char* file, int line, size_t bytes) {
        int cur = 0;
        if (device < 0 || hipGetDevice(&cur) != hipSuccess || cur == device || hipSetDevice(device) != hipSuccess) return;
        previous = cur;
        static const bool log = [] {
            const char* v = std::getenv("STRATA_HIP_XDEV_LOG");
            return v != nullptr && v[0] == '1';
        }();
        static std::atomic<int> shown{0};
        if (log && shown.fetch_add(1) < 200)
            std::fprintf(stderr, "strata hip: %s of %zu bytes on device %d while device %d was current (%s:%d)\n", what,
                         bytes, device, cur, file, line);
    }
    ~DeviceScope() {
        if (previous >= 0) (void) hipSetDevice(previous);
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
};
// the device side of a copy: the source of a device-to-host copy, else the destination
inline int copy_device(const void* dst, const void* src, hipMemcpyKind kind) {
    if (kind == hipMemcpyDeviceToHost) return device_of(src);
    const int d = device_of(dst);
    return d >= 0 ? d : device_of(src);
}
inline hipError_t memcpy_on(const char* file, int line, void* dst, const void* src, size_t n, hipMemcpyKind kind) {
    if (!multi_device()) return hipMemcpy(dst, src, n, kind);
    const DeviceScope scope(copy_device(dst, src, kind), "memcpy", file, line, n);
    return hipMemcpy(dst, src, n, kind);
}
inline hipError_t memcpy_async_on(const char* file, int line, void* dst, const void* src, size_t n, hipMemcpyKind kind,
                                  hipStream_t stream = nullptr) {
    if (stream != nullptr || !multi_device()) return hipMemcpyAsync(dst, src, n, kind, stream);
    const DeviceScope scope(copy_device(dst, src, kind), "memcpy on the null stream", file, line, n);
    return hipMemcpyAsync(dst, src, n, kind, stream);
}
inline hipError_t memset_on(const char* file, int line, void* dst, int value, size_t n) {
    if (!multi_device()) return hipMemset(dst, value, n);
    const DeviceScope scope(device_of(dst), "memset", file, line, n);
    return hipMemset(dst, value, n);
}
inline hipError_t memset_async_on(const char* file, int line, void* dst, int value, size_t n, hipStream_t stream = nullptr) {
    if (stream != nullptr || !multi_device()) return hipMemsetAsync(dst, value, n, stream);
    const DeviceScope scope(device_of(dst), "memset on the null stream", file, line, n);
    return hipMemsetAsync(dst, value, n, stream);
}
}  // namespace strata::hip_compat
#define cudaMemcpy(...) strata::hip_compat::memcpy_on(__FILE__, __LINE__, __VA_ARGS__)
#define cudaMemcpyAsync(...) strata::hip_compat::memcpy_async_on(__FILE__, __LINE__, __VA_ARGS__)
#define cudaMemset(...) strata::hip_compat::memset_on(__FILE__, __LINE__, __VA_ARGS__)
#define cudaMemsetAsync(...) strata::hip_compat::memset_async_on(__FILE__, __LINE__, __VA_ARGS__)

template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                     hipGraphNode_t* error, char* log, size_t size) {
    return hipGraphInstantiate(exec, graph, error, log, size);
}
#define __trap() __builtin_trap()   // the compiled-out sm_80 paths (never selected on AMD)
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, ...) hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), __VA_ARGS__)

#include "intrinsics.hpp"

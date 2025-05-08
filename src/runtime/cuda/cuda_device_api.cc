/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <dgl/runtime/device_api.h>

#include <dmlc/thread_local.h>
#include <dgl/runtime/registry.h>
#include <cuda_runtime.h>
#include "cuda_common.h"

// Added includes
#include <map>
#include <mutex>
#include <vector>

namespace dgl {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
private: // Added private members
  std::map<int, std::map<size_t, std::vector<void *>>> free_lists_by_device_;
  std::map<int, std::map<void *, size_t>>
      allocated_block_sizes_by_device_; // New: Tracks active allocations' sizes
  std::mutex free_list_mutex_;

public:
  // Destructor to clean up pooled memory
  ~CUDADeviceAPI() {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    // Free blocks in the free lists
    for (auto const &[device_id, size_map] : free_lists_by_device_) {
      CUDA_CALL(cudaSetDevice(device_id));
      for (auto const &[size, ptr_vector] : size_map) {
        for (void *ptr : ptr_vector) {
          CUDA_CALL(cudaFree(ptr));
        }
      }
    }
    free_lists_by_device_.clear();

    // Free any outstanding allocated blocks not returned to the pool
    for (auto const &[device_id, ptr_size_map] :
         allocated_block_sizes_by_device_) {
      CUDA_CALL(cudaSetDevice(device_id));
      for (auto const &[ptr, size] : ptr_size_map) {
        LOG(WARNING)
            << "CUDADeviceAPI destructor: Cleaning up unfreed block of size "
            << size << " at address " << ptr << " on device " << device_id;
        CUDA_CALL(cudaFree(ptr));
      }
    }
    allocated_block_sizes_by_device_.clear();
  }

  void SetDevice(DGLContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
  }
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value = (
            cudaDeviceGetAttribute(
                &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id)
            == cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxSharedMemoryPerBlock, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        cudaDeviceProp props;
        CUDA_CALL(cudaGetDeviceProperties(&props, ctx.device_id));
        *rv = std::string(props.name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMultiProcessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[0], cudaDevAttrMaxBlockDimX, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[1], cudaDevAttrMaxBlockDimY, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[2], cudaDevAttrMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] <<", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
    }
    *rv = value;
  }
  void *AllocDataSpace(DGLContext ctx,
                       size_t nbytes, // User requested size
                       size_t alignment, DGLType type_hint) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    // CUDA memory is generally aligned to at least 256 bytes.
    // This check ensures the requested alignment is compatible.
    CHECK_EQ(256 % alignment, 0U)
        << "CUDA space is aligned at 256 bytes. Requested alignment="
        << alignment << " is not compatible.";

    void *ptr_to_return = nullptr;

    { // Scope for lock
      std::lock_guard<std::mutex> lock(free_list_mutex_);
      auto &device_free_lists = free_lists_by_device_[ctx.device_id];
      auto it_size_list = device_free_lists.find(nbytes);

      if (it_size_list != device_free_lists.end() &&
          !it_size_list->second.empty()) {
        // Found a suitable block in the free list
        std::vector<void *> &size_list = it_size_list->second;
        ptr_to_return = size_list.back();
        size_list.pop_back();
        if (size_list.empty()) {
          device_free_lists.erase(it_size_list);
        }
        // Track this allocation
        allocated_block_sizes_by_device_[ctx.device_id][ptr_to_return] = nbytes;
      } else {
        // Not found in free list, allocate new
        CUDA_CALL(cudaMalloc(&ptr_to_return, nbytes));
        if (ptr_to_return == nullptr) {
          // CUDA_CALL should handle errors, but an explicit check is safer.
          LOG(FATAL) << "cudaMalloc failed to allocate " << nbytes
                     << " bytes on device " << ctx.device_id;
          // This LOG(FATAL) will typically terminate. If not, an exception or
          // error return is needed.
        }
        // Track this new allocation
        allocated_block_sizes_by_device_[ctx.device_id][ptr_to_return] = nbytes;
      }
    } // Lock released
    return ptr_to_return;
  }

  void FreeDataSpace(DGLContext ctx, void *user_ptr) final {
    if (user_ptr == nullptr) {
      return;
    }

    void *raw_ptr = user_ptr; // user_ptr is the raw pointer, no header offset
    size_t nbytes = 0;

    { // Scope for lock
      std::lock_guard<std::mutex> lock(free_list_mutex_);
      auto &device_allocations =
          allocated_block_sizes_by_device_[ctx.device_id];
      auto it_alloc = device_allocations.find(raw_ptr);

      if (it_alloc == device_allocations.end()) {
        LOG(WARNING)
            << "Attempting to free an untracked or already freed pointer: "
            << raw_ptr << " on device " << ctx.device_id
            << ". This might indicate a double free or freeing an invalid "
               "pointer.";
        // Depending on desired strictness, could be LOG(FATAL) or an error
        // throw. For now, log a warning and return to avoid crashing if it's a
        // non-critical error.
        return;
      }
      nbytes = it_alloc->second;
      device_allocations.erase(it_alloc); // Remove from active allocations

      // Add raw_ptr to the free list for reuse
      free_lists_by_device_[ctx.device_id][nbytes].push_back(raw_ptr);
    } // Lock released
    // cudaFree is not called here; the block is kept in the pool.
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      DGLContext ctx_from,
                      DGLContext ctx_to,
                      DGLType type_hint,
                      DGLStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLGPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, ctx_to.device_id,
                            from, ctx_from.device_id,
                            size, cu_stream);
      }
    } else if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLCPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLGPU) {
      CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  DGLStreamHandle CreateStream(DGLContext ctx) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t retval;
    // make sure the legacy default stream won't block on this stream
    CUDA_CALL(cudaStreamCreateWithFlags(&retval, cudaStreamNonBlocking));
    return static_cast<DGLStreamHandle>(retval);
  }

  void FreeStream(DGLContext ctx, DGLStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(DGLContext ctx, DGLStreamHandle event_src, DGLStreamHandle event_dst) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    CUDA_CALL(cudaEventCreate(&evt));
    CUDA_CALL(cudaEventRecord(evt, src_stream));
    CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(cudaEventDestroy(evt));
  }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void SetStream(DGLContext ctx, DGLStreamHandle stream) final {
    CUDAThreadEntry::ThreadLocal()
        ->stream = static_cast<cudaStream_t>(stream);
  }

  void* AllocWorkspace(DGLContext ctx, size_t size, DGLType type_hint) final {
    return CUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(DGLContext ctx, void* data) final {
    CUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<CUDADeviceAPI>& Global() {
    static std::shared_ptr<CUDADeviceAPI> inst =
        std::make_shared<CUDADeviceAPI>();
    return inst;
  }

 private:
  static void GPUCopy(const void* from,
                      void* to,
                      size_t size,
                      cudaMemcpyKind kind,
                      cudaStream_t stream) {
    CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    if (stream == 0 && kind == cudaMemcpyDeviceToHost) {
      // only wait for the copy, when it's on the default stream, and it's to host memory
      CUDA_CALL(cudaStreamSynchronize(stream));
    }
  }
};

typedef dmlc::ThreadLocalStore<CUDAThreadEntry> CUDAThreadStore;

CUDAThreadEntry::CUDAThreadEntry()
    : pool(kDLGPU, CUDADeviceAPI::Global()) {
}

CUDAThreadEntry* CUDAThreadEntry::ThreadLocal() {
  return CUDAThreadStore::Get();
}

DGL_REGISTER_GLOBAL("device_api.gpu")
.set_body([](DGLArgs args, DGLRetValue* rv) {
    DeviceAPI* ptr = CUDADeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace runtime
}  // namespace dgl

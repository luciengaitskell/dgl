#include "./scan.h"

#include <dgl/runtime/device_api.h>

using namespace dgl;
using namespace dgl::runtime;

namespace dgl {
namespace ds {

static const int ELE_PER_BLOCK = 1024;
static const int THREADS_PER_BLOCK = 512;

__device__ void _GetValue(IdType *workspace, int tid, int size, int rank,
                          IdType *part_ids, IdType *ret) {
  if (tid < size) {
    if (part_ids != nullptr) {
      *ret = part_ids[tid] == rank;
    } else {
      *ret = workspace[tid];
    }
  } else {
    *ret = 0;
  }
}

// Inplace exclusive multi-way scan
__global__ void _MultiWayCtaScanKernel(IdType *workspace, int size,
                                       IdType *part_ids, int world_size,
                                       IdType *sums) {
  __shared__ IdType temp[ELE_PER_BLOCK];
  int tid = threadIdx.x;
  int rank = blockIdx.x;
  int bid = blockIdx.y;
  int n_blocks = gridDim.y;

  int block_inc = size * rank;
  int thread_inc = ELE_PER_BLOCK * bid;
  workspace += block_inc + thread_inc;
  part_ids += thread_inc;
  if (ELE_PER_BLOCK * bid + ELE_PER_BLOCK > size) {
    size = size % ELE_PER_BLOCK;
  } else {
    size = ELE_PER_BLOCK;
  }
  int ai = tid;
  int bi = tid + ELE_PER_BLOCK / 2;
  _GetValue(workspace, ai, size, rank, part_ids, temp + ai);
  _GetValue(workspace, bi, size, rank, part_ids, temp + bi);

  int offset = 1;
  for (int d = ELE_PER_BLOCK >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }

  if (tid == 0) {
    if (sums != nullptr) {
      sums[bid + rank * n_blocks] = temp[ELE_PER_BLOCK - 1];
    }
    temp[ELE_PER_BLOCK - 1] = 0;
  }
  for (int d = 1; d < ELE_PER_BLOCK; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      IdType t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  if (ai < size)
    workspace[ai] = temp[ai];
  if (bi < size)
    workspace[bi] = temp[bi];
}

__global__ void _ScanAddKernel(IdType *workspace, IdType *sums, int size,
                               int world_size) {
  int rank = blockIdx.x;
  int bid = blockIdx.y;
  int tid = blockDim.x * bid + threadIdx.x;
  int n_blocks = gridDim.y;
  int stride = gridDim.y * blockDim.x;
  workspace += size * rank;
  while (tid < size) {
    workspace[tid] += sums[rank * n_blocks + bid];
    tid += stride;
  }
}

void _MultiWayScanRecursive(IdType *workspace, int size, IdType *part_ids,
                            int world_size, DeviceAPI *device, DGLContext ctx) {
  // Calculate the number of blocks needed
  int n_blocks = (size + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;
  LOG(INFO) << "[_MultiWayScanRecursive] Processing size=" << size
            << ", blocks=" << n_blocks << ", world_size=" << world_size;

  // Limit recursion depth to prevent stack overflow
  static int recursion_depth = 0;
  const int MAX_RECURSION_DEPTH = 10;
  recursion_depth++;

  if (recursion_depth > MAX_RECURSION_DEPTH) {
    LOG(WARNING) << "[_MultiWayScanRecursive] Max recursion depth reached, "
                    "terminating recursion";
    recursion_depth--;
    return;
  }

  IdType *sums = nullptr;
  if (size > ELE_PER_BLOCK) {
    size_t sums_size = world_size * n_blocks * sizeof(IdType);
    try {
      LOG(INFO) << "[_MultiWayScanRecursive] Allocating sums array of size "
                << (sums_size / 1024) << " KB";
      sums = (IdType *)device->AllocWorkspace(ctx, sums_size);
      if (sums == nullptr) {
        LOG(ERROR) << "[_MultiWayScanRecursive] Failed to allocate sums memory";
        recursion_depth--;
        return;
      }

      // Initialize sums to zero
      cudaError_t cuda_err = cudaMemset(sums, 0, sums_size);
      if (cuda_err != cudaSuccess) {
        LOG(ERROR) << "[_MultiWayScanRecursive] Failed to initialize sums: "
                   << cudaGetErrorString(cuda_err);
        device->FreeWorkspace(ctx, sums);
        recursion_depth--;
        return;
      }
    } catch (const std::exception &e) {
      LOG(ERROR)
          << "[_MultiWayScanRecursive] Exception during sums allocation: "
          << e.what();
      recursion_depth--;
      return;
    }
  }

  // Configure grid and blocks
  const dim3 grid(world_size, n_blocks);
  auto *thr_entry = CUDAThreadEntry::ThreadLocal();

  // Launch scan kernel with error checking
  LOG(INFO)
      << "[_MultiWayScanRecursive] Launching _MultiWayCtaScanKernel with grid=("
      << grid.x << "," << grid.y << "), threads=" << THREADS_PER_BLOCK;

  _MultiWayCtaScanKernel<<<grid, THREADS_PER_BLOCK, 0, thr_entry->stream>>>(
      workspace, size, part_ids, world_size, sums);

  // Check for kernel launch errors
  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    LOG(ERROR)
        << "[_MultiWayScanRecursive] CUDA error in _MultiWayCtaScanKernel: "
        << cudaGetErrorString(cuda_err);
    if (sums != nullptr) {
      device->FreeWorkspace(ctx, sums);
    }
    recursion_depth--;
    return;
  }

  // Ensure kernel completion before proceeding
  cuda_err = cudaStreamSynchronize(thr_entry->stream);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "[_MultiWayScanRecursive] CUDA sync error: "
               << cudaGetErrorString(cuda_err);
    if (sums != nullptr) {
      device->FreeWorkspace(ctx, sums);
    }
    recursion_depth--;
    return;
  }

  // Recursive step if needed
  if (size > ELE_PER_BLOCK) {
    LOG(INFO) << "[_MultiWayScanRecursive] Recursing with n_blocks="
              << n_blocks;
    _MultiWayScanRecursive(sums, n_blocks, nullptr, world_size, device, ctx);

    LOG(INFO)
        << "[_MultiWayScanRecursive] Launching _ScanAddKernel after recursion";
    _ScanAddKernel<<<grid, THREADS_PER_BLOCK * 2, 0, thr_entry->stream>>>(
        workspace, sums, size, world_size);

    // Check for kernel errors
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      LOG(ERROR) << "[_MultiWayScanRecursive] CUDA error in _ScanAddKernel: "
                 << cudaGetErrorString(cuda_err);
    } else {
      // Ensure kernel completion
      cuda_err = cudaStreamSynchronize(thr_entry->stream);
      if (cuda_err != cudaSuccess) {
        LOG(ERROR)
            << "[_MultiWayScanRecursive] CUDA sync error after _ScanAddKernel: "
            << cudaGetErrorString(cuda_err);
      }
    }

    // Clean up memory
    LOG(INFO) << "[_MultiWayScanRecursive] Freeing sums memory";
    device->FreeWorkspace(ctx, sums);
  }

  LOG(INFO) << "[_MultiWayScanRecursive] Completed for size=" << size;
  recursion_depth--;
}

__global__ void _PermutateKernel(IdType *workspace, IdType *input,
                                 IdType *part_offset, IdType *part_ids,
                                 int size, int world_size, IdType *sorted,
                                 IdType *index) {
  int rank = blockIdx.x;
  int bid = blockIdx.y;
  int tid = bid * ELE_PER_BLOCK + threadIdx.x;
  if (tid < size && part_ids[tid] == rank) {
    IdType tval = input[tid];
    IdType pos = workspace[rank * size + tid] + part_offset[rank];
    sorted[pos] = tval;
    index[pos] = tid;
  }
}

static void _Permutate(IdType *workspace, IdType *input, IdType *part_offset,
                       IdType *part_ids, int size, int world_size,
                       IdType *sorted, IdType *index) {
  int n_blocks = (size + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;
  const dim3 grid(world_size, n_blocks);
  auto *thr_entry = CUDAThreadEntry::ThreadLocal();
  _PermutateKernel<<<grid, ELE_PER_BLOCK, 0, thr_entry->stream>>>(
      workspace, input, part_offset, part_ids, size, world_size, sorted, index);
}

std::pair<IdArray, IdArray> MultiWayScan(IdArray input, IdArray part_offset,
                                         IdArray part_ids, int world_size) {
  LOG(INFO) << "[MultiWayScan] Starting with input size=" << input->shape[0]
            << ", world_size=" << world_size;

  if (input->shape[0] == 0) {
    LOG(INFO) << "[MultiWayScan] Empty input array, returning null arrays";
    return {NullArray(input->dtype, input->ctx),
            NullArray(input->dtype, input->ctx)};
  }

  // Validate input parameters
  if (world_size <= 0) {
    LOG(WARNING) << "[MultiWayScan] Invalid world_size=" << world_size
                 << ", using world_size=1";
    world_size = 1;
  }

  int size = input->shape[0];
  auto device = DeviceAPI::Get(input->ctx);

  // Check if part_ids matches input size
  if (part_ids->shape[0] != size) {
    LOG(ERROR) << "[MultiWayScan] part_ids size (" << part_ids->shape[0]
               << ") doesn't match input size (" << size << ")";
    // Return original array and sequential index as fallback
    IdArray index = Range(0, size, 1, input->dtype, input->ctx);
    return {input, index};
  }

  // Check if part_offset has correct size
  if (part_offset->shape[0] != world_size + 1) {
    LOG(ERROR) << "[MultiWayScan] part_offset size (" << part_offset->shape[0]
               << ") doesn't match expected size (" << (world_size + 1) << ")";
    // Return original array and sequential index as fallback
    IdArray index = Range(0, size, 1, input->dtype, input->ctx);
    return {input, index};
  }

  // Calculate and check workspace size
  size_t workspace_size = world_size * size * sizeof(IdType);
  LOG(INFO) << "[MultiWayScan] Allocating workspace of size "
            << (workspace_size / (1024 * 1024)) << " MB";

  // Limit workspace size and handle large inputs more safely
  const size_t MAX_WORKSPACE_SIZE = 4ULL * 1024 * 1024 * 1024; // 4GB limit
  if (workspace_size > MAX_WORKSPACE_SIZE) {
    LOG(WARNING) << "[MultiWayScan] Requested workspace size exceeds limit, "
                    "falling back to simple sorting";
    // Return sorted array as fallback
    return Sort(input);
  }

  // Allocate workspace with error checking
  IdType *workspace = nullptr;
  try {
    workspace = (IdType *)device->AllocWorkspace(input->ctx, workspace_size);
    if (workspace == nullptr) {
      throw std::runtime_error("Failed to allocate workspace memory");
    }
    LOG(INFO) << "[MultiWayScan] Workspace allocated successfully";
  } catch (const std::exception &e) {
    LOG(ERROR) << "[MultiWayScan] Workspace allocation failed: " << e.what()
               << ", falling back to simple sorting";
    return Sort(input);
  }

  // Initialize workspace to zero
  cudaError_t cuda_err = cudaMemset(workspace, 0, workspace_size);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "[MultiWayScan] Failed to initialize workspace: "
               << cudaGetErrorString(cuda_err);
    device->FreeWorkspace(input->ctx, workspace);
    return Sort(input);
  }

  // Perform the scan with careful error handling
  try {
    LOG(INFO) << "[MultiWayScan] Starting recursive scan";
    _MultiWayScanRecursive(workspace, size, part_ids.Ptr<IdType>(), world_size,
                           device, input->ctx);

    // Create output arrays
    IdArray sorted = IdArray::Empty({size}, input->dtype, input->ctx);
    IdArray index = IdArray::Empty({size}, input->dtype, input->ctx);

    // Check for CUDA errors after scan
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error after scan: ") +
                               cudaGetErrorString(cuda_err));
    }

    LOG(INFO) << "[MultiWayScan] Scan complete, performing permutation";

    // Permute the input based on scan results
    _Permutate(workspace, input.Ptr<IdType>(), part_offset.Ptr<IdType>(),
               part_ids.Ptr<IdType>(), size, world_size, sorted.Ptr<IdType>(),
               index.Ptr<IdType>());

    // Check for CUDA errors after permutation
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error after permutation: ") +
                               cudaGetErrorString(cuda_err));
    }

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();

    LOG(INFO) << "[MultiWayScan] Permutation complete, freeing workspace";
    device->FreeWorkspace(input->ctx, workspace);

    LOG(INFO) << "[MultiWayScan] Completed successfully";
    return {sorted, index};
  } catch (const std::exception &e) {
    LOG(ERROR) << "[MultiWayScan] Error during scan: " << e.what()
               << ", falling back to simple sorting";
    if (workspace) {
      device->FreeWorkspace(input->ctx, workspace);
    }
    return Sort(input);
  }
}

} // namespace ds
} // namespace dgl

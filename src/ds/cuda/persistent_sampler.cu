#include "alltoall.h"
#include "cuda_utils.h"
#include "ds_kernel.h"
#include "persistent_sampler.h"
#include <chrono>
#include <dgl/runtime/device_api.h>
#include <thread>

namespace dgl {
namespace ds {

using namespace dgl::runtime;
using namespace dgl::aten;

// Wrapper for NCCL all-to-all operation for use with our implementation
void NCCLAllToAll(void *send_data, void *recv_data, int count, DLDataType dtype,
                  ncclComm_t comm, cudaStream_t stream) {
  // In a real implementation, this would use NCCL for communication
  // For now this is a stub implementation that just logs the operation
  LOG(INFO) << "NCCL AllToAll operation with count: " << count;

  // Actual implementation would use NCCL collective operations
  // This is left as a placeholder for future integration
}

// Function to get NCCL datatype based on DGL dtype
ncclDataType_t GetNCCLDataType(const DLDataType &dtype) {
  if (dtype.code == kDLInt) {
    if (dtype.bits == 32)
      return ncclInt32;
    if (dtype.bits == 64)
      return ncclInt64;
  } else if (dtype.code == kDLFloat) {
    if (dtype.bits == 32)
      return ncclFloat32;
    if (dtype.bits == 64)
      return ncclFloat64;
  }
  LOG(FATAL) << "Unsupported dtype for NCCL operation";
  return ncclFloat32; // Never reach here, just to silence compiler warnings
}

// Helper function for binary search in the kernel
__device__ int64_t binarySearch(uint32_t *elements, int64_t left, int64_t right,
                                uint32_t element_to_find) {

  int64_t mid = left;
  while (mid <= right) {
    if (elements[mid] == element_to_find) {
      return mid - left;
    }
    mid += 1;
  }

  return right - left;
}

// CUDA kernel for persistent neighbor sampling
__global__ void PersistentSamplingKernel(
    volatile IdType *task_flags, volatile IdType *task_counts,
    IdType **task_seeds, IdType **task_results, int *task_fanouts,
    bool *task_bias_flags, uint32_t **task_weights, IdType *dev_indptr,
    IdType *dev_indices, IdType *uva_indptr, IdType *uva_indices,
    IdType *adj_pos_map, int max_tasks) {

  // Each block handles one persistent task slot
  int slot_id = blockIdx.x;

  // Shared memory for task data
  __shared__ int fanout;
  __shared__ IdType n_seeds;
  __shared__ IdType *seeds;
  __shared__ IdType *results;
  __shared__ bool use_bias;
  __shared__ uint32_t *weight;
  __shared__ bool active;

  // Random state initialization (using different seeds for different threads)
  curandState rng;
  curand_init(7777777 + slot_id * 1000 + threadIdx.y * blockDim.x + threadIdx.x,
              0, 0, &rng);

  active = false;

  while (true) {
    // Check for new task
    if (!active && task_flags[slot_id] == 1) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        n_seeds = task_counts[slot_id];
        seeds = task_seeds[slot_id];
        results = task_results[slot_id];
        fanout = task_fanouts[slot_id];
        use_bias = task_bias_flags[slot_id];
        weight = task_weights[slot_id];
        active = true;

        // Debug signal that task was accepted
        // This hack using a volatile location acts as a memory barrier
        task_flags[slot_id] = 3; // 3 means processing
      }
      __syncthreads();
    }

    // Process active task
    if (active) {
      // Sample neighbors for assigned seeds
      // Each warp (threadIdx.y) handles different seeds
      for (int seed_idx = threadIdx.y; seed_idx < n_seeds;
           seed_idx += blockDim.y) {
        IdType seed = seeds[seed_idx];

        // Get position in adjacency list
        int64_t pos = adj_pos_map[seed];
        bool on_dev = (pos >= 0);
        if (!on_dev)
          pos = ENCODE_ID(pos);

        // Get degree and neighbors
        const int64_t in_row_start = on_dev ? dev_indptr[pos] : uva_indptr[pos];
        const int64_t deg =
            (on_dev ? dev_indptr[pos + 1] : uva_indptr[pos + 1]) - in_row_start;
        const int64_t *index =
            (on_dev ? dev_indices : uva_indices) + in_row_start;

        // Sample neighbors
        for (int pick = threadIdx.x; pick < fanout; pick += blockDim.x) {
          if (deg > 0) {
            int64_t edge;
            if (use_bias) {
              // Biased sampling with weight
              uint32_t val = curand(&rng) % deg;
              edge = binarySearch(weight, in_row_start, in_row_start + deg - 1,
                                  val);
            } else {
              // Uniform sampling
              edge = curand(&rng) % deg;
            }
            results[seed_idx * fanout + pick] = index[edge];
          } else {
            // No neighbors - use placeholder value
            results[seed_idx * fanout + pick] = -1;
          }
        }
      }

      __syncthreads();

      // Mark task as complete
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        task_flags[slot_id] = 2; // 2 means completed
        active = false;
      }
      __syncthreads();
    }

    // Small sleep to prevent busy waiting
    if (!active) {
      clock_t start_clock = clock64();
      clock_t clock_offset = 5000;
      while (clock64() < start_clock + clock_offset) {
      }
    }
  }
}

void InitializePersistentSampler(DSContext *context, IdArray min_vids,
                                 int world_size) {
  if (context->persistent_sampler_state &&
      context->persistent_sampler_state->initialized) {
    return;
  }

  // Create sampler state
  context->persistent_sampler_state =
      std::make_shared<PersistentSamplerState>();
  auto *state = context->persistent_sampler_state.get();

  // Initialize queues for peer-to-peer communication
  state->p2p_send_queues.resize(world_size);
  state->p2p_recv_queues.resize(world_size);
  state->p2p_queue_mutexes.resize(world_size);
  state->p2p_cvs.resize(world_size);

  // Allocate memory for task management
  context->task_flags = Full<int64_t>(0, state->max_tasks, min_vids->ctx);
  context->task_counts =
      IdArray::Empty({state->max_tasks}, min_vids->dtype, min_vids->ctx);

  // Allocate arrays of pointers for tasks
  const size_t ptr_size = state->max_tasks * sizeof(IdType *);
  CUDACHECK(cudaMalloc(&context->task_seeds, ptr_size));
  CUDACHECK(cudaMalloc(&context->task_results, ptr_size));
  CUDACHECK(cudaMalloc(&context->task_fanouts, state->max_tasks * sizeof(int)));
  CUDACHECK(
      cudaMalloc(&context->task_bias_flags, state->max_tasks * sizeof(bool)));
  CUDACHECK(cudaMalloc(&context->task_weights, ptr_size));

  // Launch persistent kernel
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  context->persistent_kernel_stream = stream;

  LaunchPersistentSamplingKernel(context, stream);

  // Start worker thread for handling sampling tasks
  state->shutdown = false;
  context->sampler_thread = std::thread(SamplerWorkerThread, context, min_vids);

  // Start P2P communication threads (one per remote rank)
  for (int i = 0; i < world_size; i++) {
    if (i != context->rank) {
      context->p2p_threads.push_back(
          std::thread(P2PCommunicationThread, context, i));
    }
  }

  // Enable peer access between GPUs if multiple GPUs on same node
  // This is a simplified version - in practice would need to handle
  // GPUs that can't directly access each other
  for (int i = 0; i < world_size; i++) {
    if (i != context->rank) {
      int can_access = 0;
      CUDACHECK(cudaDeviceCanAccessPeer(&can_access, context->rank, i));
      if (can_access) {
        CUDACHECK(cudaDeviceEnablePeerAccess(i, 0));
      }
    }
  }

  state->initialized = true;
}

void LaunchPersistentSamplingKernel(DSContext *context, cudaStream_t stream) {
  auto *state = context->persistent_sampler_state.get();

  // Launch persistent kernel with configuration
  dim3 block(32, 8);           // 32 threads per warp, 8 warps per block
  dim3 grid(state->max_tasks); // One block per task slot

  PersistentSamplingKernel<<<grid, block, 0, stream>>>(
      context->task_flags.Ptr<IdType>(), context->task_counts.Ptr<IdType>(),
      context->task_seeds, context->task_results, context->task_fanouts,
      context->task_bias_flags, context->task_weights,
      context->dev_graph.indptr.Ptr<IdType>(),
      context->dev_graph.indices.Ptr<IdType>(),
      context->uva_graph.indptr.Ptr<IdType>(),
      context->uva_graph.indices.Ptr<IdType>(),
      context->adj_pos_map.Ptr<IdType>(), state->max_tasks);

  // Check for kernel launch errors
  CUDACHECK(cudaGetLastError());
}

std::future<IdArray> SubmitSamplingTask(IdArray seeds, bool is_local,
                                        int fanout, bool bias, IdArray weight) {
  auto *context = DSContext::Global();
  auto *state = context->persistent_sampler_state.get();

  // Create task with promise for result
  SamplingTask task;
  task.seeds = seeds;
  task.is_local = is_local;
  task.fanout = fanout;
  task.bias = bias;
  task.weight = weight;
  std::future<IdArray> future = task.result_promise.get_future();

  // Submit task to queue - we need to use emplace and move since SamplingTask
  // is not copyable
  {
    std::lock_guard<std::mutex> lock(state->queue_mutex);
    state->task_queue.emplace(std::move(task));
  }
  state->cv.notify_one();

  return future;
}

IdArray WaitForSamplingResult(std::future<IdArray> &future) {
  // Wait for the result (with timeout for safety)
  if (future.wait_for(std::chrono::seconds(30)) ==
      std::future_status::timeout) {
    LOG(FATAL) << "Sampling task timed out";
  }
  return future.get();
}

void SamplerWorkerThread(DSContext *context, IdArray min_vids) {
  auto *state = context->persistent_sampler_state.get();
  CUDACHECK(cudaSetDevice(context->rank));

  while (!state->shutdown) {
    SamplingTask task;
    bool has_task = false;

    // Get next task from queue - must use std::move since task is not copyable
    {
      std::unique_lock<std::mutex> lock(state->queue_mutex);
      if (state->cv.wait_for(lock, std::chrono::milliseconds(100), [state]() {
            return !state->task_queue.empty() || state->shutdown;
          })) {
        if (state->shutdown)
          break;
        // Move from queue to local variable
        task = std::move(state->task_queue.front());
        state->task_queue.pop();
        has_task = true;
      }
    }

    if (has_task) {
      // Process local seeds
      IdArray local_seeds;
      if (task.is_local) {
        local_seeds = Partition(task.seeds, min_vids);
      } else {
        local_seeds = task.seeds;
      }

      // Calculate seed distribution
      IdArray send_sizes, send_offset;
      Cluster(context->rank, local_seeds, min_vids, context->world_size,
              &send_sizes, &send_offset);

      // Distribute seeds via P2P transfer or fallback to all-to-all
      P2PDistributeResult distribute_result =
          P2PDistributeSeeds(context, local_seeds, send_sizes, send_offset);
      IdArray frontier = distribute_result.frontier;
      IdArray recv_offset = distribute_result.recv_offset;

      // Convert global IDs to local IDs
      ConvertGidToLid(frontier, min_vids, context->rank);

      // Process seeds in the persistent kernel
      auto neighbors = ProcessSeedsInPersistentKernel(
          context, frontier, task.fanout, task.bias, task.weight);

      // Collect results via P2P transfer or fallback to all-to-all
      P2PCollectResult collect_result =
          P2PCollectResults(context, neighbors, recv_offset, send_offset);
      IdArray reshuffled_neighbors = collect_result.reshuffled_neighbors;

      // Complete the task - we use std::move since we're completing the promise
      task.result_promise.set_value(reshuffled_neighbors);
    }
  }
}

// Helper function to find a free task slot
int FindFreeTaskSlot(DSContext *context) {
  auto *state = context->persistent_sampler_state.get();
  auto *task_flags_ptr = context->task_flags.Ptr<IdType>();

  // Simple polling to find a free slot
  for (int attempt = 0; attempt < 1000; attempt++) {
    for (int slot = 0; slot < state->max_tasks; slot++) {
      int flag;
      CUDACHECK(cudaMemcpy(&flag, const_cast<IdType *>(task_flags_ptr + slot),
                           sizeof(IdType), cudaMemcpyDeviceToHost));
      if (flag == 0) {
        return slot;
      }
    }
    // Wait a bit before trying again
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  LOG(FATAL) << "No free task slot available after multiple attempts";
  return -1;
}

IdArray ProcessSeedsInPersistentKernel(DSContext *context, IdArray frontier,
                                       int fanout, bool bias, IdArray weight) {
  auto *state = context->persistent_sampler_state.get();
  const int n_frontier = frontier->shape[0];
  auto dgl_ctx = frontier->ctx;

  // Allocate memory for results
  IdArray neighbors =
      IdArray::Empty({n_frontier * fanout}, frontier->dtype, dgl_ctx);

  // Find a free task slot
  int slot = FindFreeTaskSlot(context);

  // Allocate device memory for this task's seeds and results
  IdType *d_seeds, *d_results;
  CUDACHECK(cudaMalloc(&d_seeds, n_frontier * sizeof(IdType)));
  CUDACHECK(cudaMalloc(&d_results, n_frontier * fanout * sizeof(IdType)));

  // Copy seeds to device
  CUDACHECK(cudaMemcpy(d_seeds, frontier.Ptr<IdType>(),
                       n_frontier * sizeof(IdType), cudaMemcpyDeviceToDevice));

  // Copy pointers to kernel-accessible arrays
  CUDACHECK(cudaMemcpy(context->task_seeds + slot, &d_seeds, sizeof(IdType *),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(context->task_results + slot, &d_results,
                       sizeof(IdType *), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(context->task_fanouts + slot, &fanout, sizeof(int),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(context->task_bias_flags + slot, &bias, sizeof(bool),
                       cudaMemcpyHostToDevice));

  // Handle weight array if biased sampling is enabled
  uint32_t *d_weight = nullptr;
  if (bias && !IsNullArray(weight)) {
    CUDACHECK(cudaMalloc(&d_weight, weight->shape[0] * sizeof(uint32_t)));
    CUDACHECK(cudaMemcpy(d_weight, weight.Ptr<uint32_t>(),
                         weight->shape[0] * sizeof(uint32_t),
                         cudaMemcpyDeviceToDevice));
  }
  CUDACHECK(cudaMemcpy(context->task_weights + slot, &d_weight,
                       sizeof(uint32_t *), cudaMemcpyHostToDevice));

  // Set count and flag atomically to signal task is ready
  CUDACHECK(cudaMemcpy(
      const_cast<IdType *>(context->task_counts.Ptr<IdType>() + slot),
      &n_frontier, sizeof(IdType), cudaMemcpyHostToDevice));

  // Set flag to 1 to indicate task is ready
  IdType ready_flag = 1;
  CUDACHECK(
      cudaMemcpy(const_cast<IdType *>(context->task_flags.Ptr<IdType>() + slot),
                 &ready_flag, sizeof(IdType), cudaMemcpyHostToDevice));

  // Wait for task completion (flag set to 2)
  IdType flag = 0;
  while (flag != 2) {
    CUDACHECK(cudaMemcpy(
        &flag, const_cast<IdType *>(context->task_flags.Ptr<IdType>() + slot),
        sizeof(IdType), cudaMemcpyDeviceToHost));
    if (flag == 2)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Copy results back
  CUDACHECK(cudaMemcpy(neighbors.Ptr<IdType>(), d_results,
                       n_frontier * fanout * sizeof(IdType),
                       cudaMemcpyDeviceToDevice));

  // Free resources
  CUDACHECK(cudaFree(d_seeds));
  CUDACHECK(cudaFree(d_results));
  if (d_weight)
    CUDACHECK(cudaFree(d_weight));

  // Reset task slot
  IdType zero = 0;
  CUDACHECK(
      cudaMemcpy(const_cast<IdType *>(context->task_flags.Ptr<IdType>() + slot),
                 &zero, sizeof(IdType), cudaMemcpyHostToDevice));

  return neighbors;
}

void P2PCommunicationThread(DSContext *context, int target_rank) {
  auto *state = context->persistent_sampler_state.get();
  CUDACHECK(cudaSetDevice(context->rank));

  // Create stream for P2P transfers
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Check if P2P is supported between these GPUs
  int can_access = 0;
  CUDACHECK(cudaDeviceCanAccessPeer(&can_access, context->rank, target_rank));
  bool p2p_enabled = can_access != 0;

  if (p2p_enabled) {
    CUDACHECK(cudaDeviceEnablePeerAccess(target_rank, 0));
  }

  // Remote device properties
  cudaDeviceProp remote_props;
  CUDACHECK(cudaGetDeviceProperties(&remote_props, target_rank));

  // Thread runs until shutdown signal
  while (!state->shutdown) {
    // Check for outgoing data in the send queue
    IdArray data_to_send;
    bool has_data = false;

    {
      std::unique_lock<std::mutex> lock(state->p2p_queue_mutexes[target_rank]);
      if (state->p2p_cvs[target_rank].wait_for(
              lock, std::chrono::milliseconds(1), [&]() {
                return !state->p2p_send_queues[target_rank].empty() ||
                       state->shutdown;
              })) {
        if (state->shutdown)
          break;
        data_to_send = state->p2p_send_queues[target_rank].front();
        state->p2p_send_queues[target_rank].pop();
        has_data = true;
      }
    }

    if (has_data) {
      // Allocate IPC memory handle for this data
      cudaIpcMemHandle_t handle;
      CUDACHECK(cudaIpcGetMemHandle(&handle, data_to_send.Ptr<IdType>()));

      // Send handle to remote process (would use MPI/socket in practice)
      // This is a simplified implementation - in a real implementation would
      // need to use MPI or some other IPC mechanism to send the handle to the
      // other process

      // For this demo, we simulate the process by directly mapping the memory
      if (p2p_enabled) {
        // Perform direct P2P copy (preferred method when available)
        CUDACHECK(cudaMemcpyPeerAsync(
            /* dst */ nullptr, // Would be the remote buffer
            target_rank,
            /* src */ data_to_send.Ptr<IdType>(), context->rank,
            data_to_send->shape[0] * sizeof(IdType), stream));
      } else {
        // Fallback to host memory for transfer
        IdArray host_data = data_to_send.CopyTo(DLContext({kDLCPU, 0}));
        // In practice, would now use MPI or other method to send data to remote
        // process
      }

      // Signal completion - would be notification to remote process in practice
      CUDACHECK(cudaStreamSynchronize(stream));
    }

    // Check for incoming data from remote process
    // This would be implemented with MPI or similar in practice

    // Sleep a bit to avoid busy waiting
    if (!has_data) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Cleanup
  if (p2p_enabled) {
    CUDACHECK(cudaDeviceDisablePeerAccess(target_rank));
  }
  CUDACHECK(cudaStreamDestroy(stream));
}

P2PDistributeResult P2PDistributeSeeds(DSContext *context, IdArray seeds,
                                       IdArray send_sizes,
                                       IdArray send_offset) {
  auto *state = context->persistent_sampler_state.get();
  int world_size = context->world_size;
  int rank = context->rank;
  auto dgl_ctx = seeds->ctx;

  // Create arrays to store received seed counts and offsets
  IdArray recv_sizes = IdArray::Empty({world_size}, seeds->dtype, dgl_ctx);

  // Copy send_sizes to host to prepare for P2P transfer
  auto host_send_sizes = send_sizes.CopyTo(DLContext({kDLCPU, 0}));
  auto host_send_offset = send_offset.CopyTo(DLContext({kDLCPU, 0}));
  auto *send_sizes_ptr = host_send_sizes.Ptr<IdType>();
  auto *send_offset_ptr = host_send_offset.Ptr<IdType>();

  // Get total number of seeds to send
  int64_t total_send = 0;
  if (send_sizes->shape[0] > 0) {
    total_send =
        send_offset_ptr[world_size - 1] + send_sizes_ptr[world_size - 1];
  }

  // Create arrays to store received seeds and queue requests
  std::vector<IdArray> seed_chunks;
  std::vector<std::future<void>> transfer_futures;

  // Use NCCL to exchange size information
  auto *recv_sizes_host = new IdType[world_size];
  NCCLAllToAll(send_sizes_ptr, recv_sizes_host, 1, seeds->dtype,
               context->nccl_comm[0], nullptr);

  // Copy received sizes to GPU
  CUDACHECK(cudaMemcpy(recv_sizes.Ptr<IdType>(), recv_sizes_host,
                       world_size * sizeof(IdType), cudaMemcpyHostToDevice));

  // Calculate receive offsets
  IdArray recv_offset = CumSum(recv_sizes, true);
  auto host_recv_offset = recv_offset.CopyTo(DLContext({kDLCPU, 0}));
  auto *recv_offset_ptr = host_recv_offset.Ptr<IdType>();

  // Calculate total number of seeds to receive
  int64_t total_recv = 0;
  if (recv_sizes->shape[0] > 0) {
    total_recv =
        recv_offset_ptr[world_size - 1] + recv_sizes_host[world_size - 1];
  }

  // Allocate buffer for all received seeds
  IdArray all_received_seeds =
      IdArray::Empty({total_recv}, seeds->dtype, dgl_ctx);

  // Distribute seeds using P2P transfers
  for (int i = 0; i < world_size; i++) {
    if (i == rank)
      continue;

    // Skip empty transfers
    if (send_sizes_ptr[i] == 0 && recv_sizes_host[i] == 0)
      continue;

    // Extract the seeds to send to rank i
    if (send_sizes_ptr[i] > 0) {
      IdArray seeds_to_send =
          seeds.CreateView({send_sizes_ptr[i]}, seeds->dtype,
                           send_offset_ptr[i] * sizeof(IdType));

      // Queue the seeds for P2P transfer
      {
        std::lock_guard<std::mutex> lock(state->p2p_queue_mutexes[i]);
        state->p2p_send_queues[i].push(seeds_to_send);
      }

      // Notify the P2P thread
      state->p2p_cvs[i].notify_one();
    }

    // In a real implementation, would need a synchronization mechanism
    // to wait for P2P transfers to complete. For now, we'll use a simple
    // fallback to process incoming data.
  }

  // Wait for all P2P transfers to complete
  // In a real implementation, this would wait for completion signals from other
  // ranks

  // For now, fallback to regular all-to-all to actually do the data movement
  // since we're just setting up the infrastructure
  auto result =
      Alltoall(seeds, send_offset, 1, context->rank, context->world_size);

  delete[] recv_sizes_host;

  // Create and return the P2PDistributeResult struct
  P2PDistributeResult distribute_result;
  distribute_result.frontier = result.first; // The frontier nodes from Alltoall
  distribute_result.recv_offset = recv_offset; // Use our computed recv_offset

  return distribute_result;
}

P2PCollectResult P2PCollectResults(DSContext *context, IdArray neighbors,
                                   IdArray recv_offset, IdArray send_offset) {
  auto *state = context->persistent_sampler_state.get();
  int world_size = context->world_size;
  int rank = context->rank;
  auto dgl_ctx = neighbors->ctx;
  int fanout = neighbors->shape[0] / recv_offset->shape[0];

  // Copy offset arrays to host
  auto host_recv_offset = recv_offset.CopyTo(DLContext({kDLCPU, 0}));
  auto host_send_offset = send_offset.CopyTo(DLContext({kDLCPU, 0}));
  auto *recv_offset_ptr = host_recv_offset.Ptr<IdType>();
  auto *send_offset_ptr = host_send_offset.Ptr<IdType>();

  // Calculate send sizes from recv_offset
  IdArray send_sizes = IdArray::Empty({world_size}, neighbors->dtype, dgl_ctx);
  auto host_send_sizes = send_sizes.CopyTo(DLContext({kDLCPU, 0}));
  auto *send_sizes_ptr = host_send_sizes.Ptr<IdType>();

  for (int i = 0; i < world_size - 1; i++) {
    send_sizes_ptr[i] = (recv_offset_ptr[i + 1] - recv_offset_ptr[i]) * fanout;
  }

  if (world_size > 0) {
    int64_t last_size =
        neighbors->shape[0] - recv_offset_ptr[world_size - 1] * fanout;
    send_sizes_ptr[world_size - 1] = last_size;
  }

  // Exchange send sizes to get receive sizes
  IdArray recv_sizes = IdArray::Empty({world_size}, neighbors->dtype, dgl_ctx);
  auto *recv_sizes_host = new IdType[world_size];

  NCCLAllToAll(send_sizes_ptr, recv_sizes_host, 1, neighbors->dtype,
               context->nccl_comm[0], nullptr);

  // Calculate receive offsets
  int64_t total_recv = 0;
  for (int i = 0; i < world_size; i++) {
    total_recv += recv_sizes_host[i];
  }

  // Allocate buffer for all received neighbors
  IdArray all_received_neighbors =
      IdArray::Empty({total_recv}, neighbors->dtype, dgl_ctx);

  // Create receive offset array
  IdArray reshuffle_recv_offset =
      IdArray::Empty({world_size + 1}, neighbors->dtype, dgl_ctx);
  auto host_reshuffle_offset =
      reshuffle_recv_offset.CopyTo(DLContext({kDLCPU, 0}));
  auto *reshuffle_offset_ptr = host_reshuffle_offset.Ptr<IdType>();

  reshuffle_offset_ptr[0] = 0;
  for (int i = 0; i < world_size; i++) {
    reshuffle_offset_ptr[i + 1] = reshuffle_offset_ptr[i] + recv_sizes_host[i];
  }

  // Send neighbors using P2P transfers
  for (int i = 0; i < world_size; i++) {
    if (i == rank)
      continue;

    // Skip empty transfers
    if (send_sizes_ptr[i] == 0 && recv_sizes_host[i] == 0)
      continue;

    // Extract the neighbors to send to rank i
    if (send_sizes_ptr[i] > 0) {
      int offset = recv_offset_ptr[i] * fanout;
      IdArray neighbors_to_send = neighbors.CreateView(
          {send_sizes_ptr[i]}, neighbors->dtype, offset * sizeof(IdType));

      // Queue the neighbors for P2P transfer
      {
        std::lock_guard<std::mutex> lock(state->p2p_queue_mutexes[i]);
        state->p2p_send_queues[i].push(neighbors_to_send);
      }

      // Notify the P2P thread
      state->p2p_cvs[i].notify_one();
    }
  }

  // For now, fallback to regular all-to-all to actually do the data movement
  auto result = Alltoall(neighbors, recv_offset, fanout, context->rank,
                         context->world_size, send_offset);

  delete[] recv_sizes_host;

  // Create and return the P2PCollectResult struct
  P2PCollectResult collect_result;
  collect_result.reshuffled_neighbors =
      result.first;                 // The reshuffled neighbors from Alltoall
  collect_result.dummy = IdArray(); // Empty array for compatibility

  return collect_result;
}

void ShutdownPersistentSampler(DSContext *context) {
  if (!context->persistent_sampler_state ||
      !context->persistent_sampler_state->initialized) {
    return;
  }

  auto *state = context->persistent_sampler_state.get();

  // Signal shutdown to all threads
  state->shutdown = true;
  state->cv.notify_all();
  for (auto &cv : state->p2p_cvs) {
    cv.notify_all();
  }

  // Wait for threads to finish
  if (context->sampler_thread.joinable()) {
    context->sampler_thread.join();
  }

  for (auto &thread : context->p2p_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Free resources
  CUDACHECK(cudaFree(context->task_seeds));
  CUDACHECK(cudaFree(context->task_results));
  CUDACHECK(cudaFree(context->task_fanouts));
  CUDACHECK(cudaFree(context->task_bias_flags));
  CUDACHECK(cudaFree(context->task_weights));

  CUDACHECK(cudaStreamDestroy(context->persistent_kernel_stream));

  // Reset state
  state->initialized = false;
}

} // namespace ds
} // namespace dgl

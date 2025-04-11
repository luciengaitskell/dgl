#ifndef DGL_DS_CUDA_PERSISTENT_SAMPLER_H_
#define DGL_DS_CUDA_PERSISTENT_SAMPLER_H_

#include "../context.h"
#include <atomic>
#include <condition_variable>
#include <dgl/array.h>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace dgl {
namespace ds {

using namespace dgl::runtime;
using namespace dgl::aten;

// Structure to represent a sampling task
struct SamplingTask {
  IdArray seeds;
  int fanout;
  bool is_local;
  bool bias;
  IdArray weight;
  std::promise<IdArray> result_promise;
};

// Structure to hold the state of the persistent sampler
struct PersistentSamplerState {
  // Queue for incoming sampling tasks
  std::queue<SamplingTask> task_queue;
  std::mutex queue_mutex;
  std::condition_variable cv;

  // Peer-to-peer communication queues (one per remote rank)
  std::vector<std::queue<IdArray>> p2p_send_queues;
  std::vector<std::queue<IdArray>> p2p_recv_queues;
  std::vector<std::mutex> p2p_queue_mutexes;
  std::vector<std::condition_variable> p2p_cvs;

  // Control flags
  std::atomic<bool> shutdown{false};
  std::atomic<bool> initialized{false};

  // Max number of concurrent task slots
  int max_tasks{32};
};

// Initialize the persistent sampler
void InitializePersistentSampler(DSContext *context, IdArray min_vids,
                                 int world_size);

// Submit a sampling task to the persistent sampler and get a future for the
// result
std::future<IdArray> SubmitSamplingTask(IdArray seeds, bool is_local,
                                        int fanout, bool bias = false,
                                        IdArray weight = aten::NullArray());

// Wait for the sampling result
IdArray WaitForSamplingResult(std::future<IdArray> &future);

// Worker thread that processes sampling tasks
void SamplerWorkerThread(DSContext *context, IdArray min_vids);

// Thread that handles P2P communication with a specific rank
void P2PCommunicationThread(DSContext *context, int target_rank);

// Launch the persistent sampling kernel
void LaunchPersistentSamplingKernel(DSContext *context, cudaStream_t stream);

// Distribute seeds via P2P transfer
std::tuple<IdArray, IdArray> P2PDistributeSeeds(DSContext *context,
                                                IdArray seeds,
                                                IdArray send_sizes,
                                                IdArray send_offset);

// Collect results via P2P transfer
IdArray P2PCollectResults(DSContext *context, IdArray neighbors,
                          IdArray recv_offset, IdArray send_offset);

// Process seeds in the persistent kernel
IdArray ProcessSeedsInPersistentKernel(DSContext *context, IdArray frontier,
                                       int fanout, bool bias = false,
                                       IdArray weight = aten::NullArray());

// Shutdown the persistent sampler
void ShutdownPersistentSampler(DSContext *context);

} // namespace ds
} // namespace dgl

#endif // DGL_DS_CUDA_PERSISTENT_SAMPLER_H_

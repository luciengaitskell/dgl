#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <atomic>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dmlc/thread_local.h>
#include <memory>
#include <nccl.h>
#include <thread>
#include <vector>

#include "coordinator.h"
#include "./comm/comm_info.h"
#include "./profiler.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace ds {

// Forward declare the PersistentSamplerState
struct PersistentSamplerState;

enum FeatMode { kFeatModeAllCache, kFeatModePartitionCache, kFeatModeReplicateCache };

#define ENCODE_ID(i) (-(i)-2)

#define SAMPLER_ROLE 0
#define LOADER_ROLE 1

#define THREAD_LOCAL_PINNED_ARRAY_SIZE 256
#define N_PINNED_ARRAY 3

struct DSThreadEntry {
  IdArray pinned_array[N_PINNED_ARRAY];
  int pinned_array_counter;
  static DSThreadEntry* ThreadLocal();
};

struct DSContext {
  bool initialized = false;
  int world_size;
  int rank;
  int thread_num;
  std::vector<ncclComm_t> nccl_comm;
  std::vector<std::unique_ptr<CommInfo> > comm_info;
  std::unique_ptr<Coordinator> coordinator;
  std::unique_ptr<Coordinator> comm_coordinator;

  // Feature related arrays
  bool feat_loaded = false;
  FeatMode feat_mode;
  IdArray dev_feats, shared_feats, feat_pos_map;
  int feat_dim;


  // Graph related arrays
  bool graph_loaded = false;
  CSRMatrix dev_graph, uva_graph;
  int64_t n_cached_nodes, n_uva_nodes;
  IdArray adj_pos_map;

  // Kernel controller
  bool enable_kernel_control;
  std::atomic<int> sampler_queue_size{0}, loader_queue_size{0};

  // Communication control
  bool enable_comm_control;

  // Profiler
  bool enable_profiler;
  std::unique_ptr<Profiler> profiler;

  // Persistent sampler related
  std::shared_ptr<PersistentSamplerState> persistent_sampler_state;
  bool persistent_sampler_initialized = false;
  std::thread sampler_thread;
  std::vector<std::thread> p2p_threads;

  // Persistent kernel variables
  IdArray task_flags;
  IdArray task_counts;
  IdType **task_seeds = nullptr;
  IdType **task_results = nullptr;
  int *task_fanouts = nullptr;
  bool *task_bias_flags = nullptr;
  uint32_t **task_weights = nullptr;
  cudaStream_t persistent_kernel_stream;

  static DSContext* Global() {
    static DSContext instance;
    return &instance;
  }
};

}
}

#endif

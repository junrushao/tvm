#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <driver_types.h>
#include <nccl.h>
#include <tvm/runtime/registry.h>

#include <mutex>
#include <vector>

#include "../../cuda/cuda_common.h"
#include "../session.h"
#include "./utils.h"

namespace tvm {
namespace runtime {
namespace nccl {

struct NCCLGlobalContext {
  std::vector<int> device_ids;
  std::vector<ncclComm_t> communicators;
  std::vector<cudaStream_t> streams;

  ~NCCLGlobalContext() {}

  void Clear() {
    for (ncclComm_t comm : this->communicators) {
      NCCL_CALL(ncclCommDestroy(comm));
    }
    device_ids.clear();
    communicators.clear();
  }

  static NCCLGlobalContext* Get() {
    static NCCLGlobalContext ctx;
    return &ctx;
  }

  void Initialize(std::vector<int> device_ids) {
    DiscoWorker* worker = DiscoWorker::ThreadLocal();
    int num_workers = worker->num_workers;
    CHECK_EQ(device_ids.size(), num_workers)
        << "ValueError: There are " << num_workers << " worker(s), but " << device_ids.size()
        << " device id(s) are provided.";
    ncclUniqueId id;
    NCCL_CALL(ncclGetUniqueId(&id));
    NCCL_CALL(ncclGroupStart());
    for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
      int device_id = device_ids[worker_id];
      ncclComm_t comm;
      cudaStream_t stream;
      CUDA_CALL(cudaSetDevice(device_id));
      CUDA_CALL(cudaStreamCreate(&stream));
      NCCL_CALL(ncclCommInitRank(&comm, num_workers, id, worker_id));
      this->streams.push_back(stream);
      this->communicators.push_back(comm);
    }
    NCCL_CALL(ncclGroupEnd());
    this->device_ids = std::move(device_ids);
  }

  static ncclComm_t ThreadLocalCommunicator() {
    thread_local static ncclComm_t comm =
        NCCLGlobalContext::Get()->communicators[DiscoWorker::ThreadLocal()->worker_id];
    return comm;
  }

  static cudaStream_t ThreadLocalStream() {
    thread_local static cudaStream_t stream =
        NCCLGlobalContext::Get()->streams[DiscoWorker::ThreadLocal()->worker_id];
    return stream;
  }
};

NDArray AllReduce(NDArray send, int _reduce_kind) {
  ShapeTuple shape = send.Shape();
  NDArray recv = NDArray::Empty(shape, send->dtype, send->device);
  ReduceKind reduce_kind = static_cast<ReduceKind>(_reduce_kind);
  size_t numel = 1;
  for (int64_t d : shape) {
    numel *= d;
  }
  NCCL_CALL(ncclAllReduce(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*op=*/AsNCCLRedOp(reduce_kind),
                          /*comm=*/NCCLGlobalContext::ThreadLocalCommunicator(),
                          /*stream=*/NCCLGlobalContext::ThreadLocalStream()));
  return recv;
}

TVM_REGISTER_GLOBAL("runtime.disco.nccl.init").set_body([](TVMArgs args, TVMRetValue* rv) -> void {
  // Parse the inputs into device ids
  std::vector<int> device_ids;
  for (int i = 0; i < args.num_args; ++i) {
    device_ids.push_back(args[i].operator int());
  }
  // Set the `default_device` and `ccl` for the current worker
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  worker->default_device = Device{DLDeviceType::kDLCUDA, device_ids[worker->worker_id]};
  worker->ccl = "nccl";
  // Setup global context only once
  static std::once_flag flag;
  std::call_once(flag, [&]() { NCCLGlobalContext::Get()->Initialize(device_ids); });
});

TVM_REGISTER_GLOBAL("runtime.disco.nccl.allreduce").set_body_typed(AllReduce);

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm

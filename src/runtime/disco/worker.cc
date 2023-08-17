/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "./worker.h"

#include <tvm/runtime/registry.h>

#include <thread>

namespace tvm {
namespace runtime {

struct ThreadLocalDiscoWorker {
  DiscoWorker* worker;

  static ThreadLocalDiscoWorker* Get() {
    thread_local static ThreadLocalDiscoWorker worker;
    return &worker;
  }
};

DiscoWorker* DiscoWorker::ThreadLocal() { return ThreadLocalDiscoWorker::Get()->worker; }

struct DiscoWorker::Impl {
  static void MainLoop(DiscoWorker* self) {
    ThreadLocalDiscoWorker::Get()->worker = self;
    LOG(INFO) << "[Thread " << std::this_thread::get_id() << "] Worker #" << self->worker_id
              << " Launched";
    while (true) {
      TVMArgs args = self->channel->Recv();
      DiscoAction action = static_cast<DiscoAction>(args[0].operator int());
      int64_t reg_id = args[1];
      switch (action) {
        case DiscoAction::kShutDown: {
          Shutdown(self);
          return;
        }
        case DiscoAction::kKillReg: {
          GetReg(self, reg_id) = nullptr;
          break;
        }
        case DiscoAction::kGetGlobalFunc: {
          GetGlobalFunc(self, reg_id, args[2]);
          break;
        }
        case DiscoAction::kCallPacked: {
          int func_reg_id = args[2];
          uint64_t is_dref = args[3];
          PackedFunc func = GetReg(self, func_reg_id);
          CallPacked(self, reg_id, func, is_dref,
                     TVMArgs(args.values + 4, args.type_codes + 4, args.num_args - 4));
          break;
        }
        case DiscoAction::kCopyFromWorker0: {
          CopyFromWorker0(self, reg_id);
          break;
        }
        case DiscoAction::kCopyToWorker0: {
          CopyToWorker0(self, reg_id);
          break;
        }
        case DiscoAction::kSyncWorker: {
          SyncWorker(self, reg_id);
          break;
        }
      }
    }
  }

  static void Shutdown(DiscoWorker* self) {}

  static void GetGlobalFunc(DiscoWorker* self, int reg_id, const std::string& name) {
    const PackedFunc* pf = runtime::Registry::Get(name);
    CHECK(pf) << "ValueError: Cannot find global function: " << name;
    if (reg_id != 0) {
      GetReg(self, reg_id) = *pf;
    }
  }

  static NDArray GetNDArrayFromHost(DiscoWorker* self) {
    std::lock_guard<std::mutex> lock(self->worker_zero_data->queue_mutex_);
    NDArray array = self->worker_zero_data->host_arrays.front();
    self->worker_zero_data->host_arrays.pop();
    return array;
  }

  static void CopyFromWorker0(DiscoWorker* self, int reg_id) {
    if (self->worker_zero_data != nullptr) {
      NDArray tgt = GetNDArrayFromHost(self);
      NDArray src = GetReg(self, reg_id);
      tgt.CopyFrom(src);
    }
  }

  static void CopyToWorker0(DiscoWorker* self, int reg_id) {
    if (self->worker_zero_data != nullptr) {
      NDArray src = GetNDArrayFromHost(self);
      NDArray tgt = GetReg(self, reg_id);
      tgt.CopyFrom(src);
    }
  }

  static void SyncWorker(DiscoWorker* self, int worker_id) {
    if (worker_id == self->worker_id) {
      TVMValue values[2];
      int type_codes[2];
      PackArgs(values, type_codes, static_cast<int>(DiscoAction::kSyncWorker), worker_id);
      self->channel->Reply(TVMArgs(values, type_codes, 2));
    }
  }

  static void CallPacked(DiscoWorker* self, int64_t ret_reg_id, PackedFunc func, uint64_t is_dref,
                         const TVMArgs& args) {
    TVMValue* values = const_cast<TVMValue*>(args.values);
    int* type_codes = const_cast<int*>(args.type_codes);
    int num_args = args.num_args;
    TVMArgsSetter setter(values, type_codes);
    for (int i = 0; i < num_args; ++i) {
      if ((is_dref >> i) & 1) {
        int64_t reg_id = TVMArgValue(values[i], type_codes[i]).operator int64_t();
        ICHECK(0 <= reg_id && reg_id <= static_cast<int>(self->register_file.size()));
        setter(i, GetReg(self, reg_id));
      }
    }
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values, type_codes, num_args), &rv);
    GetReg(self, ret_reg_id) = std::move(rv);
  }

  static TVMRetValue& GetReg(DiscoWorker* self, int64_t reg_id) {
    if (reg_id >= static_cast<int64_t>(self->register_file.size())) {
      self->register_file.resize(reg_id + 1);
    }
    return self->register_file[reg_id];
  }
};

void DiscoWorker::MainLoop() { DiscoWorker::Impl::MainLoop(this); }

}  // namespace runtime
}  // namespace tvm

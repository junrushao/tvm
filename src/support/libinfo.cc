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
#include <tvm/node/container.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#ifndef TVM_GIT_COMMIT_HASH
#define TVM_GIT_COMMIT_HASH "not-found"
#endif

namespace tvm {

Map<String, String> GetLibInfo() {
  Map<String, String> result = {
      {"GIT_COMMIT_HASH", TVM_GIT_COMMIT_HASH},
  };
  return result;
}

TVM_REGISTER_GLOBAL("support.GetLibInfo").set_body_typed(GetLibInfo);

}  // namespace tvm

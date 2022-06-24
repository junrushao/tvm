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
#include <tvm/ir/module.h>

#include "./builder.h"

namespace tvm {
namespace script {
namespace builder {

void FrameNode::EnterWithScope() { Builder::Current()->frames.push_back(GetRef<Frame>(this)); }

void FrameNode::ExitWithScope() {
  for (auto it = callbacks.rbegin(); it != callbacks.rend(); ++it) {
    (*it)();
  }
  this->callbacks.clear();
  Builder::Current()->frames.pop_back();
}

TVM_REGISTER_NODE_TYPE(FrameNode);
TVM_REGISTER_GLOBAL("script.builder.FrameEnter").set_body_method<Frame>(&FrameNode::EnterWithScope);
TVM_REGISTER_GLOBAL("script.builder.FrameExit").set_body_method<Frame>(&FrameNode::ExitWithScope);

}  // namespace builder
}  // namespace script
}  // namespace tvm

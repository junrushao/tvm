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
#include "./builder.h"

namespace tvm {
namespace script {
namespace builder {

std::vector<Builder>* ThreadLocalBuilderStack() {
  thread_local std::vector<Builder> stack;
  return &stack;
}

void Builder::EnterWithScope() {
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  stack->push_back(*this);
}

void Builder::ExitWithScope() {
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  CHECK(!stack->empty());
  stack->pop_back();
}

Builder Builder::Current() {
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  CHECK(!stack->empty());
  return stack->back();
}

TVM_REGISTER_NODE_TYPE(BuilderNode);

}  // namespace builder
}  // namespace script
}  // namespace tvm

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
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {
namespace stree {

/********** S-Tree related **********/

inline StmtSRef Scope(const StmtSRef& sref) {
  for (const StmtSRefNode* p = sref->parent; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<BlockNode>()) {
      return GetRef<StmtSRef>(p);
    }
  }
  throw;
}

}  // namespace stree

/********** AST related **********/

namespace ast {}

/********** Block related **********/

namespace block {

inline bool IsOutput(const BlockNode* block, const BlockNode* scope_block) {
  for (const TensorRegion& scope_write : scope_block->writes) {
    const Buffer& scope_buffer = scope_write->buffer;
    for (const TensorRegion& write : block->writes) {
      if (write->buffer.same_as(scope_buffer)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace block

}  // namespace tir
}  // namespace tvm

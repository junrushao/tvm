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
#include "./block_frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

BlockFrame::BlockFrame(String name) {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->name = name;
  n->iter_vars.clear();
  n->reads = NullOpt;
  n->writes = NullOpt;
  n->init = NullOpt;
  n->alloc_buffers.clear();
  n->match_buffers.clear();
  n->annotations.clear();
  n->iter_values.clear();
  n->predicate = NullOpt;
  data_ = n;
}

namespace axis {

// TODO(@junrushao1994): figure out the Block syntax without BlockRealize

tvm::tir::IterVar PushBlockVar(tvm::tir::IterVar iter_var, PrimExpr binding) {
  if (const BlockFrameNode* opt_frame = Builder::Current()->frames.back().as<BlockFrameNode>()) {
    BlockFrame frame = GetRef<BlockFrame>(opt_frame);
    frame->iter_vars.push_back(iter_var);
    frame->iter_values.push_back(binding);
  } else {
    LOG(FATAL) << "TypeError: The last frame is not BlockFrame";
  }
  return iter_var;
}

tvm::tir::IterVar Spatial(Range dom, PrimExpr binding, DataType dtype) {
  using namespace tvm::tir;
  ICHECK(dom.defined()) << "Spatial axis must have a domain";
  int bits = std::max({dom->min.dtype().bits(), dom->extent.dtype().bits(), dtype.bits()});
  return PushBlockVar(IterVar(/*dom=*/dom,                              //
                              /*var=*/Var("_", dtype.with_bits(bits)),  //
                              /*iter_type=*/IterVarType::kDataPar,      //
                              /*thread_tag=*/""),
                      binding);
}

tvm::tir::IterVar Reduce(Range dom, PrimExpr binding, DataType dtype) {
  using namespace tvm::tir;
  ICHECK(dom.defined()) << "Spatial axis must have a domain";
  int bits = std::max({dom->min.dtype().bits(), dom->extent.dtype().bits(), dtype.bits()});
  return PushBlockVar(IterVar(/*dom=*/dom,                              //
                              /*var=*/Var("_", dtype.with_bits(bits)),  //
                              /*iter_type=*/IterVarType::kCommReduce,   //
                              /*thread_tag=*/""),
                      binding);
}

tvm::tir::IterVar Remap(String kinds, Array<PrimExpr> bindings, DataType dtype) {
  //
  throw;
}

}  // namespace axis

TVM_REGISTER_NODE_TYPE(BlockFrameNode);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

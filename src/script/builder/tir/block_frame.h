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
#ifndef TVM_SCRIPT_BUILDER_TIR_BLOCK_FRAME_H_
#define TVM_SCRIPT_BUILDER_TIR_BLOCK_FRAME_H_

#include "./tir.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class BlockFrameNode : public TIRFrameNode {
 public:
  String name;
  Array<tvm::tir::IterVar> iter_vars;
  Optional<Array<tvm::tir::BufferRegion>> reads;
  Optional<Array<tvm::tir::BufferRegion>> writes;
  Optional<tvm::tir::Stmt> init;
  Array<tvm::tir::Buffer> alloc_buffers;
  Array<tvm::tir::MatchBufferRegion> match_buffers;
  Map<String, ObjectRef> annotations;

  Array<PrimExpr> iter_values;
  Optional<PrimExpr> predicate;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("iter_vars", &iter_vars);
    v->Visit("reads", &reads);
    v->Visit("writes", &writes);
    v->Visit("init", &init);
    v->Visit("alloc_buffers", &alloc_buffers);
    v->Visit("match_buffers", &match_buffers);
    v->Visit("annotations", &annotations);
    v->Visit("iter_values", &iter_values);
    v->Visit("predicate", &predicate);
  }

  static constexpr const char* _type_key = "script.builder.tir.BlockFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockFrameNode, TIRFrameNode);
};

class BlockFrame : public TIRFrame {
 public:
  explicit BlockFrame(String name);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockFrame, TIRFrame, BlockFrameNode);
};

namespace axis {
tvm::tir::IterVar Spatial(Range dom, PrimExpr binding, DataType dtype);
tvm::tir::IterVar Reduce(Range dom, PrimExpr binding, DataType dtype);
tvm::tir::IterVar Remap(String kinds, Array<PrimExpr> bindings, DataType dtype);
}  // namespace axis
}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_BLOCK_FRAME_H_

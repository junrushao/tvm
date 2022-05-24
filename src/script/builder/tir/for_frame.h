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
#ifndef TVM_SCRIPT_BUILDER_TIR_FRAME_H_
#define TVM_SCRIPT_BUILDER_TIR_FRAME_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../builder.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class ForFrameNode : public FrameNode {
 public:
  using FMakeForLoop =
      runtime::TypedPackedFunc<tvm::tir::Stmt(Array<tvm::tir::Var>, tvm::tir::Stmt)>;

  Array<tvm::tir::Var> loop_vars;
  FMakeForLoop f_make_for_loop;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("loop_vars", &loop_vars);
    // `f_make_for_loop` is not visited.
  }

  static constexpr const char* _type_key = "script.builder.tir.ForFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForFrameNode, FrameNode);
};

class ForFrame : public Frame {
 public:
  static ForFrame Serial(PrimExpr min, PrimExpr extent, Map<String, ObjectRef> attrs);
  static ForFrame Parallel(PrimExpr min, PrimExpr extent, Map<String, ObjectRef> attrs);
  static ForFrame Vectorized(PrimExpr min, PrimExpr extent, Map<String, ObjectRef> attrs);
  static ForFrame Unroll(PrimExpr min, PrimExpr extent, Map<String, ObjectRef> attrs);
  static ForFrame ThreadBinding(PrimExpr min, PrimExpr extent, String thread,
                                Map<String, ObjectRef> attrs);
  static ForFrame Grid(Array<PrimExpr> extents);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ForFrame, Frame, ForFrameNode);
};

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_FRAME_H_

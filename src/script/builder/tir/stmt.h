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
#ifndef TVM_SCRIPT_BUILDER_TIR_STMT_H_
#define TVM_SCRIPT_BUILDER_TIR_STMT_H_

#include "./base.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class AssertFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;
  PrimExpr message;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
    v->Visit("message", &message);
  }

  static constexpr const char* _type_key = "script.builder.tir.AssertFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssertFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AssertFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AssertFrame, TIRFrame, AssertFrameNode);
};

class LetFrameNode : public TIRFrameNode {
 public:
  tvm::tir::Var var;
  PrimExpr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("var", &var);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.builder.tir.LetFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class LetFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LetFrame, TIRFrame, LetFrameNode);
};

class AllocateFrameNode : public TIRFrameNode {
 public:
  Array<PrimExpr> extents;
  DataType dtype;
  String storage_scope_str;
  PrimExpr condition;
  Map<String, ObjectRef> annotations;
  tvm::tir::Buffer buffer;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("extents", &extents);
    v->Visit("dtype", &dtype);
    v->Visit("storage_scope_str", &storage_scope_str);
    v->Visit("condition", &condition);
    v->Visit("annotations", &annotations);
    v->Visit("buffer", &buffer);
  }

  static constexpr const char* _type_key = "script.builder.tir.AllocateFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AllocateFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateFrame, TIRFrame, AllocateFrameNode);
};

class AllocateConstFrameNode : public TIRFrameNode {
 public:
  DataType dtype;
  Array<PrimExpr> extents;
  ObjectRef data_or_idx;
  tvm::tir::Buffer buffer;
  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("dtype", &dtype);
    v->Visit("extents", &extents);
    v->Visit("data_or_idx", &data_or_idx);
    v->Visit("buffer", &buffer);
  }

  static constexpr const char* _type_key = "script.builder.tir.AllocateConstFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateConstFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AllocateConstFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateConstFrame, TIRFrame,
                                                    AllocateConstFrameNode);
};

// class LaunchThreadFrameNode : public TIRFrameNode {
//  public:
//   tvm::tir::Var env_var;
//   PrimExpr extent;
//   void VisitAttrs(tvm::AttrVisitor* v) { TIRFrameNode::VisitAttrs(v); }

//   static constexpr const char* _type_key = "script.builder.tir.LaunchThreadFrame";
//   TVM_DECLARE_FINAL_OBJECT_INFO(LaunchThreadFrameNode, TIRFrameNode);

//  public:
//   void ExitWithScope() final;
// };

// class LaunchThreadFrame : public TIRFrame {
//  public:
//   TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LaunchThreadFrame, TIRFrame,
//                                                     LaunchThreadFrameNode);
// };

class RealizeFrameNode : public TIRFrameNode {
 public:
  tvm::tir::BufferRegion buffer_slice;
  String storage_scope_str;
  PrimExpr condition;

  void VisitAttrs(tvm::AttrVisitor* v) { TIRFrameNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.builder.tir.RealizeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(RealizeFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class RealizeFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RealizeFrame, TIRFrame, RealizeFrameNode);
};

class AttrFrameNode : public TIRFrameNode {
 public:
  ObjectRef node;
  String attr_key;
  PrimExpr value;
  void VisitAttrs(tvm::AttrVisitor* v) { TIRFrameNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.builder.tir.AttrFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AttrFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AttrFrame, TIRFrame, AttrFrameNode);
};

AssertFrame Assert(PrimExpr condition, PrimExpr message);
LetFrame Let(tvm::tir::Var var, PrimExpr value);
AllocateFrame Allocate_(Array<PrimExpr> extents, DataType dtype, String storage_scope_str = "",
                        PrimExpr condition = true, Map<String, ObjectRef> annotations = {});
AllocateConstFrame AllocateConst_(ObjectRef data_or_idx, DataType dtype, Array<PrimExpr> extents);
RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope_str,
                     PrimExpr condition);
AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value);
}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_STMT_H_

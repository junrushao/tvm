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

AssertFrame Assert(PrimExpr condition, PrimExpr message);
LetFrame Let(tvm::tir::Var var, PrimExpr value);
}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_STMT_H_

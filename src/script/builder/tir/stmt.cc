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
#include "./stmt.h"

#include <tvm/runtime/registry.h>

namespace tvm {
namespace script {
namespace builder {
namespace tir {

void AssertFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(AssertStmt(condition, message, AsStmt(stmts)));
}

void LetFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(LetStmt(var, value, AsStmt(stmts)));
}

AssertFrame Assert(PrimExpr condition, PrimExpr message) {
  ObjectPtr<AssertFrameNode> n = make_object<AssertFrameNode>();
  n->condition = condition;
  n->message = message;
  return AssertFrame(n);
}

LetFrame Let(tvm::tir::Var var, PrimExpr value) {
  ObjectPtr<LetFrameNode> n = make_object<LetFrameNode>();
  n->var = var;
  n->value = value;
  return LetFrame(n);
}

TVM_REGISTER_NODE_TYPE(AssertFrameNode);
TVM_REGISTER_NODE_TYPE(LetFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.AssertFrame").set_body_typed(Assert);
TVM_REGISTER_GLOBAL("script.builder.tir.LetFrame").set_body_typed(Let);
}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

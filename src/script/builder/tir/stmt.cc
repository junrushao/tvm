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

#include "./var.h"

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

void AllocateFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  Buffer flattened_buffer = buffer.GetFlattenedBuffer();
  AddToParent(Allocate(buffer->data, flattened_buffer->dtype, flattened_buffer->shape, condition,
                       AsStmt(stmts), annotations));
}

void AllocateConstFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(AllocateConst(buffer->data, dtype, extents, data_or_idx, AsStmt(stmts)));
}

void RealizeFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(AttrStmt(
      buffer_slice->buffer, "realize_scope", PrimExpr(),
      BufferRealize(buffer_slice->buffer, buffer_slice->region, condition, AsStmt(stmts))));
}

void AttrFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(AttrStmt(node, attr_key, value, AsStmt(stmts)));
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

AllocateFrame Allocate_(Array<PrimExpr> extents, DataType dtype, String storage_scope_str,
                        PrimExpr condition, Map<String, ObjectRef> annotations) {
  ObjectPtr<AllocateFrameNode> n = make_object<AllocateFrameNode>();
  n->extents = extents;
  n->dtype = dtype;
  n->storage_scope_str = storage_scope_str;
  n->condition = condition;
  n->annotations = annotations;
  n->buffer = DeclBuffer(extents, dtype, "", NullOpt, {}, PrimExpr(), storage_scope_str, 0, 0,
                         "default", {}, Span());
  return AllocateFrame(n);
}

AllocateConstFrame AllocateConst_(ObjectRef data_or_idx, DataType dtype, Array<PrimExpr> extents) {
  ObjectPtr<AllocateConstFrameNode> n = make_object<AllocateConstFrameNode>();
  n->dtype = dtype;
  n->extents = extents;
  n->data_or_idx = data_or_idx;
  n->buffer =
      DeclBuffer(extents, dtype, "", NullOpt, {}, PrimExpr(), "", 0, 0, "default", {}, Span());
  return AllocateConstFrame(n);
}

RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope_str,
                     PrimExpr condition) {
  ObjectPtr<RealizeFrameNode> n = make_object<RealizeFrameNode>();
  n->buffer_slice = buffer_slice;
  n->storage_scope_str = storage_scope_str;
  n->condition = condition;
  return RealizeFrame(n);
}

AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value) {
  ObjectPtr<AttrFrameNode> n = make_object<AttrFrameNode>();
  n->node = node;
  n->attr_key = attr_key;
  n->value = value;
  return AttrFrame(n);
}

TVM_REGISTER_NODE_TYPE(AssertFrameNode);
TVM_REGISTER_NODE_TYPE(LetFrameNode);
TVM_REGISTER_NODE_TYPE(AllocateFrameNode);
TVM_REGISTER_NODE_TYPE(AllocateConstFrameNode);
TVM_REGISTER_NODE_TYPE(RealizeFrameNode);
TVM_REGISTER_NODE_TYPE(AttrFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.AssertFrame").set_body_typed(Assert);
TVM_REGISTER_GLOBAL("script.builder.tir.LetFrame").set_body_typed(Let);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocateFrame").set_body_typed(Allocate_);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocateConstFrame").set_body_typed(AllocateConst_);
TVM_REGISTER_GLOBAL("script.builder.tir.RealizeFrame").set_body_typed(Realize);
TVM_REGISTER_GLOBAL("script.builder.tir.AttrFrame").set_body_typed(Attr);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

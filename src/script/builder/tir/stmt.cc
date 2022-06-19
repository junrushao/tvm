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
#include <tvm/tir/op.h>

#include "./prim_func_frame.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

void AssertFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AssertStmt(condition, message, AsStmt(stmts)));
}

void LetFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::LetStmt(var, value, AsStmt(stmts)));
}

void AllocateFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  Buffer flattened_buffer = buffer.GetFlattenedBuffer();
  AddToParent(Allocate(buffer->data, flattened_buffer->dtype, flattened_buffer->shape, condition,
                       AsStmt(stmts), annotations));
}

void AllocateConstFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AllocateConst(buffer->data, dtype, extents, data, AsStmt(stmts)));
}

void LaunchThreadFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(env_var, attr_key, extent, AsStmt(stmts)));
}

void RealizeFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(AttrStmt(
      buffer_slice->buffer, "realize_scope", StringImm(storage_scope_str),
      BufferRealize(buffer_slice->buffer, buffer_slice->region, condition, AsStmt(stmts))));
}

void AttrFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(node, attr_key, value, AsStmt(stmts)));
}

void WhileFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::While(condition, AsStmt(stmts)));
}

AssertFrame Assert(PrimExpr condition, String message) {
  ObjectPtr<AssertFrameNode> n = make_object<AssertFrameNode>();
  n->condition = condition;
  n->message = tvm::tir::StringImm(message);
  return AssertFrame(n);
}

LetFrame Let(tvm::tir::Var var, PrimExpr value) {
  ObjectPtr<LetFrameNode> n = make_object<LetFrameNode>();
  n->var = var;
  n->value = value;
  return LetFrame(n);
}

AllocateFrame Allocate_(Array<PrimExpr> extents, DataType dtype, String storage_scope_str,
                        PrimExpr condition, Optional<Map<String, ObjectRef>> annotations) {
  ObjectPtr<AllocateFrameNode> n = make_object<AllocateFrameNode>();
  n->extents = extents;
  n->dtype = dtype;
  n->storage_scope_str = storage_scope_str;
  n->condition = condition->dtype.is_bool() ? condition : tvm::cast(DataType::Bool(), condition);
  n->annotations = annotations.value_or(Map<String, ObjectRef>());
  n->buffer = DeclBuffer(extents, dtype, "", NullOpt, {}, PrimExpr(), storage_scope_str, 0, 0,
                         "default", {}, Span());
  return AllocateFrame(n);
}

AllocateConstFrame AllocateConst_(tvm::runtime::NDArray data, DataType dtype,
                                  Array<PrimExpr> extents) {
  ObjectPtr<AllocateConstFrameNode> n = make_object<AllocateConstFrameNode>();
  n->dtype = dtype;
  n->extents = extents;
  n->data = data;
  n->buffer =
      DeclBuffer(extents, dtype, "", NullOpt, {}, PrimExpr(), "", 0, 0, "default", {}, Span());
  return AllocateConstFrame(n);
}

LaunchThreadFrame LaunchThread(tvm::tir::IterVar env_var, PrimExpr extent) {
  ObjectPtr<LaunchThreadFrameNode> n = make_object<LaunchThreadFrameNode>();
  n->env_var =
      tvm::tir::IterVar(Range(0, extent), env_var->var, env_var->iter_type, env_var->thread_tag);
  n->extent = extent;
  n->attr_key = env_var->thread_tag == "vthread" ? "virtual_thread" : "thread_extent";
  return LaunchThreadFrame(n);
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

WhileFrame While_(PrimExpr condition) {
  ObjectPtr<WhileFrameNode> n = make_object<WhileFrameNode>();
  n->condition = condition;
  return WhileFrame(n);
}

tvm::tir::IterVar EnvThread(String thread_tag) {
  using namespace tvm::tir;
  PrimFuncFrame frame = FindPrimFuncFrame("T.env_thread");
  return IterVar(Range(0, 0), Var("", DataType::Int(32)), IterVarType::kThreadIndex, thread_tag);
}

void BufferStore_(tvm::tir::Buffer buffer, PrimExpr value, Array<PrimExpr> indices) {
  AddToParent(tvm::tir::BufferStore(buffer, value, indices));
}

void Prefetch_(tvm::tir::Buffer buffer, Array<Range> bounds) {
  AddToParent(tvm::tir::Prefetch(buffer, bounds));
}

void Seq(Array<tvm::tir::Stmt> seq) { AddToParent(tvm::tir::SeqStmt(seq)); }

void IfThenElse_(PrimExpr condition, tvm::tir::Stmt then_case, tvm::tir::Stmt else_case) {
  AddToParent(tvm::tir::IfThenElse(condition, then_case, else_case));
}

void Evaluate_(PrimExpr value) { AddToParent(tvm::tir::Evaluate(value)); }

TVM_REGISTER_NODE_TYPE(AssertFrameNode);
TVM_REGISTER_NODE_TYPE(LetFrameNode);
TVM_REGISTER_NODE_TYPE(AllocateFrameNode);
TVM_REGISTER_NODE_TYPE(AllocateConstFrameNode);
TVM_REGISTER_NODE_TYPE(LaunchThreadFrameNode);
TVM_REGISTER_NODE_TYPE(RealizeFrameNode);
TVM_REGISTER_NODE_TYPE(AttrFrameNode);
TVM_REGISTER_NODE_TYPE(WhileFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.AssertFrame").set_body_typed(Assert);
TVM_REGISTER_GLOBAL("script.builder.tir.LetFrame").set_body_typed(Let);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocateFrame").set_body_typed(Allocate_);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocateConstFrame").set_body_typed(AllocateConst_);
TVM_REGISTER_GLOBAL("script.builder.tir.RealizeFrame").set_body_typed(Realize);
TVM_REGISTER_GLOBAL("script.builder.tir.AttrFrame").set_body_typed(Attr);
TVM_REGISTER_GLOBAL("script.builder.tir.WhileFrame").set_body_typed(While_);
TVM_REGISTER_GLOBAL("script.builder.tir.LaunchThreadFrame").set_body_typed(LaunchThread);
TVM_REGISTER_GLOBAL("script.builder.tir.EnvThread").set_body_typed(EnvThread);
TVM_REGISTER_GLOBAL("script.builder.tir.BufferStore").set_body_typed(BufferStore_);
TVM_REGISTER_GLOBAL("script.builder.tir.Prefetch").set_body_typed(Prefetch_);
TVM_REGISTER_GLOBAL("script.builder.tir.Seq").set_body_typed(Seq);
TVM_REGISTER_GLOBAL("script.builder.tir.IfThenElse").set_body_typed(IfThenElse_);
TVM_REGISTER_GLOBAL("script.builder.tir.Evaluate").set_body_typed(Evaluate_);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

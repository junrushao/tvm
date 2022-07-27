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

#include "./prim_func_frame.h"

#include <tvm/tir/function.h>

#include "../../../tir/ir/script/script_complete.h"
#include "../ir/ir.h"
#include "./utils.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

PrimFuncFrame FindPrimFuncFrame(const String& method) {
  Builder builder = Builder::Current();
  if (Optional<PrimFuncFrame> prim_func_frame = builder->GetLastFrame<PrimFuncFrame>()) {
    return prim_func_frame.value();
  }
  LOG(FATAL) << "ValueError: PrimFunc frame not find. Please ensure '" << method
             << "' is called under T.prim_func()";
  throw;
}

void PrimFuncFrameNode::ExitWithScope() {
  using ir::IRModuleFrame;
  TIRFrameNode::ExitWithScope();
  tvm::tir::PrimFunc func(/*params=*/args,
                          /*body=*/AsStmt(stmts),
                          /*ret_type=*/ret_type.value_or(TupleType::Empty()),
                          /*buffer_map=*/buffer_map,
                          /*preflattened_buffer_map=*/preflattened_buffer_map,
                          /*attrs=*/attrs.empty() ? NullValue<DictAttrs>() : DictAttrs(attrs));
  func = tvm::tir::ScriptComplete(func, root_alloc_buffers);
  Builder builder = Builder::Current();
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = func;
  } else if (Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    IRModuleFrame frame = opt_frame.value();
    frame->global_vars.push_back(GlobalVar(name.value_or("")));
    frame->functions.push_back(func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert PrimFunc";
  }
}

PrimFuncFrame PrimFunc() {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = NullOpt;
  n->args.clear();
  n->ret_type = NullOpt;
  n->buffer_map.clear();
  n->preflattened_buffer_map.clear();
  n->attrs.clear();
  n->root_alloc_buffers.clear();
  return PrimFuncFrame(n);
}

tvm::tir::Var Arg(String name, tvm::tir::Var var) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  Namer::Name(var, name);
  frame->args.push_back(var);
  return var;
}

tvm::tir::Buffer Arg(String name, tvm::tir::Buffer buffer) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  Namer::Name(buffer, name);
  tvm::tir::Var handle(buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

void FuncName(String name) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_name");
  if (frame->name.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func name, previous one is " << frame->name.value();
  }
  frame->name = name;
}

void FuncAttrs(Map<String, ObjectRef> attrs) {
  using namespace tvm::tir;
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_attr");
  if (!frame->attrs.empty()) {
    LOG(FATAL) << "ValueError: Duplicate prim func annotations, previous one is " << frame->attrs;
  }
  frame->attrs = attrs;
}

tvm::Type FuncRet(tvm::Type ret_type) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.ret_type");
  if (frame->ret_type.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func return type, previous one is "
               << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
  return ret_type;
}

tvm::tir::Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape, DataType dtype,
                             Optional<tvm::tir::Var> data, Array<PrimExpr> strides,
                             PrimExpr elem_offset, String storage_scope, int align,
                             int offset_factor, String buffer_type_str,
                             Array<IntImm> axis_separators) {
  tvm::tir::Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope,
                                       align, offset_factor, buffer_type_str, axis_separators);
  if (const auto* var = param.as<tvm::tir::VarNode>()) {
    PrimFuncFrame frame = FindPrimFuncFrame("T.match_buffer");
    tvm::tir::Var v = GetRef<tvm::tir::Var>(var);
    for (auto const& arg : frame->args) {
      if (arg.same_as(v)) {
        frame->buffer_map.Set(v, buffer);
        return buffer;
      }
    }
    LOG(FATAL) << "ValueError: Can not bind non-input param to buffer.";
  } else if (const auto* buffer_load = param.as<tvm::tir::BufferLoadNode>()) {
    BlockFrame frame = FindBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(tvm::tir::MatchBufferRegion(
        buffer, BufferRegionFromLoad(GetRef<tvm::tir::BufferLoad>(buffer_load))));
  } else if (const auto* buffer_region = param.as<tvm::tir::BufferRegionNode>()) {
    BlockFrame frame = FindBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(
        tvm::tir::MatchBufferRegion(buffer, GetRef<tvm::tir::BufferRegion>(buffer_region)));
  } else {
    LOG(FATAL) << "ValueError: Unexpected type for TIR MatchBuffer.";
  }
  return buffer;
};

void PreflattenedBuffer(tvm::tir::Buffer postflattened_buffer, Array<PrimExpr> shape,
                        DataType dtype, Optional<tvm::tir::Var> data, Array<PrimExpr> strides,
                        PrimExpr elem_offset, String storage_scope, int align, int offset_factor,
                        String buffer_type_str, Array<IntImm> axis_separators) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.preflattened_buffer");
  for (auto const& p : frame->buffer_map) {
    if (p.second.same_as(postflattened_buffer)) {
      String buffer_name(postflattened_buffer->name + "_preflatten");
      tvm::tir::Buffer buffer =
          BufferDecl(shape, dtype, buffer_name, data.value_or(p.second->data), strides, elem_offset,
                     storage_scope, align, offset_factor, buffer_type_str, axis_separators);
      Namer::Name(buffer, buffer_name);
      frame->preflattened_buffer_map.Set(p.first, buffer);
      return;
    }
  }
  LOG(FATAL) << "ValueError: postflattened buffer " << postflattened_buffer->name
             << " does not exist.";
};

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.PrimFunc").set_body_typed(PrimFunc);
TVM_REGISTER_GLOBAL("script.builder.tir.Arg")
    .set_body_typed([](String name, ObjectRef obj) -> ObjectRef {
      using namespace tvm::tir;
      if (const auto* var = obj.as<VarNode>()) {
        return Arg(name, GetRef<tvm::tir::Var>(var));
      }
      if (const auto* buffer = obj.as<BufferNode>()) {
        return Arg(name, GetRef<Buffer>(buffer));
      }
      LOG(FATAL) << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("script.builder.tir.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.builder.tir.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.builder.tir.FuncRet").set_body_typed(FuncRet);
TVM_REGISTER_GLOBAL("script.builder.tir.MatchBuffer").set_body_typed(MatchBuffer);
TVM_REGISTER_GLOBAL("script.builder.tir.PreflattenedBuffer").set_body_typed(PreflattenedBuffer);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

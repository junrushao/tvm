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

namespace tvm {
namespace script {
namespace builder {
namespace tir {

void PrimFuncFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  Builder builder = Builder::Current();
  PrimFunc func(/*params=*/args,
                /*body=*/AsStmt(stmts),
                /*ret_type=*/ret_type,
                /*buffer_map=*/buffer_map);
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = func;
  } else if (Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    IRModuleFrame frame = opt_frame.value();
    frame->global_vars.push_back(GlobalVar(name));
    frame->functions.push_back(func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert PrimFunc";
  }
}

PrimFuncFrame PrimFunc_(String name) {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = name;
  n->args.clear();
  n->ret_type = TupleType::Empty();
  n->buffer_map.clear();
  return PrimFuncFrame(n);
}

tvm::tir::Var Arg(String name, tvm::tir::Var var) {
  Namer::Name(var, name);
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  frame->args.push_back(var);
  return var;
}

tvm::tir::Buffer Arg(String name, tvm::tir::Buffer buffer) {
  using namespace tvm::tir;
  Namer::Name(buffer, name);
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  Var handle(buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);

TVM_REGISTER_GLOBAL("script.builder.tir.PrimFuncFrame")
  .set_body_typed([](String name){
    return PrimFunc_(name);
  });

TVM_REGISTER_GLOBAL("script.builder.tir.ExitPrimFuncFrame")
  .set_body_method<PrimFuncFrame>(&PrimFuncFrameNode::ExitWithScope);

TVM_REGISTER_GLOBAL("script.builder.tir.ArgVar")
  .set_body_typed([](tvm::tir::Var var){
    Arg(var);
  });

TVM_REGISTER_GLOBAL("script.builder.tir.ArgBuffer")
  .set_body_typed([](tvm::tir::Buffer buffer){
    Arg(buffer);
  });

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

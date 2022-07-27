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

#include <tvm/runtime/registry.h>

#include "./for_frame.h"
#include "./prim_func_frame.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

BlockFrame Block(String name, bool no_realize) {
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
  n->no_realize = no_realize;
  return BlockFrame(n);
}

void BlockFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  Array<tvm::tir::Buffer> tir_alloc_buffers;
  for (const tvm::tir::Buffer& buffer : alloc_buffers) {
    tir_alloc_buffers.push_back(buffer);
  }
  if (int detect_access = (!reads.defined()) | (!writes.defined() << 1)) {
    annotations.Set("tir.script_parsing_detect_access",
                    tvm::IntImm(DataType::Int(64), detect_access));
  }
  tvm::tir::Block block(iter_vars, reads.value_or(Array<tvm::tir::BufferRegion>()),
                        writes.value_or(Array<tvm::tir::BufferRegion>()), name, AsStmt(stmts), init,
                        tir_alloc_buffers, match_buffers, annotations);
  if (no_realize) {
    CHECK(iter_values.empty())
        << "ValueError: Block bindings are not allowed when `no_realize=True`";
    CHECK(!predicate.defined()) << "ValueError: `T.where` is not allowed when `no_realize=True`";
    AddToParent(block);
  } else {
    AddToParent(tvm::tir::BlockRealize(iter_values, predicate.value_or(Bool(true)), block));
  }
}

BlockInitFrame Init() {
  ObjectPtr<BlockInitFrameNode> n = make_object<BlockInitFrameNode>();
  return BlockInitFrame(n);
}

void BlockInitFrameNode::EnterWithScope() {
  BlockFrame frame = FindBlockFrame("T.init");
  if (frame->init.defined()) {
    LOG(FATAL) << "ValueError: Duplicate block init declaration";
  }
  TIRFrameNode::EnterWithScope();
}

void BlockInitFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  BlockFrame frame = FindBlockFrame("T.init");
  frame->init = AsStmt(stmts);
}

BlockFrame FindBlockFrame(const String& method) {
  if (Optional<BlockFrame> block_frame = Builder::Current()->GetLastFrame<BlockFrame>()) {
    return block_frame.value();
  }
  LOG(FATAL) << "ValueError: Block frame not find. Please ensure '" << method
             << "' is called under T.block()";
  throw;
}

void Where(PrimExpr predicate) {
  BlockFrame frame = FindBlockFrame("T.where");
  if (frame->predicate.defined()) {
    LOG(FATAL) << "ValueError: Duplicate block predicate declaration, previous one is "
               << frame->predicate.value();
  }
  frame->predicate = predicate;
}

void Reads(Array<ObjectRef> buffer_slices) {
  using namespace tvm::tir;
  BlockFrame frame = FindBlockFrame("T.reads");
  if (frame->reads.defined()) {
    LOG(FATAL) << "ValueError: Duplicate read region declaration, previous one is " << frame->reads;
  }
  Array<tvm::tir::BufferRegion> reads;
  for (const ObjectRef& obj : buffer_slices) {
    if (const auto* buffer_region = obj.as<BufferRegionNode>()) {
      reads.push_back(GetRef<BufferRegion>(buffer_region));
    } else if (const auto* buffer_load = obj.as<BufferLoadNode>()) {
      reads.push_back(BufferRegionFromLoad(GetRef<BufferLoad>(buffer_load)));
    } else {
      LOG(FATAL) << "Invalid type for buffer reads.";
    }
  }
  frame->reads = reads;
}

void Writes(Array<ObjectRef> buffer_slices) {
  using namespace tvm::tir;
  BlockFrame frame = FindBlockFrame("T.writes");
  if (frame->writes.defined()) {
    LOG(FATAL) << "ValueError: Duplicate write region declaration, previous one is "
               << frame->writes;
  }
  Array<tvm::tir::BufferRegion> writes;
  for (const ObjectRef& obj : buffer_slices) {
    if (const auto* buffer_region = obj.as<BufferRegionNode>()) {
      writes.push_back(GetRef<BufferRegion>(buffer_region));
    } else if (const auto* buffer_load = obj.as<BufferLoadNode>()) {
      writes.push_back(BufferRegionFromLoad(GetRef<BufferLoad>(buffer_load)));
    } else {
      LOG(FATAL) << "Invalid type for buffer writes.";
    }
  }
  frame->writes = writes;
}

void BlockAttrs(Map<String, ObjectRef> attrs) {
  BlockFrame frame = FindBlockFrame("T.block_attr");
  if (!frame->annotations.empty()) {
    LOG(FATAL) << "ValueError: Duplicate block annotations, previous one is " << frame->annotations;
  }
  frame->annotations = attrs;
}

tvm::tir::Buffer AllocBuffer(Array<PrimExpr> shape, DataType dtype, Optional<tvm::tir::Var> data,
                             Array<PrimExpr> strides, PrimExpr elem_offset, String storage_scope,
                             int align, int offset_factor, String buffer_type_str,
                             Array<IntImm> axis_separators) {
  tvm::tir::Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope,
                                       align, offset_factor, buffer_type_str, axis_separators);
  if (Optional<BlockFrame> block_frame = Builder::Current()->GetLastFrame<BlockFrame>()) {
    block_frame.value()->alloc_buffers.push_back(buffer);
  } else if (Optional<PrimFuncFrame> prim_func_frame =
                 Builder::Current()->GetLastFrame<PrimFuncFrame>()) {
    prim_func_frame.value()->root_alloc_buffers.push_back(buffer);
  } else {
    LOG(FATAL) << "ValueError: Block frame or PrimFunc frame not find. Please ensure "
                  "'T.alloc_buffer' is called under T.block() or T.prim_func()";
  }
  return buffer;
};

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

#define TVM_SCRIPT_BUILDER_TIR_AXIS_CREATE(Method, Kind, Name)                                \
  tvm::tir::IterVar Method(Range dom, PrimExpr binding, DataType dtype) {                     \
    using namespace tvm::tir;                                                                 \
    ICHECK(dom.defined()) << Name << " axis must have a domain";                              \
    int bits = std::max({dom->min.dtype().bits(), dom->extent.dtype().bits(), dtype.bits()}); \
    return PushBlockVar(IterVar(/*dom=*/dom, /*var=*/Var("_", dtype.with_bits(bits)),         \
                                /*iter_type=*/Kind, /*thread_tag=*/""),                       \
                        binding);                                                             \
  }
TVM_SCRIPT_BUILDER_TIR_AXIS_CREATE(Spatial, IterVarType::kDataPar, "Spatial");
TVM_SCRIPT_BUILDER_TIR_AXIS_CREATE(Reduce, IterVarType::kCommReduce, "Reduction");
TVM_SCRIPT_BUILDER_TIR_AXIS_CREATE(Scan, IterVarType::kOrdered, "Scan");
TVM_SCRIPT_BUILDER_TIR_AXIS_CREATE(Opaque, IterVarType::kOpaque, "Opaque");
#undef TVM_SCRIPT_BUILDER_TIR_AXIS_CREATE

Array<tvm::tir::IterVar> Remap(String kinds, Array<PrimExpr> bindings, DataType dtype) {
  using namespace tvm::tir;
  Array<IterVar> results;
  ICHECK_EQ(kinds.size(), bindings.size());
  int n = bindings.size();
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    char c = kinds.c_str()[i];
    PrimExpr e = bindings[i];
    const VarNode* v = e.as<VarNode>();
    ICHECK(v) << "TypeError: Only Var is supported in T.axis.remap";
    Range dom{nullptr};
    for (const auto& frame : Builder::Current()->frames) {
      if (const auto* for_frame = frame.as<ForFrameNode>()) {
        ICHECK_EQ(for_frame->doms.size(), for_frame->vars.size());
        int n = for_frame->doms.size();
        for (int i = 0; i < n; ++i) {
          if (for_frame->vars[i].get() == v) {
            dom = for_frame->doms[i];
            break;
          }
        }
        if (dom.defined()) {
          break;
        }
      }
    }
    ICHECK(dom.defined()) << "TypeError: Variable is not in the loop: " << GetRef<Var>(v);
    DataType dtype = v->dtype;
    if (c == 'S') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/Var("_", dtype),
                                             /*iter_type=*/IterVarType::kDataPar,
                                             /*thread_tag=*/""),
                                     e));
    } else if (c == 'R') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/Var("_", dtype),
                                             /*iter_type=*/IterVarType::kCommReduce,
                                             /*thread_tag=*/""),
                                     e));
    } else {
      LOG(FATAL) << "Unknown axis kind: " << c;
    }
  }
  return results;
}

}  // namespace axis

TVM_REGISTER_NODE_TYPE(BlockFrameNode);
TVM_REGISTER_NODE_TYPE(BlockInitFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.BlockFrame").set_body_typed(Block);
TVM_REGISTER_GLOBAL("script.builder.tir.BlockInitFrame").set_body_typed(Init);
TVM_REGISTER_GLOBAL("script.builder.tir.Where").set_body_typed(Where);
TVM_REGISTER_GLOBAL("script.builder.tir.Reads").set_body_typed(Reads);
TVM_REGISTER_GLOBAL("script.builder.tir.Writes").set_body_typed(Writes);
TVM_REGISTER_GLOBAL("script.builder.tir.BlockAttrs").set_body_typed(BlockAttrs);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocBuffer").set_body_typed(AllocBuffer);
TVM_REGISTER_GLOBAL("script.builder.tir.AxisSpatial").set_body_typed(axis::Spatial);
TVM_REGISTER_GLOBAL("script.builder.tir.AxisReduce").set_body_typed(axis::Reduce);
TVM_REGISTER_GLOBAL("script.builder.tir.AxisScan").set_body_typed(axis::Scan);
TVM_REGISTER_GLOBAL("script.builder.tir.AxisOpaque").set_body_typed(axis::Opaque);
TVM_REGISTER_GLOBAL("script.builder.tir.AxisRemap").set_body_typed(axis::Remap);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

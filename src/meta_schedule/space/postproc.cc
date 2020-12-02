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
#include "./postproc.h"  // NOLINT(build/include)

#include <tvm/tir/transform.h>

#include "../analysis.h"

namespace tvm {
namespace meta_schedule {

/********** Constructor **********/

Postproc::Postproc(String name, FProc proc) {
  ObjectPtr<PostprocNode> n = make_object<PostprocNode>();
  n->name = std::move(name);
  n->proc_ = std::move(proc);
  data_ = std::move(n);
}

/********** Postproc **********/

bool PostprocNode::Apply(const SearchTask& task, const Schedule& sch, Sampler* sampler) {
  return proc_(task, sch, sampler);
}

/********** Utility helpers **********/

Optional<String> GetAnn(const tir::StmtSRef& sref, const String& ann_key) {
  const Array<tir::Annotation>* annotations;
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  for (const tir::Annotation& ann : *annotations) {
    if (ann->attr_key == ann_key) {
      if (const auto* str_imm = ann->value.as<tir::StringImmNode>()) {
        return str_imm->value;
      }
    }
  }
  return NullOpt;
}

bool HasAnn(const tir::StmtSRef& loop_sref, const String& ann_key, const String& ann_val) {
  Optional<String> result = GetAnn(loop_sref, ann_key);
  return result.defined() && result.value() == ann_val;
}

void RemoveAnn(const tir::Schedule& sch, const tir::StmtSRef& sref, const String& ann_key) {
  // Extract annotation
  const Array<tir::Annotation>* annotations;
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Remove the annotation
  Array<tir::Annotation> new_ann;
  int n = annotations->size();
  new_ann.reserve(n - 1);
  for (int i = 0; i < n; ++i) {
    const tir::Annotation& ann = annotations->operator[](i);
    if (ann->attr_key != ann_key) {
      new_ann.push_back(ann);
    }
  }
  // Create the new stmt
  if (const auto* loop = sref->GetStmt<tir::LoopNode>()) {
    ObjectPtr<tir::LoopNode> n = make_object<tir::LoopNode>(*loop);
    n->annotations = new_ann;
    sch->Replace(sref, tir::Loop(n));
  } else if (const auto* block = sref->GetStmt<tir::BlockNode>()) {
    ObjectPtr<tir::BlockNode> n = make_object<tir::BlockNode>(*block);
    n->annotations.clear();
    tir::Block p(n);
    sch->Replace(sref, p, {{p, GetRef<tir::Block>(block)}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

PrimExpr GetLoopExtent(const tir::StmtSRef& loop_sref) {
  const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
  CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
  return loop->extent;
}

std::vector<tir::StmtSRef> CollectAllBlocks(const Schedule& sch) {
  std::vector<tir::StmtSRef> all_blocks;
  const auto* root_block = sch->sch->root->GetStmt<tir::BlockNode>();
  CHECK(root_block) << "TypeError: Expects Block, but gets: " << root_block;
  tir::PreOrderVisit(root_block->body, [&all_blocks, &sch](const ObjectRef& obj) -> bool {
    if (const auto* block = obj.as<tir::BlockNode>()) {
      all_blocks.push_back(sch->sch->stmt2ref.at(block));
    }
    return true;
  });
  return all_blocks;
}

/********** RewriteParallel **********/

Postproc RewriteParallel() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    std::vector<tir::StmtSRef> all_blocks = CollectAllBlocks(sch);
    arith::Analyzer analyzer;
    for (const tir::StmtSRef& block_sref : all_blocks) {
      Array<tir::StmtSRef> loop_srefs = sch->sch->GetLoopsInScope(block_sref);
      std::vector<int> parallel_ids;
      {
        int i = 0;
        for (const tir::StmtSRef& loop_sref : loop_srefs) {
          if (HasAnn(loop_sref, tir::attr::loop_type, "lazy_parallel")) {
            parallel_ids.push_back(i);
          }
          ++i;
        }
      }
      if (parallel_ids.empty()) {
        continue;
      }
      for (int id : parallel_ids) {
        RemoveAnn(sch->sch, loop_srefs[id], tir::attr::loop_type);
      }
      Array<Integer> loop_types = GetLoopType(sch->sch, block_sref, loop_srefs);
      int n = parallel_ids.size();
      tir::StmtSRef fused{nullptr};
      for (int i = 0; i < n; ++i) {
        int loop_type = loop_types[i];
        if (loop_type != tir::IterVarType::kDataPar) {
          break;
        }
        if (fused.defined()) {
          fused = sch->sch->fuse(fused, loop_srefs[parallel_ids[i]]);
        } else {
          fused = loop_srefs[parallel_ids[i]];
        }
      }
      if (fused.defined() && !analyzer.CanProve(GetLoopExtent(fused) <= 1)) {
        sch->sch->parallel(fused);
      }
    }
    return true;
  };
  return Postproc("rewrite_parallel", f_proc);
}

/********** RewriteVectorize **********/

Postproc RewriteVectorize() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    std::vector<tir::StmtSRef> all_blocks = CollectAllBlocks(sch);
    arith::Analyzer analyzer;
    for (const tir::StmtSRef& block_sref : all_blocks) {
      Array<tir::StmtSRef> loop_srefs = sch->sch->GetLoopsInScope(block_sref);
      std::vector<int> vectorize_ids;
      {
        int i = 0;
        for (const tir::StmtSRef& loop_sref : loop_srefs) {
          if (HasAnn(loop_sref, tir::attr::loop_type, "lazy_vectorize")) {
            vectorize_ids.push_back(i);
          }
          ++i;
        }
      }
      if (vectorize_ids.empty()) {
        continue;
      }
      for (int id : vectorize_ids) {
        RemoveAnn(sch->sch, loop_srefs[id], tir::attr::loop_type);
      }
      Array<Integer> loop_types = GetLoopType(sch->sch, block_sref, loop_srefs);
      int n = vectorize_ids.size();
      tir::StmtSRef fused{nullptr};
      for (int i = n - 1; i >= 0; --i) {
        int loop_type = loop_types[i];
        if (loop_type != tir::IterVarType::kDataPar) {
          break;
        }
        if (fused.defined()) {
          fused = sch->sch->fuse(loop_srefs[vectorize_ids[i]], fused);
        } else {
          fused = loop_srefs[vectorize_ids[i]];
        }
      }
      if (fused.defined() && !analyzer.CanProve(GetLoopExtent(fused) <= 1)) {
        sch->sch->vectorize(fused);
      }
    }
    return true;
  };
  return Postproc("rewrite_vectorize", f_proc);
}

/********** RewriteTensorize **********/

class PostprocRewriteTensorize {
 public:
  Array<tir::TensorIntrin> tensor_intrins;

  explicit PostprocRewriteTensorize(Array<tir::TensorIntrin> tensor_intrins)
      : tensor_intrins(tensor_intrins) {}

  Optional<tir::StmtSRef> FindTensorized(const Schedule& sch) {
    Optional<tir::StmtSRef> result = NullOpt;
    tir::PreOrderVisit(sch->sch->func->body, [&result, &sch](const ObjectRef& obj) -> bool {
      if (const auto* block = obj.as<tir::BlockNode>()) {
        tir::StmtSRef block_sref = sch->sch->stmt2ref.at(block);
        if (HasAnn(block_sref, tir::attr::block_type, "lazy_tensorize")) {
          result = block_sref;
          return false;
        }
      }
      return true;
    });
    return result;
  }

  bool CanTensorize(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                    const tir::TensorIntrin& intrin) {
    Optional<TensorizeInfo> opt_tensorize_info =
        GetTensorizeLoopMapping(sch, block_sref, intrin->description);
    if (!opt_tensorize_info.defined()) {
      return false;
    }
    const auto* info = opt_tensorize_info.value().get();
    arith::Analyzer analyzer;
    for (const auto& kv : info->loop_map) {
      const tir::StmtSRef& block_loop_sref = kv.first;
      const auto* block_loop = block_loop_sref->GetStmt<tir::LoopNode>();
      const tir::Loop& desc_loop = kv.second;
      if (!analyzer.CanProve(block_loop->extent == desc_loop->extent)) {
        return false;
      }
    }
    return true;
  }

  bool Proc(const Schedule& sch) {
    while (Optional<tir::StmtSRef> opt_block_sref = FindTensorized(sch)) {
      tir::StmtSRef block_sref = opt_block_sref.value();
      // Remove the annotation
      RemoveAnn(sch->sch, block_sref, tir::attr::block_type);
      // Get the surrounding loops
      Array<tir::StmtSRef> loop_srefs = sch->sch->GetLoopsInScope(block_sref);
      // Decompose Reduction
      {
        const auto* block = block_sref->GetStmt<tir::BlockNode>();
        CHECK(block) << "TypeError: Expects BlockNode, but gets: "
                     << block_sref->stmt->GetTypeKey();
        if (block->body->IsInstance<tir::ReduceStepNode>()) {
          sch->sch->decompose_reduction(block_sref, loop_srefs[0]);
        }
      }
      // Tensorize
      for (const tir::TensorIntrin& intrin : tensor_intrins) {
        if (CanTensorize(sch->sch, block_sref, intrin)) {
          sch->sch->tensorize(loop_srefs[0], intrin);
          return true;
        }
      }
    }
    return false;
  }
};

Postproc RewriteTensorize(Array<tir::TensorIntrin> tensor_intrins) {
  auto f_proc = [tensor_intrins{std::move(tensor_intrins)}](SearchTask task, Schedule self,
                                                            void* _sampler) -> bool {
    return PostprocRewriteTensorize(tensor_intrins).Proc(self);
  };
  return Postproc("rewrite_tensorize", f_proc);
}

/********** RewriteCudaThreadBind **********/

inline tir::IterVar MakeThreadIdx(const String& name) {
  return tir::IterVar(Range(nullptr), tir::Var(name), tir::kThreadIndex, name);
}

class PostprocRewriteCudaThreadBind {
 public:
  bool BindMultiLevelTiled(const Schedule& sch, const BlockRV& block_rv, int warp_size) const {
    arith::Analyzer analyzer;
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    std::vector<tir::StmtSRef> loop_srefs;
    // The indices of `blockIdx.x` / `vthread` / `threadIdx.x` annotation
    std::vector<int> block_idx;
    std::vector<int> vthread_idx;
    std::vector<int> thread_idx;
    PrimExpr prod_extent = Integer(1);
    for (const LoopRV& loop_rv : loop_rvs) {
      int i = loop_srefs.size();
      // Evaluate to a TIR loop
      tir::StmtSRef loop_sref = sch->Eval(loop_rv);
      loop_srefs.push_back(loop_sref);
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
      // Check the annotation
      if (HasAnn(loop_sref, tir::attr::loop_type, "lazy_blockIdx.x")) {
        block_idx.push_back(i);
      } else if (HasAnn(loop_sref, tir::attr::loop_type, "lazy_vthread")) {
        vthread_idx.push_back(i);
      } else if (HasAnn(loop_sref, tir::attr::loop_type, "lazy_threadIdx.x")) {
        thread_idx.push_back(i);
      } else {
        continue;
      }
      prod_extent = prod_extent * loop->extent;
    }
    prod_extent = analyzer.Simplify(prod_extent);
    // Check if the block is annotated
    if (block_idx.empty() || vthread_idx.empty() || thread_idx.empty()) {
      return false;
    }
    // Check if `prod_extent <= warp_size`
    if (analyzer.CanProve(prod_extent <= warp_size)) {
      // If so, only bind `threadIdx.x`
      std::vector<int> indices;
      indices.insert(indices.end(), block_idx.begin(), block_idx.end());
      indices.insert(indices.end(), vthread_idx.begin(), vthread_idx.end());
      indices.insert(indices.end(), thread_idx.begin(), thread_idx.end());
      std::sort(indices.begin(), indices.end());
      // Remove the annotation on the loop
      for (int idx : indices) {
        RemoveAnn(sch->sch, loop_srefs[idx], tir::attr::loop_type);
      }
      // Do fusion
      std::vector<LoopRV> to_fuse;
      for (int idx : indices) {
        to_fuse.push_back(loop_rvs[idx]);
      }
      LoopRV fused = sch->Fuse(to_fuse);
      // Do binding
      sch->sch->bind(sch->Eval(fused), MakeThreadIdx("threadIdx.x"));
    } else {
      // Otherwise, bind `blockIdx.x`, `vthread` and `threadIdx.x`
      {
        // bind `blockIdx.x`
        // Remove the annotation on the loop
        for (int idx : block_idx) {
          RemoveAnn(sch->sch, loop_srefs[idx], tir::attr::loop_type);
        }
        // Do fusion
        std::vector<LoopRV> to_fuse;
        for (int idx : block_idx) {
          to_fuse.push_back(loop_rvs[idx]);
        }
        LoopRV fused = sch->Fuse(to_fuse);
        // Do binding
        sch->sch->bind(sch->Eval(fused), MakeThreadIdx("blockIdx.x"));
      }
      {
        // bind `vthread`
        // Remove the annotation on the loop
        for (int idx : vthread_idx) {
          RemoveAnn(sch->sch, loop_srefs[idx], tir::attr::loop_type);
        }
        // Do fusion
        std::vector<LoopRV> to_fuse;
        for (int idx : vthread_idx) {
          to_fuse.push_back(loop_rvs[idx]);
        }
        LoopRV fused = sch->Fuse(to_fuse);
        // Do binding
        sch->sch->bind(sch->Eval(fused), MakeThreadIdx("vthread"));
      }
      {
        // bind `threadIdx.x`
        // Remove the annotation on the loop
        for (int idx : thread_idx) {
          RemoveAnn(sch->sch, loop_srefs[idx], tir::attr::loop_type);
        }
        // Do fusion
        std::vector<LoopRV> to_fuse;
        for (int idx : thread_idx) {
          to_fuse.push_back(loop_rvs[idx]);
        }
        LoopRV fused = sch->Fuse(to_fuse);
        // Do binding
        sch->sch->bind(sch->Eval(fused), MakeThreadIdx("threadIdx.x"));
      }
    }
    return true;
  }

  bool BindSpatial(const Schedule& sch, const BlockRV& block_rv) const {
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    Array<tir::StmtSRef> loop_srefs;
    for (const LoopRV& loop_rv : loop_rvs) {
      tir::StmtSRef loop_sref = sch->Eval(loop_rv);
      loop_srefs.push_back(loop_sref);
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
      if (!loop->annotations.empty()) {
        return false;
      }
    }
    Array<Integer> loop_types = GetLoopType(sch->sch, block_sref, loop_srefs);
    int n_spatial = 0;
    for (const Integer& _loop_type : loop_types) {
      int loop_type = _loop_type;
      if (loop_type == tir::kDataPar) {
        ++n_spatial;
      } else {
        break;
      }
    }
    CHECK(n_spatial > 0) << "NotImplementedError: binding no spatial loops is not supported yet";
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + n_spatial});
    Array<LoopRV> splits = sch->Split(fused, {NullOpt, Integer(32)});
    CHECK_EQ(splits.size(), 2);
    sch->sch->bind(sch->Eval(splits[0]), MakeThreadIdx("blockIdx.x"));
    sch->sch->bind(sch->Eval(splits[1]), MakeThreadIdx("threadIdx.x"));
    return true;
  }

  void BindCooperativeFetch(const Schedule& sch, const tir::StmtSRef& block_sref) const {
    Array<tir::StmtSRef> axes = sch->sch->GetLoopsInScope(block_sref);
    for (const tir::StmtSRef& loop_sref : axes) {
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
      if (!HasAnn(loop_sref, tir::attr::loop_type, "lazy_cooperative_fetch")) {
        continue;
      }
      RemoveAnn(sch->sch, loop_sref, tir::attr::loop_type);
      tir::StmtSRef threadIdx_sref{nullptr};
      for (tir::StmtSRefNode* upper = loop_sref->parent; upper; upper = upper->parent) {
        tir::StmtSRef upper_sref = GetRef<tir::StmtSRef>(upper);
        if (HasAnn(upper_sref, tir::attr::loop_type, "threadIdx.x")) {
          threadIdx_sref = upper_sref;
          break;
        }
      }
      CHECK(threadIdx_sref.defined())
          << "ValueError: Cannot find 'threadIdx.x' above cooperative fetching";
      PrimExpr factor = GetLoopExtent(threadIdx_sref);
      PrimExpr nparts = floordiv(GetLoopExtent(loop_sref) + factor - 1, factor);
      Array<tir::StmtSRef> splits = sch->sch->split(loop_sref, nparts, factor);
      CHECK_EQ(splits.size(), 2);
      sch->sch->bind(splits[1], MakeThreadIdx("threadIdx.x"));
    }
  }

  bool Proc(const SearchTask& task, const Schedule& sch) const {
    int warp_size = task->target->GetAttr<Integer>("thread_warp_size").value_or(Integer(-1));
    CHECK(warp_size != -1) << "ValueError: Target does not have attribute \"thread_warp_size\"";
    Array<BlockRV> root_block_rvs = sch->GetRootBlocks();
    for (const BlockRV& block_rv : root_block_rvs) {
      if (BindMultiLevelTiled(sch, block_rv, warp_size)) {
        continue;
      }
      if (BindSpatial(sch, block_rv)) {
        continue;
      }
    }
    // TODO(@junrushao1994): replace with the utility function
    // Collect all the blocks
    std::vector<tir::StmtSRef> all_blocks;
    const auto* root_block = sch->sch->root->GetStmt<tir::BlockNode>();
    CHECK(root_block) << "TypeError: Expects Block, but gets: " << root_block;
    tir::PreOrderVisit(root_block->body, [&all_blocks, &sch](const ObjectRef& obj) -> bool {
      if (const auto* block = obj.as<tir::BlockNode>()) {
        all_blocks.push_back(sch->sch->stmt2ref.at(block));
      }
      return true;
    });
    for (const tir::StmtSRef& block_sref : all_blocks) {
      BindCooperativeFetch(sch, block_sref);
    }
    return true;
  }
};

Postproc RewriteCudaThreadBind() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteCudaThreadBind().Proc(task, sch);
  };
  return Postproc("rewrite_cuda_thread_bind", f_proc);
}

/********** RewriteAutoUnroll **********/

class PostprocRewriteAutoUnroll {
 public:
  bool Proc(SearchTask task, Schedule sch) {
    std::vector<tir::StmtSRef> all_blocks = CollectAllBlocks(sch);
    for (const tir::StmtSRef& block_sref : all_blocks) {
      int max_step = -1;
      int unroll_explicit = -1;
      if (Optional<String> unroll = GetAnn(block_sref, tir::attr::auto_unroll_explicit)) {
        max_step = std::atoi(unroll.value().operator std::string().c_str());
        unroll_explicit = 1;
        RemoveAnn(sch->sch, block_sref, tir::attr::auto_unroll_explicit);
      } else if (Optional<String> unroll = GetAnn(block_sref, tir::attr::auto_unroll_implicit)) {
        max_step = std::atoi(unroll.value().operator std::string().c_str());
        unroll_explicit = 0;
        RemoveAnn(sch->sch, block_sref, tir::attr::auto_unroll_implicit);
      } else {
        continue;
      }
      Array<tir::StmtSRef> loop_srefs = sch->sch->GetLoopsInScope(block_sref);
      if (loop_srefs.empty()) {
        continue;
      }
      tir::StmtSRef loop_sref = loop_srefs[0];
      sch->sch->pragma(loop_sref, "auto_unroll_max_step", Integer(max_step));
      sch->sch->pragma(loop_sref, "unroll_explicit", Integer(unroll_explicit));
    }
    return true;
  }
};

Postproc RewriteAutoUnroll() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteAutoUnroll().Proc(task, sch);
  };
  return Postproc("rewrite_auto_unroll", f_proc);
}

/********** VerifyGPUCode **********/

class PostprocVerifyGPUCode {
 public:
  static Integer Extract(const Target& target, const char* name) {
    if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
      return v.value();
    }
    LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
    throw;
  }

  static tir::transform::Sequential MakePasses(const Target& target) {
    return tir::transform::Sequential(
        {tir::transform::InjectPrefetch(),       //
         tir::transform::BufferFlatten(),        //
         tir::transform::NarrowDataType(32),     //
         tir::transform::Simplify(),             //
         tir::transform::VectorizeLoop(true),    //
         tir::transform::InjectVirtualThread(),  //
         tir::transform::StorageRewrite(),       //
         tir::transform::Simplify(),             //
         tir::transform::VerifyGPUCode({
             {"max_shared_memory_per_block", Extract(target, "shared_memory_per_block")},
             {"max_local_memory_per_block", Extract(target, "registers_per_block")},
             {"max_threads_per_block", Extract(target, "max_threads_per_block")},
             {"max_vector_bytes", Extract(target, "vector_unit_bytes")},
             {"max_vthread", Integer(8)},
         })});
  }

  bool Proc(const SearchTask& task, const Schedule& sch) const {
    tir::transform::Sequential passes = MakePasses(task->target);
    IRModule mod({{GlobalVar("main"), sch->sch->func}});
    try {
      passes(std::move(mod));
    } catch (const dmlc::Error& e) {
      return false;
    }
    return true;
  }
};

Postproc VerifyGPUCode() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocVerifyGPUCode().Proc(task, sch);
  };
  return Postproc("verify_gpu_code", f_proc);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief FFI function for PostProcNode::Apply
   * \sa PostProcNode::Apply
   */
  static bool Apply(Postproc self, SearchTask task, Schedule sch, Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->Apply(task, sch, &seeded);
  }
};

TVM_REGISTER_NODE_TYPE(PostprocNode);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.Apply").set_body_typed(Internal::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteParallel").set_body_typed(RewriteParallel);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteVectorize").set_body_typed(RewriteVectorize);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteTensorize").set_body_typed(RewriteTensorize);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteCudaThreadBind")
    .set_body_typed(RewriteCudaThreadBind);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.VerifyGPUCode").set_body_typed(VerifyGPUCode);

}  // namespace meta_schedule
}  // namespace tvm

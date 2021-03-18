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
#include "./utils.h"

namespace tvm {
namespace tir {

template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

/**************** Utility functions ****************/

/*!
 * \brief Set the `StmtSRefNode::seq_index` field for stmt
 * \param self The schedule class
 * \param stmt The statement, or the realize node of the statement whose sref to be set
 * \param seq_index The seq_index to be set
 */
void SetSeqIndex(ScheduleStateNode* self, const Stmt& stmt, int seq_index) {
  if (const auto* realize = stmt.as<BlockRealizeNode>()) {
    const BlockNode* block = realize->block.get();
    ICHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* block = stmt.as<BlockNode>()) {
    ICHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* loop = stmt.as<ForNode>()) {
    ICHECK(self->stmt2ref.count(loop));
    self->stmt2ref.at(loop)->seq_index = seq_index;
  } else {
    // do nothing
  }
}

/*!
 * \brief Update seq_index of the children of a SeqStmt
 * \param self The schedule class
 * \param seq_stmt The SeqStmt whose children needs updating
 */
void SetSeqIndex(ScheduleStateNode* self, const SeqStmtNode* seq_stmt) {
  int i = 0;
  for (const Stmt& stmt : seq_stmt->seq) {
    SetSeqIndex(self, stmt, i);
    ++i;
  }
}

/*!
 * \brief Update the sref information on the schedule class, as well as the statement of sref itself
 * More specifically, update
 *  `sref->stmt` to `new_stmt`
 *  `self->stmt2ref`, remove the old statement that sref points to, and add the new statement
 * \param self The schedule class to be updated
 * \param sref The sref to be updated
 * \param new_stmt The statement that replaces the statement inside the sref
 */
void UpdateSRef(ScheduleStateNode* self, StmtSRefNode* sref, const StmtNode* new_stmt) {
  ICHECK(new_stmt->IsInstance<BlockNode>() || new_stmt->IsInstance<ForNode>());
  const StmtNode* old_stmt = sref->stmt;
  ICHECK_NE(new_stmt, old_stmt);
  self->stmt2ref[new_stmt] = GetRef<StmtSRef>(sref);
  self->stmt2ref.erase(sref->stmt);
  sref->stmt = new_stmt;
}

/*!
 * \brief Get PrimFunc and GlobalVar that the root block belongs to
 * \param mod The IRModule
 * \param root_block The root block of the PrimFunc
 * \param result_g_var The result GlobalVar
 * \return The result PrimFunc where the root block belongs to
 */
const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var) {
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == root_block) {
          *result_g_var = g_var;
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the correpsonding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(root_block);
  throw;
}

/**************** Creation ****************/

/*! \brief A helper class to create a new ScheduleStateNode */
class StateCreator : private StmtVisitor {
 public:
  /*!
   * \brief The entry function
   * \param self The schedule state to be completed
   */
  static ObjectPtr<ScheduleStateNode> Create(IRModule mod, bool debug_mode) {
    ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
    ScheduleStateNode* self = n.get();
    // Set `n->mod`
    n->mod = std::move(mod);
    // Set `n->debug_mode`
    n->debug_mode = debug_mode;
    // Set `n->stmt2ref` and `n->block_scopes`
    StateCreator creator(self);
    for (const auto& kv : n->mod->functions) {
      const BaseFunc& base_func = kv.second;
      if (const auto* func = base_func.as<PrimFuncNode>()) {
        creator.VisitStmt(func->body);
      }
    }
    return n;
  }

 private:
  explicit StateCreator(ScheduleStateNode* self)
      : self_(self), srefs_{}, realizes_{}, block_frames_{} {
    block_frames_.emplace({});
  }

  /*!
   * \brief Add a new statement to the stack, which becomes the current scope
   * \param stmt A loop statement or a block statement
   */
  StmtSRef PushSRef(const StmtNode* stmt) {
    if (srefs_.empty()) {
      srefs_.push_back(
          StmtSRef(stmt,
                   /*parent=*/nullptr,
                   /*seq_index=*/-1));  // `seq_index` will be set properly in SetSeqIndex
    } else {
      StmtSRefNode* parent = srefs_.back().get();
      srefs_.push_back(
          StmtSRef(stmt, parent,
                   /*seq_index=*/-1));  // `seq_index` will be set properly in SetSeqIndex
    }
    return srefs_.back();
  }

  /*! \brief Pop the top of the scope and record it in stmt2ref map */
  StmtSRef PopSRef() {
    StmtSRef sref = std::move(srefs_.back());
    self_->stmt2ref[sref->stmt] = sref;
    srefs_.pop_back();
    return sref;
  }

  BlockInfo MakeBlockInfo() {
    // Calculate `BlockInfo::scope`
    BlockScope scope(std::move(block_frames_.back()));
    // Calculate `BlockInfo::affine_binding`
    int n = static_cast<int>(srefs_.size());
    Map<Var, Range> loop_var_ranges;
    for (int i = n - 1; i >= 0; --i) {
      const StmtSRef& sref = srefs_[i];
      if (const auto* loop = sref->StmtAs<ForNode>()) {
        loop_var_ranges.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      } else {
        break;
      }
    }
    bool affine_binding =
        ValidateBlockBinding(GetRef<BlockRealize>(realizes_.back()), loop_var_ranges);
    return BlockInfo(std::move(scope), affine_binding);
  }

  void VisitStmt_(const ForNode* loop) final {
    PushSRef(loop);
    VisitStmt(loop->body);
    PopSRef();
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    realizes_.push_back(realize);
    block_frames_.emplace_back();
    const BlockNode* block = realize->block.get();
    // Recursive visit
    PushSRef(block);
    VisitStmt(block->body);  // `stmt->init` is not visited
    StmtSRef sref = PopSRef();
    // Create BlockInfo for the block
    self_->block_info[sref] = MakeBlockInfo();
    // Update parent scope
    block_frames_.pop_back();
    block_frames_.back().push_back(sref);
    realizes_.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    // Set `seq_index` information for SeqStmtNode
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndex(self_, seq_stmt);
  }

  /*! \brief The result stmt2ref */
  ScheduleStateNode* self_;
  /*! \brief The stack frame used to indicate the current scope */
  std::vector<StmtSRef> srefs_;
  /*! \brief The BlockRealize in the ancestors */
  std::vector<const BlockRealizeNode*> realizes_;
  /*! \brief The stack frames of blocks in the DFS visit. */
  std::vector<Array<StmtSRef>> block_frames_;
};

/**************** Constructor ****************/

ScheduleState::ScheduleState(IRModule mod, bool debug_mode) {
  data_ = StateCreator::Create(mod, debug_mode);
}

ScheduleState::ScheduleState(PrimFunc func, bool debug_mode)
    : ScheduleState(IRModule({{GlobalVar("main"), func}}), debug_mode) {}

/**************** Replace ****************/

/*!
 * \brief Record the different sref reuse types in the replacement
 *
 * 1) Intact: the subtree appears as the same object on both `src_stmt` and `tgt_stmt`,
 * which, given the immutability of the IR, means the entire subtree is unchanged,
 * and we do not need to recurse into the subtree.
 *
 * 2) Loop/Block sref reuse: for two different objects (`src`, `tgt`),
 * which are both loops or both blocks,
 * there is correspondence between them,
 * which makes us to reuse the sref pointing to `src`, and changes it to point to `tgt`.
 */
struct ReuseInfo {
  /*! \brief Kind 1. Intact reuse */
  std::unordered_set<const StmtNode*> intact;
  /*! \brief Kind 2.1. Loop sref reuse */
  std::unordered_set<const VarNode*> loop_sref_possible_reuse;
  /*! \brief Kind 2.2. Block sref reuse.
   * Maps an old Block in `src_stmt` to a new block in `tgt_stmt`
   */
  std::unordered_map<const BlockNode*, const BlockNode*> block_sref_reuse;
};

/*!
 * \brief A helper visitor which collects two cases of sref reuses in the `tgt_stmt`:
 *
 * 1) Intact: the subtree represented by `intact` appears on both old and new IR.
 * Given the immutability of the IR, we can quickly decide that the entire subtree is unchanged,
 * which means we do not need to visit into the subtree of the old statement.
 *
 * 2) Reused block/loop: for two different objects (`src`, `tgt`),
 * which are both loops or both blocks,
 * and there is correspondence between them,
 * which makes us to reuse the sref pointing to `src`, and changes it to point to `tgt`,
 */
class ReuseCollector : public StmtVisitor {
 public:
  static ReuseInfo Collect(ScheduleStateNode* self, const Stmt& tgt_stmt,
                           const Map<Block, Block>& block_sref_reuse) {
    ReuseCollector collector(self);
    collector.VisitStmt(tgt_stmt);
    ReuseInfo result;
    result.intact = {collector.intact_.begin(), collector.intact_.end()};
    result.loop_sref_possible_reuse = {collector.loop_vars_.begin(), collector.loop_vars_.end()};
    for (const auto& kv : block_sref_reuse) {
      const Block& old_block = kv.first;
      const Block& new_block = kv.second;
      result.block_sref_reuse.emplace(old_block.get(), new_block.get());
    }
    return result;
  }

 private:
  explicit ReuseCollector(ScheduleStateNode* self) : self_(self) {}

  void VisitStmt_(const ForNode* op) final {
    if (self_->stmt2ref.count(op)) {
      intact_.push_back(op);
    } else {
      // Collect loop vars for detecting reuse of loop sref
      loop_vars_.push_back(op->loop_var.get());
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    if (self_->stmt2ref.count(op)) {
      intact_.push_back(op);
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  ScheduleStateNode* self_;
  std::vector<const StmtNode*> intact_;
  std::vector<const VarNode*> loop_vars_;
};

/*!
 * \brief A helper visitor which removes the stale srefs in the `src_stmt`
 * that are useless after the replacement.
 *
 * It uses the reuse information previously collected to
 * 1) delete those srefs that are not reused.
 * 2) return the sref objects that are loop/block sref reuses, but not intact reuses
 */
class SRefTreePruner : public StmtVisitor {
 public:
  /*!
   * \brief The entry function
   * \param self The schedule class
   * \param info The reuse info about loop reuses and intact reuse
   * \param src_stmt The `src_stmt` where stale srefs to be removed
   * \return Mapping from the reuse elements to reused srefs, more specifically:
   * 1) Loop reuse: maps a loop var to the reused sref
   * 2) Block reuse: maps a block stmt to the reused sref,
   * where the block comes from the subtree of `tgt_stmt`
   * 3) Intact reuse: not returned
   */
  static std::unordered_map<const Object*, StmtSRef> Prune(ScheduleStateNode* self,
                                                           const ReuseInfo& reuse_info,
                                                           const Stmt& src_stmt) {
    SRefTreePruner pruner(self, reuse_info);
    pruner.VisitStmt(src_stmt);
    return std::move(pruner.reused_srefs_);
  }

 private:
  explicit SRefTreePruner(ScheduleStateNode* self, const ReuseInfo& reuse_info)
      : self_(self), reuse_info_(reuse_info) {}

  void VisitStmt_(const ForNode* op) final {
    if (reuse_info_.intact.count(op)) {
      return;
    }
    auto it = self_->stmt2ref.find(op);
    ICHECK(it != self_->stmt2ref.end())
        << "IndexError: Cannot find correpsonding StmtSRef for the loop:\n"
        << GetRef<For>(op);
    StmtSRef& sref = it->second;
    // Detect reuse
    const VarNode* loop_var = op->loop_var.get();
    if (reuse_info_.loop_sref_possible_reuse.count(loop_var)) {
      // sref can be reused
      reused_srefs_.emplace(loop_var, std::move(sref));
    } else {
      sref->Reset(/*stmt=*/nullptr, /*parent=*/nullptr,
                  /*seq_index=*/-1);
    }
    // erase the statement
    self_->stmt2ref.erase(it);
    // detect recursively
    VisitStmt(op->body);
  }

  void VisitStmt_(const BlockNode* op) final {
    if (reuse_info_.intact.count(op)) {
      return;
    }
    auto it = self_->stmt2ref.find(op);
    ICHECK(it != self_->stmt2ref.end())
        << "IndexError: Cannot find correpsonding StmtSRef for the block:\n"
        << GetRef<Block>(op);
    StmtSRef& sref = it->second;
    // Detect reuse
    auto reuse_it = reuse_info_.block_sref_reuse.find(op);
    if (reuse_it != reuse_info_.block_sref_reuse.end()) {
      // sref can be reused
      reused_srefs_.emplace(reuse_it->second, std::move(sref));
    } else {
      sref->Reset(/*stmt=*/nullptr, /*parent=*/nullptr, /*seq_index=*/-1);
      self_->block_info.erase(sref);
    }
    // erase the statement
    self_->stmt2ref.erase(it);
    // detect recursively
    // op->init is omitted
    VisitStmt(op->body);
  }

  ScheduleStateNode* self_;
  const ReuseInfo& reuse_info_;
  /*!
   * \brief Reused srefs:
   * 1) loop var -> StmtSRef
   * 2) block stmt -> StmtSRef, where the block comes from the subtree of `tgt_stmt`
   */
  std::unordered_map<const Object*, StmtSRef> reused_srefs_;
};

/*!
 * \brief Update the sref in the `tgt_stmt` given the reuse information
 *
 * After being updated, in the `tgt_stmt` subtree,
 * 1) all `parent`s are correct
 * 2) all `seq_index`s are correct, except for the root
 * 3) all `stmt`s are correct, except for the root
 */
class SRefUpdater : public StmtVisitor {
 public:
  static void Update(ScheduleStateNode* self, StmtSRefNode* root_parent,
                     const std::unordered_map<const Object*, StmtSRef>& reused_srefs,
                     const Stmt& tgt_stmt) {
    SRefUpdater(self, root_parent, reused_srefs).VisitStmt(tgt_stmt);
  }

 private:
  explicit SRefUpdater(ScheduleStateNode* self, StmtSRefNode* root_parent,
                       const std::unordered_map<const Object*, StmtSRef>& reused_srefs)
      : self_(GetRef<ScheduleState>(self)), parents_{root_parent}, reused_srefs_(reused_srefs) {}

  void VisitStmt_(const ForNode* op) final {
    StmtSRef& sref = self_->stmt2ref[op];
    // Detect intact
    if (sref.defined()) {
      sref->parent = parents_.back();
      sref->seq_index = -1;  // `seq_index` will be set properly in SetSeqIndex
      return;
    }
    // Detect reuse
    auto it = reused_srefs_.find(op->loop_var.get());
    if (it != reused_srefs_.end()) {
      // Update `stmt2ref[op]` to `reused_srefs_[op->loop_var]`
      sref = it->second;
      sref->Reset(op, parents_.back(),
                  /*seq_index=*/-1);  // `seq_index` will be set properly in SetSeqIndex
    } else {
      sref = StmtSRef(op, parents_.back(),
                      /*seq_index=*/-1);  // `seq_index` will be set properly in SetSeqIndex
    }
    // Recursive visit
    parents_.push_back(sref.get());
    VisitStmt(op->body);
    parents_.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    StmtSRef& sref = self_->stmt2ref[op];
    // Detect intact
    if (sref.defined()) {
      sref->parent = parents_.back();
      sref->seq_index = -1;  // `seq_index` will be set properly in SetSeqIndex
      return;
    }
    // Detect reuse
    auto it = reused_srefs_.find(op);
    if (it != reused_srefs_.end()) {
      // Update `stmt2ref[op]` to `reused_srefs_[op]`
      sref = it->second;
      sref->Reset(op, parents_.back(),
                  /*seq_index=*/-1);  // `seq_index` will be set properly in SetSeqIndex
    } else {
      sref = StmtSRef(op, parents_.back(),
                      /*seq_index=*/-1);  // `seq_index` will be set properly in SetSeqIndex
    }
    // Recursive visit
    parents_.push_back(sref.get());
    VisitStmt(op->body);
    parents_.pop_back();
    // Additionally, need to update the scope because the block is changed
    // TODO: affine
    BlockScope scope = BlockScope(tir::GetChildBlocks(self_, sref));
    bool affine_binding = true;
    self_->block_info[sref] = BlockInfo(std::move(scope), affine_binding);
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndex(self_.get(), seq_stmt);
  }

  ScheduleState self_;
  std::vector<StmtSRefNode*> parents_;
  const std::unordered_map<const Object*, StmtSRef>& reused_srefs_;
};

/*!
 * \brief A helper that
 * 1) returns a new copy of `child_sref->parent->stmt`,
 * where `child_sref->stmt` is replaced with `child_tgt_stmt`.
 * 2) Then it points `child_sref` to `child_tgt_stmt`.
 * 3) Finally it makes the subtree of the srefs pointing to the returned AST correct,
 * except for the root.
 */
class ChildReplacer : private StmtMutator {
 public:
  static Stmt Mutate(const StmtNode* parent_stmt, const StmtNode* child_src_stmt,
                     const Stmt& child_tgt_stmt, int seq_index, bool allow_copy_on_write) {
    // Check the invariant
    ICHECK(child_src_stmt->IsInstance<BlockNode>() ||  //
           child_src_stmt->IsInstance<ForNode>());
    ICHECK(child_tgt_stmt->IsInstance<BlockNode>() ||  //
           child_tgt_stmt->IsInstance<ForNode>() ||    //
           child_tgt_stmt->IsInstance<BlockRealizeNode>());
    ChildReplacer replacer(child_src_stmt, child_tgt_stmt, seq_index);
    replacer.allow_copy_on_write_ = allow_copy_on_write;
    return replacer.CopyOnWriteAndVisit(parent_stmt);
  }

 private:
  explicit ChildReplacer(const StmtNode* src_stmt, const Stmt& tgt_stmt, int seq_index)
      : src_stmt_(src_stmt), tgt_stmt_(tgt_stmt), seq_index_(seq_index) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (stmt.get() == src_stmt_) {
      // if the statement matches the replace tgt_stmt
      // just return the tgt_stmt
      return tgt_stmt_;
    } else {
      return StmtMutator::VisitStmt(stmt);
    }
  }

  //  Skipping sibling blocks and loops other than `src_stmt_`
  Stmt VisitStmt_(const BlockNode* op) final { return GetRef<Stmt>(op); }
  Stmt VisitStmt_(const ForNode* op) final { return GetRef<Stmt>(op); }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    int i = this->seq_index_;
    int n = static_cast<int>(op->seq.size());
    if (0 <= i && i < n) {
      const Stmt& stmt = op->seq[i];
      Optional<Stmt> new_stmt = NullOpt;
      // `stmt` can be Loop or BlockRealize
      // `src_stmt` can be Loop or Block
      // so the match from `stmt` to `src_stmt` can be
      // 1) Loop -> Loop
      // 2) BlockRealize -> Block
      if (stmt.get() == this->src_stmt_) {
        // Case 1. src_stmt is Loop, stmt is Loop
        new_stmt = tgt_stmt_;
      } else if (const auto* realize = stmt.as<BlockRealizeNode>()) {
        // Case 2. stmt is BlockRealize, src_stmt is Block
        if (realize->block.get() == src_stmt_) {
          const auto* tgt_block = TVM_TYPE_AS(tgt_block, tgt_stmt_, BlockNode);
          ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
          new_realize->block = GetRef<Block>(tgt_block);
          new_stmt = BlockRealize(std::move(new_realize));
        }
      }
      // Move new_stmt to position i
      if (new_stmt.defined()) {
        ObjectPtr<SeqStmtNode> new_seq_stmt = CopyOnWrite(op);
        new_seq_stmt->seq.Set(i, new_stmt.value());
        return SeqStmt(std::move(new_seq_stmt));
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt CopyOnWriteAndVisit(const StmtNode* parent_stmt) {
    // Step 1. Copy-on-write the `parent_stmt` and extract its `body`,
    // where `body` means the body of either a block or a loop
    // Step 2. Mutate the `block/loop->body`, searching for `child_old_stmt`
    // and replace it with `child_tgt_stmt`
    if (parent_stmt->IsInstance<BlockNode>()) {
      auto* block = const_cast<BlockNode*>(static_cast<const BlockNode*>(parent_stmt));
      ObjectPtr<BlockNode> new_block = CopyOnWrite(block);
      new_block->body = this->VisitStmt(new_block->body);
      return Block(std::move(new_block));
    } else if (parent_stmt->IsInstance<ForNode>()) {
      auto* loop = const_cast<ForNode*>(static_cast<const ForNode*>(parent_stmt));
      ObjectPtr<ForNode> new_loop = CopyOnWrite(loop);
      new_loop->body = this->VisitStmt(new_loop->body);
      return For(std::move(new_loop));
    }
    LOG(FATAL) << "TypeError: Unexpected type: " << parent_stmt->GetTypeKey();
    throw;
  }

  const StmtNode* src_stmt_;
  const Stmt& tgt_stmt_;
  int seq_index_;
};

void ScheduleStateNode::Replace(const tir::StmtSRef& _src_sref, const Stmt& tgt_stmt,
                                const Map<Block, Block>& block_sref_reuse) {
  {
    const StmtNode* src_stmt = _src_sref->stmt;
    bool input_correct =
        (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<ForNode>()) ||
        (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<BlockRealizeNode>()) ||
        (src_stmt->IsInstance<BlockNode>() && tgt_stmt->IsInstance<BlockNode>());
    if (!input_correct) {
      LOG(FATAL) << "TypeError: src_stmt has type: " << src_stmt->GetTypeKey()
                 << ". tgt_stmt has type: " << tgt_stmt->GetTypeKey() << ".\nsrc_stmt:\n"
                 << GetRef<Stmt>(src_stmt) << "\ntgt_stmt:\n"
                 << tgt_stmt;
    }
  }
  // Rule out the case that no replacement happens
  if (_src_sref->stmt == tgt_stmt.get()) {
    return;
  }
  // Reset sref as a new sref so that its content won't be affected by subsequent changes
  StmtSRef src_sref(_src_sref->stmt, _src_sref->parent, _src_sref->seq_index);
  Stmt src_stmt = GetRef<Stmt>(src_sref->stmt);
  // Step 1. Create all the nodes needed for the new sref tree.
  //   The `SRefCreator` visits the AST `tgt_stmt`, creating new nodes along the way.
  //   It deals with 3 cases:
  //
  //   Case 1.1: Visiting a node already present in the old AST
  //     It means we can skip the entire subtree, leaving it untouched
  //     Mark those nodes as `intact`
  //   Case 1.2: Can somehow infer the node being visited is mutated from the old AST, including
  //     (a) The loop var appears in the old AST
  //     (b) The node is explicitly present in `block_sref_reuse`
  //     It means we need to retain and reuse the sref, so that those srefs users hold won't
  //     expire Mark those nodes as `reuse`
  //   Case 1.3: It is a completely new node
  //     Create a new node for it
  //
  // After this step
  // 1) all `parent`s are correct
  // 2) all `seq_index`s are correct, except for the root
  // 3) all `stmt`s are correct, except for the root
  {
    // Step 1.1. Collect info for different kinds of reuses
    // 1) intact
    // 2) loop/block reuse
    ReuseInfo reuse_info = ReuseCollector::Collect(this, tgt_stmt, block_sref_reuse);
    // Step 1.2. Collect loop/block reuse to their corresponding srefs
    // and remove those srefs in the `src_stmt` that are no longer used after replacement
    std::unordered_map<const Object*, StmtSRef> reused_srefs =
        SRefTreePruner::Prune(this, reuse_info, src_stmt);
    // Step 1.3. Update the sref tree, inserting newly created srefs and properly handle reused
    // srefs in `tgt_stmt`
    SRefUpdater::Update(this, /*root_parent=*/src_sref->parent, reused_srefs, tgt_stmt);
  }
  // Step 2. Set the ancestors' children properly
  //   Iteratively visit the ancestors, creating new ones whose `body`s are properly fixed.
  //   The visit stops when all the ancestors are uniquely referenced, i.e. can mutate inplace.
  //   Along the way, because we create a new ancestor path,
  //   we need to update those sref points from old ancestors to newly created ones
  // Variables:
  // 1) `num_copy_steps`. The maximum number of hops until we need to copy. To reach a node that
  // can be mutated in-place, it needs `num_copy_steps + 1` hops. 2) `need_module_copy`. If
  // true, need to mutate the PrimFunc and IRModule the sref belongs to. 3) `g_var` and
  // `g_func`. Indicate which GlobalVar and PrimFunc the sref corresponds to
  int num_copy_steps = -1;
  bool need_module_copy = false;
  const PrimFuncNode* g_func = nullptr;
  GlobalVar g_var;
  {
    int i = 0;
    const StmtSRefNode* p = src_sref.get();
    while (true) {
      if (!p->stmt->unique()) {
        num_copy_steps = i;
      }
      if (p->parent == nullptr) {
        break;
      }
      ++i;
      p = p->parent;
    }
    // Find `g_func` and `g_var` where the `src_sref` is in
    g_func = GetRootPrimFunc(this->mod, p->stmt, &g_var);
    need_module_copy = num_copy_steps == i ||             //
                       !this->mod.unique() ||             //
                       !this->mod->functions.unique() ||  //
                       !g_func->unique();
  }
  // Loop invariant:
  //
  // Before step `i`:
  // 1) `child_sref` is `src_sref` going up by `i` steps
  // 2) `child_tgt_stmt` is the subtree that `child_sref` should correspond to after replacement
  // 3) except for the subtree root, srefs that point to the subtree of `child_tgt_stmt` are
  // correct 4) for the subtree root of `child_tgt_stmt`, `child_sref` has not pointed to it yet
  // 5) `tgt_stmt` is of type Loop, Block or BlockRealize
  //
  // During step `i`:
  // 1) Create `parent_stmt` that corresponds to `child_sref->parent
  // 2) Point `child_sref` to `child_tgt_stmt`
  // 3) `tgt_stmt` is of type Loop or Block
  StmtSRefNode* child_sref = src_sref.get();
  Stmt child_tgt_stmt = std::move(tgt_stmt);
  for (int i = 0; (need_module_copy || i <= num_copy_steps) && child_sref->parent != nullptr; ++i) {
    bool can_directly_mutate_parent = !need_module_copy && i == num_copy_steps;
    // replacing `child_sref->stmt` to `child_tgt_stmt`.
    const StmtNode* parent_stmt = child_sref->parent->stmt;
    const StmtNode* child_src_stmt = child_sref->stmt;
    // Step 2.1. Link `child_sref` to `child_tgt_stmt`
    if (i == 0) {
      // As the invariance of SRefUpdater,
      // the `seq_index` of the root of `tgt_stmt` is set as -1,
      // which might be incorrect
      SetSeqIndex(this, child_tgt_stmt, child_sref->seq_index);
    } else {
      // Point `child_sref` to `child_tgt_stmt`
      UpdateSRef(this, child_sref, child_tgt_stmt.get());
    }
    // Step 2.2. Create `new_parent_stmt`, by mutating the body of `parent_stmt`,
    Stmt new_parent_stmt =
        ChildReplacer::Mutate(parent_stmt, child_src_stmt, child_tgt_stmt,
                              /*seq_index=*/child_sref->seq_index,
                              /*allow_copy_on_write=*/can_directly_mutate_parent);
    // Step 2.3. Go to next parent
    if (can_directly_mutate_parent) {
      // If the node can be directly mutated inplace,
      // then there is no need to update its parent and the function
      break;
    }
    child_tgt_stmt = std::move(new_parent_stmt);
    child_sref = child_sref->parent;
  }
  // Step 3. Handle the case that we mutate the root
  if (need_module_copy) {
    // From the loop invariant, upon exit, while its subtree is properly set,
    // `child_sref` is not properly to `child_tgt_stmt` yet.
    if (src_sref->parent != nullptr) {
      // Not replacing a root
      UpdateSRef(this, child_sref, child_tgt_stmt.get());
    }
    // Ensure the uniqueness of `this->mod` and `this->mod->functions`
    IRModuleNode* new_mod = this->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    // Move out the PrimFunc where the sref belong while ensuring uniqueness
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    ICHECK(ref_new_func.get() == g_func);
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    // If `g_func` was not unique, after the 3 lines above:
    //   `ref_new_func` points to a unique PrimFunc
    //   `g_func` points to the previous PrimFunc if it is not unique
    // If `g_func` was unique, after the 3 lines above:
    //   `ref_new_func` points to the same unique function that `g_func` points to
    // Update the body of the function the sref belongs to Assign
    const auto* realize = TVM_TYPE_AS(realize, g_func->body, BlockRealizeNode);
    // Make `child_tgt_stmt` the root block
    const auto* child_block = TVM_TYPE_AS(child_block, child_tgt_stmt, BlockNode);
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
    new_realize->block = GetRef<Block>(child_block);
    new_func->body = BlockRealize(std::move(new_realize));
    // Finally, move the `ref_new_func` back and update `this->mod`
    new_map->at(g_var) = std::move(ref_new_func);
    this->mod = GetRef<IRModule>(new_mod);
  }
  if (this->debug_mode) {
    VerifySRefTree(GetRef<ScheduleState>(this));
  }
}

/**************** Scope-related Property ****************/

bool IsDominant(const BlockScope& self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Cond 1. Block is the only writer to its outputs
  const SMap<Buffer, Array<StmtSRef>>& buffer_writers = self->buffer_writers;
  for (const BufferRegion& write_region : block->writes) {
    ICHECK(buffer_writers.count(write_region->buffer))
        << "InternalError: buffer \"" << write_region->buffer->name
        << "\" does not exist in the current scope, when querying block:\n"
        << GetRef<Block>(block);
    // Check if the buffer is only written once (by the given block)
    if (buffer_writers.at(write_region->buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Check if each reduction instance is valid. Particularly, check:
 * 1) Each iteration variable is either data parallel or reduction
 * 2) Indices used to access the output buffer are not related to or affected by reduction iteration
 * variables.
 * \param iter_vars Iteration variables of the reduction
 * \param output_buffer_indices Indices used to access the output buffer
 * \return A boolean indicating if the reduction instance is valid
 */
bool CheckReductionInstance(const Array<IterVar>& iter_vars,
                            const Array<PrimExpr>& output_buffer_indices) {
  std::unordered_set<const VarNode*> reduction_block_vars;
  reduction_block_vars.reserve(iter_vars.size());
  // Check 1. Each iter_var can only be data parallel or reduction
  for (const IterVar& iter_var : iter_vars) {
    IterVarType kind = iter_var->iter_type;
    if (kind != kDataPar && kind != kCommReduce) {
      return false;
    }
    if (kind == kCommReduce) {
      reduction_block_vars.insert(iter_var->var.get());
    }
  }
  // Check 2. Each reduction iter_var should not be used to index output buffer
  for (const PrimExpr& idx : output_buffer_indices) {
    if (ContainsVar(idx, reduction_block_vars)) {
      return false;
    }
  }
  return true;
}

BlockInfo ScheduleStateNode::GetBlockInfo(const StmtSRef& block_sref) const {
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  auto it = this->block_info.find(block_sref);
  CHECK(it != this->block_info.end())
      << "IndexError: Cannot find the corresponding BlockScope to the block sref:\n"
      << GetRef<Stmt>(block_sref->stmt);
  return it->second;
}

BlockScope ScheduleStateNode::GetBlockScope(const StmtSRef& block_sref) const {
  return GetBlockInfo(block_sref).scope;
}

bool ScheduleStateNode::IsAffineBlockBinding(const StmtSRef& block_sref) const {
  return GetBlockInfo(block_sref).affine_binding;
}

bool ScheduleStateNode::IsComplete(const StmtSRef& block_sref, const StmtSRef& scope_root) const {
  BlockScope scope = GetBlockScope(scope_root);
  // Cond 2. Check if all the block vars are data parallel
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }
  // Cond 1. A complete block must be dominate
  if (!IsDominant(scope, block_sref)) {
    return false;
  }
  // Cond 3. Check if there is no overlap between buffers read and buffers written
  for (const BufferRegion& write : block->writes) {
    const Buffer& buffer = write->buffer;
    for (const BufferRegion& read : block->reads) {
      if (buffer.same_as(read->buffer)) {
        return false;
      }
    }
  }
  return true;
}

bool ScheduleStateNode::IsReduction(const StmtSRef& block_sref, const StmtSRef& scope_root) const {
  BlockScope scope = GetBlockScope(scope_root);
  // Cond 3. Block binding is valid iter affine map
  if (!this->IsAffineBlockBinding(block_sref)) {
    return false;
  }
  // Cond 4. Check whether the block body has the init statement.
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (!block->init.defined()) {
    return false;
  }
  // Cond 2. All block vars are either data parallel or reduction
  const Array<IterVar>& iter_vars = block->iter_vars;
  for (const IterVar& iter_var : iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  // Cond 1. Dominate block
  if (!IsDominant(scope, block_sref)) {
    return false;
  }
  // Cond 5. All reduction vars should not affect indexing the output buffer
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  bool not_affected = true;
  PreOrderVisit(block->body, [&not_affected, &iter_vars, &buffer_written](const ObjectRef& obj) {
    if (!not_affected) {
      return false;
    }
    if (const auto* store = obj.as<BufferStoreNode>()) {
      // Only consider buffers written by the block
      if (buffer_written.count(store->buffer.get())) {
        if (!CheckReductionInstance(iter_vars, store->indices)) {
          not_affected = false;
        }
      } else {
        LOG(FATAL) << "InternalError: A write buffer is not in the block signature: "
                   << store->buffer;
      }
      return false;
    }
    return true;
  });
  return not_affected;
}

bool ScheduleStateNode::CanMergeReduction(const StmtSRef& init_block_sref,
                                          const StmtSRef& update_block_sref,
                                          const StmtSRef& scope_root) const {
  BlockScope scope = GetBlockScope(scope_root);
  const auto* init = TVM_SREF_TO_BLOCK(init, init_block_sref);
  const auto* update = TVM_SREF_TO_BLOCK(update, update_block_sref);
  // Cond 1. Check the binding of update block is valid
  if (!this->IsAffineBlockBinding(update_block_sref)) {
    return false;
  }
  // Cond 2. Check init_block and update_block are the only two producers for their output buffer
  for (const BufferRegion& write_region : update->writes) {
    const Array<StmtSRef>& writers = scope->buffer_writers.at(write_region->buffer);
    if (writers.size() != 2) {
      return false;
    }
    if (!writers[0].same_as(init_block_sref) && !writers[0].same_as(update_block_sref)) {
      return false;
    }
    if (!writers[1].same_as(init_block_sref) && !writers[1].same_as(update_block_sref)) {
      return false;
    }
  }
  // Cond 3. init and update share the same buffer
  const auto* init_body = TVM_TYPE_AS(init_body, init->body, BufferStoreNode);
  const auto* update_body = TVM_TYPE_AS(update_body, update->body, BufferStoreNode);
  if (!init_body->buffer.same_as(update_body->buffer)) {
    return false;
  }
  // Access must be the same dimensional
  ICHECK_EQ(init_body->indices.size(), update_body->indices.size())
      << "InternalError: indexing to the same buffer with different dimensions";
  // Cond 4. All block vars of update_block are either data parallel or reduction,
  // and reduction vars of update_block should not affect indexing the output buffer
  return CheckReductionInstance(update->iter_vars, update_body->indices);
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(ScheduleStateNode);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleState")
    .set_body_typed([](ObjectRef obj, bool debug_mode) {
      if (const auto* func = obj.as<PrimFuncNode>()) {
        return ScheduleState(GetRef<PrimFunc>(func), debug_mode);
      }
      if (const auto* mod = obj.as<IRModuleNode>()) {
        return ScheduleState(GetRef<IRModule>(mod), debug_mode);
      }
      LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateGetBlockScope")
    .set_body_method<ScheduleState>(&ScheduleStateNode::GetBlockScope);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateReplace")
    .set_body_method<ScheduleState>(&ScheduleStateNode::Replace);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateGetSRef")
    .set_body_typed([](ScheduleState self, Stmt stmt) -> Optional<StmtSRef> {
      auto it = self->stmt2ref.find(stmt.get());
      return it != self->stmt2ref.end() ? it->second : Optional<StmtSRef>(NullOpt);
    });

}  // namespace tir
}  // namespace tvm

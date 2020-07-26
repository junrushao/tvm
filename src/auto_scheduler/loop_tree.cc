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

#include "./loop_tree.h"

#include "./auto_scheduler_utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(IteratorNode);
TVM_REGISTER_NODE_TYPE(LoopTreeNode);

Iterator::Iterator(String name, PrimExpr min, PrimExpr extent, IterKind kind,
                   IterAnnotation annotation) {
  ObjectPtr<IteratorNode> n = make_object<IteratorNode>();
  n->name = std::move(name);
  n->min = std::move(min);
  n->extent = std::move(extent);
  n->kind = std::move(kind);
  n->annotation = std::move(annotation);
  data_ = std::move(n);
}

LoopTree::LoopTree(Array<Iterator> iters, Optional<tir::BlockRealize> block_realize,
                   Array<ObjectRef> children) {
  ObjectPtr<LoopTreeNode> n = make_object<LoopTreeNode>();
  n->iters = std::move(iters);
  n->block_realize = std::move(block_realize);
  n->children = std::move(children);
  data_ = std::move(n);
}

/*!
 * \brief Figure out IterKind of a loop variable using the information provided in TIR
 * \param loop The TIR loop that contains the loop variable
 * \param sch The TIR schedule
 * \return IterKind indicating the kind of iteration variable
 */
IterKind IterKindFromTIRLoop(const tir::Loop& loop, const tir::Schedule& sch) {
  // TODO(@junrushao1994): introduce kOpaque?
  // Step 0. Check if the subtree under loop satisfies one-way fine-grained dataflow condition
  const tir::StmtSRef& loop_sref = sch->stmt2ref.at(loop.get());
  bool is_compact_dataflow = sch->GetParentScope(loop_sref).IsCompactDataFlow(loop_sref, sch.get());
  // Step 1. Collect direct child blocks under the loop
  std::vector<const tir::BlockRealizeNode*> child_block_realizes;
  tir::PreOrderVisit(loop, [&child_block_realizes](const ObjectRef& node) -> bool {
    if (const auto* realize = node.as<tir::BlockRealizeNode>()) {
      child_block_realizes.push_back(realize);
      return false;
    }
    return true;
  });
  // For non-compact dataflow, right now we only support a single child block
  if (!is_compact_dataflow && child_block_realizes.size() > 1) {
    return IterKind::kSpecial;
  }
  // Step 2. Check if bindings of all child blocks are validated
  for (const tir::BlockRealizeNode* realize : child_block_realizes) {
    if (!sch->stmt2ref.at(realize->block.get())->binding_valid) {
      return IterKind::kSpecial;
    }
  }
  // Step 3. Check the loop variable is bound to what kind of block variables
  bool bind_data_par = false;
  bool bind_reduction = false;
  bool bind_other = false;
  for (const tir::BlockRealizeNode* realize : child_block_realizes) {
    // Enumerate child blocks
    const tir::BlockNode* block = realize->block.get();
    CHECK_EQ(realize->binding_values.size(), block->iter_vars.size())
        << "InternalError: BlockRealize is inconsistent with its Block";
    int n = realize->binding_values.size();
    for (int i = 0; i < n; ++i) {
      const tir::IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->binding_values[i];
      // If loop variable is bound in the current binding
      if (ExprContainsVar(binding, loop->loop_var)) {
        if (iter_var->iter_type == tir::kDataPar) {
          bind_data_par = true;
        } else if (iter_var->iter_type == tir::kCommReduce) {
          bind_reduction = true;
        } else {
          bind_other = true;
        }
      }
    }
  }
  // Step 4. Check if the loop variable can be data parallel or reduction
  if (!bind_reduction && !bind_other) {
    // TODO(@junrushao1994): if it is not bound to anything, do we really consider it as data
    // parallel?
    return IterKind::kDataPar;
  }
  if (bind_reduction && !bind_other) {
    return IterKind::kReduction;
  }
  return IterKind::kSpecial;
}

/*!
 * \brief Create the loop tree recursively from a TIR AST
 * \param root The root of the AST. Should be tir::BlockRealize or tir::Loop
 * \return The corresponding loop tree created
 */
LoopTree LoopTreeFromTIR(const tir::StmtNode* root, const tir::Schedule& sch) {
  CHECK(root->IsInstance<tir::LoopNode>() || root->IsInstance<tir::BlockRealizeNode>())
      << "InternalError: Cannot create LoopTree because the TIR root provided starts with neither "
         "Loop nor BlockRealize";
  // Step 1. Collect all loops
  std::vector<Iterator> iters;
  while (root->IsInstance<tir::LoopNode>()) {
    const auto* loop = static_cast<const tir::LoopNode*>(root);
    iters.emplace_back(
        /*name=*/loop->loop_var->name_hint,
        /*min=*/loop->min,
        /*extent=*/loop->extent,
        /*kind=*/IterKindFromTIRLoop(GetRef<tir::Loop>(loop), sch),
        /*annotation=*/IterAnnotation::kNone);
    root = loop->body.get();
  }
  // Step 2. Check if there is a BlockRealize under the nested loop
  // It is possible that BlockRealize is not the direct child
  Optional<tir::BlockRealize> block_realize = NullOpt;
  if (root->IsInstance<tir::BlockRealizeNode>()) {
    const auto* block_realize_ptr = static_cast<const tir::BlockRealizeNode*>(root);
    block_realize = GetRef<tir::BlockRealize>(block_realize_ptr);
    CHECK(block_realize_ptr->block->annotations.empty())
        << "InternalError: block with pre-defined annotations are not supported";
    root = block_realize_ptr->block->body.get();
  }
  // Step 3. Collect all children of the block
  std::vector<ObjectRef> children;
  if (root->IsInstance<tir::SeqStmtNode>()) {
    // The node has many children
    const Array<tir::Stmt>& seq = static_cast<const tir::SeqStmtNode*>(root)->seq;
    children = {seq.begin(), seq.end()};
  } else {
    // The node has only one child
    children = {GetRef<ObjectRef>(root)};
  }
  // Step 4. Create children subtree
  // TODO(@junrushao1994): do we need to check if children are all blocks or all non-blocks?
  for (ObjectRef& child : children) {
    if (child->IsInstance<tir::LoopNode>() || child->IsInstance<tir::BlockRealizeNode>()) {
      child = LoopTreeFromTIR(child.as<tir::StmtNode>(), sch);
    } else {
      // `child` is a leaf statement
      // Check: Any BlockNode should be contained inside `BlockRealizeNode`
      CHECK(!child->IsInstance<tir::BlockNode>())
          << "InternalError: the IR is invalid because a BlockNode is not contained in "
             "BlockRealizeNode";
      // Check: no nested SeqStmt
      CHECK(!child->IsInstance<tir::SeqStmtNode>())
          << "InternalError: the IR is invalid because there are nested SeqStmt";
    }
  }
  return LoopTree(iters, block_realize, children);
}

LoopTree LoopTree::FromPrimFunc(const tir::PrimFunc& func) {
  const auto* realize = func->body.as<tir::BlockRealizeNode>();
  CHECK(realize != nullptr)
      << "InternalError: the PrimFunc is invalid because its body is not BlockRealizeNode";
  return LoopTreeFromTIR(realize, tir::ScheduleNode::Create(func));
}

class LoopTreeNode::Stringifier : public tir::StmtFunctor<void(const tir::Stmt&)> {
 public:
  /*!
   * \brief Entry function to stringify LoopTreeNode
   * \param root The object to be stringified
   * \return The human readable string representation
   */
  std::string Run(const LoopTreeNode* root) {
    Cout() << "LoopTree(" << root << "):\n";
    RecursivePrint(root);
    return os.str();
  }
  /*!
   * \brief Recursively print the LoopTreeNode
   * \param root The root of the tree to be stringified
   */
  void RecursivePrint(const LoopTreeNode* root) {
    constexpr int kIndentWidth = 2;
    int indent_delta = 0;
    // Step 1. Print loops with proper indentation
    for (const Iterator& iter : root->iters) {
      Cout() << "for " << iter << std::endl;
      // add one level of indentation
      indent_delta += kIndentWidth;
      this->indent += kIndentWidth;
    }
    // Step 2. Print its children
    for (const ObjectRef& child : root->children) {
      // Case 1. the child is another node in the loop tree
      if (const auto* node = child.as<LoopTreeNode>()) {
        RecursivePrint(node);
        continue;
      }
      // Case 2: the child is a leaf, and contains a certain computation
      const auto* stmt = child.as<tir::StmtNode>();
      CHECK(stmt) << "InternalError: Expect type tir::Stmt, get: " << child->GetTypeKey();
      try {
        VisitStmt(GetRef<tir::Stmt>(stmt));
      } catch (const dmlc::Error& e) {
        // If the printing function is not defined, then printing is not supported and the developer
        // need to fix this
        LOG(FATAL) << "InternalError: printing is not well supported for type: "
                   << child->GetTypeKey() << "\n"
                   << e.what();
        throw;
      }
    }
    // Cancel the indentation
    this->indent -= indent_delta;
  }
  /*! \brief Print ReduceStep */
  void VisitStmt_(const tir::ReduceStepNode* reduce) override {
    if (const auto* buffer_load = reduce->lhs.as<tir::BufferLoadNode>()) {
      Cout() << buffer_load->buffer->name << " = ..." << std::endl;
    } else {
      LOG(FATAL) << "InternalError: unknown type in ReduceStep: " << reduce->lhs->GetTypeKey();
    }
  }
  /*! \brief Print BufferStore */
  void VisitStmt_(const tir::BufferStoreNode* buffer_store) override {
    Cout() << buffer_store->buffer->name << " = ..." << std::endl;
  }
  /*!
   * \brief Prefixed function to print indented line
   * \return The ostream that can be used to continue the current line
   */
  std::ostream& Cout() { return os << std::string(indent, ' '); }
  /*! \brief The current indentation */
  int indent = 0;
  /*! \brief The ostream used to store the stringfying result temporarily */
  std::ostringstream os;
};

String LoopTreeNode::ToString() const { return LoopTreeNode::Stringifier().Run(this); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LoopTreeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = ref.as<LoopTreeNode>();
      CHECK(node);
      p->stream << node->ToString();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IteratorNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = ref.as<IteratorNode>();
      CHECK(node);
      arith::Analyzer analyzer;
      PrimExpr left_inclusive = analyzer.Simplify(node->min);
      PrimExpr right_exclusive = analyzer.Simplify(node->min + node->extent);
      p->stream  // Print name
          << node->name
          << ' '
          // Print kind
          << IterKind2String(node->kind)
          // Print loop domain
          << '[' << left_inclusive << ", " << right_exclusive
          << ')'
          // Print loop annotation
          << (node->annotation == IterAnnotation::kNone
                  ? ""
                  : " # " + IterAnnotation2String(node->annotation));
    });

TVM_REGISTER_GLOBAL("auto_scheduler.loop_tree.FromPrimFunc").set_body_typed(LoopTree::FromPrimFunc);

}  // namespace auto_scheduler
}  // namespace tvm

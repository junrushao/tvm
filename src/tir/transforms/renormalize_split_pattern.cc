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

/*!
 * \file renormalize_split_pattern.cc
 * \brief Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tir {

using namespace arith;

class SplitPatternReNormalizer : public IRMutatorWithAnalyzer {
 public:
  explicit SplitPatternReNormalizer(Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {}

  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    // floordiv(floormod(x, c1 * c2), c2) = floormod(floordiv(x, c2), c1)
    if (const auto* inner = op->a.as<FloorModNode>()) {
      if (const auto* c2 = op->b.as<IntImmNode>()) {
        if (const auto* c1c2 = inner->b.as<IntImmNode>()) {
          if (c1c2->value % c2->value == 0) {
            return analyzer_->Simplify(FloorMod(FloorDiv(inner->a, op->b),
                                                IntImm(op->b.dtype(), c1c2->value / c2->value)));
          }
        }
      }
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return FloorDiv(a, b);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }
};

namespace transform {

Pass RenormalizeSplitPattern() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    arith::Analyzer analyzer;
    n->body = SplitPatternReNormalizer(&analyzer)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RenormalizeSplitPattern", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RenormalizeSplitPattern")
    .set_body_typed(RenormalizeSplitPattern);

}  // namespace transform

}  // namespace tir
}  // namespace tvm

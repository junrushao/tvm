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
#ifndef SRC_META_SCHEDULE_SCHEDULE_H_
#define SRC_META_SCHEDULE_SCHEDULE_H_

#include <tvm/tir/schedule.h>

#include <unordered_map>

#include "./instruction.h"
#include "./sampler.h"

namespace tvm {
namespace meta_schedule {

class Schedule;

/*! \brief The meta schedule class */
class ScheduleNode : public Object {
 public:
  /*! \brief Type of the symbol table, which maps a random variable to its value */
  using TSymbolTable =
      std::unordered_map<ObjectRef, Optional<ObjectRef>, ObjectPtrHash, ObjectPtrEqual>;

 public:
  /*! \brief The original TIR PrimFunc to be scheduled */
  tir::PrimFunc orig_func;
  /*! \brief The TIR schedule in the current stage */
  tir::Schedule sch{nullptr};
  /*! \brief The trace of instructions used */
  Array<Instruction> trace;
  /*! \brief The decisions made in sampling */
  Map<Instruction, Array<ObjectRef>> decisions;
  /*! \brief The symbol table with information of all defined variables in the meta schedule */
  TSymbolTable sym_tab;
  /*! \brief The random number generator */
  Sampler sampler;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("orig_func", &orig_func);
    v->Visit("sch", &sch);
    v->Visit("trace", &trace);
    v->Visit("decisions", &decisions);
    // `sym_tab` is not visited
    // `sampler` is not visited
  }
  static constexpr const char* _type_key = "meta_schedule.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);
  /**************** Utility ****************/
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed
   */
  void Seed(int seed);
  /*!
   * \brief Copy the schedule into a new one. Operation on the new schedule won't affect the
   * original schedule, and vice versa.
   * \return A new schedule.
   */
  Schedule Copy(int new_seed) const;
  /**************** Evaluation of random variables ****************/
  /*!
   * \brief Evaluate the value of a random variable of type Block
   * \param block The block random variable to be evaluated
   * \return The TIR SRef to the block evaluated
   */
  tir::StmtSRef Eval(const BlockRV& block);
  /*!
   * \brief Evaluate the value of a random variable of type LoopAxis
   * \param loop The loop random variable to be evaluated
   * \return The TIR SRef to the block evaluated
   */
  tir::StmtSRef Eval(const LoopRV& loop);
  /*!
   * \brief Evaluate the value of a PrimExpr, containing random variable of type tir::Var
   * \param expr The expression containing random variables to be evaluated
   * \return The result of the evaluation
   */
  int Eval(const PrimExpr& expr);
  /**************** Sampling ****************/
  /*!
   * \brief Apply the instruction SamplePerfectTile
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop
   * \return An array of random variables, the result of sampling
   */
  Array<tir::Var> SamplePerfectTile(int n_splits, const LoopRV& loop,
                                    int max_innermost_factor = 16);
  /*!
   * \brief Apply the instruction SampleTileFactor
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param where The distribution of tile size to be sampled
   * \return An array of random variables, the result of sampling
   */
  Array<tir::Var> SampleTileFactor(int n_splits, const LoopRV& loop, const Array<Integer>& where);
  /**************** Block/Loop Relationship ****************/
  /*!
   * \brief Get the only consumer of a specific block
   * \param block The block to be queried
   * \return A block, its only consumer; or NullOpt if it does not exist
   */
  Optional<BlockRV> GetOnlyConsumer(const BlockRV& block);
  /*!
   * \brief Apply the instruction GetBlock
   * \param name The name of the block to get retrieved
   * \return A block random variable, the return value of the instruction
   */
  BlockRV GetBlock(const String& name);
  /*!
   * \brief Apply the instruction GetAxes
   * \param block The block used to retrieve the axes
   * \return An array of loop random variables
   */
  Array<LoopRV> GetAxes(const BlockRV& block);
  /**************** Scheduling Primitives ****************/
  /*!
   * \brief Apply the instruction Split
   * \param loop The loop to be split
   * \param factors The split factors
   * \return An array of loop random variables
   * \note If there is no NullOpt in factors, will split from inner to outer, and factors[0] is not
   * used
   */
  Array<LoopRV> Split(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors);
  /*!
   * \brief Apply the instruction Reorder
   * \param after_axes The axes to be reordered
   */
  void Reorder(const Array<LoopRV>& after_axes);
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block The block to be moved
   * \param loop The loop to be moved to
   */
  void ReverseComputeAt(const BlockRV& block, const LoopRV& loop);
  /*!
   * \brief Apply the instruction compute_inline
   * \param block The block to be computed inline
   */
  void ComputeInline(const BlockRV& block);
  /*!
   * \brief Apply the instruction reverse+compute_inline
   * \param block The block to be reverse computed inline
   */
  void ReverseComputeInline(const BlockRV& block);
  /*!
   * \brief Apply the instruction cache_write
   * \param block The block to be buffered
   * \param storage_scope The storage scope
   * \return The cache write stage
   */
  BlockRV CacheWrite(const BlockRV& block, const String& storage_scope);
  /*!
   * \brief Apply the instruction DecomposeReduction
   * \param block The block to be decomposed
   * \param loop The loop to be decomposed at
   * \return The block random variable indicating the decomposition result
   */
  BlockRV DecomposeReduction(const BlockRV& block, const LoopRV& loop);
  /**************** Trace-related ****************/
  /*!
   * \brief Mutate the decision on the specific instruction
   * \param inst The instruction whose decision is mutated
   * \param decision The decision to be mutated to. If it is NullOpt, then remove it from decisions
   * \note This method does not replay the trace and does not do any validity check
   */
  void MutateDecision(const Instruction& inst, const Optional<Array<ObjectRef>>& decision);
  /*!
   * \brief Re-sample along the trace to generatea new sequence of
   * scheduling instructions and program states
   */
  void ReSample();
  /*!
   * \brief Replay the trace with the decision stored in the schedule class.
   * If a decision has been changed using MutateDecision, then it will generate
   * different schedule. This process is theoretically deterministic if all sampling
   * instructions have decision made
   * \sa MutateDecision
   */
  void ReplayDecision();

 private:
  /*!
   * \brief Replay the trace with the decision stored in the schedule class.
   * If follow decision is true, and a decision has been changed using MutateDecision,
   * then it will generate different underlying TIR schedule
   * \param follow_decision Whether to follow existing decisions stored in the class.
   * If the flag is true, then the replay process will be deterministic
   */
  void Replay(bool follow_decision);
};

class Schedule : public ObjectRef {
 public:
  using TSymbolTable = ScheduleNode::TSymbolTable;
  /*!
   * \brief Constructor
   * \param orig_func The original TIR PrimFunc to be scheduled
   * \param sch The TIR schedule in the current stage
   * \param trace The trace of instructions used
   * \param decisions The decisions made in sampling
   * \param sym_tab The symbol table with information of all defined variables in the meta schedule
   * \param seed The random seed
   */
  explicit Schedule(tir::PrimFunc orig_func, tir::Schedule sch, Array<Instruction> trace,
                    Map<Instruction, Array<ObjectRef>> decisions, TSymbolTable sym_tab,
                    Optional<Integer> seed);
  /*!
   * \brief Constructor: other fields are created with default value
   * \param orig_func The original TIR PrimFunc to be scheduled
   * \param seed The random seed
   */
  explicit Schedule(tir::PrimFunc orig_func, Optional<Integer> seed);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, ObjectRef, ScheduleNode);
};

/*!
 * \brief Get the string representation of a schedule
 * \param sch The schedule to be stringified
 * \return The string representation of a schedule
 */
inline String Repr(const Schedule& sch) {
  const auto* f = runtime::Registry::Get("script.AsTVMScript");
  CHECK(f) << "IndexError: global function \"script.AsTVMScript\" not found";
  String s = (*f)(sch->sch->func, false);
  return s;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SCHEDULE_H_

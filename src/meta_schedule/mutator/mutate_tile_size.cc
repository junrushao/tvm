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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using tir::Instruction;
using tir::InstructionKind;
using tir::Trace;

/*!
 * \brief Downcast the decision of Sample-Perfect-Tile to an array of integers
 * \param decision The decision of Sample-Perfect-Tile
 * \return The result of downcast
 */
std::vector<int64_t> DowncastDecision(const ObjectRef& decision) {
  const auto* arr = TVM_TYPE_AS(arr, decision, runtime::ArrayNode);
  return support::AsVector<ObjectRef, int64_t>(GetRef<Array<ObjectRef>>(arr));
}

/*!
 * \brief Calculate the product of elements in an array
 * \param array The array
 * \return The product of elements in the array
 */
int64_t Product(const std::vector<int64_t>& array) {
  int64_t result = 1;
  for (int64_t x : array) {
    result *= x;
  }
  return result;
}

/*! \brief A mutator that mutates the tile size */
class MutateTileSizeNode : public MutatorNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.MutateTileSize";
  TVM_DECLARE_FINAL_OBJECT_INFO(MutateTileSizeNode, MutatorNode);

 public:
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherit from `MutatorNode`
  Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
};

/*!
 * \brief Find a sample-perfect-tile decision in the trace
 * \param trace The trace
 * \param rand_state The random state
 * \param inst The instruction selected
 * \param decision The decision selected
 * \return Whether a decision is found
 */
bool FindSamplePerfectTile(const Trace& trace, TRandState* rand_state, Instruction* inst,
                           std::vector<int64_t>* decision) {
  static const InstructionKind& inst_sample_perfect_tile =
      InstructionKind::Get("SamplePerfectTile");
  std::vector<Instruction> instructions;
  std::vector<std::vector<int64_t>> decisions;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    const ObjectRef& decision = kv.second;
    if (!inst->kind.same_as(inst_sample_perfect_tile)) {
      continue;
    }
    std::vector<int64_t> tiles = DowncastDecision(decision);
    if (tiles.size() >= 2 && Product(tiles) >= 2) {
      instructions.push_back(inst);
      decisions.push_back(tiles);
    }
  }
  int n = instructions.size();
  if (n > 0) {
    int i = tir::SampleInt(rand_state, 0, n);
    *inst = instructions[i];
    *decision = decisions[i];
    return true;
  }
  return false;
}

static std::unordered_map<int, std::vector<int>> factor_memory_;

const std::vector<int>& GetFactors(int n) {
  auto it = factor_memory_.find(n);
  if (it != factor_memory_.end()) {
    return it->second;
  }
  
  std::vector<int>& res = factor_memory_[n];
  int step = n % 2 == 0 ? 1 : 2;
  for (size_t i = 1; i < static_cast<size_t>(std::sqrt(n)) + 1; i += step) {
    if (n % i == 0) {
      res.push_back(i);
      if (n / i != i) {
        res.push_back(n / i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

Optional<Trace> MutateTileSizeNode::Apply(const Trace& trace, TRandState* rand_state) {
  Instruction inst;
  std::vector<int64_t> tiles;
  if (!FindSamplePerfectTile(trace, rand_state, &inst, &tiles)) {
    return NullOpt;
  }
  int n_splits = tiles.size();
  // Step 1. Choose two loops, `x` and `y`
  int x,y;
  //select source
  while(true) {
    x = tir::SampleInt(rand_state, 0, n_splits);
    if (tiles[x] <= 1) {
      continue;
    }

    y = tir::SampleInt(rand_state, 0, n_splits - 1);
    if (y >= x) {
      ++y;
    }

    auto factors = GetFactors(tiles[x]);
    // Step 2. Choose the divide factor
    int64_t divide_factor;
    if (y != n_splits - 1) {
      divide_factor = factors[tir::SampleInt(rand_state, 1, factors.size())];
    } else {
      int64_t limit = Downcast<Integer>(inst->attrs[1])->value;
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * tiles[y] <= limit) {
          break;
        }
      }
      if (max_factor_index == 0) {
        // Failed on this dst_idx, try next one.
        continue;
      }
      divide_factor = factors[tir::SampleInt(rand_state, 1, max_factor_index+1)];
    }
    tiles[x] /= divide_factor;
    tiles[y] *= divide_factor;
    return trace->WithDecision(inst, support::AsArray<int64_t, ObjectRef>(tiles),
                               /*remove_postproc=*/true);
  }
}

Mutator Mutator::MutateTileSize() { return Mutator(make_object<MutateTileSizeNode>()); }

TVM_REGISTER_NODE_TYPE(MutateTileSizeNode);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorMutateTileSize").set_body_typed(Mutator::MutateTileSize);

}  // namespace meta_schedule
}  // namespace tvm

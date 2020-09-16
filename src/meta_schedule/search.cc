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

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>

#include "./random_variable.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

using runtime::PackedFunc;
using runtime::TypedPackedFunc;

class Rule {
 public:
  String name;
  TypedPackedFunc<bool(Schedule, BlockRV)> meets;
  TypedPackedFunc<std::string(Schedule, BlockRV)> apply;

  explicit Rule(String name, const PackedFunc* meets, const PackedFunc* apply)
      : name(name), meets(*meets), apply(*apply) {}
};

void Search(const tir::PrimFunc& func, const std::vector<Rule>& rules, const String& policy) {
  Schedule sch = Schedule(/*orig_func=*/func, /*sch=*/tir::ScheduleNode::Create(func), /*trace=*/{},
                          /*sym_tab=*/{}, /*sampler=*/Sampler(DeviceRand));
  // TODO(@junrushao1994): deal with the nested case
  Array<tir::StmtSRef> blocks = sch->sch->Blocks(sch->sch->root);
  for (const tir::StmtSRef& tir_block : blocks) {
    BlockRV block = sch->CreateBlockRV(tir_block);
    for (const Rule& rule : rules) {
      if (rule.meets(sch, block)) {
        // TODO(@junrushao1994): deal with the return value
        rule.apply(sch, block);
      }
    }
  }
}

TVM_DLL void SearchRules(tir::PrimFunc func, Array<String> rule_names, String policy) {
  std::vector<Rule> rules;
  for (const String& rule_name : rule_names) {
    const PackedFunc* meets = runtime::Registry::Get(rule_name + ".meets");
    const PackedFunc* apply = runtime::Registry::Get(rule_name + ".apply");
    CHECK(meets != nullptr) << "ValueError: Rule not registered: " << (rule_name + ".meets");
    CHECK(apply != nullptr) << "ValueError: Rule not registered: " << (rule_name + ".apply");
    rules.emplace_back(rule_name, meets, apply);
  }
  Search(func, rules, policy);
}

TVM_REGISTER_GLOBAL("meta_schedule.SearchRules").set_body_typed(SearchRules);

}  // namespace meta_schedule
}  // namespace tvm

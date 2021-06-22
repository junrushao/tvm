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

/**************** Constructors  ****************/

Trace::Trace() { data_ = make_object<TraceNode>(); }

Trace::Trace(Array<Inst> insts, Map<Inst, ObjectRef> decisions) {
  ObjectPtr<TraceNode> n = make_object<TraceNode>();
  n->insts = std::move(insts);
  n->decisions = std::move(decisions);
  data_ = std::move(n);
}

/**************** Add/Remove ****************/

void TraceNode::Append(const Inst& inst) { insts.push_back(inst); }

void TraceNode::Append(const Inst& inst, const ObjectRef& decision) {
  insts.push_back(inst);
  decisions.Set(inst, decision);
}

Optional<Inst> TraceNode::Pop() {
  if (insts.empty()) {
    return NullOpt;
  }
  Inst inst = insts.back();
  insts.pop_back();
  if (decisions.count(inst)) {
    decisions.erase(inst);
  }
  return inst;
}

/**************** Interfacing with InstKind ****************/

void TraceNode::ApplyToSchedule(const Schedule& sch, bool remove_postproc,
                                std::function<ObjectRef(const Array<ObjectRef>& inputs,  //
                                                        const Array<ObjectRef>& attrs,   //
                                                        const ObjectRef& decision)>
                                    decision_provider) const {
  auto f_translate_inputs =
      [](const Array<ObjectRef>& inputs,
         const std::unordered_map<const Object*, const Object*>& rv_map) -> Array<ObjectRef> {
    Array<ObjectRef> result;
    result.reserve(inputs.size());
    for (const ObjectRef& input : inputs) {
      if (!input.defined() ||                   // constant: nullptr
          input->IsInstance<StringObj>() ||     // constant: string
          input->IsInstance<IntImmNode>() ||    // constant: integer
          input->IsInstance<FloatImmNode>()) {  // constant: float
        result.push_back(input);
      } else if (input->IsInstance<BlockNode>() ||   // RV: block
                 input->IsInstance<LoopRVNode>() ||  // RV: loop
                 input->IsInstance<VarNode>()) {     // RV: var
        auto it = rv_map.find(input.get());
        ICHECK(it != rv_map.end()) << "IndexError: Random variable doesn't exist: " << input;
        result.push_back(GetRef<ObjectRef>(it->second));
      } else if (const auto* expr = input.as<PrimExprNode>()) {  // RV: Expr
        result.push_back(
            Substitute(GetRef<PrimExpr>(expr), [&rv_map](const Var& var) -> Optional<PrimExpr> {
              auto it = rv_map.find(var.get());
              if (it == rv_map.end()) {
                return NullOpt;
              }
              const Object* dst = it->second;
              ICHECK(dst->IsInstance<VarNode>())
                  << "TypeError: Expect 'tir.Var', but gets: " << dst->GetTypeKey();
              return GetRef<Var>(static_cast<const VarNode*>(dst));
            }));
      } else {
        ICHECK(false) << "TypeError: Cannot recognize the type of an input random variable: "
                      << input->GetTypeKey();
        throw;
      }
    }
    return result;
  };

  auto f_translate_outputs = [](const Array<ObjectRef>& old_outputs,
                                const Array<ObjectRef>& new_outputs,
                                std::unordered_map<const Object*, const Object*>* rv_map) -> void {
    ICHECK_EQ(old_outputs.size(), new_outputs.size());
    int n = old_outputs.size();
    const ObjectRef* p_old = old_outputs.GetArrayNode()->begin();
    const ObjectRef* p_new = new_outputs.GetArrayNode()->begin();
    for (int i = 0; i < n; ++i) {
      (*rv_map)[p_old[i].get()] = p_new[i].get();
    }
  };

  static InstKind inst_enter_postproc = InstKind::Get("EnterPostProc");
  std::unordered_map<const Object*, const Object*> rv_map;
  for (const Inst& inst : this->insts) {
    if (remove_postproc && inst->kind.same_as(inst_enter_postproc)) {
      break;
    }
    Array<ObjectRef> inputs = f_translate_inputs(inst->inputs, rv_map);
    Array<ObjectRef> attrs = inst->attrs;
    ObjectRef decision = this->decisions.Get(inst);
    if (decision_provider) {
      decision = decision_provider(inputs, attrs, decision);
    }
    Array<ObjectRef> outputs = inst->kind->f_apply_to_schedule(sch, inputs, attrs, decision);
    f_translate_outputs(inst->outputs, outputs, &rv_map);
  }
}

Array<String> InputsAsNames(
    const Array<ObjectRef>& inputs,
    const std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual>& rv_names) {
  Array<String> results;
  results.reserve(inputs.size());
  for (const ObjectRef& input : inputs) {
    if (!input.defined()) {
      // Case 0. nullptr => None
      results.push_back(String("None"));
      continue;
    }
    auto it = rv_names.find(input);
    if (it != rv_names.end()) {
      // Case 1. BlockRV, LoopRV, VarRV
      results.push_back(it->second);
    } else if (const auto* str_obj = input.as<StringObj>()) {
      // Case 2. string => "content"
      results.push_back('"' + std::string(str_obj->data) + '"');
    } else if (const auto* int_imm = input.as<IntImmNode>()) {
      // Case 3. int => 8
      results.push_back(std::to_string(int_imm->value));
    } else if (const auto* float_imm = input.as<FloatImmNode>()) {
      // Case 4. float => 1.00002
      std::ostringstream os;
      os.precision(19);
      os << std::fixed << float_imm->value;
      results.push_back(os.str());
    } else {
      LOG(FATAL) << "TypeError: Stringifying is not supported for type: " << input->GetTypeKey();
      throw;
    }
  }
  return results;
}

Array<String> AddOutputsNames(
    const Array<ObjectRef>& outputs,
    std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual>* rv_names) {
  Array<String> results;
  results.reserve(outputs.size());
  for (const ObjectRef& output : outputs) {
    int i = rv_names->size();
    ICHECK(!rv_names->count(output))
        << "ValueError: The random variable has been produced once: " << rv_names->at(output);
    String result{ObjectPtr<StringObj>{nullptr}};
    if (output->IsInstance<BlockRVNode>()) {
      result = "b" + std::to_string(i);
    } else if (output->IsInstance<LoopRVNode>()) {
      result = "l" + std::to_string(i);
    } else if (output->IsInstance<VarNode>()) {
      result = "v" + std::to_string(i);
    } else {
      LOG(FATAL) << "TypeError: Cannot recognize the type of the random variable: "
                 << output->GetTypeKey();
      throw;
    }
    results.push_back(result);
    rv_names->emplace(output, std::move(result));
  }
  return results;
}

ObjectRef TraceNode::AsJSON() const {
  std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual> rv_names;
  Array<ObjectRef> json_trace;
  Array<ObjectRef> json_decision;
  json_trace.reserve(this->insts.size());
  json_decision.reserve(this->insts.size());

  int i = 0;
  for (const Inst& inst : this->insts) {
    const InstKind& kind = inst->kind;
    json_trace.push_back(Array<ObjectRef>{
        /* 0: inst name */ kind->name,
        /* 1: inputs    */ InputsAsNames(inst->inputs, rv_names),
        /* 2: attrs     */ kind->f_attrs_as_json ? kind->f_attrs_as_json(inst->attrs) : inst->attrs,
        /* 3: outputs   */ AddOutputsNames(inst->outputs, &rv_names),
    });
    if (Optional<ObjectRef> decision = this->decisions.Get(inst)) {
      json_decision.push_back(Array<ObjectRef>{
          /* 0: index    */ Integer(i),
          /* 1: decision */ decision.value(),
      });
    }
    ++i;
  }
  return Array<ObjectRef>{
      /* 0: trace    */ std::move(json_trace),
      /* 1: decision */ std::move(json_decision),
  };
}

Array<String> TraceNode::AsPython() const {
  std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual> rv_names;
  Array<String> py_trace;
  py_trace.reserve(this->insts.size());
  for (const Inst& inst : this->insts) {
    py_trace.push_back(
        inst->kind->f_as_python(/*inputs=*/InputsAsNames(inst->inputs, rv_names),
                                /*attrs=*/inst->attrs,
                                /*decision=*/this->decisions.Get(inst),
                                /*outputs=*/AddOutputsNames(inst->outputs, &rv_names)));
  }
  return py_trace;
}

void Trace::ApplyJSONToSchedule(const ObjectRef& json, const Schedule& sch) {
  throw;  //
}

/**************** Creation ****************/

Trace TraceNode::Simplified(bool remove_postproc) const {
  throw;  //
}

Trace TraceNode::WithDecision(const Inst& inst, const ObjectRef& decision,
                              bool remove_postproc) const {
  throw;  //
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(TraceNode);

}  // namespace tir
}  // namespace tvm

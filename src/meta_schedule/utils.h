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
#ifndef TVM_META_SCHEDULE_UTILS_H_
#define TVM_META_SCHEDULE_UTILS_H_

#include <dmlc/memory_io.h>
#include <tvm/arith/analyzer.h>
#include <tvm/meta_schedule/arg_info.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/cost_model.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/feature_extractor.h>
#include <tvm/meta_schedule/measure_callback.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/meta_schedule/task_scheduler.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/node/node.h>
#include <tvm/node/serialization.h>
#include <tvm/support/parallel_for.h>
#include <tvm/tir/schedule/schedule.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../printer/text_printer.h"
#include "../support/array.h"
#include "../support/base64.h"
#include "../support/nd_int_set.h"
#include "../support/utils.h"
#include "../tir/schedule/primitive.h"
#include "../tir/schedule/utils.h"

namespace tvm {
namespace tir {

inline double CountFlop(const IRModule& mod) {
  struct TResult {
    using TTable = std::unordered_map<int32_t, double>;

    TResult() = default;

    explicit TResult(const tvm::DataType& dtype) { Add(dtype); }

    void Add(const tvm::DataType& dtype) { data_[DataType2Int(dtype)] += 1; }

    TResult operator+=(const TResult& rhs) {
      for (const auto& kv : rhs.data_) {
        data_[kv.first] += kv.second;
      }
      return *this;
    }

    TResult operator*=(int64_t rhs) {
      for (auto& kv : data_) {
        kv.second *= rhs;
      }
      return *this;
    }

    TResult MaxWith(const TResult& rhs) {
      for (const auto& kv : rhs.data_) {
        double& v = data_[kv.first];
        if (v < kv.second) {
          v = kv.second;
        }
      }
      return *this;
    }

    struct DType {
      uint8_t code : 8;
      uint8_t bits : 8;
      uint16_t lanes : 16;
    };
    static_assert(sizeof(DType) == 4, "Incorrect size of DType");

    static String Int2Str(int32_t dtype) {
      union {
        DType dst;
        int32_t src;
      } converter;
      converter.src = dtype;
      static std::string type_code_tab[] = {"int", "uint", "float", "handle", "bfloat"};
      std::ostringstream os;
      os << type_code_tab[converter.dst.code];
      os << static_cast<int>(converter.dst.bits);
      if (converter.dst.lanes != 1) {
        os << "x" << static_cast<int>(converter.dst.lanes);
      }
      return os.str();
    }

    static int32_t DataType2Int(const tvm::DataType& dtype) {
      union {
        DType src;
        int32_t dst;
      } converter;
      converter.src.code = dtype.code();
      converter.src.bits = dtype.bits();
      converter.src.lanes = dtype.lanes();
      return converter.dst;
    }

    TTable data_;
  };

  class FlopCounter : public ExprFunctor<TResult(const PrimExpr& n)>,
                      public StmtFunctor<TResult(const Stmt& n)> {
   public:
    ~FlopCounter() {}

    TResult VisitExpr(const PrimExpr& expr) override { return ExprFunctor::VisitExpr(expr); }
    TResult VisitStmt(const Stmt& stmt) override { return StmtFunctor::VisitStmt(stmt); }

    TResult VisitStmt_(const IfThenElseNode* branch) override {
      TResult cond = VisitExpr(branch->condition);
      cond += VisitStmt(branch->then_case).MaxWith(VisitStmt(branch->else_case));
      return cond;
    }

    TResult VisitStmt_(const BufferStoreNode* store) override {
      TResult result = VisitExpr(store->value);
      for (const PrimExpr& e : store->indices) {
        result += VisitExpr(e);
      }
      return result;
    }

    TResult VisitStmt_(const SeqStmtNode* seq) override {
      TResult result;
      for (const Stmt& stmt : seq->seq) {
        result += VisitStmt(stmt);
      }
      return result;
    }

    TResult VisitStmt_(const BlockRealizeNode* block) override {
      return VisitStmt(block->block->body);
    }

    TResult VisitStmt_(const BlockNode* block) override {
      TResult result;
      if (block->init.defined()) {
        result += VisitStmt(block->init.value());
      }
      result += VisitStmt(block->body);
      return result;
    }

    TResult VisitStmt_(const ForNode* loop) override {
      TResult result = VisitStmt(loop->body);
      const auto* int_imm = loop->extent.as<IntImmNode>();
      ICHECK(int_imm) << "TypeError: Expect the extent of a loop to be IntImm, but gets: "
                      << loop->extent->GetTypeKey();
      result *= int_imm->value;
      return result;
    }

#define TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(Node) \
  TResult VisitExpr_(const Node* op) final {        \
    TResult result(op->dtype);                      \
    result += VisitExpr(op->a);                     \
    result += VisitExpr(op->b);                     \
    return result;                                  \
  }
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(AddNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(SubNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(MulNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(DivNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(ModNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(FloorDivNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(FloorModNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(MinNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(MaxNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(EQNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(NENode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(LTNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(LENode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(GTNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(GENode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(AndNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(OrNode);
#undef TVM_META_SCHEDULE_FLOP_COUNTER_BINARY
    TResult VisitExpr_(const CastNode* op) override { return VisitExpr(op->value); }
    TResult VisitExpr_(const VarNode* op) override { return TResult(); }
    TResult VisitExpr_(const SizeVarNode* op) override { return TResult(); }
    TResult VisitExpr_(const BufferLoadNode* op) override { return TResult(); }
    TResult VisitExpr_(const IntImmNode* op) override { return TResult(); }
    TResult VisitExpr_(const FloatImmNode* op) override { return TResult(); }
    TResult VisitExpr_(const NotNode* op) override {
      TResult result(op->dtype);
      result += VisitExpr(op->a);
      return result;
    }
    TResult VisitExpr_(const SelectNode* op) override {
      TResult cond = VisitExpr(op->condition);
      cond += VisitExpr(op->true_value).MaxWith(VisitExpr(op->false_value));
      return cond;
    }
    TResult VisitExpr_(const CallNode* op) override {
      TResult ret;
      for (const auto& x : op->args) {
        ret += VisitExpr(x);
      }
      return ret;
    }
  };
  FlopCounter counter;
  TResult result;
  for (const auto& kv : mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
      result += counter.VisitStmt(prim_func->body);
    }
  }
  double cnt = 0.0;
  int i32 = TResult::DataType2Int(tvm::DataType::Int(32));
  int i64 = TResult::DataType2Int(tvm::DataType::Int(64));
  int u1 = TResult::DataType2Int(tvm::DataType::UInt(1));
  for (const auto& kv : result.data_) {
    if (kv.first != i32 && kv.first != i64 && kv.first != u1) {
      cnt += kv.second;
    }
  }
  return cnt;
}

}  // namespace tir
namespace meta_schedule {

/*! \brief The type of the random state */
using TRandState = support::LinearCongruentialEngine::TRandState;

/*!
 * \brief Read lines from a json file.
 * \param path The path to the json file.
 * \param allow_missing Whether to create new file when the given path is not found.
 * \return An array containing lines read from the json file.
 */
inline Array<String> JSONFileReadLines(const String& path, bool allow_missing) {
  std::ifstream is(path);
  if (is.good()) {
    Array<String> results;
    for (std::string str; std::getline(is, str);) {
      results.push_back(str);
    }
    return results;
  }
  CHECK(allow_missing) << "ValueError: File doesn't exist: " << path;
  std::ofstream os(path);
  CHECK(os.good()) << "ValueError: Cannot create new file: " << path;
  return {};
}

/*!
 * \brief Append a line to a json file.
 * \param path The path to the json file.
 * \param line The line to append.
 */
inline void JSONFileAppendLine(const String& path, const std::string& line) {
  std::ofstream os(path, std::ofstream::app);
  CHECK(os.good()) << "ValueError: Cannot open the file to write: " << path;
  os << line << std::endl;
}

/*!
 * \brief Get the base64 encoded result of a string.
 * \param str The string to encode.
 * \return The base64 encoded string.
 */
inline std::string Base64Encode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&result);
  support::Base64OutStream b64stream(&m_stream);
  static_cast<dmlc::Stream*>(&b64stream)->Write(str);
  b64stream.Finish();
  return result;
}

/*!
 * \brief Get the base64 decoded result of a string.
 * \param str The string to decode.
 * \return The base64 decoded string.
 */
inline std::string Base64Decode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&str);
  support::Base64InStream b64stream(&m_stream);
  b64stream.InitPosition();
  static_cast<dmlc::Stream*>(&b64stream)->Read(&result);
  return result;
}

/*!
 * \brief Parse lines of json string into a json object.
 * \param lines The lines of json string.
 * \return Array of json objects parsed.
 * \note The function calls the python-side json parser in runtime registry.
 */
inline Array<ObjectRef> JSONStr2Obj(const Array<String>& lines) {
  static const runtime::PackedFunc* f_to_obj =
      runtime::Registry::Get("meta_schedule.batch_json_str2obj");
  ICHECK(f_to_obj) << "IndexError: Cannot find the packed function "
                      "`meta_schedule.batch_json_str2obj` in the global registry";
  return (*f_to_obj)(lines);
}

/*!
 * \brief Serialize a json object into a json string.
 * \param json_obj The json object to serialize.
 * \return A string containing the serialized json object.
 * \note The function calls the python-side json obj serializer in runtime registry.
 */
inline String JSONObj2Str(const ObjectRef& json_obj) {
  static const runtime::PackedFunc* f_to_str = runtime::Registry::Get("meta_schedule.json_obj2str");
  ICHECK(f_to_str) << "IndexError: Cannot find the packed function "
                      "`meta_schedule.json_obj2str` in the global registry";
  return (*f_to_str)(json_obj);
}

/*!
 * \brief Converts a structural hash code to string
 * \param hash_code The hash code
 * \return The string representation of the hash code
 */
inline String SHash2Str(Workload::THashCode hash_code) { return std::to_string(hash_code); }

/*!
 * \brief Find the entry function of the given IRModule, i.e, functions marked by
 * `tir::attr::kIsEntryFunc`, whose name is `main` or being the only PrimeFunc.
 * \param mod The IRModule to find the entry function.
 * \return The entry function.
 */
inline tir::PrimFunc FindEntryFunc(const IRModule& mod) {
  // Priority 1: PrimFunc marked as `tir::attr::kIsEntryFunc`
  int num_prim_func = 0;
  const tir::PrimFuncNode* main_func = nullptr;
  const tir::PrimFuncNode* last_func = nullptr;
  for (const auto& kv : mod->functions) {
    GlobalVar gv = kv.first;
    BaseFunc base_func = kv.second;
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      last_func = func;
      if (func->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        return GetRef<tir::PrimFunc>(func);
      }
      if (gv->name_hint == "main") {
        main_func = func;
      }
      ++num_prim_func;
    }
  }
  // Priority 2: PrimFunc whose name is `main`
  if (main_func != nullptr) {
    return GetRef<tir::PrimFunc>(main_func);
  }
  // Priority 3: The only PrimFunc in the IRModule
  if (num_prim_func == 0) {
    LOG(FATAL) << "ValueError: Cannot find any PrimFunc in the given IRModule: "
               << tir::AsTVMScript(mod);
  }
  if (num_prim_func > 1) {
    LOG(FATAL) << "ValueError: Multiple PrimFuncs exist in the IRModule, but none of them are "
                  "annotated with `kIsEntryFunc`, i.e. `tir.is_entry_func`"
               << tir::AsTVMScript(mod);
  }
  return GetRef<tir::PrimFunc>(last_func);
}

/*!
 * \brief Fork a random state into another, i.e. PRNG splitting.
 * The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \return The forked random state
 */
inline support::LinearCongruentialEngine::TRandState ForkSeed(
    support::LinearCongruentialEngine::TRandState* rand_state) {
  return support::LinearCongruentialEngine(rand_state).ForkSeed();
}

/*!
 * \brief Fork a random state into another ones, i.e. PRNG splitting.
 *  The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \param n The number of forks
 * \return The forked random states
 */
inline std::vector<support::LinearCongruentialEngine::TRandState> ForkSeed(
    support::LinearCongruentialEngine::TRandState* rand_state, int n) {
  std::vector<support::LinearCongruentialEngine::TRandState> results;
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    results.push_back(support::LinearCongruentialEngine(rand_state).ForkSeed());
  }
  return results;
}

/*!
 * \brief Get deep copy of an IRModule.
 * \param mod The IRModule to make a deep copy.
 * \return The deep copy of the IRModule.
 */
inline IRModule DeepCopyIRModule(IRModule mod) {
  return Downcast<IRModule>(LoadJSON(SaveJSON(mod)));
}

/*!
 * \brief Concatenate strings
 * \param strs The strings to concatenate
 * \param delim The delimiter
 * \return The concatenated string
 */
inline std::string Concat(const Array<String>& strs, const std::string& delim) {
  if (strs.empty()) {
    return "";
  }
  std::ostringstream os;
  os << strs[0];
  for (int i = 1, n = strs.size(); i < n; ++i) {
    os << delim << strs[i];
  }
  return os.str();
}

/*!
 * \brief Get the BlockRV from a block StmtSRef
 * \param sch The schedule
 * \param block_sref The block StmtSRef
 * \param global_var_name The global variable name
 * \return The BlockRV
 */
inline tir::BlockRV GetRVFromSRef(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                                  const String& global_var_name) {
  const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  return sch->GetBlock(block->name_hint, global_var_name);
}

/*!
 * \brief A helper data structure that replays a trace and collects failure counts
 * for each postprocessor
 */
struct ThreadedTraceApply {
  /*! \brief Constructor */
  explicit ThreadedTraceApply(const Array<Postproc>& postprocs)
      : n_(postprocs.size()), items_(new Item[n_]) {
    for (int i = 0; i < n_; ++i) {
      items_[i].postproc = postprocs[i];
      items_[i].fail_counter = 0;
    }
  }

  /*! \brief Destructor */
  ~ThreadedTraceApply() { delete[] items_; }

  /*!
   * \brief Apply the trace and postprocessors to an IRModule
   * \param mod The IRModule to be applied
   * \param trace The trace to apply to the IRModule
   * \param rand_state The random seed
   * \return The schedule created, or NullOpt if any postprocessor fails
   */
  Optional<tir::Schedule> Apply(const IRModule& mod, const tir::Trace& trace,
                                TRandState* rand_state) {
    tir::Schedule sch =
        tir::Schedule::Traced(mod,
                              /*rand_state=*/ForkSeed(rand_state),
                              /*debug_mode=*/0,
                              /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
    trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
    sch->EnterPostproc();
    for (int i = 0; i < n_; ++i) {
      Item& item = items_[i];
      if (!item.postproc->Apply(sch)) {
        ++item.fail_counter;
        return NullOpt;
      }
    }
    return sch;
  }

  /*! \brief Returns a string summarizing the failures on each postprocessor */
  std::string SummarizeFailures() const {
    std::ostringstream os;
    for (int i = 0; i < n_; ++i) {
      const Item& item = items_[i];
      os << "Postproc #" << i << " [" << item.postproc  //
         << "]: " << item.fail_counter.load() << " failure(s)";
      if (i != n_ - 1) {
        os << "\n";
      }
    }
    return os.str();
  }

 private:
  /*! \brief A helper data structure that stores the fail count for each postprocessor. */
  struct Item {
    /*! \brief The postprocessor. */
    Postproc postproc{nullptr};
    /*! \brief The thread-safe postprocessor failure counter. */
    std::atomic<int> fail_counter{0};
  };

  /*! \brief The number of total postprocessors. */
  int n_;
  /*! \brief The pointer to the list of postprocessor items. */
  Item* items_;
};

/*!
 * \brief Get the number of cores in CPU
 * \param target The target
 * \return The number of cores.
 */
inline int GetTargetNumCores(const Target& target) {
  int num_cores = target->GetAttr<Integer>("num-cores").value_or(-1);
  if (num_cores == -1) {
    static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule.cpu_count");
    ICHECK(f_cpu_count)
        << "ValueError: Cannot find the packed function \"meta_schedule._cpu_count\"";
    num_cores = (*f_cpu_count)(false);
    LOG(FATAL)
        << "Target does not have attribute \"num-cores\", physical core number must be "
           "defined! For example, on the local machine, the target must be \"llvm -num-cores "
        << num_cores << "\"";
  }
  return num_cores;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_UTILS_H_

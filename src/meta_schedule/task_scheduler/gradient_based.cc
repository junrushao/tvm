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

/*! \brief The gradient based task scheduler. */
class GradientBasedNode final : public TaskSchedulerNode {
 public:
  int backward_window_size;  // the backward windows size for backward gradient computation
  double alpha, beta;        // alpha, beta as in gradient computation
  bool done_round_robin;     // whether the round-robin warm up has been done

  int task_id = -1;  // the current task id processing

  support::LinearCongruentialEngine::TRandState rand_state;  // the random state
  std::vector<int> task_cnts;                                // task tuning counts
  std::vector<double> task_weights;                          // task weights
  std::vector<double> task_best_latencies;                   // best latency achived by the task
  std::vector<double> task_flop_counts;                      // total flop count of the task
  std::vector<std::vector<double>> task_latency_history;     // all task latency history

  std::vector<int> task_group_id;          // group id of the task
  std::vector<std::set<int>> task_groups;  // the task ids in a given group

  TaskSchedulerNode::FObjectiveFunc objective_func;  // the objective function

  void VisitAttrs(tvm::AttrVisitor* v) {
    TaskSchedulerNode::VisitAttrs(v);
    v->Visit("backward_window_size", &backward_window_size);
    v->Visit("alpha", &alpha);
    v->Visit("beta", &beta);
    v->Visit("task_id", &task_id);
  }

  static constexpr const char* _type_key = "meta_schedule.GradientBased";
  TVM_DECLARE_FINAL_OBJECT_INFO(GradientBasedNode, TaskSchedulerNode);

 protected:
  /*!
   * \brief Compute the objective function score for given workload latencies.
   * \param latencies The current best latencies of each workload.
   * \return The computed objective function score.
   */
  double _compute_score(const std::vector<double>& latencies) {
    Array<FloatImm> input_latencies;
    for (double latency : latencies)
      input_latencies.push_back(FloatImm(DataType::Float(32), latency));
    return objective_func(input_latencies)->value;
  }

  /*!
   * \brief Adjuest the similarity group information for given task.
   * \param task_id The id of the task to adjust similarity group.
   */
  void _adjust_similarity_group(int task_id) {
    int group_id = task_group_id[task_id];
    if (task_groups[group_id].size() <= 1) return;

    double best_flops = -1.0;
    int max_cnt[3] = {-1, -1, -1};  // to find the 2nd largest
    for (int i : task_groups[group_id]) {
      best_flops = std::max(best_flops, task_flop_counts[i] / task_best_latencies[i]);
      max_cnt[0] = task_cnts[i];
      std::sort(max_cnt, max_cnt + 3);  // place the 2nd largest to middle position
    }
    double cur_flops = task_flop_counts[task_id] / task_best_latencies[task_id];
    // if we tune a task for many times but it still cannot achieve
    // a similar speed to the fastest one in its group, this means this task
    // is actually not similar to other tasks in its group.
    // So we will move it from the current group to a standalone group.

    if (cur_flops < best_flops / beta && task_cnts[task_id] > 5 + max_cnt[1]) {
      task_groups[group_id].erase(task_id);
      task_group_id[task_id] = task_groups.size();
      task_groups.push_back(std::set<int>{task_id});
    }
  }

  int NextTaskId() final {
    int n_tasks = this->tasks.size();
    // Check if warmed up with round robin
    // have to do it manually because the search is async
    if (!done_round_robin) {
      task_id++;
      if (task_id == n_tasks) {
        done_round_robin = true;
      } else {
        return task_id;
      }
    }
    // Calculate gradients if already warmed up
    double max_gradient = -1e30, min_gradient = 1e30;
    int arg_min_gradient = -1;
    std::vector<int> tasks_alive;
    tasks_alive.reserve(n_tasks);
    for (task_id = 0; task_id < n_tasks; ++task_id) {
      if (!tasks[task_id]->is_stopped) {
        tasks_alive.push_back(task_id);
        // compute gradient from chain rule : (\Delta f / \Delta g_i)
        // here f is given as objective function, default weighted sum
        double delta = 1e-4;
        std::vector<double> new_latencies(task_best_latencies);
        new_latencies[task_id] -= delta;
        double chain_grad =
            (_compute_score(task_best_latencies) - _compute_score(new_latencies)) / delta;

        // compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
        // which is approximated by (g_i(t_i) - g_t(t_i - window_size)) / window_size
        double backward_grad;
        if (task_cnts[task_id] - 1 - backward_window_size >= 0) {
          backward_grad =
              (task_latency_history[task_id][task_cnts[task_id] - 1] -
               task_latency_history[task_id][task_cnts[task_id] - 1 - backward_window_size]) /
              backward_window_size;
        } else {
          backward_grad = 0;
        }

        // compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
        // which is approximated by
        // Min_{ - g_i(t_i) / t_i, \Beta \frac{C_i}{Max_{k \in N_i}(V_k)} - g_i(t_i)}

        // Part I: g_i_[t_i]- g_i(t_i) / t_i
        double g_next_1;
        if (task_cnts[task_id] > 0) {
          g_next_1 =
              task_best_latencies[task_id] - (task_best_latencies[task_id] / task_cnts[task_id]);
        } else {
          g_next_1 = beta * 1e30;
        }
        // Part II:  \Beta \frac{C_i}{Max_{k \in N_i}(V_k)}
        double best_flops = -1.0;
        int group_id = task_group_id[task_id];
        for (int groupmate_id : task_groups[group_id]) {
          best_flops = std::max(best_flops,
                                task_flop_counts[groupmate_id] / task_best_latencies[groupmate_id]);
        }
        ICHECK(best_flops > 0) << "Best flops should be greater or equal to zero!";
        double g_next_2 = beta * task_flop_counts[task_id] / best_flops;

        double g_next = std::min(g_next_1, g_next_2);
        double forward_grad = std::min(g_next - task_best_latencies[task_id], 0.0);

        double gradient = chain_grad * (alpha * backward_grad + (1 - alpha) * forward_grad);
        ICHECK(gradient <= 0)
            << "Wrong gradient calculated, should be less than or equal to 0. Chain_grad: "
            << chain_grad << ", backward_grad: " << backward_grad
            << ", forward_grad: " << forward_grad << ".";
        if (gradient > max_gradient) {
          max_gradient = gradient;
        }
        if (gradient < min_gradient) {
          min_gradient = gradient;
          arg_min_gradient = task_id;
        }
      }
    }
    // all tasks done
    if (tasks_alive.size() == 0) return -1;
    // same gradient, sample any task
    if (std::abs(max_gradient - min_gradient) < 1e-6) {
      task_id = tasks_alive[tir::SampleInt(&rand_state, 0, tasks_alive.size())];
    } else {
      task_id = arg_min_gradient;
    }
    // check if task is running
    if (IsTaskRunning(task_id)) {
      JoinRunningTask(task_id);
    }
    return task_id;
  }

  void JoinRunningTask(int task_id) final {
    TuneContext task = tasks[task_id];
    ICHECK(task->runner_futures.defined());
    Array<RunnerFuture> futures = task->runner_futures.value();
    int n = futures.size();
    Array<RunnerResult> results;
    task_cnts[task_id]++;
    results.reserve(n);
    double trial_best_latency = 1e30;
    for (const RunnerFuture future : task->runner_futures.value()) {
      RunnerResult result = future->Result();
      results.push_back(result);
      if (!result->error_msg.defined() && result->run_secs.defined()) {
        double sum = 0;
        int count = 0;
        for (const FloatImm& run_sec : result->run_secs.value()) {
          count += 1;
          sum += run_sec->value;
        }
        if (count > 0) trial_best_latency = std::min(trial_best_latency, sum / count);
      }
    }

    if (task_latency_history[task_id].size() == 0 ||
        trial_best_latency < task_best_latencies[task_id]) {
      task_best_latencies[task_id] = trial_best_latency;
    }
    task_latency_history[task_id].push_back(task_best_latencies[task_id]);
    _adjust_similarity_group(task_id);
    task->search_strategy.value()->NotifyRunnerResults(task, task->measure_candidates.value(),
                                                       results);
    // Invoke the callbacks
    ICHECK(task->measure_candidates.defined());
    ICHECK(task->builder_results.defined());
    ICHECK_EQ(results.size(), task->measure_candidates.value().size());
    ICHECK_EQ(results.size(), task->builder_results.value().size());
    for (const MeasureCallback& callback : this->measure_callbacks) {
      callback->Apply(GetRef<TaskScheduler>(this), task_id, task->measure_candidates.value(),
                      task->builder_results.value(), results);
    }
    task->measure_candidates = NullOpt;
    task->builder_results = NullOpt;
    task->runner_futures = NullOpt;
  }
};

TaskScheduler TaskScheduler::GradientBased(Array<TuneContext> tasks,                            //
                                           Builder builder,                                     //
                                           Runner runner,                                       //
                                           Database database,                                   //
                                           double alpha,                                        //
                                           double beta,                                         //
                                           int backward_window_size,                            //
                                           support::LinearCongruentialEngine::TRandState seed,  //
                                           Array<FloatImm> task_weights,                        //
                                           String objective_func_name,                          //
                                           String tag_generation_func_name,                     //
                                           Optional<CostModel> cost_model,                      //
                                           Optional<Array<MeasureCallback>> measure_callbacks) {
  ObjectPtr<GradientBasedNode> n = make_object<GradientBasedNode>();
  n->alpha = alpha;
  n->beta = beta;
  n->backward_window_size = backward_window_size;
  if (seed == -1) seed = std::random_device()();
  support::LinearCongruentialEngine(&n->rand_state).Seed(seed);

  n->done_round_robin = false;
  n->task_id = -1;
  n->tasks = tasks;
  n->builder = builder;
  n->runner = runner;
  n->database = database;
  n->cost_model = cost_model;
  n->measure_callbacks = measure_callbacks.value_or({});

  n->task_cnts.assign(n->tasks.size(), 0);
  n->task_flop_counts.assign(n->tasks.size(), 0);
  n->task_best_latencies.assign(n->tasks.size(), 1e30);
  n->task_latency_history.assign(n->tasks.size(), std::vector<double>());
  n->task_weights.assign(n->tasks.size(), 1);

  n->task_groups.reserve(n->tasks.size());
  n->task_group_id.reserve(n->tasks.size());

  const auto* objective_func_ptr = runtime::Registry::Get(objective_func_name);
  CHECK(objective_func_ptr) << "The given objective function is undefined!";
  n->objective_func = *objective_func_ptr;

  const auto* tag_generation_func_ptr = runtime::Registry::Get(tag_generation_func_name);
  CHECK(tag_generation_func_ptr) << "The given tag generation function is undefined!";
  TaskSchedulerNode::FTagGenerationFunc tag_generation_func = *tag_generation_func_ptr;

  if (task_weights.defined()) {
    CHECK(task_weights.size() == n->tasks.size())
        << "The given task weights number does not equal to task number!";
    int cnt = 0;
    for (const FloatImm& weight : task_weights) {
      n->task_weights[cnt++] = weight->value;
    }
  }

  std::map<std::string, int> tag_to_group;

  int task_id = -1;
  for (const TuneContext& task : tasks) {
    task->task_scheduler = n.get();
    IRModule mod = task->mod.value_or({});
    n->task_flop_counts[++task_id] = tir::CountFlop(mod);
    std::string tag = tag_generation_func(mod);

    if (tag_to_group.find(tag) == tag_to_group.end()) {
      tag_to_group[tag] = tag_to_group.size();
      n->task_groups.push_back(std::set<int>());
    }
    int group_id = tag_to_group[tag];
    n->task_group_id.push_back(group_id);
    n->task_groups[group_id].insert(task_id);
  }
  return TaskScheduler(n);
}

TVM_REGISTER_NODE_TYPE(GradientBasedNode);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerGradientBased")
    .set_body_typed(TaskScheduler::GradientBased);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerFlopCount").set_body_typed(tir::CountFlop);

}  // namespace meta_schedule
}  // namespace tvm

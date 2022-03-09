# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitatios
# under the License.
# pylint: disable=missing-docstring
import argparse
import os

import tvm
from tvm import auto_scheduler
from tvm.meta_schedule.runner import RPCConfig
from tvm.meta_schedule.testing.te_workload import CONFIGS


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-workers",
        type=int,
        required=True,
    )
    args.add_argument(
        "--log-dir",
        type=str,
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    return parsed


ARGS = _parse_args()


def main():
    log_file = os.path.join(ARGS.log_dir, f"{ARGS.workload}.json")
    workload_func, params = CONFIGS[ARGS.workload]
    params = params[0]  # type: ignore
    workload_func = auto_scheduler.register_workload(workload_func)

    if ARGS.target.kind.name == "llvm":
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=int(ARGS.target.attrs["num-cores"]),
            target=ARGS.target,
        )
    elif ARGS.target.kind.name == "cuda":
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(ARGS.target.attrs["max_shared_memory_per_block"]),
            max_threads_per_block=int(ARGS.target.attrs["max_threads_per_block"]),
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    else:
        raise NotImplementedError(f"Unsupported target {ARGS.target}")
    task = auto_scheduler.SearchTask(
        func=workload_func,
        args=params,
        target=ARGS.target,
        hardware_params=hardware_params,
    )
    runner = auto_scheduler.RPCRunner(
        key=ARGS.rpc_key,
        host=ARGS.rpc_host,
        port=ARGS.rpc_port,
        n_parallel=ARGS.rpc_workers,
        number=3,
        repeat=1,
        min_repeat_ms=100,
        enable_cpu_cache_flush=False,
    )

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=ARGS.num_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
        runner=runner,
    )
    print("Running AutoTuning:")
    task.tune(tune_option)
    print("History Best:")
    print(task.print_best(log_file))
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))


if __name__ == "__main__":
    main()

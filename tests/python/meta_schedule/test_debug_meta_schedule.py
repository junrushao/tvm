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
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring

from typing import List

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm._ffi import get_global_func
from tvm.ir import IRModule
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.measure_callback import EchoStatistics
from tvm.meta_schedule.postproc import Postproc
from tvm.meta_schedule.runner import EvaluatorConfig
from tvm.meta_schedule.testing import create_te_workload
from tvm.meta_schedule.tune import DefaultCUDA, DefaultLLVM
from tvm.meta_schedule.utils import remove_build_dir
from tvm.target import Target
from tvm.tir import Schedule


RPC_HOST = "192.168.6.66"
RPC_PORT = 4445
RPC_KEY = "jetson-agx-xavier"
TARGET = Target("nvidia/jetson-agx-xavier")
WORKLOAD = "GMM"
POSTPROCS: List[Postproc] = DefaultCUDA._postproc()  # pylint: disable=protected-access

TARGET = tvm.target.Target("nvidia/jetson-agx-xavier")


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    import os

    if not os.path.exists("/tmp/perf"):
        os.mkdir("/tmp/perf")
    with open("/tmp/perf/tir.cu", "w") as f:
        f.write(code)
    return code


def schedule_fn(sch: Schedule):
    # pylint: disable=invalid-name,line-too-long,unused-variable
    # fmt: off
    b0 = sch.get_block(name="Z", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11])
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[4, 1, 8, 2, 2])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21])
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[4, 2, 16, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31])
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[4, 2, 16])
    l40, l41, l42 = sch.split(loop=l6, factors=[v37, v38, v39])
    sch.reorder(l12, l22, l32, l13, l23, l33, l14, l24, l34, l40, l41, l15, l25, l35, l42, l16, l26, l36)
    l43 = sch.fuse(l12, l22, l32)
    sch.bind(loop=l43, thread_axis="blockIdx.x")
    l44 = sch.fuse(l13, l23, l33)
    sch.bind(loop=l44, thread_axis="vthread.x")
    l45 = sch.fuse(l14, l24, l34)
    sch.bind(loop=l45, thread_axis="threadIdx.x")

    b46 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b46, loop=l40, preserve_unit_loops=True)
    l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b46)
    l54 = sch.fuse(l51, l52, l53)
    v55, v56 = sch.sample_perfect_tile(loop=l54, n=2, max_innermost_factor=4, decision=[512, 1])
    sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)

    b57 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True)
    l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b57)
    l65 = sch.fuse(l62, l63, l64)
    v66, v67 = sch.sample_perfect_tile(loop=l65, n=2, max_innermost_factor=4, decision=[1024, 2])
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)

    print(sch.mod.script())
    import sys; sys.exit()
    sch.reverse_compute_at(block=b2, loop=l45, preserve_unit_loops=True)
    v68 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v68)
    # fmt: on
    # pylint: enable=invalid-name,line-too-long,unused-variable
    return sch


def _make_mod() -> IRModule:
    prim_func = create_te_workload(WORKLOAD, 0)
    prim_func = prim_func.with_attr("global_symbol", "main")
    prim_func = prim_func.with_attr("tir.noalias", True)
    return IRModule({"main": prim_func})


def _apply_postproc(sch: Schedule):
    sch.enter_postproc()
    ctx = TuneContext(target=TARGET)
    for p in POSTPROCS:
        p.initialize_with_tune_context(ctx)
        assert p.apply(sch)


def run_sch(sch: Schedule, mod: IRModule):
    print(sch.mod.script())
    print(sch.trace)
    print(tvm.lower(sch.mod).script())
    tvm.build(sch.mod, target=TARGET)
    builder = ms.builder.LocalBuilder()
    runner = ms.runner.RPCRunner(
        rpc_config=ms.runner.RPCConfig(
            tracker_host=RPC_HOST,
            tracker_port=RPC_PORT,
            tracker_key=RPC_KEY,
            session_timeout_sec=60,
        ),
        evaluator_config=EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=200,
            enable_cpu_cache_flush=False,
        ),
        alloc_repeat=10,
        max_workers=5,
    )
    (builder_result,) = builder.build(  # pylint: disable=unbalanced-tuple-unpacking
        [ms.builder.BuilderInput(sch.mod, TARGET)]
    )
    if builder_result.error_msg is not None:
        print(builder_result.error_msg)
        return
    try:
        runner_input = ms.runner.RunnerInput(
            builder_result.artifact_path,
            device_type=TARGET.kind.name,
            args_info=ms.arg_info.ArgInfo.from_prim_func(sch.mod["main"]),
        )
        (runner_future,) = runner.run([runner_input])  # pylint: disable=unbalanced-tuple-unpacking
        runner_result = runner_future.result()
        if runner_result.error_msg is not None:
            print(runner_result.error_msg)
            return
        else:
            result = [float(x) * 1000.0 for x in runner_result.run_secs]
    finally:
        remove_build_dir(builder_result.artifact_path)

    flop = get_global_func("meta_schedule._CountFlop")(mod)
    print(flop)
    result = [flop / 1e6 / s for s in result]
    print(result)
    print(np.mean(result))


def main():
    mod = _make_mod()
    sch = Schedule(mod, debug_mask="all")
    sch = schedule_fn(sch)
    _apply_postproc(sch)
    run_sch(sch, mod)


if __name__ == "__main__":
    main()

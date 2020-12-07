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
""" Test multi-level tiling """
# pylint: disable=missing-function-docstring
import os

import pytest
import te_workload
import tvm
from tvm import meta_schedule as ms
from tvm import te

TARGET = tvm.target.Target("llvm --num_cores 16")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=True),
        ms.rule.multi_level_tiling_and_fusion(
            structure="SSRSRS",
            must_cache_read=False,
            cache_read_scope="global",
            can_cache_write=True,
            must_cache_write=False,
            cache_write_scope="global",
            fusion_levels=[1, 2],
        ),
        ms.rule.random_compute_location(),
        ms.rule.mark_parallelize_outer(max_jobs_per_core=8),
        ms.rule.mark_vectorize_inner(max_extent=32),
        ms.rule.mark_auto_unroll(max_steps=[0, 16, 64, 512], unroll_explicit=True),
    ],
    postprocs=[
        ms.postproc.rewrite_parallel(),
        ms.postproc.rewrite_vectorize(),
        ms.postproc.rewrite_auto_unroll(),
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_matmul_post_order_apply():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(
            workload=te.create_func(te_workload.matmul(1024, 1024, 1024)),
            target=TARGET,
            task_name="cpu_matmul",
            log_file="./cpu_matmul.json",
        ),
        space=SPACE,
        strategy=ms.strategy.Replay(num_trials=32),
        measurer=ms.ProgramMeasurer(
            measure_callbacks=[
                ms.RecordToFile(),
            ]
        ),
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_relu_post_order_apply():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(
            workload=te.create_func(te_workload.matmul_relu(1024, 1024, 1024)),
            target=TARGET,
            task_name="cpu_matmul_relu",
            log_file="./cpu_matmul_relu.json",
        ),
        space=SPACE,
        strategy=ms.strategy.Replay(num_trials=32),
        measurer=ms.ProgramMeasurer(
            measure_callbacks=[
                ms.RecordToFile(),
            ]
        ),
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.sch.func))


if __name__ == "__main__":
    test_matmul_post_order_apply()
    test_matmul_relu_post_order_apply()

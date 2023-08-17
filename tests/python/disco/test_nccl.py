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
"""Tests for NCCL"""
import tempfile

import numpy as np

import tvm
from tvm import dlight as dl
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.script import relax as R


def test_init():
    num_workers = 2
    devices = [1, 2]

    sess = di.Session.threaded_session(num_workers=num_workers)
    sess.init_ccl("nccl", *devices)


def test_allreduce_basic():
    num_workers = 2
    devices = [1, 2]
    sess = di.Session.threaded_session(num_workers=num_workers)
    sess.init_ccl("nccl", *devices)
    d_array = sess.empty((3, 4), "float32")

    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.arange(start=0, stop=-12, step=-1, dtype="float32").reshape(3, 4)

    d_array.debug_get_from_remote(0).copyfrom(array_1)
    d_array.debug_get_from_remote(1).copyfrom(array_2)

    result = sess.get_global_func("runtime.disco.nccl.allreduce")(d_array, 0)
    sess.sync_worker(0)
    result = result.debug_get_from_remote(0).numpy()
    np.testing.assert_equal(result, np.zeros((3, 4), "float32"))


def test_allreduce_in_relax_ir():
    num_workers = 2
    devices = [1, 2]

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class TestMod:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 64), "float32"),  # shard along axis 1
            W2: R.Tensor((64, 128), "float32"),  # shard along axis 0
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((128, 64), "float32") = R.matmul(x, W1)
                lv1: R.Tensor((128, 64), "float32") = R.nn.gelu(lv0)
                lv2: R.Tensor((128, 128), "float32") = R.matmul(lv1, W2)
                lv3: R.Tensor((128, 128), "float32") = R.call_pure_packed(
                    "runtime.disco.allreduce",
                    lv2,
                    R.shape((0,)),
                    sinfo_args=R.Tensor((128, 128), "float32"),
                )
                R.output(lv3)
            return lv3

    # pylint: enable=invalid-name
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": 49152,
            "max_threads_per_block": 1024,
            "thread_warp_size": 32,
            "registers_per_block": 65536,
            "arch": "sm_80",
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        with target:
            mod = rx.get_pipeline("zero")(TestMod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            mod.show(black_format=False)
            rx.build(mod, target="cuda").export_library(path)
        sess = di.Session.threaded_session(num_workers=num_workers)
        sess.init_ccl("nccl", *devices)
        mod = sess.load_vm_module(path)
        sess.sync_worker(0)
        sess.sync_worker(1)


if __name__ == "__main__":
    test_init()
    test_allreduce_basic()
    test_allreduce_in_relax_ir()

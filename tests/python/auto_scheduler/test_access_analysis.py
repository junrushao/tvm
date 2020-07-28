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
""" Testing tvm.auto_scnheduler.AccessAnalysis. """
import tvm


@tvm.tir.hybrid.script
def _matmul_with_relu(a, b, c, d):
    A = buffer_bind(a, (1024, 1024), "float32")
    B = buffer_bind(b, (1024, 1024), "float32")
    C = buffer_bind(d, (1024, 1024), "float32")
    D = buffer_bind(d, (1024, 1024), "float32")
    reducer = comm_reducer(lambda x, y: x + y, float32(0))

    with block({},
               writes=[C[0:1024, 0:1024], D[0:1024, 0:1024]],
               reads=[A[0:1024, 0:1024], B[0:1024, 0:1024]],
               name="root"):

        for i in range(0, 1024):
            for j in range(0, 1024):
                for k in range(0, 1024):
                    with block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)],
                                      B[vj:(vj + 1), vk:(vk + 1)]], name="C"):
                        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])

        for i in range(0, 1024):
            for j in range(0, 1024):
                with block({vi(0, 1024): i, vj(0, 1024): j},
                           writes=[D[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[C[vi:(vi + 1), vj:(vj + 1)]], name="D"):
                    # TODO(@junrushao1994): change it to `max` once supported
                    D[vi, vj] = C[vi, vj] + 1


def test_matmul_with_relu():
    module = tvm.tir.hybrid.create_module({"hybrid_func": _matmul_with_relu})
    func = module["hybrid_func"]
    assert isinstance(func, tvm.tir.PrimFunc)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    analysis_result = tvm.auto_scheduler.access_analysis.analyze(loop_tree)
    analysis_mm = analysis_result[loop_tree.children[0]]
    analysis_relu = analysis_result[loop_tree.children[1]]


if __name__ == "__main__":
    test_matmul_with_relu()

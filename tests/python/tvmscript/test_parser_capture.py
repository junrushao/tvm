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

from tvm.script.builder import ir as I
from tvm.script.builder import tir as T


def test_capture_func():
    from tvm.script.builder.tir import axis as ax
    from tvm.script.builder.tir import block, match_buffer

    @T.prim_func
    def scalar_func(a: T.handle, b: T.handle, c: T.Buffer((128,))):
        A = match_buffer(a, (128, 128))
        B = match_buffer(b, (128, 128))
        with block():
            for i, j in T.grid(128, 128):
                with block("inner_block"):
                    vi, vj = ax.remap("SR", [i, j])
                    A[i, j] = B[i - 1, j + 1] + A[i - 1, j - 1]

    print(scalar_func.script())


def test_capture_class():
    from tvm.script.builder.tir import axis as ax
    from tvm.script.builder.tir import block, match_buffer

    @I.ir_module
    class Module:
        @T.prim_func
        def scalar_func(a: T.handle, b: T.handle, c: T.Buffer((128,))):
            A = match_buffer(a, (128, 128))
            B = match_buffer(b, (128, 128))
            with block():
                for i, j in T.grid(128, 128):
                    with block("inner_block"):
                        vi, vj = ax.remap("SR", [i, j])
                        A[i, j] = B[i - 1, j + 1] + A[i - 1, j - 1]

    print(Module.script())


if __name__ == "__main__":
    test_capture_func()
    test_capture_class()

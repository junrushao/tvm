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

import tvm
from tvm.ir import Range
from tvm.script import tir as T
from tvm.script.builder import Builder



def test_builder_basic():
    code = """
    def elementwise(
        A: T.Buffer(shape=(128, 128, 128), dtype="float32"),
        B: T.Buffer(shape=(128, 128, 128), dtype="float32"),
    ) -> None:
        for i, j, *vvv, k in T.grid(128, 128, 128, 128, 128, 128, 128):
            with T.block("inner_block"):
                # vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                vi = T.axis.S(128, i + 1)
                vj = T.axis.S(128, j + 20)
                vk = T.axis.R(128, k - i)
                ...
    """
    ir_builder = Builder()
    with ir_builder:
        print(0)

if __name__ == "__main__":
    test_builder_basic()

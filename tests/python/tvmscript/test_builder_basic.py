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
from tvm.script.builder import Builder, def_, def_many
from tvm.script.builder import tir as T
from tvm.tir import BufferLoad, BufferRegion
from tvm.ir import Range


def test_builder_basic():
    with Builder() as b:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"global_symbol": "main"})
            T.func_ret(tvm.ir.PrimType("int8"))
            arg_a = T.arg("a", T.handle())
            arg_b = T.arg("b", T.handle())
            buffer_c = T.Buffer((128,), "float32")
            buffer_d = T.Buffer((128,), "float32")
            arg_c = T.arg("c", buffer_c)
            arg_d = T.arg("d", buffer_d)
            T.func_ret(tvm.ir.PrimType("int8"))
            A = def_("A", T.match_buffer(arg_a, (128, 128, 128), "int32"))
            B = def_("B", T.match_buffer(arg_b, (128, 128, 128), "int32"))
            T.preflattened_buffer(buffer_c, (128,), data=buffer_c.data)
            T.preflattened_buffer(buffer_d, (128,), data=buffer_d.data)
            with T.block(name="root"):
                with T.grid(128, 128, 128) as (i, j, k):
                    def_many(["i", "j", "k"], [i, j, k])
                    with T.block(name="block"):
                        T.block_attr({"axis": 1})
                        T.where(i > 1)
                        with T.init():
                            pass
                        vi = def_("vi", T.axis.spatial(128, i))
                        vj = def_("vj", T.axis.spatial(128, j))
                        vk = def_("vk", T.axis.reduce(128, k))
                        T.reads(
                            BufferRegion(
                                A,
                                [
                                    Range(vi, vi + 1),
                                    Range.from_min_extent(vj, 2),
                                    Range(1, 1 + BufferLoad(B, [1, 2, BufferLoad(A, [3, 4, 5])])),
                                ],
                            )
                        )
                        T.writes([BufferLoad(A, [100, 101, 102])])
                        E = def_("E", T.alloc_buffer((128, 128)))
                        F = def_("F", T.alloc_buffer((128, 128)))
    print(b.get().script())


if __name__ == "__main__":
    test_builder_basic()

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
from typing import Tuple

import tvm
from tvm import te, topi


TARGET = tvm.target.Target("nvidia/jetson-agx-xavier")


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    import os

    if not os.path.exists("/tmp/perf"):
        os.mkdir("/tmp/perf")
    with open("/tmp/perf/te.cu", "w") as f:
        f.write(code)
    return code


def func(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    N: int,
    M: int,
    K: int,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, N, K), name="X")
    y = te.placeholder((B, K, M), name="Y")
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(  # pylint: disable=invalid-name
        (B, N, M),
        lambda b, i, j: te.sum(x[b][i][k] * y[b][k][j], axis=[k]),
        name="Z",
    )
    return (x, y, z)


def main():
    X, Y, Z = func(1, 128, 128, 128)
    s = te.create_schedule(Z.op)
    # fmt: off
    Z_b, Z_i, Z_j, Z_k = tuple(Z.op.axis) + tuple(Z.op.reduce_axis)
    Z_local, = s.cache_write([Z], "local")
    Z_local_b_c, Z_local_i_c, Z_local_j_c, Z_local_k = tuple(Z_local.op.axis) + tuple(Z_local.op.reduce_axis)

    Z_local_b_c_o_i, Z_local_b_c_i = s[Z_local].split(Z_local_b_c, factor=1)
    Z_local_b_c_o_o_i, Z_local_b_c_o_i = s[Z_local].split(Z_local_b_c_o_i, factor=1)
    Z_local_b_c_o_o_o_i, Z_local_b_c_o_o_i = s[Z_local].split(Z_local_b_c_o_o_i, factor=1)
    Z_local_b_c_o_o_o_o, Z_local_b_c_o_o_o_i = s[Z_local].split(Z_local_b_c_o_o_o_i, factor=1)

    Z_local_i_c_o_i, Z_local_i_c_i = s[Z_local].split(Z_local_i_c, factor=2)
    Z_local_i_c_o_o_i, Z_local_i_c_o_i = s[Z_local].split(Z_local_i_c_o_i, factor=2)
    Z_local_i_c_o_o_o_i, Z_local_i_c_o_o_i = s[Z_local].split(Z_local_i_c_o_o_i, factor=8)
    Z_local_i_c_o_o_o_o, Z_local_i_c_o_o_o_i = s[Z_local].split(Z_local_i_c_o_o_o_i, factor=1)

    Z_local_j_c_o_i, Z_local_j_c_i = s[Z_local].split(Z_local_j_c, factor=1)
    Z_local_j_c_o_o_i, Z_local_j_c_o_i = s[Z_local].split(Z_local_j_c_o_i, factor=1)
    Z_local_j_c_o_o_o_i, Z_local_j_c_o_o_i = s[Z_local].split(Z_local_j_c_o_o_i, factor=16)
    Z_local_j_c_o_o_o_o, Z_local_j_c_o_o_o_i = s[Z_local].split(Z_local_j_c_o_o_o_i, factor=2)

    Z_local_k_o_i, Z_local_k_i = s[Z_local].split(Z_local_k, factor=16)
    Z_local_k_o_o, Z_local_k_o_i = s[Z_local].split(Z_local_k_o_i, factor=2)

    s[Z_local].reorder(Z_local_b_c_o_o_o_o, Z_local_i_c_o_o_o_o, Z_local_j_c_o_o_o_o, Z_local_b_c_o_o_o_i, Z_local_i_c_o_o_o_i, Z_local_j_c_o_o_o_i, Z_local_b_c_o_o_i, Z_local_i_c_o_o_i, Z_local_j_c_o_o_i, Z_local_k_o_o, Z_local_k_o_i, Z_local_b_c_o_i, Z_local_i_c_o_i, Z_local_j_c_o_i, Z_local_k_i, Z_local_b_c_i, Z_local_i_c_i, Z_local_j_c_i)
    Z_b_o_i, Z_b_i = s[Z].split(Z_b, factor=1)
    Z_b_o_o_i, Z_b_o_i = s[Z].split(Z_b_o_i, factor=1)
    Z_b_o_o_o, Z_b_o_o_i = s[Z].split(Z_b_o_o_i, factor=1)
    Z_i_o_i, Z_i_i = s[Z].split(Z_i, factor=4)
    Z_i_o_o_i, Z_i_o_i = s[Z].split(Z_i_o_i, factor=8)
    Z_i_o_o_o, Z_i_o_o_i = s[Z].split(Z_i_o_o_i, factor=1)
    Z_j_o_i, Z_j_i = s[Z].split(Z_j, factor=1)
    Z_j_o_o_i, Z_j_o_i = s[Z].split(Z_j_o_i, factor=16)
    Z_j_o_o_o, Z_j_o_o_i = s[Z].split(Z_j_o_o_i, factor=2)
    s[Z].reorder(Z_b_o_o_o, Z_i_o_o_o, Z_j_o_o_o, Z_b_o_o_i, Z_i_o_o_i, Z_j_o_o_i, Z_b_o_i, Z_i_o_i, Z_j_o_i, Z_b_i, Z_i_i, Z_j_i)
    s[Z_local].compute_at(s[Z], Z_j_o_i)
    Y_shared = s.cache_read(Y, "shared", [Z_local])
    Y_shared_ax0, Y_shared_ax1, Y_shared_ax2 = tuple(Y_shared.op.axis)
    s[Y_shared].compute_at(s[Z_local], Z_local_k_o_o)
    X_shared = s.cache_read(X, "shared", [Z_local])
    X_shared_ax0, X_shared_ax1, X_shared_ax2 = tuple(X_shared.op.axis)
    s[X_shared].compute_at(s[Z_local], Z_local_k_o_o)
    Z_b_o_o_o_i_o_o_o_fused_j_o_o_o_fused = s[Z].fuse(Z_b_o_o_o, Z_i_o_o_o, Z_j_o_o_o)
    s[Z].bind(Z_b_o_o_o_i_o_o_o_fused_j_o_o_o_fused, te.thread_axis("blockIdx.x"))
    Z_b_o_o_i_i_o_o_i_fused_j_o_o_i_fused = s[Z].fuse(Z_b_o_o_i, Z_i_o_o_i, Z_j_o_o_i)
    s[Z].bind(Z_b_o_o_i_i_o_o_i_fused_j_o_o_i_fused, te.thread_axis("vthread"))
    Z_b_o_i_i_o_i_fused_j_o_i_fused = s[Z].fuse(Z_b_o_i, Z_i_o_i, Z_j_o_i)
    s[Z].bind(Z_b_o_i_i_o_i_fused_j_o_i_fused, te.thread_axis("threadIdx.x"))

    Y_shared_ax0_ax1_fused_ax2_fused = s[Y_shared].fuse(Y_shared_ax0, Y_shared_ax1, Y_shared_ax2)
    Y_shared_ax0_ax1_fused_ax2_fused_o, Y_shared_ax0_ax1_fused_ax2_fused_i = s[Y_shared].split(Y_shared_ax0_ax1_fused_ax2_fused, factor=4)
    s[Y_shared].vectorize(Y_shared_ax0_ax1_fused_ax2_fused_i)
    Y_shared_ax0_ax1_fused_ax2_fused_o_o, Y_shared_ax0_ax1_fused_ax2_fused_o_i = s[Y_shared].split(Y_shared_ax0_ax1_fused_ax2_fused_o, factor=128)
    s[Y_shared].bind(Y_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))

    X_shared_ax0_ax1_fused_ax2_fused = s[X_shared].fuse(X_shared_ax0, X_shared_ax1, X_shared_ax2)
    X_shared_ax0_ax1_fused_ax2_fused_o, X_shared_ax0_ax1_fused_ax2_fused_i = s[X_shared].split(X_shared_ax0_ax1_fused_ax2_fused, factor=1)
    s[X_shared].vectorize(X_shared_ax0_ax1_fused_ax2_fused_i)
    X_shared_ax0_ax1_fused_ax2_fused_o_o, X_shared_ax0_ax1_fused_ax2_fused_o_i = s[X_shared].split(X_shared_ax0_ax1_fused_ax2_fused_o, factor=128)
    s[X_shared].bind(X_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))

    s[Z_local].pragma(Z_local_b_c_o_o_o_o, "auto_unroll_max_step", 0)
    s[Z_local].pragma(Z_local_b_c_o_o_o_o, "unroll_explicit", True)
    # fmt: on

    print(tvm.lower(s, [X, Y, Z]).script())
    tvm.build(s, [X, Y, Z], target=TARGET)


if __name__ == "__main__":
    main()

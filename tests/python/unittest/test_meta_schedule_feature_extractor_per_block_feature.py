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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import sys
from typing import Callable, List

from numpy.testing import assert_allclose

import tvm
from tvm import meta_schedule as ms
from tvm import te, tir
from tvm.script import tir as T

# N_FEATURES = 164


@T.prim_func
def matmul(
    A: T.Buffer[(512, 512), "float32"],
    B: T.Buffer[(512, 512), "float32"],
    C: T.Buffer[(512, 512), "float32"],
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

# from tvm.script import tir as T
@tvm.script.ir_module
class LayoutTransform:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 16, 7, 7, 32), "float32"], placeholder_1: T.Buffer[(25088,), "float32"], T_layout_trans: T.Buffer[(1, 1, 7, 7, 512), "float32"]) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        for i0_i1_i2_i3_i4_fused in T.parallel(25088, annotations={"pragma_auto_unroll_max_step":64, "pragma_unroll_explicit":1}):
            with T.block("T_layout_trans_1"):
                ax0 = T.axis.spatial(1, 0)
                ax1 = T.axis.spatial(1, 0)
                ax2 = T.axis.spatial(7, i0_i1_i2_i3_i4_fused // 3584)
                ax3 = T.axis.spatial(7, i0_i1_i2_i3_i4_fused % 3584 // 512)
                ax4 = T.axis.spatial(512, i0_i1_i2_i3_i4_fused % 512)
                T.reads(placeholder[0, (ax4 * 49 + ax2 * 7 + ax3) % 25088 // 1568, (ax2 * 7 + ax3) % 49 // 7, ax3 % 7, (ax4 * 49 + ax2 * 7 + ax3) % 1568 // 49], placeholder_1[(ax4 * 49 + ax2 * 7 + ax3) % 25088])
                T.writes(T_layout_trans[ax0, ax1, ax2, ax3, ax4])
                T_layout_trans[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(ax0 < 1 and ax1 * 512 + ax4 < 512 and ax2 < 7 and ax3 < 7, T.Select(T.float32(0) < T.if_then_else(0 < 1 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 < 512 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7 < 7 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7 < 7, placeholder[0, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 // 32, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 % 32], T.float32(0), dtype="float32"), T.if_then_else(0 < 1 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 < 512 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7 < 7 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7 < 7, placeholder[0, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 // 32, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 % 32], T.float32(0), dtype="float32"), T.if_then_else(0 < 1 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 < 512 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7 < 7 and ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7 < 7, placeholder[0, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 // 32, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 49 // 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 7, ((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088 % 25088 // 49 % 32], T.float32(0), dtype="float32") * placeholder_1[((ax1 * 512 + ax4) * 49 + ax2 * 7 + ax3) % 25088]), T.float32(0), dtype="float32")


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _make_context(target) -> ms.TuneContext:
    return ms.TuneContext(
        target=target,
        num_threads=1,
    )


def _make_candidate(f_sch: Callable[[], tir.Schedule]) -> ms.MeasureCandidate:
    return ms.MeasureCandidate(sch=f_sch(), args_info=[])


def _feature_names(  # pylint: disable=invalid-name
    buffers_per_block: int = 5,
    arith_intensity_curve_num_samples: int = 10,
) -> List[str]:
    result = [
        "float_mad",
        "float_addsub",
        "float_mul",
        "float_divmod",
        "float_cmp",
        "float_mathfunc",
        "float_otherfunc",
        "int_mad",
        "int_addsub",
        "int_mul",
        "int_divmod",
        "int_cmp",
        "int_mathfunc",
        "int_otherfunc",
        "bool_op",
        "select_op",
        "vec_num",
        "vec_prod",
        "vec_len",
        "vec_type.kPosNone",
        "vec_type.kPosInnerSpatial",
        "vec_type.kPosMiddleSpatial",
        "vec_type.kPosOuterSpatial",
        "vec_type.kPosInnerReduce",
        "vec_type.kPosMiddleReduce",
        "vec_type.kPosOuterReduce",
        "vec_type.kPosMixed",
        "unroll_num",
        "unroll_prod",
        "unroll_len",
        "unroll_type.kPosNone",
        "unroll_type.kPosInnerSpatial",
        "unroll_type.kPosMiddleSpatial",
        "unroll_type.kPosOuterSpatial",
        "unroll_type.kPosInnerReduce",
        "unroll_type.kPosMiddleReduce",
        "unroll_type.kPosOuterReduce",
        "unroll_type.kPosMixed",
        "parallel_num",
        "parallel_prod",
        "parallel_len",
        "parallel_type.kPosNone",
        "parallel_type.kPosInnerSpatial",
        "parallel_type.kPosMiddleSpatial",
        "parallel_type.kPosOuterSpatial",
        "parallel_type.kPosInnerReduce",
        "parallel_type.kPosMiddleReduce",
        "parallel_type.kPosOuterReduce",
        "parallel_type.kPosMixed",
        "is_gpu",
        "blockIdx_x_len",
        "blockIdx_y_len",
        "blockIdx_z_len",
        "threadIdx_x_len",
        "threadIdx_y_len",
        "threadIdx_z_len",
        "vthread_len",
    ]
    for i in range(buffers_per_block):
        result.extend(
            f"B{i}.{s}"
            for s in [
                "acc_type.kRead",
                "acc_type.kWrite",
                "acc_type.kReadWrite",
                "bytes",
                "unique_bytes",
                "lines",
                "unique_lines",
                "reuse_type.kLoopMultipleRead",
                "reuse_type.kSerialMultipleReadWrite",
                "reuse_type.kNoReuse",
                "reuse_dis_iter",
                "reuse_dis_bytes",
                "reuse_ct",
                "bytes_d_reuse_ct",
                "unique_bytes_d_reuse_ct",
                "lines_d_reuse_ct",
                "unique_lines_d_reuse_ct",
                "stride",
            ]
        )
    result.extend(f"arith_intensity_curve_{i}" for i in range(arith_intensity_curve_num_samples))
    result.extend(
        [
            "alloc_size",
            "alloc_prod",
            "alloc_outer_prod",
            "alloc_inner_prod",
            "outer_prod",
            "num_loops",
            "auto_unroll_max_step",
        ]
    )
    # 57 + 18 * 5 + 10 + 4 + 3
    # assert len(result) == N_FEATURES
    return result


def _zip_feature(feature, names):
    assert feature.ndim == 1
    # assert feature.shape[0] == N_FEATURES
    # assert len(names) == N_FEATURES
    return list(zip(names, feature))


def _print_feature(feature, st, ed):  # pylint: disable=invalid-name
    named_feature = _zip_feature(feature, _feature_names())
    for k, v in named_feature[st:ed]:
        print("\t", k, v)


def test_cpu_matmul():
    def _create_schedule():
        func = matmul
        sch = tir.Schedule(func, debug_mask="all")
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        i_o, i_i = sch.split(i, factors=[None, 16])  # outer: 32
        j_o, j_i = sch.split(j, factors=[None, 8])  # outer: 64
        sch.reorder(i_o, j_o, k, j_i, i_i)
        sch.vectorize(j_i)
        sch.parallel(i_o)
        sch.parallel(j_o)
        sch.unroll(k)
        return sch

    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    print(feature.shape)
    print(feature)
    # assert feature.shape == (1, N_FEATURES)


def test_cpu_fusion():
    # pylint: disable=all
    @T.prim_func
    def func(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [64, 32], dtype="float32")
        B = T.match_buffer(b, [64, 32], dtype="float32")
        C = T.match_buffer(c, [64, 32], dtype="float32")
        for i, j in T.grid(64, 32):  # type: ignore
            with T.block():
                T.reads([A[i, j], B[i, j]])  # type: ignore
                T.writes([B[i, j], C[i, j]])  # type: ignore
                with T.block("B"):
                    T.reads([A[i, j]])  # type: ignore
                    T.writes([B[i, j]])  # type: ignore
                    B[i, j] = A[i, j]  # type: ignore
                with T.block("C"):
                    T.reads([B[i, j]])  # type: ignore
                    T.writes([C[i, j]])  # type: ignore
                    C[i, j] = B[i, j]  # type: ignore

    # pylint: enable=all

    def _create_schedule():
        return tir.Schedule(func, debug_mask="all")

    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    # assert feature.shape == (2, N_FEATURES)


def test_gpu():
    def _create_schedule():
        func = matmul
        sch = tir.Schedule(func, debug_mask="all")
        c = sch.get_block("C")
        c_local = sch.cache_write(c, 0, "local")
        i, j, k = sch.get_loops(c)
        # pylint: disable=invalid-name
        i0, i1, i2, i3, i4 = sch.split(i, factors=[None, 1, 16, 32, 1])  # outer: 1
        j0, j1, j2, j3, j4 = sch.split(j, factors=[None, 4, 1, 1, 16])  # outer: 8
        k0, k1, k2 = sch.split(k, factors=[None, 1, 2])  # outer: 256
        # pylint: enable=invalid-name
        # fmt: off
        sch.reorder(
            i0, j0,  # S
            i1, j1,  # S
            i2, j2,  # S
            k0,      # R
            k1,      # R
            i3, j3,  # S
            k2,      # R
            i4, j4,  # S
        )
        # fmt: on
        # thread binding
        i0_j0 = sch.fuse(i0, j0)
        i1_j1 = sch.fuse(i1, j1)
        i2_j2 = sch.fuse(i2, j2)
        sch.bind(i0_j0, "blockIdx.x")
        sch.bind(i1_j1, "vthread.x")
        sch.bind(i2_j2, "threadIdx.x")
        # fusion
        sch.reverse_compute_at(c_local, i2_j2)
        # cache read 'A'
        a_shared = sch.cache_read(c, 1, "shared")
        sch.compute_at(a_shared, k0)
        _, _, _, _, a_i, a_j = sch.get_loops(a_shared)
        a_ij = sch.fuse(a_i, a_j)
        _, a_j = sch.split(a_ij, factors=[None, 16])  # outer: 64
        sch.bind(a_j, "threadIdx.x")
        # cache read 'B'
        b_shared = sch.cache_read(c, 2, "shared")
        sch.compute_at(b_shared, k0)
        _, _, _, _, b_i, b_j = sch.get_loops(b_shared)
        b_ij = sch.fuse(b_i, b_j)
        _, b_j = sch.split(b_ij, factors=[None, 16])  # outer: 8
        sch.bind(b_j, "threadIdx.x")
        # auto unroll
        sch.annotate(i0_j0, "pragma_auto_unroll_max_step", tir.IntImm("int32", 1024))
        sch.annotate(i0_j0, "pragma_unroll_explicit", tir.IntImm("int32", 1))
        return sch

    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("cuda")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    # assert feature.shape == (4, N_FEATURES)


def test_cpu_layout_transform():
    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("llvm")),
        candidates=[_make_candidate(lambda: tir.Schedule(LayoutTransform))],
    )


if __name__ == "__main__":
    test_cpu_matmul()

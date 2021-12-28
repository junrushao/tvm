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

from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.schedule_rule import (
    multi_level_tiling_memhammer,
    multi_level_tiling_memhammer_tensor_core,
)
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.te import create_prim_func
from tvm.meta_schedule.testing import te_workload
from tvm.target import Target
from tvm.meta_schedule.testing import tir_tensor_intrin


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx

def test_cuda_matmul():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'l1, l2, l3 = sch.get_loops(block=b0)',
            'v4, v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l1, n=5, max_innermost_factor=64)',
            'l9, l10, l11, l12, l13 = sch.split(loop=l1, factors=[v4, v5, v6, v7, v8])',
            'v14, v15, v16, v17, v18 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)',
            'l19, l20, l21, l22, l23 = sch.split(loop=l2, factors=[v14, v15, v16, v17, v18])',
            'v24, v25, v26 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=64)',
            'l27, l28, l29 = sch.split(loop=l3, factors=[v24, v25, v26])',
            'sch.reorder(l9, l19, l10, l20, l11, l21, l27, l28, l12, l22, l29, l13, l23)',
            'l30 = sch.fuse(l9, l19)',
            'sch.bind(loop=l30, thread_axis="blockIdx.x")',
            'l31 = sch.fuse(l10, l20)',
            'sch.bind(loop=l31, thread_axis="vthread.x")',
            'l32 = sch.fuse(l11, l21)',
            'sch.bind(loop=l32, thread_axis="threadIdx.x")',
            'b33 = sch.read_at(loop=l27, block=b0, read_buffer_index=1, storage_scope="shared")',
            'v34 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b0, ann_key="vector_bytes", ann_val=v34)',
            'sch.annotate(block_or_loop=b33, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b33, ann_key="double_buffer_scope", ann_val=0)',
            'b35 = sch.read_at(loop=l27, block=b0, read_buffer_index=2, storage_scope="shared")',
            'v36 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b0, ann_key="vector_bytes", ann_val=v36)',
            'sch.annotate(block_or_loop=b35, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b35, ann_key="double_buffer_scope", ann_val=0)',
            'sch.annotate(block_or_loop=l27, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 0, 1])',
            'sch.annotate(block_or_loop=l27, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3, 4])',
            'b37 = sch.write_at(loop=l32, block=b0, write_buffer_index=0, storage_scope="local")',
        ]
    ]
    # pylint: enable=line-too-long
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_memhammer(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'l1, l2, l3 = sch.get_loops(block=b0)',
            'v4, v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l1, n=5, max_innermost_factor=64)',
            'l9, l10, l11, l12, l13 = sch.split(loop=l1, factors=[v4, v5, v6, v7, v8])',
            'v14, v15, v16, v17, v18 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)',
            'l19, l20, l21, l22, l23 = sch.split(loop=l2, factors=[v14, v15, v16, v17, v18])',
            'v24, v25, v26 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=64)',
            'l27, l28, l29 = sch.split(loop=l3, factors=[v24, v25, v26])',
            'sch.reorder(l9, l19, l10, l20, l11, l21, l27, l28, l12, l22, l29, l13, l23)',
            'l30 = sch.fuse(l9, l19)',
            'sch.bind(loop=l30, thread_axis="blockIdx.x")',
            'l31 = sch.fuse(l10, l20)',
            'sch.bind(loop=l31, thread_axis="vthread.x")',
            'l32 = sch.fuse(l11, l21)',
            'sch.bind(loop=l32, thread_axis="threadIdx.x")',
            'b33 = sch.read_at(loop=l27, block=b0, read_buffer_index=1, storage_scope="shared")',
            'v34 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b0, ann_key="vector_bytes", ann_val=v34)',
            'sch.annotate(block_or_loop=b33, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b33, ann_key="double_buffer_scope", ann_val=0)',
            'b35 = sch.read_at(loop=l27, block=b0, read_buffer_index=2, storage_scope="shared")',
            'v36 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b0, ann_key="vector_bytes", ann_val=v36)',
            'sch.annotate(block_or_loop=b35, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b35, ann_key="double_buffer_scope", ann_val=0)',
            'sch.annotate(block_or_loop=l27, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 0, 1])',
            'sch.annotate(block_or_loop=l27, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3, 4])',
            'b37 = sch.write_at(loop=l32, block=b0, write_buffer_index=0, storage_scope="local")',
        ],
    ]
    # pylint: enable=line-too-long
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_memhammer(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_tensor_core_matmul():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'l1, l2, l3 = sch.get_loops(block=b0)',
            'l4, l5 = sch.split(loop=l1, factors=[32, 16])',
            'l6, l7 = sch.split(loop=l2, factors=[32, 16])',
            'l8, l9 = sch.split(loop=l3, factors=[32, 16])',
            'l10, l11, l12, l13, l14, l15 = sch.get_loops(block=b0)',
            'sch.reorder(l12, l14, l5, l7, l9)',
            'b16 = sch.blockize(loop=l5)',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")',
            'sch.annotate(block_or_loop=b16, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")',
            'b17 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b17, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")',
            'b18 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b18, ann_key="warp_execution", ann_val=1)',
            'l19, l20, l21 = sch.get_loops(block=b16)',
            'v22, v23, v24, v25, v26 = sch.sample_perfect_tile(loop=l19, n=5, max_innermost_factor=64)',
            'l27, l28, l29, l30, l31 = sch.split(loop=l19, factors=[v22, v23, v24, v25, v26])',
            'v32, v33, v34, v35, v36 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64)',
            'l37, l38, l39, l40, l41 = sch.split(loop=l20, factors=[v32, v33, v34, v35, v36])',
            'v42, v43, v44 = sch.sample_perfect_tile(loop=l21, n=3, max_innermost_factor=64)',
            'l45, l46, l47 = sch.split(loop=l21, factors=[v42, v43, v44])',
            'sch.reorder(l27, l37, l28, l38, l29, l39, l45, l46, l30, l40, l47, l31, l41)',
            'l48 = sch.fuse(l27, l37)',
            'sch.bind(loop=l48, thread_axis="blockIdx.x")',
            'l49 = sch.fuse(l28, l38)',
            'sch.bind(loop=l49, thread_axis="blockIdx.y")',
            'l50 = sch.fuse(l29, l39)',
            'sch.bind(loop=l50, thread_axis="threadIdx.y")',
            'b51 = sch.read_at(loop=l45, block=b16, read_buffer_index=1, storage_scope="shared")',
            'v52 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b16, ann_key="vector_bytes", ann_val=v52)',
            'sch.annotate(block_or_loop=b51, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b51, ann_key="double_buffer_scope", ann_val=0)',
            'b53 = sch.read_at(loop=l45, block=b16, read_buffer_index=2, storage_scope="shared")',
            'v54 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b16, ann_key="vector_bytes", ann_val=v54)',
            'sch.annotate(block_or_loop=b53, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b53, ann_key="double_buffer_scope", ann_val=0)',
            'b55 = sch.read_at(loop=l46, block=b16, read_buffer_index=1, storage_scope="wmma.matrix_a")',
            'b56 = sch.read_at(loop=l46, block=b16, read_buffer_index=2, storage_scope="wmma.matrix_b")',
            'sch.annotate(block_or_loop=l46, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])',
            'sch.annotate(block_or_loop=l46, ann_key="software_pipeline_order", ann_val=[0, 1, 2])',
            'sch.annotate(block_or_loop=l45, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 0, 0, 1, 1])',
            'sch.annotate(block_or_loop=l45, ann_key="software_pipeline_order", ann_val=[0, 3, 1, 4, 5, 2, 6])',
            'b57 = sch.write_at(loop=l50, block=b16, write_buffer_index=0, storage_scope="wmma.accumulator")',
            'v58 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b57, ann_key="vector_bytes", ann_val=v58)',
        ]
    ]
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_fp16(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_memhammer_tensor_core(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_tensor_core_matmul_relu():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'l1, l2, l3 = sch.get_loops(block=b0)',
            'l4, l5 = sch.split(loop=l1, factors=[32, 16])',
            'l6, l7 = sch.split(loop=l2, factors=[32, 16])',
            'l8, l9 = sch.split(loop=l3, factors=[32, 16])',
            'l10, l11, l12, l13, l14, l15 = sch.get_loops(block=b0)',
            'sch.reorder(l12, l14, l5, l7, l9)',
            'b16 = sch.blockize(loop=l5)',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")',
            'sch.annotate(block_or_loop=b16, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")',
            'b17 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b17, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")',
            'b18 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b18, ann_key="warp_execution", ann_val=1)',
            'l19, l20, l21 = sch.get_loops(block=b16)',
            'v22, v23, v24, v25, v26 = sch.sample_perfect_tile(loop=l19, n=5, max_innermost_factor=64)',
            'l27, l28, l29, l30, l31 = sch.split(loop=l19, factors=[v22, v23, v24, v25, v26])',
            'v32, v33, v34, v35, v36 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64)',
            'l37, l38, l39, l40, l41 = sch.split(loop=l20, factors=[v32, v33, v34, v35, v36])',
            'v42, v43, v44 = sch.sample_perfect_tile(loop=l21, n=3, max_innermost_factor=64)',
            'l45, l46, l47 = sch.split(loop=l21, factors=[v42, v43, v44])',
            'sch.reorder(l27, l37, l28, l38, l29, l39, l45, l46, l30, l40, l47, l31, l41)',
            'l48 = sch.fuse(l27, l37)',
            'sch.bind(loop=l48, thread_axis="blockIdx.x")',
            'l49 = sch.fuse(l28, l38)',
            'sch.bind(loop=l49, thread_axis="blockIdx.y")',
            'l50 = sch.fuse(l29, l39)',
            'sch.bind(loop=l50, thread_axis="threadIdx.y")',
            'b51 = sch.read_at(loop=l45, block=b16, read_buffer_index=1, storage_scope="shared")',
            'v52 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b16, ann_key="vector_bytes", ann_val=v52)',
            'sch.annotate(block_or_loop=b51, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b51, ann_key="double_buffer_scope", ann_val=0)',
            'b53 = sch.read_at(loop=l45, block=b16, read_buffer_index=2, storage_scope="shared")',
            'v54 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b16, ann_key="vector_bytes", ann_val=v54)',
            'sch.annotate(block_or_loop=b53, ann_key="local_stage", ann_val=1)',
            'sch.annotate(block_or_loop=b53, ann_key="double_buffer_scope", ann_val=0)',
            'b55 = sch.read_at(loop=l46, block=b16, read_buffer_index=1, storage_scope="wmma.matrix_a")',
            'b56 = sch.read_at(loop=l46, block=b16, read_buffer_index=2, storage_scope="wmma.matrix_b")',
            'sch.annotate(block_or_loop=l46, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])',
            'sch.annotate(block_or_loop=l46, ann_key="software_pipeline_order", ann_val=[0, 1, 2])',
            'sch.annotate(block_or_loop=l45, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 0, 0, 1, 1])',
            'sch.annotate(block_or_loop=l45, ann_key="software_pipeline_order", ann_val=[0, 3, 1, 4, 5, 2, 6])',
            'b57 = sch.write_at(loop=l50, block=b16, write_buffer_index=0, storage_scope="wmma.accumulator")',
            'v58 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331])',
            'sch.annotate(block_or_loop=b57, ann_key="vector_bytes", ann_val=v58)',
        ]
    ]
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu_fp16(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_memhammer_tensor_core(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":

    test_cuda_matmul()
    test_cuda_matmul_relu()
    test_cuda_tensor_core_matmul()
    test_cuda_tensor_core_matmul_relu()

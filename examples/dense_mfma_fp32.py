import argparse
import os
import tvm
from tvm import te, tir
from tvm import meta_schedule as ms
from tvm.topi.utils import get_const_tuple
from tvm.script import ty
import numpy as np

np.set_printoptions(threshold=10000)
parser = argparse.ArgumentParser()
parser.add_argument("--M", default=1904, type=int)
parser.add_argument("--N", default=3072, type=int)
parser.add_argument("--K", default=2304, type=int)
parser.add_argument("--target", default="rocm -mcpu=gfx908")
parser.add_argument("--tune", action="store_true", default=False)
parser.add_argument("--log_file", default="dense_mfma_32.log", type=str)
parser.add_argument("--num_trials", type=int, default=1500)
parser.add_argument('--export', default=False, action='store_true')
parser.add_argument('--run_rocblas', default=False, action='store_true')


args = parser.parse_args()

@tvm.script.tir
def intrin_mfma_load_matrix_desc(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 4), "float32", scope="shared")
    B = tir.match_buffer(b, (16, 4), "float32", scope="local.special.64")
    with tir.block([16, 4], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[vi : vi + 16, vj : vj + 4])
        tir.writes(B[vi : vi + 16, vj : vj + 4])
        for i, j in tir.grid(16, 4):
            with tir.block([16, 4], "B") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                B[vii, vjj] = A[vii, vjj]


@tvm.script.tir
def intrin_mfma_load_matrix_a_impl(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 4), "float32", scope="shared")
    B = tir.match_buffer(
        b, (16, 4), "float32", scope="local.special.64", offset_factor=1
    )
    with tir.block([16, 4], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[vi : vi + 16, vj : vj + 4])
        tir.writes(B[vi : vi + 16, vj : vj + 4])
        tx = tir.env_thread("threadIdx.x")
        tir.launch_thread(tx, 64)
        for v in tir.unroll(0, 1):
            tir.store(
                B.data,
                (
                    tir.floordiv(B.elem_offset, 64)
                    + tir.floordiv(tir.floormod(B.elem_offset, 64), 4)
                )
                + v,
                A[vi + tir.floormod(tx, 16), vj + tir.floordiv(tx, 16) + v],
                True,
            )


@tvm.script.tir
def intrin_mfma_f32_16x16x4_f32_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(
        a, (16, 4), "float32", scope="local.special.64", offset_factor=1
    )
    B = tir.match_buffer(
        b, (16, 4), "float32", scope="local.special.64", offset_factor=1
    )
    C = tir.match_buffer(
        c, (16, 16), "float32", scope="local.special.64", offset_factor=1
    )
    with tir.block([16, 16, tir.reduce_axis(0, 4)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads(
            [
                C[vi : vi + 16, vj : vj + 16],
                A[vi : vi + 16, vk : vk + 4],
                B[vj : vj + 16, vk : vk + 4],
            ]
        )
        tir.writes(C[vi : vi + 16, vj : vj + 16])
        for i, j, k in tir.grid(16, 16, 4):
            with tir.block([16, 16, tir.reduce_axis(0, 4)], "B") as [vii, vjj, vkk]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@tvm.script.tir
def intrin_mfma_f32_16x16x4_f32_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(
        a, (16, 4), "float32", scope="local.special.64", offset_factor=1
    )
    B = tir.match_buffer(
        b, (16, 4), "float32", scope="local.special.64", offset_factor=1
    )
    C = tir.match_buffer(
        c, (16, 16), "float32", scope="local.special.64", offset_factor=1
    )
    with tir.block([16, 16, tir.reduce_axis(0, 4)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads(
            [
                C[vi : vi + 16, vj : vj + 4],
                A[vi : vi + 16, vk : vk + 4],
                B[vj : vj + 16, vk : vk + 4],
            ]
        )
        tir.writes(C[vi : vi + 16, vj : vj + 4])
        tir.store(
            C.data,
            tir.ramp(
                (
                    tir.floordiv(C.elem_offset, 256)
                    + tir.floordiv(tir.floormod(C.elem_offset, 256), 16)
                )
                * 4,
                1,
                4,
            ),
            tir.call_llvm_pure_intrin(
                tir.llvm_lookup_intrinsic_id("llvm.amdgcn.mfma.f32.16x16x4f32"),
                tir.uint32(6),
                tir.load(
                    "float32",
                    B.data,
                    (
                        tir.floordiv(B.elem_offset, 64)
                        + tir.floordiv(tir.floormod(B.elem_offset, 64), 4)
                    ),
                    True,
                ),
                tir.load(
                    "float32",
                    A.data,
                    (
                        tir.floordiv(A.elem_offset, 64)
                        + tir.floordiv(tir.floormod(A.elem_offset, 64), 4)
                    ),
                    True,
                ),
                tir.load(
                    "float32x4",
                    C.data,
                    tir.ramp(
                        (
                            tir.floordiv(C.elem_offset, 256)
                            + tir.floordiv(tir.floormod(C.elem_offset, 256), 16)
                        )
                        * 4,
                        1,
                        4,
                    ),
                    tir.broadcast(True, 4),
                ),
                0,
                0,
                0,
                dtype="float32x4",
            ),
            tir.broadcast(True, 4),
        )


@tvm.script.tir
def intrin_mfma_store_matrix_desc(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32", scope="local.special.64")
    B = tir.match_buffer(b, (16, 16), "float32", scope="shared")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "B") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                B[vii, vjj] = A[vii, vjj]


@tvm.script.tir
def intrin_mfma_store_matrix_impl(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32", scope="local", offset_factor=1)
    B = tir.match_buffer(b, (16, 16), "float32", scope="shared")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.writes(B[vi : vi + 16, vj : vj + 16])
        tir.reads(A[vi : vi + 16, vj : vj + 16])
        tx = tir.env_thread("threadIdx.x")
        tir.launch_thread(tx, 64)
        for v in tir.vectorized(0, 4):
            B[tir.floormod(tx, 16) + vi, vj + tir.floordiv(tx, 16) * 4 + v] = tir.load(
                "float32",
                A.data,
                (
                    tir.floordiv(A.elem_offset, 256)
                    + tir.floordiv(tir.floormod(A.elem_offset, 256), 16)
                )
                * 4
                + v,
                True,
            )


@tvm.script.tir
def intrin_mfma_init_desc(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32", scope="local.special.64")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "B") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                A[vii, vjj] = tir.float32(0.0)


@tvm.script.tir
def intrin_mfma_init_impl(a: ty.handle) -> None:
    A = tir.match_buffer(
        a, (16, 16), "float32", scope="local.special.64", offset_factor=1
    )
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads([])
        tir.writes(A[vi : vi + 16, vj : vj + 16])
        tir.store(
            A.data,
            tir.ramp(
                (
                    tir.floordiv(A.elem_offset, 256)
                    + tir.floordiv(tir.floormod(A.elem_offset, 256), 16)
                )
                * 4,
                1,
                4,
            ),
            tir.broadcast(tir.float32(0.0), 4),
            tir.broadcast(True, 4),
        )


tir.TensorIntrin.register(
    "intrin_mfma_init",
    intrin_mfma_init_desc,
    intrin_mfma_init_impl,
)

tir.TensorIntrin.register(
    "intrin_mfma_load_a_matrix",
    intrin_mfma_load_matrix_desc,
    intrin_mfma_load_matrix_a_impl,
)


tir.TensorIntrin.register(
    "intrin_mfma_f32_16x16x4_f32",
    intrin_mfma_f32_16x16x4_f32_desc,
    intrin_mfma_f32_16x16x4_f32_impl,
)


tir.TensorIntrin.register(
    "intrin_mfma_store_matrix",
    intrin_mfma_store_matrix_desc,
    intrin_mfma_store_matrix_impl,
)


def dense_mfma_rocm(data, weight, bias=None, out_dtype=None):
    """Dense MFMA operator on ROCM"""
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    assert (
        batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0
    ), "batch, in_dim, and out_dim each must be a multiple of 16"
    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(
            data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k
        ),
        name="T_dense",
        tag="dense_mfma",
    )
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
            name="T_bias",
            tag=tag.BROADCAST,
        )
    return matmul


def get_autotir_workload(M, N, K):
    data = te.placeholder((M, K), name="data", dtype="float32")
    weight = te.placeholder((N, K), name="weight", dtype="float32")
    matmul = dense_mfma_rocm(data, weight)
    func = te.create_func(matmul)
    return func


def get_rocblas_func(M, N, K):
    data = te.placeholder((M, K), name="data", dtype="float32")
    weight = te.placeholder((N, K), name="weight", dtype="float32")
    mm = tvm.contrib.rocblas.matmul(data, weight, False, True)
    s = tvm.topi.generic.schedule_extern([mm])
    with tvm.target.Target(target):
        func = tvm.build(s, [data, weight, mm])
    return func


def schedule_fn(sch):
    matmul = sch.get_block("T_dense")

    order = sch.sample_categorical([0, 1], probs=[1.0 / 2, 1.0 / 2])

    # Explicit memory access
    AS = sch.cache_read(matmul, 1, "shared")
    BS = sch.cache_read(matmul, 2, "shared")
    AF = sch.cache_read(matmul, 1, "local.special.64")
    BF = sch.cache_read(matmul, 2, "local.special.64")

    CS = sch.cache_write(matmul, 0, "shared")
    CF = sch.cache_write(CS, 0, "local.special.64")

    warp_size = 64
    mfma_m = mfma_n = 16
    mfma_k = 4

    chunk = sch.sample_categorical(
        [1, 2, 4, 8], probs=[1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]
    )
    offset = sch.sample_categorical([0, 8], probs=[1.0 / 2, 1.0 / 2])
    offsetCS = sch.sample_categorical([0, 8], probs=[1.0 / 2, 1.0 / 2])
    vec = sch.sample_categorical(
        [1, 2, 4, 8], probs=[1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]
    )

    # Define the stride for tensorization
    chunk = sch.get(chunk)
    offset = sch.get(offset)
    offsetCS = sch.get(offsetCS)

    # Schedule for fragment store
    bb, oo, k = sch.get_axes(CF)
    bb, bbi = sch.split(bb, factors=[None, mfma_m])
    oo, ooi = sch.split(oo, factors=[None, mfma_n])

    # use max_innermost_factor to control tile size
    _, warp_row_tiles = sch.sample_perfect_tile(n=2, loop=bb, max_innermost_factor=4)
    bb, bbii = sch.split(bb, factors=[None, warp_row_tiles])
    _, warp_col_tiles = sch.sample_perfect_tile(n=2, loop=oo, max_innermost_factor=4)
    oo, ooii = sch.split(oo, factors=[None, warp_col_tiles])
    _, block_row_warps = sch.sample_perfect_tile(n=2, loop=bb, max_innermost_factor=4)
    block_b, bb = sch.split(bb, factors=[None, block_row_warps])
    _, block_col_warps = sch.sample_perfect_tile(n=2, loop=oo, max_innermost_factor=4)
    block_o, oo = sch.split(oo, factors=[None, block_col_warps])
    sch.reorder(block_b, block_o, bb, oo, bbii, ooii, bbi, ooi)
    # each warp (64 threads indexed by threadIdx.x) handles mfma_m * mfma_n elements
    # we will use warp-level premitive to tensorize, which transposes a 16x16 matrix
    # Schedule for dense computation

    sch.bind(bb, "threadIdx.y")
    sch.bind(oo, "threadIdx.z")
    sch.bind(block_b, "blockIdx.x")
    sch.bind(block_o, "blockIdx.y")

    sch.reverse_compute_at(CS, oo)
    sch.reverse_compute_at(matmul, block_o)

    AS_align = chunk * mfma_k + offset
    BS_align = chunk * mfma_k + offset
    CS_align = sch.get(warp_col_tiles) * sch.get(block_col_warps) * mfma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [mfma_k, 1]
    BF_stride = [mfma_k, 1]
    CF_stride = [warp_col_tiles * mfma_n, 1]
    CS_stride = [CS_align, 1]

    sch.storage_align(CS, 0, -2, CS_align - 1, CS_align)

    t = sch.fuse(*sch.get_axes(matmul)[-2:])
    _, tz, ty, tx, vi = sch.split(
        t, factors=[None, block_col_warps, block_row_warps, warp_size, vec]
    )
    sch.bind(tz, "threadIdx.z")
    sch.bind(ty, "threadIdx.y")
    sch.bind(tx, "threadIdx.x")
    sch.vectorize(vi)

    # Schedule for gemm computation
    warp_i, warp_j, _ii, _jj, k = sch.get_axes(CF)[-5:]
    ko, ki, _kk = sch.split(k, factors=[None, chunk, mfma_k])
    sch.reorder(ko, ki, warp_i, warp_j, _ii, _jj, _kk)

    if sch.get(order):
        sch.compute_at(BF, ki)
        sch.compute_at(AF, ki)
    else:
        sch.compute_at(AF, ki)
        sch.compute_at(BF, ki)
    # Schedule for tensorized matrix_A load
    b, i = sch.get_axes(AF)[-2:]
    b, b_ii = sch.split(b, factors=[None, mfma_m])
    i, i_jj = sch.split(i, factors=[None, mfma_k])
    sch.reorder(b, i, b_ii, i_jj)

    # Schedule for tensorized matrix_B load
    o, i = sch.get_axes(BF)[-2:]
    o, o_ii = sch.split(o, factors=[None, mfma_n])
    i, i_ii = sch.split(i, factors=[None, mfma_k])
    sch.reorder(o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_schedule(stage, stride):
        sch.compute_at(stage, ko)
        sch.storage_align(stage, 0, -2, stride - 1, stride)
        xo, yo = sch.get_axes(stage)[-2:]
        t = sch.fuse(xo, yo)
        _, tz, ty, tx, vi = sch.split(
            t, factors=[None, block_col_warps, block_row_warps, warp_size, vec]
        )
        sch.bind(tz, "threadIdx.z")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vi)

    if sch.get(order):
        shared_schedule(BS, BS_align)
        shared_schedule(AS, AS_align)
    else:
        shared_schedule(AS, AS_align)
        shared_schedule(BS, BS_align)

    init = sch.decompose_reduction(CF, ko)

    sch.tensorize(sch.get_axes(init)[-2], "intrin_mfma_init")
    sch.tensorize(b_ii, "intrin_mfma_load_a_matrix")
    sch.tensorize(o_ii, "intrin_mfma_load_a_matrix")
    sch.tensorize(_ii, "intrin_mfma_f32_16x16x4_f32")

    bb, ii = sch.get_axes(CS)[-2:]
    bb, bbi = sch.split(bb, factor=mfma_m)
    ii, iii = sch.split(ii, factor=mfma_n)
    sch.reorder(bb, ii, bbi, iii)
    sch.tensorize(bbi, "intrin_mfma_store_matrix")


def schedule(workload):
    task = ms.SearchTask(workload, "dense_mfma", target=target, log_file=args.log_file)
    space = ms.space.ScheduleFn(
        schedule_fn,
        postprocs=[ms.postproc.disallow_dynamic_loops(), ms.postproc.verify_gpu_code()],
    )

    strategy = ms.strategy.Replay(args.num_trials)
    measurer = ms.ProgramMeasurer(measure_callbacks=[ms.RecordToFile()])
    if args.tune:
        sch = ms.autotune(task=task, space=space, strategy=strategy, measurer=measurer)
        space.postprocess(task, sch)

    else:
        print("Apply history best from log file {}".format(args.log_file))
        strategy = ms.strategy.Replay(num_trials=0)
        sch = ms.autotune(
            task=task, space=space, strategy=strategy, measurer=measurer
        )
        if sch is None:
            sch = space.sample_schedule(task)
        else:
            space.postprocess(task, sch)
    print(tvm.script.asscript(sch.mod["main"]))

    return sch


if __name__ == "__main__":
    M = args.M
    N = args.N
    K = args.K
    target = args.target
    ctx = tvm.context(target, 0)
    func = get_autotir_workload(M, N, K)
    sch = schedule(func)

    # verify
    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(N, K)).astype("float32")
    c_np = a_np @ b_np.T
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    b_tvm = tvm.nd.array(b_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)


    func = tvm.build(sch.mod["main"], None, target)
    func(a_tvm, b_tvm, c_tvm)
    np.testing.assert_allclose(c_np, c_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
    evaluator = func.time_evaluator(
        func.entry_name, ctx, number=200, repeat=3, min_repeat_ms=40
    )
    ms = evaluator(a_tvm, b_tvm, c_tvm).mean * 1e3
    GFLOPS = M * N * K * 2 / (1e9) / (ms / 1e3)
    print(
        "TVM Result: M = {} N = {} K = {} {} ms {} GFLOPS".format(
            args.M, args.N, args.K, ms, GFLOPS
        )
    )

    if args.export:
        func.export_library('lib/dense_fp32_{}x{}x{}_fp32.so'.format(args.M, args.K, args.N))

    if args.run_rocblas:
        func = get_rocblas_func(M, N, K)
        c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
        func(a_tvm, b_tvm, c_tvm)
        np.testing.assert_allclose(c_np, c_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
        evaluator = func.time_evaluator(
            func.entry_name, ctx, number=200, repeat=3, min_repeat_ms=40
        )
        ms = evaluator(a_tvm, b_tvm, c_tvm).mean * 1e3
        GFLOPS = M * N * K * 2 / (1e9) / (ms / 1e3)
        print(
            "Rocblas Result: M = {} N = {} K = {} {} ms {} GFLOPS".format(
                args.M, args.N, args.K, ms, GFLOPS
            )
        )

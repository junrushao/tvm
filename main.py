import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def func(
    p0: T.Buffer[(1, 64, 56, 56), "float32"],
    p1: T.Buffer[(6, 6, 64, 64), "float32"],
    p2: T.Buffer[(1, 64, 1, 1), "float32"],
    output: T.Buffer[(1, 64, 56, 56), "float32"],
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    data_pad = T.alloc_buffer([1, 64, 58, 58], dtype="float32")
    d = T.alloc_buffer([64, 196, 6, 6], dtype="float32")
    B = T.alloc_buffer([6, 6], dtype="float32")
    data_pack = T.alloc_buffer([6, 6, 64, 196], dtype="float32")
    bgemm = T.alloc_buffer([6, 6, 64, 196], dtype="float32")
    A = T.alloc_buffer([6, 4], dtype="float32")
    inverse = T.alloc_buffer([64, 196, 4, 4], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 64, 58, 58):
        with T.block("data_pad"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(p0[i0_1, i1_1, i2_1 - 1, i3_1 - 1])
            T.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
            # fmt: off
            data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i2_1 and i2_1 < 57 and 1 <= i3_1 and i3_1 < 57, p0[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
            # fmt: on
    for i0, i1, i2, i3 in T.grid(64, 196, 6, 6):
        with T.block("d"):
            c, p, eps, nu = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(data_pad[p // 196, c, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu])
            T.writes(d[c, p, eps, nu])
            d[c, p, eps, nu] = data_pad[p // 196, c, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu]
    for i0, i1 in T.grid(6, 6):
        with T.block("B"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads()
            T.writes(B[i, j])
            # T.block_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
            # fmt: off
            B[i, j] = T.Select(i % 6 == 5 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 5 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 6 == 5, T.float32(1.5), T.Select(i % 6 == 4 and j % 6 == 4, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 3, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 2, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 6 == 5, T.float32(-2), T.Select(i % 6 == 3 and j % 6 == 4, T.float32(-0.5), T.Select(i % 6 == 3 and j % 6 == 3, T.float32(2), T.Select(i % 6 == 3 and j % 6 == 2, T.float32(2.5), T.Select(i % 6 == 3 and j % 6 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 6 == 0, T.float32(1.5), T.Select(i % 6 == 2 and j % 6 == 5, T.float32(-1.5), T.Select(i % 6 == 2 and j % 6 == 4, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 3, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 2, T.float32(0.5), T.Select(i % 6 == 2 and j % 6 == 1, T.float32(-2.5), T.Select(i % 6 == 2 and j % 6 == 0, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 4, T.float32(0.5), T.Select(i % 6 == 1 and j % 6 == 3, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 2, T.float32(-1), T.Select(i % 6 == 1 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 0, T.float32(-1.5), T.Select(i % 6 == 0 and j % 6 == 5, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
            # fmt: on
    for i0, i1, i2, i3, i4, i5 in T.grid(6, 6, 64, 196, 6, 6):
        with T.block("data_pack"):
            eps, nu, ci, p, r_a, r_a_1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            T.reads(
                d[ci, p, r_a, r_a_1],
                B[T.min(r_a, r_a_1) : T.max(r_a, r_a_1) + 1, T.min(eps, nu) : T.max(eps, nu) + 1],
            )
            T.writes(data_pack[eps, nu, ci, p])
            # T.block_attr({"schedule_rule":"meta_schedule.winograd_data_pack.nchw.cuda"})
            with T.init():
                data_pack[eps, nu, ci, p] = T.float32(0)
            data_pack[eps, nu, ci, p] = (
                data_pack[eps, nu, ci, p] + d[ci, p, r_a, r_a_1] * B[r_a, eps] * B[r_a_1, nu]
            )
    for i0, i1, i2, i3, i4 in T.grid(6, 6, 64, 196, 64):
        with T.block("bgemm"):
            eps, nu, co, p, ci = T.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
            T.reads(p1[eps, nu, ci, co], data_pack[eps, nu, ci, p])
            T.writes(bgemm[eps, nu, co, p])
            with T.init():
                bgemm[eps, nu, co, p] = T.float32(0)
            bgemm[eps, nu, co, p] = (
                bgemm[eps, nu, co, p] + p1[eps, nu, ci, co] * data_pack[eps, nu, ci, p]
            )
    for i0, i1 in T.grid(6, 4):
        with T.block("A"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads()
            T.writes(A[i, j])
            # T.block_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
            # fmt: off
            A[i, j] = T.Select(i % 6 == 5 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 5 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 4 == 3, T.float32(-8), T.Select(i % 6 == 4 and j % 4 == 2, T.float32(4), T.Select(i % 6 == 4 and j % 4 == 1, T.float32(-2), T.Select(i % 6 == 4 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 4 == 3, T.float32(0.125), T.Select(i % 6 == 3 and j % 4 == 2, T.float32(0.25), T.Select(i % 6 == 3 and j % 4 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
            # fmt: on
    for i0, i1, i2, i3, i4, i5 in T.grid(64, 196, 4, 4, 6, 6):
        with T.block("inverse"):
            co, p, vh, vw, r_a_2, r_a_3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            T.reads(
                bgemm[r_a_2, r_a_3, co, p],
                A[T.min(r_a_2, r_a_3) : T.max(r_a_2, r_a_3) + 1, T.min(vh, vw) : T.max(vh, vw) + 1],
            )
            T.writes(inverse[co, p, vh, vw])
            # T.block_attr({"schedule_rule":"meta_schedule.winograd_inverse.nchw.cuda"})
            with T.init():
                inverse[co, p, vh, vw] = T.float32(0)
            inverse[co, p, vh, vw] = (
                inverse[co, p, vh, vw] + bgemm[r_a_2, r_a_3, co, p] * A[r_a_2, vh] * A[r_a_3, vw]
            )
    for i0, i1, i2, i3 in T.grid(1, 64, 56, 56):
        with T.block("output"):
            n, co, h, w = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(inverse[co, n * 196 + h // 4 * 14 + w // 4, h % 4, w % 4])
            T.writes(output[n, co, h, w])
            # T.block_attr({"schedule_rule":"meta_schedule.winograd_output.nchw.cuda", "winograd_tile_size":4, "workload":["conv2d_nchw_winograd_without_weight_transform.cuda", ["TENSOR", [1, 64, 56, 56], "float32"], ["TENSOR", [6, 6, 64, 64], "float32"], [1, 1], [1, 1, 1, 1], [1, 1], "float32"]})
            output[n, co, h, w] = inverse[co, n * 196 + h // 4 * 14 + w // 4, h % 4, w % 4]


def schedule_data_pack(sch: tir.Schedule, data_pack: tir.schedule.BlockRV):
    loops = sch.get_loops(data_pack)

    # factors = sch.sample_perfect_tile(loops[2], n=2, max_innermost_factor=64)
    # t0 = sch.split(loops[2], factors)
    #
    # factors = sch.sample_perfect_tile(loops[3], n=2, max_innermost_factor=64)
    # t1 = sch.split(loops[3], factors)

    # sch.unroll(loops[0])
    # sch.unroll(loops[1])
    # sch.unroll(loops[4])
    # sch.unroll(loops[5])
    # sch.reorder(
    #     t0[0],
    #     t1[0],
    #     t0[1],
    #     t1[1],
    #     loops[0],
    #     loops[1],
    #     loops[4],
    #     loops[5],
    # )
    # return t1[1]

    # sch.unroll(loops[0])
    # sch.unroll(loops[1])
    # sch.unroll(loops[4])
    # sch.unroll(loops[5])
    t0_t1 = sch.fuse(loops[2], loops[3])
    t0, t1 = sch.split(t0_t1, factors=[None, 128])
    sch.reorder(
        t0,
        t1,
        loops[0],
        loops[1],
        loops[4],
        loops[5],
    )
    return t1


def main():
    sch = tir.Schedule(func)
    sch.compute_inline(sch.get_block("A"))
    sch.compute_inline(sch.get_block("B"))
    # data_pack
    data_pack = sch.get_block("data_pack")
    (input_tile,) = sch.get_producers(data_pack)
    (data_pad,) = sch.get_producers(input_tile)
    loop = schedule_data_pack(sch, data_pack)
    # sch->ComputeAt(input_tile, /*loop_rv=*/loop, /*preserve_unit_loops=*/true);
    # sch->SetScope(input_tile, /*buffer_index=*/0, /*storage_scope=*/"local");
    # sch->ComputeInline(data_pad);
    sch.compute_at(input_tile, loop, preserve_unit_loops=True)
    sch.set_scope(input_tile, 0, "local")
    sch.compute_inline(data_pad)

    tvm.lower(sch.mod).show()


if __name__ == "__main__":
    main()

import argparse
import os
import tvm
from tvm import te, tir, topi
from tvm import meta_schedule as ms
from tvm.topi.utils import get_const_tuple
from tvm.script import ty
import tvm.testing
import tvm.topi.testing
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--B", default=2048, type=int)
parser.add_argument("--C", default=30528, type=int)
parser.add_argument("--target", default="rocm -mcpu=gfx908")
parser.add_argument("--tune", action="store_true", default=False)
parser.add_argument("--log_file", default="log_softmax_fp32.log", type=str)
parser.add_argument("--num_trials", type=int, default=1500)
parser.add_argument('--export', default=False, action='store_true')


args = parser.parse_args()
def get_workload(B, C):
    x = te.placeholder((B, C), name='x')
    logsoftmax = topi.nn.log_softmax(x)
    return (x, logsoftmax)

def _create_task():
    _, workload = get_workload(args.B, args.C)
    workload = te.create_func(workload)
    return ms.SearchTask(
        workload=workload,
        task_name='log_softmax',
        target=args.target,
        log_file=args.log_file,
    )


def _create_strategy():
    return ms.strategy.Replay(args.num_trials)


def rocm_space():
    return ms.space.PostOrderApply(
        stages=[
            ms.rule.inline_pure_spatial(strict_mode=False),
            ms.rule.cross_thread_reduction(),
            ms.rule.multi_level_tiling(
                structure="SSSRRSRS",
                must_cache_read=True,
                cache_read_scope="shared",
                can_cache_write=True,
                must_cache_write=True,
                cache_write_scope="local",
                consumer_inline_strict=False,
                fusion_levels=[3],
                vector_load_max_len=4,
                tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
            ),
            ms.rule.parallelize_vectorize_unroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ],
        postprocs=[
            ms.postproc.rewrite_reduction_block(),
            ms.postproc.rewrite_cooperative_fetch(),
            ms.postproc.rewrite_unbound_blocks(),
            ms.postproc.rewrite_parallel_vectorize_unroll(),
            ms.postproc.disallow_dynamic_loops(),
            ms.postproc.verify_gpu_code(),
        ],
    )

def _create_measurer():
    return ms.ProgramMeasurer(
        builder=ms.LocalBuilder(),
        runner=ms.RPCRunner(),
        measure_callbacks=[
            ms.RecordToFile(),
        ],
    )


def main():
    sch = ms.autotune(
        task=_create_task(),
        space=rocm_space(),
        strategy=_create_strategy(),
        measurer=_create_measurer(),
    )
    if sch is None:
        print("No valid schedule found")
        return

    print(tvm.script.asscript(sch.mod))

    ctx = tvm.context('rocm', 0)
    a_np = np.random.uniform(size=(args.B, args.C)).astype('float32')
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    c_tvm = tvm.nd.empty(shape=a_tvm.shape, dtype=a_tvm.dtype, ctx=ctx)
    with tvm.target.Target(args.target):
        func = tvm.build(sch.mod['main'])

    func(a_tvm, c_tvm)
    c_ref = tvm.topi.testing.log_softmax_python(a_np)
    tvm.testing.assert_allclose(c_tvm.asnumpy(), c_ref, rtol=1e-4)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=20, repeat=1, min_repeat_ms=40)
    time_ms = evaluator(a_tvm, c_tvm).mean * 1e3
    print('TVM: {} ms'.format(time_ms))

    if args.export:
        with tvm.target.Target(args.target):
            filename = 'log_softmax_{}x{}.so'.format(args.B, args.C)
            print('Export to {}'.format(filename))
            func.export_library(filename)



if __name__=='__main__':
    main()

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
import argparse
import logging
from os import cpu_count
from typing import Optional

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.testing.te_workload import create_te_workload


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-workers",
        type=int,
        required=True,
    )
    args.add_argument(
        "--as-db",
        type=str,
        required=True,
    )
    args.add_argument(
        "--ms-db",
        type=str,
        required=True,
    )
    args.add_argument(
        "--ms-wkl",
        type=str,
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=60,
    )
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()
FIRST_NUM = 0


def load_db():
    return
    # pylint: disable=import-outside-toplevel,invalid-name
    import json
    import os
    import tempfile

    def fix_tile(length, tiles):
        p = 1
        for x in tiles:
            p *= x
        assert length % p == 0
        return [length // p] + tiles

    TILE_LENS = [
        # spatial
        1,
        8,
        8,
        4,
        4,
        32,
        # reduction
        3,
        3,
        4,
        32,
    ]
    TILE_IDX = [
        # spatial
        5,
        7,
        9,
        11,
        13,
        15,
        # reduction
        17,
        19,
        21,
        23,
    ]
    VECTOR_LOAD_IDX = [40, 46]
    UNROLL_IDX = 49
    # pylint: enable=import-outside-toplevel,invalid-name
    with open(ARGS.as_db) as json_file:  # pylint: disable=W1514
        lines = [json.loads(i) for i in json_file.readlines()][:FIRST_NUM]
    as_decisions = []
    for trace_id, data in enumerate(lines):
        insts = data["i"][1][1]
        tiles = [
            # spatial
            insts[1][4],
            insts[2][4],
            insts[3][4],
            insts[4][4],
            insts[5][4],
            insts[6][4],
            # reduction
            insts[7][4],
            insts[8][4],
            insts[9][4],
            insts[10][4],
        ]
        tiles = [[i, fix_tile(len, tile)] for i, len, tile in zip(TILE_IDX, TILE_LENS, tiles)]
        vector_load = [
            min(3, insts[-10][4][0] - 1),
            min(3, insts[-5][4][0] - 1),
        ]
        vector_load = [[i, x] for i, x in zip(VECTOR_LOAD_IDX, vector_load)]
        unroll = [
            [
                UNROLL_IDX,
                {
                    0: 0,
                    16: 1,
                    64: 2,
                    512: 3,
                    1024: 4,
                }[int(insts[-1][-1].split("$")[1])],
            ]
        ]
        as_decisions.append(tiles + vector_load + unroll)
    with open(ARGS.ms_db) as json_file:  # pylint: disable=W1514
        line = json.loads(json_file.readline())
    assert line[1][0][0][51][0] == "EnterPostproc"
    line[1][0][0] = line[1][0][0][:51]

    with tempfile.TemporaryDirectory() as work_dir:
        path_tuning_record = os.path.join(work_dir, "records.json")
        with open(path_tuning_record, "w") as o_f:  # pylint: disable=W1514
            for decision in as_decisions:
                line[1][0][1] = decision
                o_f.write(json.dumps(line) + "\n")
        database = ms.database.JSONDatabase(
            path_workload=ARGS.ms_wkl,
            path_tuning_record=path_tuning_record,
        )

    prim_func = create_te_workload(ARGS.workload, 0)
    prim_func = prim_func.with_attr("global_symbol", "main")
    prim_func = prim_func.with_attr("tir.noalias", True)
    mod = tvm.ir.IRModule({"main": prim_func})
    ARGS.records = database.get_top_k(
        workload=database.commit_workload(mod),
        top_k=20000,
    )
    print("Done!!!")


@tvm._ffi.register_func("meta_schedule.inject_traces")  # pylint: disable=protected-access
def inject_traces(st: int, ed: int):  # pylint: disable=invalid-name
    result = []
    for i in range(st, ed):
        if i < FIRST_NUM:
            result.append(ARGS.records[i].trace)
    print(f"[{st}:{ed}): inject {len(result)} traces")
    return result


def main():
    load_db()
    # if ARGS.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
    #     alloc_repeat = 3
    # else:
    #     alloc_repeat = 1
    alloc_repeat = 1

    runner = ms.runner.RPCRunner(
        rpc_config=ARGS.rpc_config,
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        ),
        alloc_repeat=alloc_repeat,
        max_workers=ARGS.rpc_workers,
    )
    sch: Optional[tir.Schedule] = ms.tune_tir(
        mod=create_te_workload(ARGS.workload, 0),
        target=ARGS.target,
        config=ms.EvolutionarySearchConfig(
            num_trials_per_iter=64,
            num_trials_total=ARGS.num_trials,
            init_min_unmeasured=50,
        ),
        runner=runner,  # type: ignore
        task_name=ARGS.workload,
        work_dir=ARGS.work_dir,
        num_threads=cpu_count(),
    )
    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    main()

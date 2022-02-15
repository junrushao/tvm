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
"""Gradient Based Task Scheduler"""
import math

from typing import TYPE_CHECKING, List, Optional, Callable
from tvm._ffi import register_object

from tvm.ir import IRModule
from tvm.tir import Schedule
from ..measure_callback import MeasureCallback
from ..builder import Builder
from ..runner import Runner
from ..database import Database
from ..cost_model import CostModel
from .task_scheduler import TaskScheduler

from .. import _ffi_api

if TYPE_CHECKING:
    from ..tune_context import TuneContext


def derive_similarity_tag(log_base=1.618):
    def compute(mod: IRModule):
        ret = ""
        sch = Schedule(mod)
        for func in mod.get_global_vars:
            sref = sch.get_sref(sch.get_block(func))
            if (
                sref is not None
                and sref.stmt is not None
                and "meta_scheduler_task_scheduler_tag" in sref.stmt.annotations
            ):
                ret += sref.stmt.annotations["meta_scheduler_task_scheduler_tag"] + "_"
        if ret:
            flop_count = _ffi_api.TaskSchedulerFlopCount(mod)  # type: ignore # pylint: disable=no-member
            ret += "%d" % int(math.log(flop_count + 1, log_base))
        return ret

    return compute


@register_object("meta_schedule.GradientBased")
class GradientBased(TaskScheduler):
    """Gradient Based Task Scheduler"""

    def __init__(
        self,
        tasks: List["TuneContext"],
        builder: Builder,
        runner: Runner,
        database: Database,
        *,
        alpha: float = 0.2,
        beta: float = 2.0,
        backward_window_size: int = 3,
        seed: int = -1,
        task_weights: List[float] = None,
        objective_func: Callable[[List[float]], float] = None,
        tag_generation_func: Callable[[IRModule], str] = derive_similarity_tag(),
        cost_model: Optional[CostModel] = None,
        measure_callbacks: Optional[List[MeasureCallback]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        tasks : List[TuneContext]
            List of tasks to schedule.
        builder : Builder
            The builder.
        runner : Runner
            The runner.
        database : Database
            The database.
        alpha: float
            The parameter alpha to control gradient computation.
        beta: float
             The parameter beta to control gradient computation.
        backward_window_size: int
            The parameter to control backward window size.
        seed: int
            The random seed.
        task_weights: Optional[List[float]]
            The weights of each task.
        objective_func:
            The objective function for gradient optimization.
        tag_generation_func
            The function to generate similarity tag for workloads.
        cost_model: CostModel
            The cost model of the scheduler.
        measure_callbacks: Optional[List[MeasureCallback]]
            The list of measure callbacks of the scheduler.
        """
        if task_weights is None:
            task_weights = [1.0 for _ in tasks]
        assert len(task_weights) == len(
            tasks
        ), "The given task weights should be same length as tasks."
        if objective_func is None:
            objective_func = lambda l: sum([l[i] * w for i, w in enumerate(task_weights)])
        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerGradientBased,  # type: ignore # pylint: disable=no-member
            tasks,
            builder,
            runner,
            database,
            cost_model,
            measure_callbacks,
            task_weights,
            alpha,
            beta,
            backward_window_size,
            seed,
            objective_func,
            tag_generation_func,
        )

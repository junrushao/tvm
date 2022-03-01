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

from typing import TYPE_CHECKING, List, Optional
from tvm._ffi import register_object
from tvm._ffi.registry import register_func

from tvm.ir import IRModule
from ..measure_callback import MeasureCallback
from ..builder import Builder
from ..runner import Runner
from ..database import Database
from ..cost_model import CostModel
from .task_scheduler import TaskScheduler

from .. import _ffi_api

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_func("meta_schedule.task_scheduler.derive_similarity_tag")
def derive_similarity_tag(mod: IRModule, log_base: float = 1.618) -> str:
    """Get the tags for smilarity group creation

    Parameters
    ----------
    mod : IRModule
        The input workload.
    log_base : float
        The log base to normalize the flop count. Default natural (1.618).

    Return
    ------
    tag : str
        The generated similarity tag.
    """
    ret = ""
    for var in mod.get_global_vars():
        if "meta_scheduler_task_scheduler_tag" in mod[var].attrs:
            ret += mod[var].attrs.meta_scheduler_task_scheduler_tag + "_"
    if ret:
        flop_count = _ffi_api.TaskSchedulerFlopCount(mod)  # type: ignore # pylint: disable=no-member
        ret += "%d" % int(math.log(flop_count + 1, log_base))
    return ret


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
        objective_func_name: str = "meta_schedule.task_scheduler.objective_func",
        tag_generation_func_name: str = "meta_schedule.task_scheduler.derive_similarity_tag",
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
        alpha : float, default 0.2.
            The parameter alpha to control gradient computation.
        beta : float, default 2.0.
             The parameter beta to control gradient computation.
        backward_window_size : int, default 3.
            The parameter to control backward window size.
        seed : int, default -1, meaning random seed.
            The random seed.
        task_weights : Optional[List[float]], default None, meaning equal weight.
            The weights of each task.
        objective_func_name : str, default "meta_schedule.task_scheduler.objective_func"
            The name of objective function for gradient optimization.
        tag_generation_func_name : str,
            default "meta_schedule.task_scheduler.derive_similarity_tag"
            The name of function to generate similarity tag for workloads.
        cost_model : CostModel, default None.
            The cost model of the scheduler.
        measure_callbacks : Optional[List[MeasureCallback]], default None.
            The list of measure callbacks of the scheduler.
        """

        @register_func("meta_schedule.task_scheduler.objective_func")
        def weighted_sum(latency: List[float]) -> float:  # pylint: disable= unused-variable,
            """Get the weighted sum as objective function

            Parameters
            ----------
            latency : List[float]
                The current latency of each workload.

            Returns
            -------
            result : float
                The computed objective function value.
            """
            return sum([latency[i] * w for i, w in enumerate(self.task_weights)])

        if task_weights is None:
            task_weights = [1.0 for _ in tasks]
        self.task_weights = task_weights

        assert len(task_weights) == len(
            tasks
        ), "The given task weights should be same length as tasks."

        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerGradientBased,  # type: ignore # pylint: disable=no-member
            tasks,
            builder,
            runner,
            database,
            alpha,
            beta,
            backward_window_size,
            seed,
            task_weights,
            objective_func_name,
            tag_generation_func_name,
            cost_model,
            measure_callbacks,
        )

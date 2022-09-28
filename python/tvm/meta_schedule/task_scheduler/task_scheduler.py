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
"""Auto-tuning Task Scheduler"""
from typing import Callable, List, Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on

from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api
from ..builder import Builder, BuilderResult
from ..cost_model import CostModel
from ..database import Database
from ..logging import get_logger, make_logging_func
from ..measure_callback import MeasureCallback
from ..runner import Runner, RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext

logger = get_logger(__name__)  # pylint: disable=invalid-name


@register_object("meta_schedule.TaskRecord")
class TaskRecord(Object):
    """The running record of a task."""

    ctx: TuneContext
    task_weight: float
    flop: float
    is_terminated: bool
    build_error_count: int
    run_error_count: int
    measure_candidates: List[MeasureCandidate]
    builder_results: List[BuilderResult]
    runner_results: List[RunnerResult]


@register_object("meta_schedule.TaskScheduler")
class TaskScheduler(Object):
    """The abstract task scheduler interface."""

    tasks_: List[TaskRecord]
    measure_callbacks_: List[MeasureCallback]
    database_: Optional[Database]
    cost_model_: Optional[CostModel]
    remaining_tasks_: int

    TaskSchedulerType = Union["TaskScheduler", Literal["gradient", "round-robin"]]

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        return _ffi_api.TaskSchedulerNextTaskId(self)  # type: ignore # pylint: disable=no-member

    def join_running_task(self, task_id: int) -> List[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.

        Returns
        -------
        results : List[RunnerResult]
            The list of results.
        """
        return _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def tune(
        self,
        tasks: List[TuneContext],
        task_weights: List[float],
        max_trials_global: int,
        max_trials_per_task: int,
        num_trials_per_iter: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: List[MeasureCallback],
        database: Optional[Database],
        cost_model: Optional[CostModel],
    ) -> None:
        """Auto-tuning."""
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self,
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            num_trials_per_iter,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
        )

    def terminate_task(self, task_id: int) -> None:
        """Terminate the task

        Parameters
        ----------
        task_id : int
            The task id to be terminated.
        """
        _ffi_api.TaskSchedulerTerminateTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def touch_task(self, task_id: int) -> None:
        """Touch the task and update its status

        Parameters
        ----------
        task_id : int
            The task id to be checked.
        """
        _ffi_api.TaskSchedulerTouchTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def tuning_statistics(self) -> str:
        """Returns a human-readable string of the tuning statistics.

        Returns
        -------
        tuning_statistics : str
            The tuning statistics.
        """
        return _ffi_api.TaskSchedulerTuningStatistics(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: Literal["round-robin", "gradient"] = "gradient",
        *args,
        **kwargs,
    ) -> "TaskScheduler":
        """Create a task scheduler."""
        from . import (  # pylint: disable=import-outside-toplevel
            GradientBased,
            RoundRobin,
        )

        if kind == "round-robin":
            return RoundRobin(*args, **kwargs)  # type: ignore
        if kind == "gradient":
            return GradientBased(*args, **kwargs)
        raise ValueError(f"Unknown TaskScheduler name: {kind}")


create = TaskScheduler.create  # pylint: disable=invalid-name


@register_object("meta_schedule.PyTaskScheduler")
class _PyTaskScheduler(TaskScheduler):
    """
    A TVM object task scheduler to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyTaskScheduler
    """

    def __init__(
        self,
        f_next_task_id: Callable,
        f_join_running_task: Callable,
        f_tune: Callable,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerPyTaskScheduler,  # type: ignore # pylint: disable=no-member
            make_logging_func(logger),
            f_next_task_id,
            f_join_running_task,
            f_tune,
        )


class PyTaskScheduler:
    """
    An abstract task scheduler with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyTaskScheduler,
        "fields": [],
        "methods": ["next_task_id", "join_running_task", "tune"],
    }

    def __init__(self):
        ...

    def tune(
        self,
        tasks: List[TuneContext],
        task_weights: List[float],
        max_trials_global: int,
        max_trials_per_task: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: List[MeasureCallback],
        database: Optional[Database],
        cost_model: Optional[CostModel],
    ) -> None:
        """Auto-tuning."""
        # Using self._outer to replace the self pointer
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self._outer(),  # type: ignore # pylint: disable=no-member
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
        )

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        raise NotImplementedError

    def join_running_task(self, task_id: int) -> List[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.
        """
        # Using self._outer to replace the self pointer
        return _ffi_api.TaskSchedulerJoinRunningTask(self._outer(), task_id)  # type: ignore # pylint: disable=no-member

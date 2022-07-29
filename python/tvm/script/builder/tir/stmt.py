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
"""TVMScript TIR statements"""
from typing import List, Union

import numpy as np
from tvm._ffi import register_object as _register_object
from tvm.arith import Analyzer
from tvm.runtime import Object, convert
from tvm.runtime import ndarray as nd
from tvm.tir import Buffer, BufferRegion, IterVar, PrimExpr, StringImm, Var, Let, Ramp, IntImm
from tvm.tir.generic import cast

from .. import _ffi_api as _base_ffi_api
from . import _ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.AssertFrame")
class AssertFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.LetFrame")
class LetFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.AllocateFrame")
class AllocateFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        _base_ffi_api.FrameEnter(self)  # pylint: disable=no-member # type: ignore
        return self.buffer


@_register_object("script.builder.tir.AllocateConstFrame")
class AllocateConstFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        _base_ffi_api.FrameEnter(self)  # pylint: disable=no-member # type: ignore
        return self.buffer


@_register_object("script.builder.tir.LaunchThreadFrame")
class LaunchThreadFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.RealizeFrame")
class RealizeFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.AttrFrame")
class AttrFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.WhileFrame")
class WhileFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.IfFrame")
class IfFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.ThenFrame")
class ThenFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.ElseFrame")
class ElseFrame(TIRFrame):
    ...


def Assert(condition: PrimExpr, message: str) -> AssertFrame:
    return _ffi_api.AssertFrame(condition, message)  # pylint: disable=no-member # type: ignore


def let(var: Var, value: PrimExpr, body: PrimExpr = None) -> LetFrame:
    if body is None:
        return _ffi_api.LetFrame(var, value)  # pylint: disable=no-member # type: ignore
    return Let(var, value, body)


def allocate(
    extents: List[PrimExpr],
    dtype: str,
    scope: str = "",
    condition: PrimExpr = None,
    annotations=None,
) -> AllocateFrame:
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.AllocateFrame(
        extents, dtype, scope, condition, annotations
    )  # pylint: disable=no-member # type: ignore


def allocate_const(
    data: List[PrimExpr], dtype: str, extents: List[PrimExpr], annotations=None
) -> AllocateConstFrame:
    return _ffi_api.AllocateConstFrame(
        nd.array(np.asarray(data, dtype)), dtype, extents, annotations
    )  # pylint: disable=no-member # type: ignore


def launch_thread(iter_var: IterVar, extent: PrimExpr) -> LaunchThreadFrame:
    return _ffi_api.LaunchThreadFrame(iter_var, extent)  # pylint: disable=no-member # type: ignore


def realize(
    buffer_slice: BufferRegion, storage_scope: str, condition: PrimExpr = True
) -> RealizeFrame:
    return _ffi_api.RealizeFrame(
        buffer_slice, storage_scope, condition
    )  # pylint: disable=no-member # type: ignore


def attr(node: Object, attr_key: str, value: Union[PrimExpr, str]) -> AttrFrame:
    node = convert(node)
    value = convert(value)
    return _ffi_api.AttrFrame(node, attr_key, value)  # pylint: disable=no-member # type: ignore


def while_(condition: PrimExpr) -> WhileFrame:
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.WhileFrame(condition)  # pylint: disable=no-member # type: ignore


def if_(condition: PrimExpr) -> IfFrame:
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.IfFrame(condition)  # pylint: disable=no-member # type: ignore


def then_() -> ThenFrame:
    return _ffi_api.ThenFrame()  # pylint: disable=no-member # type: ignore


def else_() -> ElseFrame:
    return _ffi_api.ElseFrame()  # pylint: disable=no-member # type: ignore


def env_thread(thread_tag: str) -> IterVar:
    return _ffi_api.EnvThread(thread_tag)  # pylint: disable=no-member # type: ignore


def buffer_store(buffer: Buffer, value: PrimExpr, indices: List[Union[PrimExpr, slice]]) -> None:
    expr_indices = []
    for index in indices:
        if isinstance(index, slice):
            step = 1 if index.step is None else index.step
            lanes = Analyzer().simplify((index.stop - index.start + step - 1) // step)
            if lanes == 1:
                expr_indices.append(index.start)
            else:
                expr_indices.append(Ramp(index.start, step, int(lanes)))
        else:
            expr_indices.append(index)
    if isinstance(value, bool) and buffer.dtype == "bool":
        value = IntImm("bool", value)
    return _ffi_api.BufferStore(
        buffer, value, expr_indices
    )  # pylint: disable=no-member # type: ignore


def prefetch(buffer: Buffer, indices: List[PrimExpr]) -> None:
    return _ffi_api.Prefetch(buffer, indices)  # pylint: disable=no-member # type: ignore


def evaluate(value: PrimExpr) -> None:
    if isinstance(value, str):
        value = StringImm(value)
    return _ffi_api.Evaluate(value)  # pylint: disable=no-member # type: ignore

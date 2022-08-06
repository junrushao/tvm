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
"""IRBuilder for TIR"""
import inspect
from numbers import Integral
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tvm.ir import Range, Type
from tvm.runtime import convert

from . import _ffi_ir_builder_api as _ffi_api
from . import ir_builder_frame as frame
from .buffer import Buffer
from .expr import (
    BufferLoad,
    CommReducer,
    IntImm,
    IterVar,
    Let,
    PrimExpr,
    StringImm,
    Var,
)
from .generic import cast  # pylint: disable=unused-import
from .stmt import BufferRegion, type_annotation  # pylint: disable=unused-import


def buffer_decl(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    align=0,
    offset_factor=0,
    buffer_type="",
    axis_separators=None,
) -> Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    return _ffi_api.BufferDecl(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        "",
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def ptr(dtype, storage_scope="global"):
    return _ffi_api.Ptr(dtype, storage_scope)  # pylint: disable=no-member # type: ignore


def block(name: str = "", no_realize: bool = False) -> frame.BlockFrame:
    return _ffi_api.Block(name, no_realize)  # pylint: disable=no-member # type: ignore


def init() -> frame.BlockInitFrame:
    return _ffi_api.Init()  # pylint: disable=no-member # type: ignore


def where(predicate) -> None:
    if isinstance(predicate, bool):
        predicate = IntImm("bool", predicate)
    _ffi_api.Where(predicate)  # pylint: disable=no-member # type: ignore


def reads(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore
        else:
            buffer_slices = [buffer_slices[0]]  # type: ignore
    else:
        buffer_slices = list(buffer_slices)  # type: ignore
    _ffi_api.Reads(buffer_slices)  # pylint: disable=no-member # type: ignore


def writes(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore
    _ffi_api.Writes(buffer_slices)  # pylint: disable=no-member # type: ignore


def block_attr(attrs: Dict[str, Any]) -> None:
    return _ffi_api.BlockAttrs(attrs)  # pylint: disable=no-member # type: ignore


def alloc_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    return _ffi_api.AllocBuffer(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def _as_range(dom) -> Range:
    if isinstance(dom, Range):
        return dom
    if isinstance(dom, (list, tuple)):
        return Range(dom[0], dom[1])
    return Range(0, dom)


class axis:  # pylint: disable=invalid-name
    @staticmethod
    def spatial(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisSpatial(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def reduce(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisReduce(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def scan(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisScan(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def opaque(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisOpaque(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def remap(kinds, bindings, dtype="int32") -> Union[List[IterVar], IterVar]:
        iter_vars = _ffi_api.AxisRemap(  # pylint: disable=no-member # type: ignore
            kinds, bindings, dtype
        )
        return iter_vars[0] if len(iter_vars) == 1 else iter_vars

    S = spatial  # pylint: disable=invalid-name
    R = reduce  # pylint: disable=invalid-name


def serial(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Serial(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def parallel(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Parallel(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def vectorized(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Vectorized(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def unroll(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Unroll(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def thread_binding(
    start,
    stop=None,
    thread=None,
    *,
    annotations=None,
) -> frame.ForFrame:
    if thread is None:
        if not isinstance(stop, str):
            raise ValueError("Thread cannot be None for thread_binding")
        thread = stop
        stop = start
        start = 0
    elif stop is None:
        stop = start
        start = 0
    return _ffi_api.ThreadBinding(  # pylint: disable=no-member # type: ignore
        start, stop, thread, annotations
    )


def grid(*extents) -> frame.ForFrame:
    return _ffi_api.Grid(extents)  # pylint: disable=no-member # type: ignore


def prim_func() -> frame.PrimFuncFrame:
    return _ffi_api.PrimFunc()  # pylint: disable=no-member # type: ignore


def arg(name, obj):
    return _ffi_api.Arg(name, obj)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> str:
    return _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, Any]) -> None:
    return _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def func_ret(ret_type) -> Type:
    return _ffi_api.FuncRet(ret_type)  # pylint: disable=no-member # type: ignore


def match_buffer(
    param,
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="global",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    return _ffi_api.MatchBuffer(  # pylint: disable=no-member # type: ignore
        param,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def preflattened_buffer(
    postflattened,
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="global",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> None:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    _ffi_api.PreflattenedBuffer(  # pylint: disable=no-member # type: ignore
        postflattened,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def Assert(condition: PrimExpr, message: str) -> frame.AssertFrame:  # pylint: disable=invalid-name
    return _ffi_api.Assert(condition, message)  # pylint: disable=no-member # type: ignore


def let(
    v: Var,
    value: PrimExpr,
    body: PrimExpr = None,
) -> frame.LetFrame:
    if body is None:
        return _ffi_api.Let(v, value)  # pylint: disable=no-member # type: ignore
    return Let(v, value, body)


def allocate(
    extents: List[PrimExpr],
    dtype: str,
    scope: str = "",
    condition: PrimExpr = None,
    annotations=None,
) -> frame.AllocateFrame:
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.Allocate(  # pylint: disable=no-member # type: ignore
        extents, dtype, scope, condition, annotations
    )


def allocate_const(
    data: List[PrimExpr],
    dtype: str,
    extents: List[PrimExpr],
    annotations=None,
) -> frame.AllocateConstFrame:
    from tvm.runtime.ndarray import array  # pylint: disable=import-outside-toplevel

    return _ffi_api.AllocateConst(  # pylint: disable=no-member # type: ignore
        array(np.asarray(data, dtype)), dtype, extents, annotations
    )


def realize(
    buffer_slice: BufferRegion,
    storage_scope: str,
    condition: PrimExpr = True,
) -> frame.RealizeFrame:
    return _ffi_api.Realize(  # pylint: disable=no-member # type: ignore
        buffer_slice, storage_scope, condition
    )


def attr(node: Any, attr_key: str, value: Union[PrimExpr, str]) -> frame.AttrFrame:
    node = convert(node)
    value = convert(value)
    return _ffi_api.Attr(node, attr_key, value)  # pylint: disable=no-member # type: ignore


def While(condition: PrimExpr) -> frame.WhileFrame:  # pylint: disable=invalid-name
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.While(condition)  # pylint: disable=no-member # type: ignore


def If(condition: PrimExpr) -> frame.IfFrame:  # pylint: disable=invalid-name
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.If(condition)  # pylint: disable=no-member # type: ignore


def Then() -> frame.ThenFrame:  # pylint: disable=invalid-name
    return _ffi_api.Then()  # pylint: disable=no-member # type: ignore


def Else() -> frame.ElseFrame:  # pylint: disable=invalid-name
    return _ffi_api.Else()  # pylint: disable=no-member # type: ignore


def launch_thread(
    iter_var: IterVar,  # pylint: disable=redefined-outer-name
    extent: PrimExpr,
) -> frame.LaunchThreadFrame:
    return _ffi_api.LaunchThread(iter_var, extent)  # pylint: disable=no-member # type: ignore


def env_thread(thread_tag: str) -> IterVar:
    return _ffi_api.EnvThread(thread_tag)  # pylint: disable=no-member # type: ignore


def buffer_store(buffer: Buffer, value: PrimExpr, indices: List[Union[PrimExpr, slice]]) -> None:
    from tvm.arith import Analyzer  # pylint: disable=import-outside-toplevel
    from tvm.tir import Ramp  # pylint: disable=import-outside-toplevel

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
    return _ffi_api.BufferStore(  # pylint: disable=no-member # type: ignore
        buffer, value, expr_indices
    )


def prefetch(buffer: Buffer, indices: List[PrimExpr]) -> None:
    return _ffi_api.Prefetch(buffer, indices)  # pylint: disable=no-member # type: ignore


def evaluate(value: PrimExpr) -> None:
    if isinstance(value, str):
        value = StringImm(value)
    return _ffi_api.Evaluate(value)  # pylint: disable=no-member # type: ignore


def int8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int8(expr)  # pylint: disable=no-member # type: ignore


def int16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int16(expr)  # pylint: disable=no-member # type: ignore


def int32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32(expr)  # pylint: disable=no-member # type: ignore


def int64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int64(expr)  # pylint: disable=no-member # type: ignore


def uint8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt8(expr)  # pylint: disable=no-member # type: ignore


def uint16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt16(expr)  # pylint: disable=no-member # type: ignore


def uint32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt32(expr)  # pylint: disable=no-member # type: ignore


def uint64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt64(expr)  # pylint: disable=no-member # type: ignore


def float8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float8(expr)  # pylint: disable=no-member # type: ignore


def float16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float16(expr)  # pylint: disable=no-member # type: ignore


def float32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float32(expr)  # pylint: disable=no-member # type: ignore


def float64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float64(expr)  # pylint: disable=no-member # type: ignore


def boolean(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Boolean(expr)  # pylint: disable=no-member # type: ignore


def handle(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Handle(expr)  # pylint: disable=no-member # type: ignore


def void(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Void(expr)  # pylint: disable=no-member # type: ignore


def min(a, b):  # pylint: disable=redefined-builtin
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api.min(a, b)  # pylint: disable=no-member # type: ignore


def max(a, b):  # pylint: disable=redefined-builtin
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api.max(a, b)  # pylint: disable=no-member # type: ignore


def var(dtype, name="") -> Var:
    return Var(name, dtype)  # pylint: disable=no-member # type: ignore


def iter_var(v, dom, iter_type, thread_tag):
    iter_type = getattr(IterVar, iter_type)
    return IterVar(dom, v, iter_type, thread_tag)


def comm_reducer(combiner, identity):
    """Create a CommReducer from lambda inputs/outputs and the identities"""
    params = inspect.signature(combiner).parameters
    num_args = len(params)
    args = []
    for name, i in zip(params.keys(), identity + identity):
        args.append(Var(name, i.dtype))
    res = combiner(*args)
    if not isinstance(res, tuple):
        res = (res,)
    return CommReducer(args[: num_args // 2], args[num_args // 2 :], res, identity)


def llvm_lookup_intrinsic_id(name):
    # pylint: disable=import-outside-toplevel
    from tvm.target.codegen import llvm_lookup_intrinsic_id as f

    # pylint: enable=import-outside-toplevel
    return f(name)


def Select(condition, true_value, false_value):  # pylint: disable=invalid-name
    from tvm.tir import Select as _Select  # pylint: disable=import-outside-toplevel

    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _Select(condition, true_value, false_value)

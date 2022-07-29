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
"""TVM Script TIR Buffer"""
from numbers import Integral

from tvm._ffi import register_object as _register_object
from tvm.ir import Array, PrimExpr, Range, PrimType
from tvm.runtime import DataType, Object
from tvm.tir import BufferLoad, BufferRegion, IntImm, Var, IterVar
from tvm import tir

from . import _ffi_api


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
) -> tir.Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    return _ffi_api.BufferDecl(
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


class BufferProxy:
    def __call__(
        self,
        shape,
        dtype="float32",
        name="buffer",
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> tir.Buffer:
        shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
        return _ffi_api.BufferDecl(
            shape,
            dtype,
            name,
            data,
            strides,
            elem_offset,
            scope,
            align,
            offset_factor,
            buffer_type,
            axis_separators,
        )

    def __getitem__(self, keys) -> tir.Buffer:
        return self(*keys)  # pylint: disable=no-member # type: ignore


def var(dtype, name="") -> Var:
    return Var(name, dtype)  # pylint: disable=no-member # type: ignore


def iter_var(var, dom, iter_type, thread_tag):
    iter_type = getattr(IterVar, iter_type)
    return IterVar(dom, var, iter_type, thread_tag)


class Ptr_:
    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) == 1:
            args = (args[0], "global")
        return _ffi_api.Ptr(PrimType(args[0]().dtype), args[1])


Buffer = BufferProxy()
Ptr = Ptr_()

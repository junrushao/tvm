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

import inspect
from typing import Callable, Union

from tvm.tir import Buffer, PrimFunc
from tvm.tir.ir_builder import buffer_decl, ptr

from ..entry import parse
from ..utils import inspect_function_capture


def _is_defined_in_class(frames):
    if len(frames) > 2:
        maybe_class_frame = frames[2]
        statement_list = maybe_class_frame[4]
        first_statement = statement_list[0]
        line = first_statement.strip()
        if line.startswith("class "):
            return True
        if line.startswith("@") and "ir_module" in line:
            return True
    return False


def prim_func(f: Callable) -> Union[PrimFunc, Callable]:
    if not inspect.isfunction(f):
        raise TypeError(f"Expect a function, but got: {f}")
    if _is_defined_in_class(inspect.stack()):
        return f
    return parse(f, inspect_function_capture(f))


setattr(prim_func, "dispatch_token", "tir")


class BufferProxy:
    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> Buffer:
        return buffer_decl(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    def __getitem__(self, keys) -> Buffer:
        return self(*keys)  # pylint: disable=no-member # type: ignore


class PtrProxy:
    def __call__(self, dtype, storage_scope="global"):
        if callable(dtype):
            dtype = dtype().dtype
        return ptr(dtype, storage_scope)  # pylint: disable=no-member # type: ignore

    def __getitem__(self, keys):
        return self(*keys)


Buffer = BufferProxy()
Ptr = PtrProxy()
